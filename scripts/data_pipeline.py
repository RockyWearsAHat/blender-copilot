#!/usr/bin/env python3
"""Unified data pipeline: download → extract → render → VL label → cache.

Each item flows through the full pipeline in isolation so failures are
contained.  Downloads run in a parallel pool to saturate available bandwidth;
extraction/render/label/cache run per-item in worker threads.

Flow per item
─────────────
  1. Source yields item descriptor  {source, url, raw_path, metadata}
  2. Download raw file  (parallel, pool_size workers, rate-limit retry)
  3. Extract → JSON via Blender headless  (blend_extractor.py)
  4. Quick multi-view render → PNG images  (blender_render.py)
  5. Feed everything into qwen2.5vl:32b → label
  6. Build .pt cache entry  (mesh_tokenizer + bpe_tokenizer)
  7. Cleanup raw file (unless --keep-raw)

Usage
─────
  python scripts/data_pipeline.py                           # full pipeline, all sources (scraping ON)
  python scripts/data_pipeline.py --local                   # disk-only; reprocess existing .blend files
  python scripts/data_pipeline.py --sources blendswap objaverse
  python scripts/data_pipeline.py --test                    # 1 item per source, then stop
  python scripts/data_pipeline.py --pull-max 500            # stop after 500 items
  python scripts/data_pipeline.py --pull-behave concurrent  # default: all sources in parallel
  python scripts/data_pipeline.py --pull-behave batch       # one source at a time
  python scripts/data_pipeline.py --workers 24              # download parallelism
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, cast

import requests
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.mesh_tokenizer import MeshTokenizer
from processing.bpe_tokenizer import BPETokenizer
from processing.generate_synthetic import normalize_mesh
from processing.labeler_smart import generate_smart_label, compute_bbox_aspect
from scripts.rebuild_cache import decimate_mesh

logger = logging.getLogger("data_pipeline")

BASE        = Path(__file__).parent.parent
RAW_DIR     = BASE / "data" / "raw"
PROC_DIR    = BASE / "data" / "processed"
CACHE_DIR   = PROC_DIR / ".mesh_cache"
RENDER_DIR  = BASE / "data" / "renders"
PARSE_MARKER_NAME = ".parsed_complete.json"
BLENDER_EXE = "/Applications/Blender.app/Contents/MacOS/Blender"
EXTRACTOR      = BASE / "processing" / "blend_extractor.py"
RENDERER       = BASE / "processing" / "blender_render.py"
SCENE_RENDERER = BASE / "processing" / "blender_scene_render.py"

PIPELINE_SCENE_FULL_SIZE = max(0, int(os.environ.get("PIPELINE_SCENE_FULL_SIZE", "0")))
PIPELINE_SCENE_MAX_SAMPLES = max(0, int(os.environ.get("PIPELINE_SCENE_MAX_SAMPLES", "0")))
PIPELINE_VIEWPORT_SIZE = max(512, int(os.environ.get("PIPELINE_VIEWPORT_SIZE", "1024")))
PIPELINE_RENDER_TIMEOUT = max(600, int(os.environ.get("PIPELINE_RENDER_TIMEOUT", "1200")))
VL_QUALITY_MIN_FACES = max(0, int(os.environ.get("VL_QUALITY_MIN_FACES", "0")))

OLLAMA_URL  = "http://localhost:11434"
VL_MODEL    = "qwen2.5vl:32b"

ALL_SOURCES = ["blender_official", "blendswap", "github", "objaverse",
               "objaverse_xl", "open3dlab", "smutbase", "youtube"]

# Auto-detect default workers: network-bound, so scale with CPU count
# Blender extraction is the bottleneck (~1 process per worker), keep reasonable
_DEFAULT_WORKERS = min(24, max(4, (os.cpu_count() or 4) * 2))

# ── Shared tokenisers (loaded once, shared across threads) ────────────────

_tok_lock   = threading.Lock()
_mesh_tok: MeshTokenizer | None  = None
_bpe_tok:  BPETokenizer  | None  = None

# Semaphore: only 1 Qwen VL call at a time.
# Qwen 32B with 15 images takes ~3-5 min; parallel calls all queue up and timeout.
_vl_sem = threading.Semaphore(1)

HIGH_POLY_FACE_THRESHOLD = 50_000

_cfg_lock = threading.Lock()
_cfg_cache: dict | None = None

_fingerprint_lock = threading.Lock()
_seen_fingerprints: set[str] = set()
_fingerprint_store = CACHE_DIR / "_seen_fingerprints.txt"


def _load_seen_fingerprints() -> None:
    with _fingerprint_lock:
        if _seen_fingerprints:
            return
        try:
            if _fingerprint_store.exists():
                for line in _fingerprint_store.read_text().splitlines():
                    val = line.strip()
                    if val:
                        _seen_fingerprints.add(val)
        except Exception as e:
            logger.debug(f"fingerprint load failed: {e}")


def _register_fingerprint(fp: str) -> bool:
    """Return True if new, False if already seen (global cross-source dedup)."""
    with _fingerprint_lock:
        if fp in _seen_fingerprints:
            return False
        _seen_fingerprints.add(fp)
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with _fingerprint_store.open("a") as f:
                f.write(fp + "\n")
        except Exception as e:
            logger.debug(f"fingerprint persist failed: {e}")
        return True


def _mesh_fingerprint(json_data: dict) -> str:
    objects = json_data.get("objects", [json_data]) if isinstance(json_data, dict) else [json_data]
    total_v = 0
    total_f = 0
    bbox_vals: list[float] = []
    for obj in objects:
        mesh = obj.get("mesh", {})
        verts = mesh.get("vertices", [])
        faces = mesh.get("faces", [])
        total_v += len(verts)
        total_f += len(faces)
        if verts:
            try:
                xs = [float(v[0]) for v in verts]
                ys = [float(v[1]) for v in verts]
                zs = [float(v[2]) for v in verts]
                bbox_vals.extend([
                    round(min(xs), 4), round(max(xs), 4),
                    round(min(ys), 4), round(max(ys), 4),
                    round(min(zs), 4), round(max(zs), 4),
                ])
            except Exception:
                pass
    bbox_hash = hashlib.sha1(json.dumps(bbox_vals, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{total_v}_{total_f}_{bbox_hash}"


def _get_tokenizers():
    global _mesh_tok, _bpe_tok
    with _tok_lock:
        if _mesh_tok is None:
            cfg = _load_pipeline_config()
            token_cfg = cfg.get("tokenization", {}) if isinstance(cfg, dict) else {}
            unified_cfg = cfg.get("unified", {}) if isinstance(cfg, dict) else {}
            geom_cfg = unified_cfg.get("geometry", {}) if isinstance(unified_cfg, dict) else {}

            token_faces = int(token_cfg.get("max_faces", 0) or 0)
            max_seq = int(geom_cfg.get("max_seq_length", 0) or 0)
            seq_faces = max(0, (max_seq - 2) // 9)
            default_budget = max(20_000, token_faces, seq_faces)
            cache_face_budget = int(os.environ.get("CACHE_FACE_BUDGET", default_budget))
            if cache_face_budget < 2_048:
                cache_face_budget = 2_048

            _mesh_tok = MeshTokenizer(
                vocab_size=8192,
                coord_range=(-1.0, 1.0),
                max_faces=cache_face_budget,
            )
            _bpe_tok  = BPETokenizer.load(str(BASE / "data/datasets/geometry/bpe_tokenizer"))
    return _mesh_tok, _bpe_tok


def _load_pipeline_config() -> dict:
    global _cfg_cache
    with _cfg_lock:
        if _cfg_cache is not None:
            return cast(dict, _cfg_cache)
        cfg_path = BASE / "config.yaml"
        try:
            _cfg_cache = yaml.safe_load(cfg_path.read_text()) or {}
        except Exception:
            _cfg_cache = {}
        return cast(dict, _cfg_cache)


def _training_max_tokens() -> int:
    cfg = _load_pipeline_config()
    unified_cfg = cfg.get("unified", {}) if isinstance(cfg, dict) else {}
    geom_cfg = unified_cfg.get("geometry", {}) if isinstance(unified_cfg, dict) else {}
    max_seq = int(geom_cfg.get("max_seq_length", 36_002) or 36_002)
    return max(1_024, max_seq)


def _source_marker_path(source_key: str) -> Path:
    return RAW_DIR / source_key / PARSE_MARKER_NAME


def _source_is_parse_marked(source_key: str) -> bool:
    marker = _source_marker_path(source_key)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text())
        if isinstance(payload, dict) and payload.get("freeze_download", True) is False:
            return False
    except Exception:
        pass
    return True


# ── Source adapters ───────────────────────────────────────────────────────
# Each adapter is a generator that yields item dicts:
#   {
#     "source":    str,          # source key
#     "source_url": str,         # human URL of the listing page (for VL label)
#     "raw_path":  Path | None,  # pre-existing file, or None to download
#     "download_url": str | None,# URL to download if raw_path is None
#     "metadata":  dict,         # name, description, tags, categories, ...
#     "raw_dir":   Path,         # where to save downloaded file
#   }


def _write_sidecar(blend_path: Path, meta: dict) -> None:
    """Persist scraper metadata alongside a .blend as a .blend.meta.json sidecar.
    This ensures future disk-scan runs (scraping disabled) still have full
    listing metadata: title, description, tags, source URL, creator, etc.
    """
    try:
        sidecar = blend_path.parent / (blend_path.name + ".meta.json")
        sidecar.write_text(json.dumps(meta, indent=2, default=str))
    except Exception as e:
        logger.debug(f"_write_sidecar failed for {blend_path.name}: {e}")


def _iter_blender_official(raw_dir: Path, proc_dir: Path,
                           download: bool = True) -> Iterator[dict]:
    if not download:
        return
    from scrapers.blender_official import crawl_apache_directory
    base_url = "https://download.blender.org/demo/"
    seen = {p.stem for p in proc_dir.glob("*.json")}
    for url in crawl_apache_directory(base_url, extensions={".blend"}, max_depth=4):
        stem = Path(url.split("/")[-1]).stem
        if stem in seen:
            continue
        yield {
            "source": "blender_official",
            "source_url": url,
            "raw_path": None,
            "download_url": url,
            "metadata": {"name": stem.replace("_", " ").replace("-", " ")},
            "raw_dir": raw_dir / "blender_official",
        }


def _iter_blendswap(raw_dir: Path, proc_dir: Path,
                    download: bool = True) -> Iterator[dict]:
    from scrapers.blendswap_scraper import (
        CATEGORIES, create_session, get_listing_items, get_blend_detail,
    )
    from scrapers.smutbase_scraper import _extract_blend_from_archive
    seen = {p.stem for p in proc_dir.glob("*.json")}
    out_dir = raw_dir / "blendswap"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Disk-scan: yield .blend files kept from previous runs (--keep-raw)
    for blend_path in out_dir.rglob("*.blend"):
        if blend_path.stem in seen:
            continue
        mp = blend_path.parent / (blend_path.name + ".meta.json")
        m = json.loads(mp.read_text()) if mp.exists() else {}
        yield {
            "source":     "blendswap",
            "source_url": m.get("url", "https://blendswap.com"),
            "raw_path":   blend_path,
            "download_url": None,
            "metadata": {
                "name":        m.get("title", blend_path.stem.replace("_", " ")),
                "description": m.get("description", ""),
                "tags":        m.get("tags", []),
                "creator":     m.get("creator", ""),
                "license":     m.get("license", ""),
            },
            "raw_dir": out_dir,
        }

    if not download:
        return

    try:
        session = create_session()
    except Exception:
        session = requests.Session()

    for cat_name in CATEGORIES:
        try:
            for item in get_listing_items(session, cat_name, CATEGORIES[cat_name]):
                stem = re.sub(r'[^\w]', '_', item.get("title", item.get("id", "")))
                if stem and stem in seen:
                    continue
                # Enrich with download URL if missing
                dl_url = item.get("download_url")
                if not dl_url:
                    try:
                        detail = get_blend_detail(item["url"], session)
                        if detail:
                            item.update(detail)
                            dl_url = item.get("download_url")
                    except Exception:
                        pass
                if not dl_url:
                    continue

                # Download using authenticated session (returns ZIP or .blend)
                blend_id = item.get("id", re.sub(r'[^\w]', '_', item.get("title","u")))
                blend_path: Path | None = None
                blend_check = out_dir / f"{blend_id}.blend"
                zip_check   = out_dir / f"{blend_id}.zip"
                if blend_check.exists():
                    blend_path = blend_check
                elif not zip_check.exists():
                    try:
                        resp = session.get(dl_url, stream=True, timeout=120)
                        resp.raise_for_status()
                        data = resp.content
                        ct = resp.headers.get("content-type", "")
                        cd = resp.headers.get("content-disposition", "")
                        # Sniff magic bytes to determine real type
                        magic = data[:4]
                        if magic == b'BLEN':
                            # Raw .blend file
                            archive_path = out_dir / f"{blend_id}.blend"
                            archive_path.write_bytes(data)
                            blend_path = archive_path
                        elif magic[:2] == b'PK':
                            # ZIP file
                            archive_path = out_dir / f"{blend_id}.zip"
                            archive_path.write_bytes(data)
                        elif data[:5] in (b'<!DOC', b'<html', b'<HTML'):
                            # HTML response — not logged in or redirect
                            logger.debug(f"blendswap {blend_id}: got HTML response (not logged in?)")
                            continue
                        else:
                            # Unknown — try to guess from content-type
                            cd_match = re.search(r'filename=["\']?([^"\';\s]+)', cd)
                            if cd_match:
                                ext = Path(cd_match.group(1)).suffix.lower() or ".zip"
                            elif "zip" in ct:
                                ext = ".zip"
                            else:
                                ext = ".blend"
                            archive_path = out_dir / f"{blend_id}{ext}"
                            archive_path.write_bytes(data)
                    except Exception as e:
                        logger.debug(f"blendswap download {blend_id}: {e}")
                        continue

                # Extract .blend if archive (applies to zip_check path too)
                if blend_path is None:
                    archive_path = zip_check if zip_check.exists() else out_dir / f"{blend_id}.zip"
                    if archive_path.exists() and archive_path.suffix.lower() in (".zip", ".rar", ".7z"):
                        extracted = _extract_blend_from_archive(archive_path, out_dir)
                        if extracted:
                            blend_path = extracted[0]
                        archive_path.unlink(missing_ok=True)

                if not blend_path or not blend_path.exists():
                    continue

                _write_sidecar(blend_path, item)
                yield {
                    "source":     "blendswap",
                    "source_url": item.get("url", "https://blendswap.com"),
                    "raw_path":   blend_path,
                    "download_url": None,
                    "metadata": {
                        "name":        item.get("title", blend_path.stem.replace("_", " ")),
                        "description": item.get("description", ""),
                        "tags":        item.get("tags", []),
                        "creator":     item.get("creator", ""),
                        "license":     item.get("license", ""),
                    },
                    "raw_dir": out_dir,
                }
        except Exception as e:
            logger.debug(f"blendswap cat {cat_name}: {e}")


def _iter_objaverse(raw_dir: Path, proc_dir: Path,
                    download: bool = True) -> Iterator[dict]:
    if not download:
        return
    try:
        import objaverse
        annotations = objaverse.load_annotations()
    except Exception:
        logger.warning("objaverse package not available; skipping source")
        return
    seen_ids = {p.stem for p in proc_dir.glob("*.json")}
    # Sort by likeCount desc so most-popular models come first.
    # Annotations may be a plain dict (uid→ann) or a DataFrame-like object.
    try:
        items = sorted(
            annotations.items(),
            key=lambda kv: (kv[1].get("likeCount") or kv[1].get("viewCount") or 0),
            reverse=True,
        )
    except Exception:
        items = list(annotations.items())
    for uid, ann in items:
        if uid in seen_ids:
            continue
        dl_url = ann.get("glb_url") or ann.get("thumbnail_url")
        if not dl_url:
            continue   # skip items with no downloadable asset
        yield {
            "source":       "objaverse",
            "source_url":   f"https://sketchfab.com/3d-models/{uid}",
            "raw_path":     None,
            "download_url": dl_url,
            "metadata": {
                "name":        ann.get("name", uid),
                "description": ann.get("description", ""),
                "tags":        ann.get("tags", []),
                "categories":  ann.get("categories", []),
                "likeCount":   ann.get("likeCount", 0),
                "viewCount":   ann.get("viewCount", 0),
            },
            "raw_dir": raw_dir / "objaverse",
        }


def _iter_github(raw_dir: Path, proc_dir: Path,
                  download: bool = True) -> Iterator[dict]:
    """Yield .blend files scraped by `python run.py scrape --sources github`.
    GitHub files are scraped separately; this iterator does disk-scan only.
    Metadata comes from .blend.meta.json sidecars written by the scraper."""
    seen = {p.stem for p in proc_dir.glob("*.json")}
    out = raw_dir / "github"
    out.mkdir(parents=True, exist_ok=True)
    for blend_path in out.rglob("*.blend"):
        if blend_path.stem in seen:
            continue
        meta_path = blend_path.parent / (blend_path.name + ".meta.json")
        m = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        repo = m.get("repo", "")
        file_path = m.get("file_path", "")
        src_url = (
            m.get("html_url")
            or (f"https://github.com/{repo}/blob/HEAD/{file_path}" if repo else "")
            or "https://github.com"
        )
        yield {
            "source":       "github",
            "source_url":   src_url,
            "raw_path":     blend_path,
            "download_url": None,
            "metadata": {
                "name":        blend_path.stem.replace("__", " / ").replace("_", " ").strip(),
                "description": m.get("description", ""),
                "tags":        m.get("topics", []),
                "repo":        repo,
                "license":     m.get("license", ""),
                "stars":       m.get("stars", 0),
                "file_path":   file_path,
            },
            "raw_dir": out,
        }


def _iter_smutbase(raw_dir: Path, proc_dir: Path,
                   download: bool = True) -> Iterator[dict]:
    from scrapers.smutbase_scraper import (
        create_session as smut_session, get_listing_page,
        get_project_details, download_project_file, SITES,
    )
    seen = {p.stem for p in proc_dir.glob("*.json")}
    out_dir = raw_dir / "smutbase"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Disk-scan: yield .blend files kept from previous runs (--keep-raw)
    for blend_path in out_dir.rglob("*.blend"):
        if blend_path.stem in seen:
            continue
        mp = blend_path.parent / (blend_path.name + ".meta.json")
        m = json.loads(mp.read_text()) if mp.exists() else {}
        yield {
            "source":     "smutbase",
            "source_url": m.get("url", "https://smutba.se"),
            "raw_path":   blend_path,
            "download_url": None,
            "metadata": {
                "name":        m.get("title", blend_path.stem.replace("_", " ")),
                "description": m.get("description", ""),
                "tags":        m.get("tags", []),
                "creator":     m.get("creator", ""),
                "downloads":   m.get("downloads", 0),
                "license":     m.get("license", ""),
            },
            "raw_dir": out_dir,
        }

    if not download:
        return

    try:
        session = smut_session("smutbase")
        base_url = SITES["smutbase"]["base_url"]
        for page in range(1, 6):
            projects = get_listing_page(session, base_url, page=page,
                                        software_tag="blender",
                                        sort_by="popular")
            if not projects:
                break
            for proj in projects:
                stem = re.sub(r'[^\w]', '_', proj.get("title", ""))[:60]
                if stem and stem in seen:
                    continue
                detail = get_project_details(session, proj["url"],
                                             proj.get("project_id", ""))
                if not detail:
                    continue
                proj.update(detail)
                blend_path = download_project_file(session, proj, out_dir)
                if not blend_path or not Path(blend_path).exists():
                    continue
                blend_path = Path(blend_path)
                _write_sidecar(blend_path, proj)
                yield {
                    "source":     "smutbase",
                    "source_url": proj.get("url", "https://smutba.se"),
                    "raw_path":   blend_path,
                    "download_url": None,
                    "metadata": {
                        "name":        proj.get("title", blend_path.stem.replace("_", " ")),
                        "description": proj.get("description", ""),
                        "tags":        proj.get("tags", []),
                        "creator":     proj.get("creator", ""),
                        "downloads":   proj.get("downloads", 0),
                        "license":     proj.get("license", ""),
                    },
                    "raw_dir": out_dir,
                }
    except Exception as e:
        logger.debug(f"smutbase: {e}")


def _iter_open3dlab(raw_dir: Path, proc_dir: Path,
                    download: bool = True) -> Iterator[dict]:
    from scrapers.smutbase_scraper import (
        create_session as smut_session, get_listing_page,
        get_project_details, SITES,
    )
    seen = {p.stem for p in proc_dir.glob("*.json")}
    out = raw_dir / "open3dlab"

    # Build filename→metadata lookup from metadata/UUID.meta.json directory
    meta_by_filename: dict[str, dict] = {}
    meta_dir = out / "metadata"
    if meta_dir.exists():
        for mf in meta_dir.glob("*.meta.json"):
            try:
                m = json.loads(mf.read_text())
                fname = m.get("filename", "")
                if fname:
                    meta_by_filename[fname] = m
            except Exception:
                pass

    def _normalize_open3dlab(m: dict, blend_path: Path) -> dict:
        tags = m.get("tags", {})
        if isinstance(tags, dict):
            tags = list(tags.keys())
        return {
            "name":        m.get("title", blend_path.stem),
            "description": m.get("description", ""),
            "tags":        tags,
            "creator":     m.get("creator", ""),
            "downloads":   m.get("downloads", 0),
            "license":     m.get("license", ""),
            "project_id":  m.get("project_id", ""),
        }

    # Disk-scan: yield already-downloaded .blend files
    for blend_path in out.rglob("*.blend"):
        if blend_path.stem in seen:
            continue
        raw_meta = (
            meta_by_filename.get(blend_path.name)
            or meta_by_filename.get(blend_path.stem + ".blend")
        )
        if not raw_meta:
            sp = blend_path.parent / (blend_path.name + ".meta.json")
            raw_meta = json.loads(sp.read_text()) if sp.exists() else {}
        yield {
            "source":       "open3dlab",
            "source_url":   raw_meta.get("source_url") or raw_meta.get("url", "https://open3dlab.com"),
            "raw_path":     blend_path,
            "download_url": raw_meta.get("download_url"),
            "metadata":     _normalize_open3dlab(raw_meta, blend_path),
            "raw_dir":      out,
        }

    if not download:
        return

    # Live-scrape popular items
    try:
        from scrapers.smutbase_scraper import download_project_file
        session = smut_session("open3dlab")
        base_url = SITES["open3dlab"]["base_url"]
        for page in range(1, 6):
            projects = get_listing_page(session, base_url, page=page,
                                        software_tag="blender",
                                        sort_by="popular")
            if not projects:
                break
            for proj in projects:
                stem = re.sub(r'[^\w]', '_', proj.get("title", ""))[:60]
                if stem and stem in seen:
                    continue
                detail = get_project_details(session, proj["url"],
                                             proj.get("project_id", ""))
                if not detail:
                    continue
                proj.update(detail)
                blend_path = download_project_file(session, proj, out)
                if not blend_path or not Path(blend_path).exists():
                    continue
                blend_path = Path(blend_path)
                # Persist to metadata/ dir keyed by project_id
                proj["source_url"] = proj.get("url", "")
                pid = proj.get("project_id", blend_path.stem)
                meta_dir.mkdir(parents=True, exist_ok=True)
                (meta_dir / f"{pid}.meta.json").write_text(
                    json.dumps(proj, indent=2, default=str))
                yield {
                    "source":       "open3dlab",
                    "source_url":   proj.get("url", "https://open3dlab.com"),
                    "raw_path":     blend_path,
                    "download_url": None,
                    "metadata":     _normalize_open3dlab(proj, blend_path),
                    "raw_dir":      out,
                }
    except Exception as e:
        logger.debug(f"open3dlab: {e}")


def _iter_youtube(raw_dir: Path, proc_dir: Path,
                   download: bool = True) -> Iterator[dict]:
    out = raw_dir / "youtube"
    seen = {p.stem for p in proc_dir.glob("*.json")}
    for blend_path in out.rglob("*.blend"):
        if blend_path.stem in seen:
            continue
        meta_path = blend_path.parent / (blend_path.name + ".meta.json")
        if not meta_path.exists():
            meta_path = blend_path.with_suffix(".json")  # legacy fallback
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        yield {
            "source":       "youtube",
            "source_url":   meta.get("url", "https://youtube.com"),
            "raw_path":     blend_path,
            "download_url": None,
            "metadata": {
                "name":        meta.get("title", blend_path.stem.replace("_", " ")),
                "description": meta.get("description", ""),
                "tags":        meta.get("tags", []),
            },
            "raw_dir": out,
        }


def _iter_objaverse_xl(raw_dir: Path, proc_dir: Path,
                        download: bool = True) -> Iterator[dict]:
    """Iterate Objaverse-XL (Sketchfab source) sorted by viewCount desc.

    Unlike other iterators, this one triggers the actual XL API download
    synchronously (1 item at a time) because the XL API uses HuggingFace
    Hub paths that aren't plain HTTP URLs.  Each yielded item has raw_path
    already set to the downloaded local file.
    """
    if not download:
        return
    try:
        import objaverse.xl as oxl
        import objaverse as obv1
    except ImportError:
        logger.warning("objaverse package not available; skipping objaverse_xl source")
        return

    seen_ids = {p.stem for p in proc_dir.glob("*.json")}
    out_dir  = raw_dir / "objaverse_xl"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(out_dir / ".xl_cache")

    # Load v1 annotations for name/desc/tags/categories enrichment
    v1_ann: dict = {}
    try:
        v1_ann = obv1.load_annotations()
        logger.info(f"objaverse_xl: loaded {len(v1_ann)} v1 annotations")
    except Exception as e:
        logger.debug(f"objaverse_xl: v1 annotations unavailable ({e})")

    try:
        downloader = oxl.downloaders["sketchfab"]
        annotations = downloader.get_annotations()
        # Filter to processable types and sort by viewCount desc (popular first)
        processable = {"glb", "gltf", "obj", "stl", "ply", "blend"}
        annotations = annotations[annotations["fileType"].isin(processable)]
        try:
            annotations = annotations.sort_values("viewCount", ascending=False)
        except Exception:
            pass
        logger.info(f"objaverse_xl: {len(annotations)} processable models to iterate")
    except Exception as e:
        logger.warning(f"objaverse_xl: could not load annotations: {e}")
        return

    for _, row in annotations.iterrows():
        file_id  = str(row.get("fileIdentifier", "")).strip()
        if not file_id:
            continue
        uid = _uid_from_xl_row(row, "sketchfab")
        if uid in seen_ids:
            continue

        # Download ONE item via XL API — this uses HuggingFace Hub internally
        try:
            batch = annotations[annotations["fileIdentifier"] == file_id].head(1)
            results = downloader.download_objects(
                objects=batch,
                download_dir=cache_dir,
                processes=1,
            )
        except Exception as e:
            logger.debug(f"objaverse_xl: download error for {uid}: {e}")
            continue

        # XL API results may be keyed by fileIdentifier or UID — check both.
        # NOTE: if already-cached, download_objects() returns {} but the file
        # still exists in cache_dir.  Fall back to scanning cache_dir for the UID.
        local_path_str = None
        if results:
            local_path_str = (results.get(file_id)
                              or results.get(uid)
                              or next(iter(results.values()), None))
        if not local_path_str or not Path(local_path_str).exists():
            # Scan cache_dir recursively for a filename containing uid
            for candidate in Path(cache_dir).rglob(f"*{uid}*"):
                if candidate.is_file():
                    local_path_str = str(candidate)
                    break
        if not local_path_str or not Path(local_path_str).exists():
            logger.debug(f"objaverse_xl: download returned no file for {uid}; results keys={list((results or {}).keys())[:3]}")
            continue

        # Copy to our managed raw_dir with a clean filename
        file_type = str(row.get("fileType", "glb"))
        dst = out_dir / f"{uid}.{file_type}"
        if not dst.exists():
            try:
                import shutil
                shutil.copy2(local_path_str, dst)
            except Exception as e:
                logger.debug(f"objaverse_xl: copy failed {uid}: {e}")
                continue

        # Enrich metadata from v1 annotations
        meta: dict = {
            "name":      str(row.get("name", "")),
            "file_type": file_type,
            "uid":       uid,
            "source":    "objaverse_xl",
        }
        v1 = v1_ann.get(uid, {})
        if v1.get("name"):
            meta["name"] = v1["name"]
        if v1.get("description"):
            meta["description"] = v1["description"][:500]
        if v1.get("tags"):
            meta["tags"] = [
                t["name"] if isinstance(t, dict) else t for t in v1["tags"][:10]
            ]
        if v1.get("categories"):
            meta["categories"] = [
                c["name"] if isinstance(c, dict) else c for c in v1["categories"][:5]
            ]
        if v1.get("likeCount"):
            meta["likeCount"] = v1["likeCount"]

        yield {
            "source":       "objaverse_xl",
            "source_url":   f"https://sketchfab.com/3d-models/{uid}",
            "raw_path":     dst,
            "download_url": None,
            "metadata":     meta,
            "raw_dir":      out_dir,
        }


def _uid_from_xl_row(row, source: str) -> str:
    """Extract a clean UID string from an XL annotation row."""
    file_id = str(row.get("fileIdentifier", "")).strip()
    if source == "sketchfab":
        parts = file_id.rstrip("/").split("/")
        for part in reversed(parts):
            if len(part) > 10:
                return part[:36]
        return file_id[-36:]
    return re.sub(r"[^\w]", "_", file_id)[-60:]


SOURCE_ITERS = {
    "blender_official": _iter_blender_official,
    "blendswap":        _iter_blendswap,
    "objaverse":        _iter_objaverse,
    "objaverse_xl":     _iter_objaverse_xl,
    "github":           _iter_github,
    "smutbase":         _iter_smutbase,
    "open3dlab":        _iter_open3dlab,
    "youtube":          _iter_youtube,
}


# ── Download ──────────────────────────────────────────────────────────────

def _download_item(item: dict) -> Path | None:
    """Download raw file for item. Returns local path or None on failure.
    Handles rate-limiting (HTTP 429/503) with exponential back-off and retries."""
    if item.get("raw_path") and Path(item["raw_path"]).exists():
        return Path(item["raw_path"])

    url = item.get("download_url")
    if not url:
        return None

    raw_dir = Path(item["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    fname = url.split("?")[0].split("/")[-1] or "model.blend"
    dest  = raw_dir / fname
    if dest.exists():
        return dest

    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, stream=True, timeout=90,
                                headers={"User-Agent": "blender-copilot-pipeline/1.0"})
            if resp.status_code == 429:
                # Respect Retry-After header if present, otherwise back off
                wait = int(resp.headers.get("Retry-After", min(120, 15 * attempt)))
                logger.warning(f"[download] Rate limited (429) on {url!r} — "
                               f"waiting {wait}s (attempt {attempt}/{max_attempts})")
                time.sleep(wait)
                continue
            if resp.status_code == 503:
                wait = min(120, 30 * attempt)
                logger.warning(f"[download] Service unavailable (503) — "
                               f"waiting {wait}s (attempt {attempt}/{max_attempts})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(65536):
                    fh.write(chunk)
            return dest
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code in (429, 503) and attempt < max_attempts:
                wait = min(120, 30 * attempt)
                logger.warning(f"[download] HTTP {code} on {url!r} — "
                               f"retrying in {wait}s ({attempt}/{max_attempts})")
                time.sleep(wait)
                continue
            logger.debug(f"Download failed {url}: {e}")
            return None
        except Exception as e:
            if attempt < max_attempts:
                wait = min(60, 5 * attempt)
                logger.debug(f"Download error (attempt {attempt}/{max_attempts}) {url}: {e} — retry in {wait}s")
                time.sleep(wait)
                continue
            logger.debug(f"Download failed after {max_attempts} attempts {url}: {e}")
            return None
    return None


# ── Extract ───────────────────────────────────────────────────────────────

_MESH_EXTS  = {".blend", ".glb", ".gltf", ".obj", ".stl", ".ply", ".fbx",
               ".off", ".3ds"}
_MESH_EXTRACTOR_SCRIPT = BASE / "processing" / "mesh_extractor.py"


def _extract_item(raw_path: Path, proc_dir: Path, timeout: int = 300) -> Path | None:
    """Extract raw file → JSON.  Returns JSON path or None.

    All supported formats (blend, glb, gltf, obj, fbx) are routed through
    Blender + blend_extractor.py which handles import natively via bpy.ops.
    """
    out_json = proc_dir / f"{raw_path.stem}.json"
    if out_json.exists():
        return out_json

    ext = raw_path.suffix.lower()
    if ext not in _MESH_EXTS:
        logger.debug(f"Unsupported extension {ext}: {raw_path.name}")
        return None

    # Route ALL formats through Blender — blend_extractor supports glb/gltf/obj/fbx too
    cmd = [
        BLENDER_EXE, "--background",
        "--python", str(EXTRACTOR),
        "--",
        "--input", str(raw_path),
        "--output", str(proc_dir),
    ]

    proc_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout)
        if out_json.exists():
            return out_json
        logger.debug(f"Extract failed {raw_path.name}: {result.stderr[-400:]}")
        return None
    except subprocess.TimeoutExpired:
        logger.debug(f"Extract timeout {raw_path.name}")
        return None
    except Exception as e:
        logger.debug(f"Extract error {raw_path.name}: {e}")
        return None


# ── Multi-view render (14 viewports + 1 full = 15 images) ────────────────


def _render_item_views(json_path: Path, render_dir: Path,
                       raw_path: Path | None = None,
                       timeout: int = PIPELINE_RENDER_TIMEOUT) -> list[Path]:
    """Render views using scene-native settings for .blend, or JSON renderer otherwise.

    .blend files → blender_scene_render.py: uses the file's own camera/engine/lighting.
      - Full render with native engine (EEVEE/Cycles) at capped quality
      - 14 viewport renders: 512×512 Workbench/Material-Preview
      - Materials-only files: creates a geometry showcase grid first
      - Framing camera only added when no camera exists in the scene

    All other formats (GLB/OBJ/FBX extracted to JSON) → blender_render.py:
      - 14 viewport renders: 512×512 Workbench/Material-Preview
      - 1 full render: 2560×1440 EEVEE

    Returns list of PNG paths (full render first, then viewports).
    """
    render_dir.mkdir(parents=True, exist_ok=True)
    stem = json_path.stem

    # Already done?
    existing_views = sorted(render_dir.glob(f"{stem}_view*.png"))
    existing_full  = list(render_dir.glob(f"{stem}_full.png"))
    if len(existing_views) >= 14 and existing_full:
        return existing_full + existing_views  # full first for VL priority

    # Route: native .blend scene renderer vs JSON mesh renderer
    if raw_path and str(raw_path).lower().endswith(".blend"):
        logger.info(f"  [render] Using scene-native renderer for {raw_path.name}")
        cmd = [
            BLENDER_EXE, str(raw_path), "--background",
            "--python", str(SCENE_RENDERER),
            "--",
            "--output",     str(render_dir),
            "--scene-id",   stem,
            "--vp-size",    str(PIPELINE_VIEWPORT_SIZE),
        ]
        if PIPELINE_SCENE_FULL_SIZE > 0:
            cmd += ["--size", str(PIPELINE_SCENE_FULL_SIZE)]
        if PIPELINE_SCENE_MAX_SAMPLES > 0:
            cmd += ["--max-samples", str(PIPELINE_SCENE_MAX_SAMPLES)]
    else:
        logger.info(f"  [render] Using JSON mesh renderer for {stem}")
        cmd = [
            BLENDER_EXE, "--background",
            "--python", str(RENDERER),
            "--",
            "--input",  str(json_path),
            "--output", str(render_dir),
            "--vp-width", str(PIPELINE_VIEWPORT_SIZE),
            "--vp-height", str(PIPELINE_VIEWPORT_SIZE),
        ]
        if PIPELINE_SCENE_FULL_SIZE > 0:
            cmd += [
                "--width", str(max(2560, PIPELINE_SCENE_FULL_SIZE * 2)),
                "--height", str(max(1440, PIPELINE_SCENE_FULL_SIZE)),
            ]
        if PIPELINE_SCENE_MAX_SAMPLES > 0:
            cmd += ["--samples", str(max(128, PIPELINE_SCENE_MAX_SAMPLES))]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode not in (0, 1):  # 1 = no-camera fallback in scene renderer
            logger.debug(f"Render exit {result.returncode} for {stem}")
            if result.stderr:
                logger.debug(f"Render stderr: {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        logger.debug(f"Render timeout {stem}")
    except Exception as e:
        logger.debug(f"Render error {stem}: {e}")

    views = sorted(render_dir.glob(f"{stem}_view*.png"))
    full  = list(render_dir.glob(f"{stem}_full.png"))
    return full + views  # full render first so VL sees best image first


# ── VL Labeling ───────────────────────────────────────────────────────────

def _encode_image(path: Path) -> str | None:
    """Return base64-encoded image string for Ollama vision API."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def _estimate_label_confidence(label: str, meta: dict) -> float:
    """Estimate label confidence via listing/label lexical agreement."""
    if not label:
        return 0.0
    label_words = {w.lower() for w in re.findall(r"[a-zA-Z0-9]+", label) if len(w) > 2}
    if not label_words:
        return 0.4

    meta_terms: set[str] = set()
    for field in ("name", "description", "creator", "license"):
        val = str(meta.get(field, ""))
        meta_terms.update(w.lower() for w in re.findall(r"[a-zA-Z0-9]+", val) if len(w) > 2)
    tags = meta.get("tags") or meta.get("topics") or []
    if isinstance(tags, dict):
        tags = list(tags.keys())
    for tag in tags[:30]:
        meta_terms.update(w.lower() for w in re.findall(r"[a-zA-Z0-9]+", str(tag)) if len(w) > 2)

    if not meta_terms:
        return 0.4

    overlap = len(label_words.intersection(meta_terms))
    ratio = overlap / max(1, len(label_words))
    if ratio >= 0.6:
        return 1.0
    if ratio >= 0.25:
        return 0.7
    return 0.4


def _vl_label(item: dict, json_data: dict, render_pngs: list[Path],
              timeout: int = 300) -> dict:
    """Call qwen2.5vl:32b with ALL context: source URL, JSON metadata,
    rendered images. Returns {'label','label_confidence','semantic_parts'}."""

    meta     = item.get("metadata", {})
    source   = item.get("source", "unknown")
    src_url  = item.get("source_url", "")

    # Build rich text context — use every piece of real listing data available
    ctx_parts = [f"Source: {source}"]
    # Only include URL if it looks like a real listing page (not just a domain root)
    if src_url and len(src_url) > 25 and src_url not in (
        "https://github.com", "https://open3dlab.com",
        "https://blendswap.com", "https://smutba.se",
    ):
        ctx_parts.append(f"Listing URL: {src_url}")
    if meta.get("name"):
        ctx_parts.append(f"Name: {meta['name']}")
    if meta.get("creator"):
        ctx_parts.append(f"Creator: {meta['creator']}")
    if meta.get("repo"):
        ctx_parts.append(f"GitHub repo: {meta['repo']}")
    if meta.get("file_path"):
        ctx_parts.append(f"File path in repo: {meta['file_path']}")
    if meta.get("description"):
        ctx_parts.append(f"Description: {str(meta['description'])[:400]}")
    tags = meta.get("tags") or meta.get("topics") or []
    if isinstance(tags, dict):
        tags = list(tags.keys())
    if tags:
        ctx_parts.append(f"Tags: {', '.join(str(t) for t in tags[:15])}")
    if meta.get("categories"):
        ctx_parts.append(f"Categories: {meta['categories']}")
    if meta.get("downloads"):
        ctx_parts.append(f"Downloads: {meta['downloads']}")
    if meta.get("stars"):
        ctx_parts.append(f"GitHub stars: {meta['stars']}")
    if meta.get("license"):
        ctx_parts.append(f"License: {meta['license']}")

    # Pull object/mesh info from extracted JSON
    objects = json_data.get("objects", [json_data]) if isinstance(json_data, dict) else []
    if objects:
        all_obj_names = [o.get("name", "") for o in objects if o.get("name")]
        all_mat_names = []
        total_faces = 0
        for o in objects:
            all_mat_names += [m.get("name", "") for m in o.get("materials", []) if m.get("name")]
            mesh = o.get("mesh", {})
            total_faces += mesh.get("num_faces") or len(mesh.get("faces", []))
        if all_obj_names:
            ctx_parts.append(f"Blender objects: {', '.join(all_obj_names[:6])}")
        if all_mat_names:
            ctx_parts.append(f"Materials: {', '.join(dict.fromkeys(all_mat_names[:8]))}")
        if total_faces:
            ctx_parts.append(f"Total faces: {total_faces}")
        if len(objects) > 1:
            ctx_parts.append(f"Object count: {len(objects)}")

    text_ctx = "\n".join(ctx_parts)

    # ── Build image manifest + encode ──────────────────────────────────────
    # View order coming from _render_item_views:
    #   index 0  = _full.png  (scene-native render, best quality)
    #   index 1  = _view00 front       index 2  = _view01 back
    #   index 3  = _view02 right       index 4  = _view03 left
    #   index 5  = _view04 top         index 6  = _view05 bottom
    #   index 7  = _view06 upper_front_right    index 8  = _view07 upper_back_right
    #   index 9  = _view08 upper_back_left      index 10 = _view09 upper_front_left
    #   index 11 = _view10 lower_front_right    index 12 = _view11 lower_back_right
    #   index 13 = _view12 lower_back_left      index 14 = _view13 lower_front_left
    _VIEW_META = [
        ("full quality render",          "native engine, scene lighting — this is the authoritative render"),
        ("FRONT view",                   "eye-level, straight on"),
        ("BACK view",                    "eye-level, rear"),
        ("RIGHT view",                   "eye-level, right side"),
        ("LEFT view",                    "eye-level, left side"),
        ("TOP view",                     "looking straight down — may show only a flat top surface for tall objects"),
        ("BOTTOM view",                  "⚠ OFTEN INVALID — may look at underside of floor, underground, or be pure black. Ignore if uninformative."),
        ("upper-front-right diagonal",   "45° azimuth, +45° elevation — best overall shape view"),
        ("upper-back-right diagonal",    "135° azimuth, +45° elevation"),
        ("upper-back-left diagonal",     "225° azimuth, +45° elevation"),
        ("upper-front-left diagonal",    "315° azimuth, +45° elevation"),
        ("lower-front-right diagonal",   "⚠ PERSPECTIVE ANGLE — slightly below horizon. May show floor or look awkward; may be from underneath if geometry extends below origin."),
        ("lower-back-right diagonal",    "⚠ same caveat — below-horizon perspective"),
        ("lower-back-left diagonal",     "⚠ same caveat — below-horizon perspective"),
        ("lower-front-left diagonal",    "⚠ same caveat — below-horizon perspective"),
    ]

    images_b64 = []
    image_manifest_lines = []
    img_num = 1
    for i, png in enumerate(render_pngs[:15]):
        enc = _encode_image(png)
        if enc:
            images_b64.append(enc)
            label_i, note_i = _VIEW_META[i] if i < len(_VIEW_META) else (f"view {i}", "")
            image_manifest_lines.append(f"  [{img_num:02d}] {label_i}: {note_i}")
            img_num += 1

    image_manifest = "\n".join(image_manifest_lines) if image_manifest_lines else "  (no renders available)"

    prompt = (
        f"{text_ctx}\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "PURPOSE: You are generating a TRAINING LABEL for a text-to-3D mesh AI.\n"
        "This label becomes the text prompt the model learns to associate with this\n"
        "geometry. It must read exactly like something a user would type into a 3D\n"
        "model generator — concise, specific, and natural.\n\n"
        "GOOD LABELS: 'medieval sword', 'wooden dining chair', 'sci-fi space station\n"
        "module', 'cartoon bear head', 'low-poly pine tree', 'iron knight helmet'\n"
        "BAD LABELS: '3D rendered object', 'detailed mesh', 'game asset',\n"
        "'beautifully crafted form', anything vague or genre-only\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        f"IMAGES PROVIDED — {len(images_b64)} images in order below:\n"
        f"{image_manifest}\n\n"
        "═══════════════════════════════════════════════════════════════\n"
        "CRITICAL RULES:\n\n"
        "1. RENDERS ARE SOURCE OF TRUTH — base your label on the actual geometry\n"
        "   visible in the renders. The full render [01] and upper-diagonal views\n"
        "   [08-11] are the most reliable. Silhouette, proportions, and structure\n"
        "   in the images override everything else.\n\n"
        "2. LISTING DATA IS STRONG CROSS-REFERENCE — the Name, Description, Tags,\n"
        "   and Category above are written by the model's own author and are usually\n"
        "   accurate. Use them to identify specific subtypes and details you can\n"
        "   confirm in the renders. If listing says 'medieval knight armor' and the\n"
        "   renders show armor → use it. If listing says 'sword' but renders clearly\n"
        "   show a chair → trust the renders and ignore the listing.\n\n"
        "3. INVALID VIEW AWARENESS — the BOTTOM view and lower-diagonal views are\n"
        "   perspective angles that may shoot underneath the floor or at empty sky.\n"
        "   They may be completely black or show only ground geometry. Ignore them\n"
        "   if they appear uninformative; do not let them mislead your label.\n\n"
        "4. NO HALLUCINATION — do not invent details not visible in any render.\n"
        "   If the shape is ambiguous, use accurate neutral terms: 'humanoid figure',\n"
        "   'quadruped animal', 'abstract organic form', 'mechanical component'.\n\n"
        "5. NO FILENAME INFERENCE — never derive the label from the filename, repo\n"
        "   name, or file path. A file 'mob_enemy_01.blend' is labeled from its\n"
        "   actual rendered geometry, not the name.\n\n"
        "Output: 3-12 words. No explanation. No quotes. No trailing punctuation.\n"
        "Just the label."
    )

    # Also include any scene-level thumbnail images from the JSON
    # Only include values that are actual base64 strings (not metadata dicts)
    scene_images = json_data.get("images", {}) if isinstance(json_data, dict) else {}
    for img_name, img_b64 in list(scene_images.items())[:2]:
        if img_b64 and isinstance(img_b64, str) and len(images_b64) < 17:
            images_b64.append(img_b64)

    if not images_b64:
        # Text-only fallback (still uses VL model)
        payload = {
            "model": VL_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 80},
        }
        endpoint = f"{OLLAMA_URL}/api/generate"
    else:
        # Multimodal with images
        payload = {
            "model": VL_MODEL,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 80},
        }
        endpoint = f"{OLLAMA_URL}/api/generate"

    # Verbose: show what we're sending to Qwen
    logger.info(
        f"\n{'='*60}\n"
        f"[Qwen INPUT] source={source}  images={len(images_b64)}\n"
        f"--- PROMPT ---\n{prompt}\n"
        f"{'='*60}"
    )

    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        try:
            logger.info(f"[Qwen] Acquiring VL semaphore (attempt {attempt}) for {source}...")
            with _vl_sem:
                resp = requests.post(endpoint, json=payload, timeout=timeout)
            raw_response = resp.json().get("response", "") if resp.status_code == 200 else ""
            logger.info(f"[Qwen RAW OUTPUT] (attempt {attempt}) {raw_response!r}")
            if resp.status_code != 200:
                logger.warning(f"VL label HTTP {resp.status_code} (attempt {attempt})")
                time.sleep(3 * attempt)
                continue
            label = raw_response.strip().strip('"\'').strip(".")
            words = label.split()
            if 2 <= len(words) <= 20:
                logger.info(f"[Qwen FINAL LABEL] {label!r}")
                confidence = _estimate_label_confidence(label, meta)
                return {
                    "label": label,
                    "label_confidence": confidence,
                    "semantic_parts": [],
                }
            if words:
                label = " ".join(words[:20])
                logger.info(f"[Qwen FINAL LABEL] (truncated) {label!r}")
                confidence = _estimate_label_confidence(label, meta)
                return {
                    "label": label,
                    "label_confidence": confidence,
                    "semantic_parts": [],
                }
            # Empty response — cold-start; wait and retry
            logger.warning(f"[Qwen] Empty response on attempt {attempt}/{max_attempts} — retrying in {3*attempt}s")
            time.sleep(3 * attempt)
        except Exception as e:
            logger.warning(f"[Qwen] Error on attempt {attempt}/{max_attempts}: {e} — retrying in {3*attempt}s")
            time.sleep(3 * attempt)

    logger.error(f"[Qwen] All {max_attempts} attempts failed for {source}")
    return {"label": None, "label_confidence": 0.0, "semantic_parts": []}


def _vl_part_list(item: dict, json_data: dict, render_pngs: list[Path],
                  short_label: str, timeout: int = 300) -> list[str]:
    """Second-pass VL call for complex/high-poly assets to extract semantic parts."""
    images_b64 = []
    for png in render_pngs[:8]:
        enc = _encode_image(png)
        if enc:
            images_b64.append(enc)
    if not images_b64:
        return []

    objects = json_data.get("objects", [json_data]) if isinstance(json_data, dict) else []
    total_faces = 0
    for obj in objects:
        mesh = obj.get("mesh", {})
        total_faces += mesh.get("num_faces") or len(mesh.get("faces", []))

    prompt = (
        "You are creating training metadata for text-to-3D decomposition.\n"
        f"Primary label: {short_label}\n"
        f"Estimated total faces: {total_faces}\n"
        "Return ONLY a JSON array of semantic part names, 3 to 12 items.\n"
        "Example: [\"body\", \"wheels\", \"windows\", \"interior\"]\n"
        "No prose, no markdown, no explanation."
    )
    payload = {
        "model": VL_MODEL,
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 80},
    }
    try:
        with _vl_sem:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
        if resp.status_code != 200:
            return []
        raw = resp.json().get("response", "").strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"\[[\s\S]*\]", raw)
            parsed = json.loads(m.group(0)) if m else []
        if not isinstance(parsed, list):
            return []
        out = []
        for p in parsed:
            if isinstance(p, str):
                s = p.strip()[:64]
                if s:
                    out.append(s)
        return out[:12]
    except Exception:
        return []


def _vl_quality_rank(item: dict, json_data: dict, render_pngs: list[Path],
                     short_label: str, timeout: int = 180) -> dict:
    """Qwen-VL quality/professionalism scoring for sample ranking."""

    images_b64 = []
    for png in render_pngs[:8]:
        enc = _encode_image(png)
        if enc:
            images_b64.append(enc)
    if not images_b64:
        return {}

    objects = json_data.get("objects", [json_data]) if isinstance(json_data, dict) else []
    total_faces = 0
    for obj in objects:
        mesh = obj.get("mesh", {})
        total_faces += int(mesh.get("num_faces") or len(mesh.get("faces", [])) or 0)

    if total_faces < VL_QUALITY_MIN_FACES:
        return {}

    meta = item.get("metadata", {})
    prompt = (
        "You are scoring 3D asset training quality for a professional Blender pipeline.\n"
        f"Primary label: {short_label}\n"
        f"Total faces: {total_faces}\n"
        f"Source name: {meta.get('name', '')}\n\n"
        "Return ONLY strict JSON object with numeric scores in [0,1]:\n"
        "{\"visual_quality\":0.0,\"professionalism\":0.0,\"texture_integrity\":0.0,\"composition_quality\":0.0}\n"
        "No prose, no markdown."
    )

    payload = {
        "model": VL_MODEL,
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 120},
    }

    try:
        with _vl_sem:
            resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
        if resp.status_code != 200:
            return {}
        raw = resp.json().get("response", "").strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", raw)
            parsed = json.loads(m.group(0)) if m else {}
        if not isinstance(parsed, dict):
            return {}

        out = {}
        for key in ("visual_quality", "professionalism", "texture_integrity", "composition_quality"):
            try:
                out[key] = float(max(0.0, min(1.0, float(parsed.get(key, 0.0)))))
            except Exception:
                pass
        return out
    except Exception:
        return {}


def _infer_scene_domain(label: str, semantic_parts: list[str], source: str) -> str:
    text = f"{label} {' '.join(semantic_parts)} {source}".lower()
    if any(k in text for k in ("character", "humanoid", "creature", "animal", "rig", "armature")):
        return "character"
    if any(k in text for k in ("car", "vehicle", "ship", "plane", "robot", "mech", "engine")):
        return "vehicle"
    if any(k in text for k in ("house", "building", "room", "interior", "architecture", "bridge", "tower")):
        return "environment"
    if any(k in text for k in ("chair", "table", "desk", "sofa", "shelf", "lamp", "furniture")):
        return "prop_set"
    return "object"


def _build_composition_supervision(
    label: str,
    source: str,
    json_data: dict,
    semantic_parts: list[str],
) -> dict:
    objects = json_data.get("objects", [json_data]) if isinstance(json_data, dict) else []
    obj_names = [str(o.get("name", "")).strip() for o in objects if str(o.get("name", "")).strip()]

    total_faces = 0
    for obj in objects:
        mesh = obj.get("mesh", {})
        total_faces += int(mesh.get("num_faces") or len(mesh.get("faces", [])) or 0)

    object_count = max(1, len(objects))
    scene_domain = _infer_scene_domain(label, semantic_parts, source)

    if object_count >= 8 or total_faces >= 120000:
        complexity_tier = "hero"
        complexity_score = 1.0
    elif object_count >= 4 or total_faces >= 50000:
        complexity_tier = "complex"
        complexity_score = 0.8
    elif object_count >= 2 or total_faces >= 12000:
        complexity_tier = "medium"
        complexity_score = 0.55
    else:
        complexity_tier = "simple"
        complexity_score = 0.3

    composition_tags = []
    if object_count > 1:
        composition_tags.append("multi_object")
    if semantic_parts:
        composition_tags.append("semantic_parts")
    if any(k in label.lower() for k in ("scene", "setup", "corner", "assembly", "workstation", "interior")):
        composition_tags.append("scene_layout")
    if not composition_tags:
        composition_tags.append("single_object")

    return {
        "scene_domain": scene_domain,
        "composition_label": label,
        "composition_tags": composition_tags[:8],
        "semantic_part_count": int(len(semantic_parts)),
        "object_count": int(object_count),
        "total_face_count": int(total_faces),
        "complexity_tier": complexity_tier,
        "scene_complexity_score": float(complexity_score),
        "object_name_hints": obj_names[:12],
    }


def _build_workflow_supervision(
    label: str,
    composition: dict,
    semantic_parts: list[str],
) -> dict:
    scene_domain = composition.get("scene_domain", "object")
    object_count = int(composition.get("object_count", 1))
    complexity_tier = composition.get("complexity_tier", "simple")

    action_sequence = [
        "decompose_prompt",
        "generate_base_mesh",
        "inspect_scene",
    ]
    if object_count > 1:
        action_sequence.append("arrange_scene")
    if scene_domain in ("vehicle", "environment", "character"):
        action_sequence.append("apply_modifiers")
    action_sequence.extend(["assign_materials", "capture_viewport", "declare_complete"])

    workflow_targets = ["modeling", "shading", "qa"]
    if scene_domain == "character":
        workflow_targets.append("rigging")
    if complexity_tier in ("complex", "hero"):
        workflow_targets.append("scene_assembly")

    return {
        "initial_state_summary": "empty_scene",
        "target_instruction": label,
        "workflow_targets": workflow_targets,
        "action_sequence": action_sequence,
        "final_state_checks": [
            "non_empty_scene",
            "reasonable_scale",
            "materials_assigned",
            "completion_declared",
        ],
        "semantic_parts": semantic_parts[:12],
    }


def _compute_texture_integrity(obj: dict, scene_images: dict) -> dict:
    """Estimate texture health from extracted material node data.

    Penalizes known broken texture nodes and missing image references.
    """
    total_image_nodes = 0
    broken_image_nodes = 0
    missing_image_refs = 0

    materials = obj.get("materials", []) if isinstance(obj, dict) else []
    available_images = set(scene_images.keys()) if isinstance(scene_images, dict) else set()

    for mat in materials:
        for node in mat.get("nodes", []) if isinstance(mat, dict) else []:
            if str(node.get("type", "")).upper() != "TEX_IMAGE":
                continue
            total_image_nodes += 1
            if node.get("image_broken"):
                broken_image_nodes += 1
                continue
            image_name = str(node.get("image_name", "")).strip()
            if image_name and available_images and image_name not in available_images:
                missing_image_refs += 1

    if total_image_nodes == 0:
        score = 0.8
    else:
        broken_ratio = broken_image_nodes / total_image_nodes
        missing_ratio = missing_image_refs / total_image_nodes
        penalty = min(0.8, 0.65 * broken_ratio + 0.35 * missing_ratio)
        score = max(0.2, 1.0 - penalty)

    return {
        "texture_image_nodes": int(total_image_nodes),
        "broken_image_nodes": int(broken_image_nodes),
        "missing_image_refs": int(missing_image_refs),
        "texture_integrity_score": float(score),
    }


def _read_render_diagnostics(render_dir: Path, stem: str) -> dict:
    """Load render manifest diagnostics and compute a trust score."""
    manifest_path = render_dir / f"{stem}_manifest.json"
    diagnostics = {
        "has_manifest": False,
        "materials_showcase": False,
        "auto_framing_camera": False,
        "fallback_light_added": False,
        "native_light_count": 0,
        "viewport_count": 0,
        "has_full_render": False,
        "render_quality_score": 0.9,
    }
    if not manifest_path.exists():
        return diagnostics

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return diagnostics

    lights = manifest.get("lights") or []
    light_names = {
        str(light_entry.get("name", ""))
        for light_entry in lights
        if isinstance(light_entry, dict)
    }
    native_lights = [n for n in light_names if n and n != "AutoFramingKey"]

    diagnostics["has_manifest"] = True
    diagnostics["materials_showcase"] = bool(manifest.get("materials_showcase", False))
    diagnostics["auto_framing_camera"] = bool((manifest.get("camera") or {}).get("auto_framing", False))
    diagnostics["fallback_light_added"] = "AutoFramingKey" in light_names
    diagnostics["native_light_count"] = int(len(native_lights))
    diagnostics["viewport_count"] = int(len(manifest.get("renders") or []))
    diagnostics["has_full_render"] = bool(manifest.get("full_render"))

    score = 1.0
    if diagnostics["materials_showcase"]:
        score -= 0.25
    if diagnostics["auto_framing_camera"]:
        score -= 0.10
    if diagnostics["fallback_light_added"] and diagnostics["native_light_count"] == 0:
        score -= 0.10
    if not diagnostics["has_full_render"]:
        score -= 0.15
    if diagnostics["viewport_count"] < 8:
        score -= 0.10
    diagnostics["render_quality_score"] = float(max(0.2, min(1.0, score)))
    return diagnostics


# ── Build cache entry ─────────────────────────────────────────────────────

def _build_cache_entry(json_path: Path, label: str, source: str,
                       json_data: dict,
                       label_confidence: float = 1.0,
                       semantic_parts: list[str] | None = None,
                       vl_quality_rank: dict | None = None,
                       render_diagnostics: dict | None = None) -> list[dict]:
    """Convert JSON → list of .pt-ready dicts (one per mesh object)."""
    mesh_tok, bpe_tok = _get_tokenizers()
    train_max_tokens = _training_max_tokens()

    objects  = json_data.get("objects", [json_data]) if isinstance(json_data, dict) else [json_data]
    images   = json_data.get("images", {}) if isinstance(json_data, dict) else {}

    items = []
    semantic_parts = semantic_parts or []
    vl_quality_rank = vl_quality_rank or {}
    render_diagnostics = render_diagnostics or {}
    render_score = float(render_diagnostics.get("render_quality_score", 0.9))
    for obj in objects:
        mesh = obj.get("mesh", {})
        verts = mesh.get("vertices", [])
        faces = mesh.get("faces", [])
        if not verts or not faces or len(faces) < 4 or len(verts) < 4:
            continue

        original_face_count = len(faces)
        original_vert_count = len(verts)

        # Base-topology target (~500 faces) for hierarchical generation.
        # Keep coarse + clean-ish topology tokens alongside primary mesh tokens.
        base_verts = verts
        base_faces = faces
        if len(base_faces) > 500:
            try:
                candidate_verts, candidate_faces, ok = decimate_mesh(
                    base_verts, base_faces, target_faces=min(500, len(base_faces)))
                if ok and candidate_faces:
                    base_verts, base_faces = candidate_verts, candidate_faces
            except Exception:
                pass

        try:
            full_verts_norm = normalize_mesh(verts, target_range=(-1.0, 1.0))
            base_verts = normalize_mesh(base_verts, target_range=(-1.0, 1.0))
        except Exception:
            continue

        full_tokens = mesh_tok.encode_mesh(full_verts_norm, faces)
        tokens = full_tokens
        training_was_decimated = False

        if len(tokens) > train_max_tokens:
            target_faces = max(32, (train_max_tokens - 2) // 9)
            if target_faces >= len(faces):
                target_faces = max(32, len(faces) - 1)
            if target_faces <= 3:
                continue
            dec_verts, dec_faces, ok = decimate_mesh(verts, faces, target_faces)
            if not ok or not dec_faces:
                continue
            try:
                dec_verts_norm = normalize_mesh(dec_verts, target_range=(-1.0, 1.0))
            except Exception:
                continue
            tokens = mesh_tok.encode_mesh(dec_verts_norm, dec_faces)
            training_was_decimated = True

        base_tokens = mesh_tok.encode_mesh(base_verts, base_faces)
        if not tokens or tokens[0] != mesh_tok.BOS or tokens[-1] != mesh_tok.EOS:
            continue
        if not base_tokens or base_tokens[0] != mesh_tok.BOS or base_tokens[-1] != mesh_tok.EOS:
            base_tokens = tokens
        if len(tokens) > train_max_tokens:
            continue

        if bpe_tok is not None:
            text_ids, text_mask = bpe_tok.encode_padded(label, 256)
        else:
            text = label[:256]
            text_ids = [ord(c) % 32000 for c in text]
            text_mask = [1] * len(text_ids)
            text_ids += [0] * (256 - len(text_ids))
            text_mask += [0] * (256 - len(text_mask))

        mat_names = [m.get("name", "") for m in obj.get("materials", []) if m.get("name")]
        tex_health = _compute_texture_integrity(obj, images)
        tex_score = float(tex_health.get("texture_integrity_score", 0.8))

        quality = 0.5
        if original_face_count > 100:
            quality += 0.2
        if original_face_count > 500:
            quality += 0.1
        if mat_names:
            quality += 0.15
        if training_was_decimated:
            quality += 0.1
        quality *= tex_score
        quality *= render_score
        quality *= max(0.0, min(1.0, float(label_confidence)))

        vl_prof = float(vl_quality_rank.get("professionalism", 0.0)) if vl_quality_rank else 0.0
        vl_vis = float(vl_quality_rank.get("visual_quality", 0.0)) if vl_quality_rank else 0.0
        vl_comp = float(vl_quality_rank.get("composition_quality", 0.0)) if vl_quality_rank else 0.0
        professionalism_score = max(
            0.2,
            min(
                1.0,
                0.45 * quality
                + 0.20 * tex_score
                + 0.15 * float(max(0.0, min(1.0, label_confidence)))
                + 0.20 * max(vl_prof, (vl_vis + vl_comp) * 0.5),
            ),
        )
        weight = max(0.2, 0.3 + quality * 1.0 + professionalism_score * 0.4)

        # Assemble scene_context with all available rendering data
        sc: dict = {}
        if obj.get("materials"):
            sc["materials"] = obj["materials"]
        if mesh.get("face_material_indices"):
            sc["face_material_indices"] = mesh["face_material_indices"]
        if mesh.get("uv_layers"):
            sc["uv_layers"] = mesh["uv_layers"]
        if mesh.get("vertex_color_layers"):
            sc["vertex_color_layers"] = mesh["vertex_color_layers"]
        if mesh.get("face_smooth"):
            sc["face_smooth"] = mesh["face_smooth"]
        if images:
            sc["images"] = images

        entry = {
            "text_ids":      torch.tensor(text_ids, dtype=torch.long),
            "text_mask":     torch.tensor(text_mask, dtype=torch.float),
            "mesh_tokens":   torch.tensor(tokens, dtype=torch.long),
            "quality_weight": torch.tensor(weight, dtype=torch.float),
            "label":         label,
            "data_source":   source,
            "cache_schema_version": "v7_professional_ranked",
        }
        if sc:
            entry["scene_context"] = sc

        # Heuristic modifier-stack supervision target (phase-2 bridge).
        label_l = label.lower()
        modifier_stack = []
        if any(k in label_l for k in ("character", "animal", "dragon", "organic", "creature", "head")):
            modifier_stack.append({"type": "subsurf", "levels": 3})
            modifier_stack.append({"type": "bevel", "width": 0.005, "segments": 2})
        elif any(k in label_l for k in ("car", "vehicle", "robot", "mech", "engine", "gun", "sword")):
            modifier_stack.append({"type": "bevel", "width": 0.01, "segments": 2})
            modifier_stack.append({"type": "weighted_normal"})
        elif any(k in label_l for k in ("wall", "building", "house", "architecture", "room")):
            modifier_stack.append({"type": "solidify", "thickness": 0.05})
            modifier_stack.append({"type": "bevel", "width": 0.015, "segments": 2})

        entry["base_mesh_tokens"] = torch.tensor(base_tokens, dtype=torch.long)
        if len(full_tokens) != len(tokens):
            entry["full_mesh_tokens"] = torch.tensor(full_tokens, dtype=torch.long)
        entry["training_decimated"] = bool(training_was_decimated)
        entry["training_token_count"] = int(len(tokens))
        entry["full_token_count"] = int(len(full_tokens))
        entry["modifier_stack"] = modifier_stack
        entry["original_face_count"] = int(original_face_count)
        entry["original_vert_count"] = int(original_vert_count)
        entry["topology_quality"] = float(1.0 if original_face_count >= 4 and original_vert_count >= 4 else 0.0)
        entry["label_confidence"] = torch.tensor(float(max(0.0, min(1.0, label_confidence))), dtype=torch.float)
        entry["texture_health"] = tex_health
        entry["texture_integrity_score"] = torch.tensor(float(tex_score), dtype=torch.float)
        entry["professionalism_score"] = torch.tensor(float(professionalism_score), dtype=torch.float)
        entry["render_diagnostics"] = render_diagnostics
        entry["render_quality_score"] = torch.tensor(float(render_score), dtype=torch.float)
        if vl_quality_rank:
            entry["vl_quality_rank"] = vl_quality_rank
        if semantic_parts:
            entry["semantic_parts"] = semantic_parts[:12]

        composition = _build_composition_supervision(
            label=label,
            source=source,
            json_data=json_data,
            semantic_parts=semantic_parts,
        )
        workflow_supervision = _build_workflow_supervision(
            label=label,
            composition=composition,
            semantic_parts=semantic_parts,
        )

        entry["composition"] = composition
        entry["scene_complexity_score"] = torch.tensor(
            float(composition.get("scene_complexity_score", 0.3)),
            dtype=torch.float,
        )
        entry["workflow_supervision"] = workflow_supervision

        items.append(entry)
        gc.collect()

    return items


# ── Per-item full pipeline ─────────────────────────────────────────────────

def process_item(item: dict, keep_raw: bool = False) -> dict:
    """Run the full pipeline for one item. Returns result summary dict."""
    source   = item["source"]
    metadata = item.get("metadata", {})
    name     = metadata.get("name", "unknown")

    result = {"source": source, "name": name, "status": "fail",
              "label": None, "cache_entries": 0}

    # 1. Download
    raw_path = _download_item(item)
    if raw_path is None:
        result["status"] = "no_download"
        return result

    # No raw-size hard cap: production pipeline preserves full-detail assets.

    # 2. Extract → JSON
    proc_dir = PROC_DIR / source
    json_path = _extract_item(raw_path, proc_dir)
    if json_path is None:
        # For .blend files that have no extractable geometry (e.g. materials-only):
        # still run the scene renderer — it will create a materials showcase, render
        # it, and Qwen will label from the renders.  Cache entry will be empty (no
        # mesh tokens) which is fine; the renders end up in data/renders/ for QA.
        if raw_path.suffix.lower() == ".blend":
            logger.info(f"  [extract] No geometry in {raw_path.name} — "
                        f"continuing as materials-only blend")
            # Synthetic json_path so render naming works (file may not exist yet)
            json_path = proc_dir / f"{raw_path.stem}.json"
            json_data: dict = {"objects": [], "metadata": dict(metadata)}
        else:
            result["status"] = "extract_fail"
            if not keep_raw:
                try:
                    raw_path.unlink()
                except Exception:
                    pass
            return result
    else:
        try:
            json_data = json.loads(json_path.read_text())
        except Exception:
            result["status"] = "json_invalid"
            return result

    # Cross-source dedup via content fingerprint.
    fp = _mesh_fingerprint(json_data)
    if not _register_fingerprint(fp):
        result["status"] = "duplicate"
        if not keep_raw:
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass
        return result

    # Keep original high-poly extract for hierarchical/base-topology training.
    try:
        objects = json_data.get("objects", [json_data]) if isinstance(json_data, dict) else [json_data]
        max_faces = 0
        for obj in objects:
            mesh = obj.get("mesh", {})
            max_faces = max(max_faces, len(mesh.get("faces", [])))
        if max_faces > HIGH_POLY_FACE_THRESHOLD:
            hp_dir = PROC_DIR / "high_poly" / source
            hp_dir.mkdir(parents=True, exist_ok=True)
            (hp_dir / f"{json_path.stem}.json").write_text(json.dumps(json_data))
    except Exception as e:
        logger.debug(f"high_poly save failed for {json_path.stem}: {e}")

    # 3. Quick render
    render_dir = RENDER_DIR / source / json_path.stem
    render_pngs = _render_item_views(json_path, render_dir, raw_path=raw_path)
    render_diagnostics = _read_render_diagnostics(render_dir, json_path.stem)

    # 4. VL label
    vl_result = _vl_label(item, json_data, render_pngs)
    label = vl_result.get("label")
    label_confidence = float(vl_result.get("label_confidence", 0.0))
    semantic_parts = vl_result.get("semantic_parts", []) or []
    vl_quality_rank: dict = {}

    if label is None:
        # Fallback: programmatic label
        objects = json_data.get("objects", [json_data]) if isinstance(json_data, dict) else [json_data]
        obj = objects[0] if objects else {}
        mesh = obj.get("mesh", {})
        mat_names = [m.get("name", "") for m in obj.get("materials", []) if m.get("name")]
        try:
            bbox_aspect = compute_bbox_aspect(mesh.get("vertices", [[0,0,0]]))
        except Exception:
            bbox_aspect = (1.0, 1.0, 1.0)
        label = generate_smart_label(
            obj_name=obj.get("name", ""),
            material_names=mat_names,
            modifier_types=[],
            num_faces=len(mesh.get("faces", [])),
            num_verts=len(mesh.get("vertices", [])),
            bbox_aspect=bbox_aspect,
            file_label=json_path.stem,
            metadata_name=metadata.get("name", ""),
            metadata_desc=str(metadata.get("description", ""))[:200],
            metadata_tags=metadata.get("tags", []),
        )
        label_confidence = max(label_confidence, 0.0)

    # Optional second-pass part decomposition for complex/high-poly meshes
    try:
        objects = json_data.get("objects", [json_data]) if isinstance(json_data, dict) else [json_data]
        total_faces = 0
        for obj in objects:
            mesh = obj.get("mesh", {})
            total_faces += mesh.get("num_faces") or len(mesh.get("faces", []))
        if total_faces > HIGH_POLY_FACE_THRESHOLD and label:
            semantic_parts = _vl_part_list(item, json_data, render_pngs, label)
    except Exception:
        pass

    result["label"] = label

    # Qwen-VL quality ranking for professional cache weighting.
    try:
        vl_quality_rank = _vl_quality_rank(item, json_data, render_pngs, label)
    except Exception:
        vl_quality_rank = {}

    # 5. Build cache entry and save
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entries = _build_cache_entry(
        json_path,
        label,
        source,
        json_data,
        label_confidence=label_confidence,
        semantic_parts=semantic_parts,
        vl_quality_rank=vl_quality_rank,
        render_diagnostics=render_diagnostics,
    )
    for i, entry in enumerate(entries):
        cache_id = f"{json_path.stem}_{i:03d}"
        torch.save(entry, CACHE_DIR / f"{cache_id}.pt")

    result["cache_entries"] = len(entries)
    result["status"] = "ok" if entries else "no_mesh"

    # 6. Cleanup
    if not keep_raw:
        try:
            raw_path.unlink(missing_ok=True)
        except Exception:
            pass

    return result


# ── Pipeline orchestrator ─────────────────────────────────────────────────

def run_pipeline(
    sources: list[str],
    pull_behave: str = "concurrent",   # "concurrent" | "batch"
    pull_max: int | None = None,
    workers: int | None = None,
    keep_raw: bool = False,
    test_mode: bool = False,
    scrape: bool = True,               # True = live scraping (default); False = disk-only (--local)
    ignore_parse_markers: bool = False,
) -> None:
    """Main entry point.  Drives downloads + processing across all sources.

    scrape=True (default): pull fresh data from live listing pages (open3dlab,
    blendswap, smutbase, objaverse, etc.).  Pass scrape=False (--local flag)
    to only process .blend files already on disk without any network requests.
    """
    if workers is None:
        workers = _DEFAULT_WORKERS

    _load_seen_fingerprints()

    active_sources = [s for s in sources if s in SOURCE_ITERS]
    if not active_sources:
        logger.error(f"No valid sources in {sources}")
        return

    mode_tag = "LOCAL/disk-only" if not scrape else "LIVE-SCRAPE"
    logger.info(f"Pipeline starting — sources={active_sources} "
                f"behave={pull_behave} workers={workers} mode={mode_tag} "
                f"{'TEST MODE (1 per source)' if test_mode else ''}")

    total_ok      = 0
    total_fail    = 0
    total_entries = 0
    counter_lock  = threading.Lock()

    def _log_result(res: dict):
        nonlocal total_ok, total_fail, total_entries
        with counter_lock:
            if res["status"] == "ok":
                total_ok += 1
                total_entries += res["cache_entries"]
                logger.info(
                    f"[{res['source']}] ✓ {res['name']!r:40s} "
                    f"→ {res['label']!r}  ({res['cache_entries']} entries)"
                )
            else:
                total_fail += 1
                logger.info(f"[{res['source']}] ✗ {res['name']!r} — {res['status']}")

    # Build item queue
    item_q: queue.Queue = queue.Queue(maxsize=workers * 4)

    def _feed_source(src: str) -> None:
        """Generator thread: push items from one source onto the queue."""
        proc_dir = PROC_DIR / src
        proc_dir.mkdir(parents=True, exist_ok=True)
        iter_fn  = SOURCE_ITERS[src]
        count    = 0
        source_marked = _source_is_parse_marked(src) if not ignore_parse_markers else False
        source_download = scrape and not source_marked
        if scrape and source_marked:
            logger.info(
                f"[{src}] parse marker detected ({_source_marker_path(src)}); "
                "network downloads disabled for this source"
            )
        try:
            for item in iter_fn(RAW_DIR, proc_dir, download=source_download):
                if pull_max is not None:
                    with counter_lock:
                        total = total_ok + total_fail
                    if total >= pull_max:
                        return
                item_q.put(item)
                count += 1
                if test_mode and count >= 1:
                    return
        except Exception as e:
            logger.warning(f"Source iterator error [{src}]: {e}")

    stopflag = threading.Event()

    def _worker():
        while not stopflag.is_set():
            try:
                item = item_q.get(timeout=5)
            except queue.Empty:
                continue
            try:
                res = process_item(item, keep_raw=keep_raw)
                _log_result(res)
            except Exception as e:
                logger.warning(f"Worker error: {e}")
            finally:
                item_q.task_done()
            if pull_max is not None:
                with counter_lock:
                    total = total_ok + total_fail
                if total >= pull_max:
                    stopflag.set()

    # Start processing workers
    worker_threads = [threading.Thread(target=_worker, daemon=True)
                      for _ in range(workers)]
    for wt in worker_threads:
        wt.start()

    if pull_behave == "batch":
        # Process one source at a time
        for src in active_sources:
            logger.info(f"── Batch: {src} ──")
            feeder = threading.Thread(target=_feed_source, args=(src,), daemon=True)
            feeder.start()
            feeder.join()
            # Drain queue for this source before moving on
            item_q.join()
            if stopflag.is_set():
                break
    else:
        # concurrent: feed ALL sources simultaneously
        feeders = [threading.Thread(target=_feed_source, args=(src,), daemon=True)
                   for src in active_sources]
        for ft in feeders:
            ft.start()
        for ft in feeders:
            ft.join()
        item_q.join()

    stopflag.set()
    for wt in worker_threads:
        wt.join(timeout=10)

    logger.info(
        f"\n{'─'*60}\n"
        f"Pipeline done.  ok={total_ok}  fail={total_fail}  "
        f"cache_entries={total_entries}\n"
        f"{'─'*60}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="*", default=ALL_SOURCES,
                        choices=ALL_SOURCES + ["all"],
                        help="Which sources to pull from (default: all)")
    parser.add_argument("--pull-max", type=int, default=None,
                        help="Stop after processing this many items total")
    parser.add_argument("--pull-behave", default="concurrent",
                        choices=["concurrent", "batch"],
                        help="concurrent=all sources at once, batch=one source at a time")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Download/process worker threads (default: auto = {_DEFAULT_WORKERS} on this machine)")
    parser.add_argument("--keep-raw", action="store_true",
                        help="Don't delete raw files after processing")
    parser.add_argument("--test", action="store_true",
                        help="Process exactly 1 item per source then exit")
    parser.add_argument("--local", action="store_true",
                        help=(
                            "Disk-only mode: do NOT run live scrapers. "
                            "Only reprocess .blend files already on disk. "
                            "Default (off): live scraping is enabled."
                        ))
    parser.add_argument("--ignore-parse-markers", action="store_true",
                        help=(
                            "Ignore data/raw/<source>/.parsed_complete.json markers and "
                            "allow live downloads even for marked sources."
                        ))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s")

    sources = args.sources
    if "all" in sources:
        sources = ALL_SOURCES

    run_pipeline(
        sources=sources,
        pull_behave=args.pull_behave,
        pull_max=args.pull_max,
        workers=args.workers,
        keep_raw=args.keep_raw,
        test_mode=args.test,
        scrape=not args.local,
        ignore_parse_markers=args.ignore_parse_markers,
    )


if __name__ == "__main__":
    main()
