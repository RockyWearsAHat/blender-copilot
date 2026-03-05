"""Targeted data pull: downloads diverse 3D models from Objaverse by category
and extracts existing .blend files. Runs alongside training — new JSONs are
picked up automatically by RealMeshStream.

Usage:
    python scripts/fetch_diverse_data.py              # all phases
    python scripts/fetch_diverse_data.py --objaverse  # only Objaverse
    python scripts/fetch_diverse_data.py --blends     # only .blend extraction
    python scripts/fetch_diverse_data.py --per-cat 80 # 80 models per category
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent

# ── Target categories + per-category aliases for label generation ────────────
TARGET_CATEGORIES: dict[str, dict] = {
    "cars-vehicles": {
        "per_cat": 80,
        "label_prefix": ["low poly car", "vehicle", "automobile", "3D car model",
                         "low poly vehicle", "sports car", "race car", "truck"],
    },
    "characters-creatures": {
        "per_cat": 80,
        "label_prefix": ["3D character", "creature", "humanoid character",
                         "game character", "fantasy creature", "character model"],
    },
    "architecture": {
        "per_cat": 80,
        "label_prefix": ["building", "architecture", "house", "structure",
                         "3D building", "low poly building", "castle", "tower"],
    },
    "nature-plants": {
        "per_cat": 60,
        "label_prefix": ["nature", "plant", "tree", "vegetation", "flower",
                         "low poly nature", "organic nature model"],
    },
    "animals-pets": {
        "per_cat": 60,
        "label_prefix": ["animal", "creature", "3D animal", "low poly animal",
                         "pet", "wildlife model"],
    },
    "furniture-home": {
        "per_cat": 60,
        "label_prefix": ["furniture", "chair", "table", "interior",
                         "3D furniture", "home object", "low poly furniture"],
    },
    "weapons-military": {
        "per_cat": 40,
        "label_prefix": ["weapon", "sword", "shield", "military",
                         "3D weapon", "low poly weapon", "game prop"],
    },
    "food-drink": {
        "per_cat": 40,
        "label_prefix": ["food", "drink", "3D food model", "low poly food",
                         "edible prop", "kitchen object"],
    },
    "people": {
        "per_cat": 40,
        "label_prefix": ["person", "human figure", "3D person",
                         "character", "humanoid model"],
    },
    "art-abstract": {
        "per_cat": 40,
        "label_prefix": ["abstract 3D object", "geometric art",
                         "artistic model", "abstract sculpture"],
    },
    "science-technology": {
        "per_cat": 40,
        "label_prefix": ["mechanical object", "sci-fi prop", "technology",
                         "machine", "industrial object", "device"],
    },
    "cultural-heritage-history": {
        "per_cat": 40,
        "label_prefix": ["historical artifact", "ancient object",
                         "cultural artifact", "archaeological model"],
    },
}

CAR_KEYWORDS = {
    "car", "cars", "vehicle", "vehicles", "automobile", "supercar",
    "sports car", "race car", "sedan", "coupe", "suv", "truck",
    "van", "bus", "lamborghini", "ferrari", "porsche", "tesla",
    "mustang", "bmw", "mercedes", "audi", "nissan", "toyota",
}


def _is_car_text(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    return any(k in t for k in CAR_KEYWORDS)


def _build_label(ann: dict, category: str) -> str:
    """Build a rich text label from objaverse annotation + category."""
    name = (ann.get("name") or "").strip()
    tags = ann.get("tags") or []
    tag_names = [t["name"] if isinstance(t, dict) else str(t) for t in tags[:8]]

    parts = []
    if name:
        parts.append(name.lower())
    if tag_names:
        parts.append(", ".join(tag_names[:4]))

    cfg = TARGET_CATEGORIES.get(category, {})
    prefixes = cfg.get("label_prefix", [category.replace("-", " ")])
    import random
    prefix = random.choice(prefixes)

    if parts:
        label = f"{prefix}: {' '.join(parts)}"
    else:
        label = prefix
    return label[:200]


def download_objaverse_by_category(
    per_cat: int = 60,
    processes: int = 4,
    category_filter: set[str] | None = None,
) -> int:
    """Download top-N models per category from Objaverse, sorted by likes."""
    try:
        import objaverse
    except ImportError:
        logger.error("objaverse not installed: pip install objaverse")
        return 0

    logger.info("Loading Objaverse v1 annotations (798K models)…")
    anns = objaverse.load_annotations()
    logger.info(f"Loaded {len(anns):,} annotations")

    # Build category → [(uid, ann, likes)] mapping
    cat_models: dict[str, list] = defaultdict(list)
    selected_categories = set(TARGET_CATEGORIES.keys())
    if category_filter:
        selected_categories = {
            c for c in selected_categories
            if c in category_filter
        }

    for uid, ann in anns.items():
        if not ann.get("name") or not ann.get("tags"):
            continue
        if not ann.get("isDownloadable", True):
            continue
        likes = ann.get("likeCount", 0)
        cats = ann.get("categories") or []
        cat_names = [c["name"] if isinstance(c, dict) else c for c in cats]
        for cat in cat_names:
            if cat in selected_categories:
                cat_models[cat].append((uid, ann, likes))

    logger.info("Category sizes (before cap):")
    for cat, items in sorted(cat_models.items()):
        logger.info(f"  {cat}: {len(items):,} total")

    # Sort each category by likes, cap at per_cat
    for cat in cat_models:
        cat_models[cat].sort(key=lambda x: x[2], reverse=True)
        cat_models[cat] = cat_models[cat][:per_cat]

    # Output dirs
    models_dir = BASE / "data" / "raw" / "objaverse" / "sketchfab" / "models"
    meta_dir = BASE / "data" / "raw" / "objaverse" / "sketchfab" / "metadata"
    proc_dir = BASE / "data" / "processed" / "objaverse"
    for d in [models_dir, meta_dir, proc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Track already-processed stems
    already = {f.stem for f in proc_dir.glob("*.json") if not f.name.endswith(".meta.json")}
    also_in_raw = {f.stem for f in models_dir.glob("*")}
    logger.info(f"Already processed: {len(already)}, in raw: {len(also_in_raw)}")

    # Collect UIDs to download (skip already done)
    uid_to_cat: dict[str, str] = {}
    uid_to_ann: dict[str, dict] = {}
    for cat, items in cat_models.items():
        for uid, ann, _ in items:
            safe = uid[:32]
            if safe not in already and safe not in also_in_raw:
                uid_to_cat[uid] = cat
                uid_to_ann[uid] = ann

    total_needed = len(uid_to_cat)
    logger.info(f"Need to download: {total_needed:,} models across {len(cat_models)} categories")
    if total_needed == 0:
        logger.info("Nothing to download — all categories satisfied!")
        return 0

    # Download in batches of 50
    batch_size = 50
    uids_list = list(uid_to_cat.keys())
    downloaded = 0
    extracted = 0

    from processing.mesh_extractor import process_file as extract_file
    from scrapers.utils import load_config
    extract_config = load_config()

    objaverse_cache = Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs"

    for batch_start in range(0, len(uids_list), batch_size):
        batch = uids_list[batch_start: batch_start + batch_size]
        logger.info(f"Batch {batch_start // batch_size + 1}/{(len(uids_list) + batch_size - 1) // batch_size}"
                    f" — downloading {len(batch)} models…")
        try:
            results = objaverse.load_objects(
                uids=batch,
                download_processes=min(processes, len(batch)),
            )
        except Exception as e:
            logger.warning(f"Batch download failed: {e}")
            continue

        for uid, local_path in results.items():
            if not local_path or not Path(local_path).exists():
                continue

            ann = uid_to_ann[uid]
            cat = uid_to_cat[uid]
            safe = uid[:32]
            ext = Path(local_path).suffix or ".glb"
            dst = models_dir / f"{safe}{ext}"

            # Copy to our raw dir
            if not dst.exists():
                try:
                    shutil.copy2(local_path, dst)
                except Exception as e:
                    logger.debug(f"Copy failed {uid}: {e}")
                    continue
            downloaded += 1

            # Build metadata
            tags = ann.get("tags") or []
            tag_names = [t["name"] if isinstance(t, dict) else t for t in tags[:15]]
            cats_raw = ann.get("categories") or []
            cat_names = [c["name"] if isinstance(c, dict) else c for c in cats_raw[:5]]
            label = _build_label(ann, cat)
            meta = {
                "source": "objaverse_xl_sketchfab",
                "uid": uid,
                "name": ann.get("name", ""),
                "description": (ann.get("description") or "")[:400],
                "tags": tag_names,
                "categories": cat_names,
                "label": label,
                "likeCount": ann.get("likeCount", 0),
                "license": str(ann.get("license", "")),
            }
            with open(meta_dir / f"{safe}.meta.json", "w") as f:
                json.dump(meta, f)

            # Extract mesh immediately → JSON in proc_dir
            out_json = proc_dir / f"{safe}.json"
            if not out_json.exists():
                try:
                    result = extract_file(dst, proc_dir, meta_dir, extract_config)
                    if result and result > 0:
                        # Inject label into the output JSON
                        for jf in proc_dir.glob(f"{safe}*.json"):
                            if jf.name.endswith(".meta.json"):
                                continue
                            try:
                                d = json.loads(jf.read_text())
                                if isinstance(d, dict) and "label" not in d:
                                    d["label"] = label
                                    jf.write_text(json.dumps(d))
                            except Exception:
                                pass
                        extracted += 1
                        logger.info(f"  ✓ [{cat}] {ann.get('name','?')[:50]} → {extracted} extracted")
                except Exception as e:
                    logger.debug(f"Extraction failed {uid}: {e}")

        # Clean objaverse cache to save disk space
        if objaverse_cache.exists():
            for glb in objaverse_cache.rglob("*.glb"):
                try:
                    glb.unlink()
                except Exception:
                    pass

        logger.info(f"Progress: {downloaded} downloaded, {extracted} extracted")
        time.sleep(0.5)

    logger.info(f"Objaverse pull complete: {downloaded} downloaded, {extracted} JSON extracted")
    return extracted


def download_blendswap_vehicles(max_pages: int = 20, max_items: int = 2000) -> int:
    """Download + extract BlendSwap vehicle category models."""
    try:
        from scrapers.blendswap_scraper import (
            CATEGORIES, create_session, get_listing_items, get_blend_detail,
        )
        from scrapers.smutbase_scraper import _extract_blend_from_archive
    except Exception as e:
        logger.warning(f"BlendSwap import failed: {e}")
        return 0

    vehicles_id = CATEGORIES.get("vehicles")
    if not vehicles_id:
        logger.warning("BlendSwap vehicles category missing")
        return 0

    raw_dir = BASE / "data" / "raw" / "blendswap" / "vehicles"
    proc_dir = BASE / "data" / "processed" / "blendswap"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    session = create_session()

    extracted = 0
    downloaded = 0
    seen_ids = set()

    from processing.mesh_extractor import process_file as extract_file
    from scrapers.utils import load_config
    extract_config = load_config()

    for item in get_listing_items(session, "vehicles", vehicles_id, max_pages=max_pages):
        blend_id = str(item.get("id", "")).strip()
        if not blend_id or blend_id in seen_ids:
            continue
        seen_ids.add(blend_id)

        detail = get_blend_detail(item.get("url", ""), session) or {}
        item.update(detail)

        title = str(item.get("title", ""))
        category = str(item.get("category", ""))
        tags = item.get("tags", []) or []
        if not _is_car_text(" ".join([title, category, " ".join(map(str, tags))])):
            continue

        dl_url = item.get("download_url")
        if not dl_url:
            continue

        blend_path = raw_dir / f"{blend_id}.blend"
        zip_path = raw_dir / f"{blend_id}.zip"
        sidecar = raw_dir / f"{blend_id}.blend.meta.json"

        if not blend_path.exists() and not zip_path.exists():
            try:
                resp = session.get(dl_url, stream=True, timeout=120)
                resp.raise_for_status()
                data = resp.content
                magic = data[:4]

                if magic == b"BLEN":
                    blend_path.write_bytes(data)
                    downloaded += 1
                elif data[:2] == b"PK":
                    zip_path.write_bytes(data)
                    downloaded += 1
                    extracted_blend = _extract_blend_from_archive(zip_path, raw_dir)
                    if extracted_blend and extracted_blend.exists():
                        blend_path = extracted_blend
                else:
                    continue
            except Exception:
                continue

        try:
            sidecar.write_text(json.dumps({
                "id": blend_id,
                "title": title,
                "description": item.get("description", ""),
                "tags": tags,
                "category": category,
                "license": item.get("license", ""),
                "url": item.get("url", ""),
            }, indent=2))
        except Exception:
            pass

        if blend_path.exists():
            out_json = proc_dir / f"{blend_path.stem}.json"
            if not out_json.exists():
                try:
                    result = extract_file(blend_path, proc_dir, raw_dir, extract_config)
                    if result and result > 0:
                        extracted += 1
                except Exception:
                    pass

        if downloaded >= max_items:
            break

    logger.info(f"BlendSwap vehicles pull complete: {downloaded} downloaded, {extracted} extracted")
    return extracted


def extract_existing_blends(max_blends: int = 138, cars_only: bool = False) -> int:
    """Extract mesh data from existing raw .blend files using Blender headless."""
    blender_bin = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if not blender_bin.exists():
        logger.warning("Blender not found at expected path — skipping .blend extraction")
        return 0

    blend_files = sorted(list(BASE.glob("data/raw/blendswap/**/*.blend")) +
                         list(BASE.glob("data/raw/open3dlab/**/*.blend")))

    if cars_only:
        filtered = []
        for blend_path in blend_files:
            meta_path = blend_path.parent / f"{blend_path.name}.meta.json"
            car_hit = _is_car_text(blend_path.stem)
            if meta_path.exists():
                try:
                    m = json.loads(meta_path.read_text())
                    joined = " ".join([
                        str(m.get("title", "")),
                        str(m.get("name", "")),
                        str(m.get("description", "")),
                        " ".join(map(str, m.get("tags", []) or [])),
                        str(m.get("category", "")),
                    ])
                    car_hit = car_hit or _is_car_text(joined)
                except Exception:
                    pass
            if car_hit:
                filtered.append(blend_path)
        blend_files = filtered

    blend_files = blend_files[:max_blends]
    if not blend_files:
        logger.info("No .blend files found in raw dirs")
        return 0

    # Find already-processed stems
    proc_dir = BASE / "data" / "processed" / "blendswap"
    proc_dir.mkdir(parents=True, exist_ok=True)
    already = {f.stem for f in proc_dir.glob("*.json")}

    to_process = [f for f in blend_files if f.stem not in already]
    logger.info(f"Blend extraction: {len(to_process)} new files (of {len(blend_files)} total)")

    extracted = 0
    for blend_path in to_process:
        out_json = proc_dir / f"{blend_path.stem}.json"
        if out_json.exists():
            continue

        # Read meta.json if it exists (for label)
        meta_path = blend_path.parent / f"{blend_path.stem}.blend.meta.json"
        label = blend_path.stem
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                label = meta.get("title", meta.get("name", label))
                tags = meta.get("tags", [])
                if tags:
                    label = f"{label}: {', '.join(tags[:5])}"
            except Exception:
                pass

        # Run blend_extractor via Blender headless
        extractor_script = BASE / "processing" / "blend_extractor.py"
        import subprocess
        cmd = [
            str(blender_bin), "--background",
            "--python", str(extractor_script),
            "--", "--input", str(blend_path),
            "--output", str(proc_dir),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0 and out_json.exists():
                extracted += 1
                logger.info(f"  ✓ [{extracted}] {blend_path.name[:50]}")
            else:
                logger.debug(f"  ✗ {blend_path.name}: {result.stderr[-200:]}")
        except subprocess.TimeoutExpired:
            logger.debug(f"  ✗ {blend_path.name}: timeout")
        except Exception as e:
            logger.debug(f"  ✗ {blend_path.name}: {e}")

    logger.info(f"Blend extraction complete: {extracted} extracted")
    return extracted


def rebuild_pt_cache() -> None:
    """Trigger a cache rebuild so new JSONs are converted to .pt files."""
    rebuild_script = BASE / "scripts" / "rebuild_cache.py"
    if not rebuild_script.exists():
        logger.warning("rebuild_cache.py not found — skipping")
        return

    import subprocess
    logger.info("Rebuilding .pt cache from new JSONs…")
    try:
        result = subprocess.run(
            [sys.executable, str(rebuild_script)],
            cwd=str(BASE), capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            logger.info("Cache rebuild done")
        else:
            logger.warning(f"Cache rebuild failed:\n{result.stderr[-400:]}")
    except subprocess.TimeoutExpired:
        logger.warning("Cache rebuild timed out (training will pick up new files anyway)")
    except Exception as e:
        logger.warning(f"Cache rebuild error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fetch diverse 3D training data")
    parser.add_argument("--objaverse", action="store_true", help="Only run Objaverse pull")
    parser.add_argument("--blends", action="store_true", help="Only extract .blend files")
    parser.add_argument("--per-cat", type=int, default=60,
                        help="Models per Objaverse category (default: 60)")
    parser.add_argument("--processes", type=int, default=4,
                        help="Download parallelism")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="Skip final cache rebuild (training auto-discovers anyway)")
    parser.add_argument("--cars-only", action="store_true",
                        help="Only pull car/vehicle-focused data")
    parser.add_argument("--bs-pages", type=int, default=20,
                        help="BlendSwap vehicles pages to scrape")
    parser.add_argument("--max-bs-items", type=int, default=2000,
                        help="Max BlendSwap vehicle downloads")
    parser.add_argument("--max-blends", type=int, default=5000,
                        help="Max existing .blend files to extract")
    args = parser.parse_args()

    run_all = not args.objaverse and not args.blends

    total = 0

    if run_all or args.objaverse:
        logger.info("=" * 60)
        logger.info("PHASE 1: Targeted Objaverse download")
        selected = {"cars-vehicles"} if args.cars_only else set(TARGET_CATEGORIES.keys())
        logger.info(f"  {len(selected)} categories × {args.per_cat} models = "
                    f"~{len(selected) * args.per_cat} target models")
        logger.info("=" * 60)
        n = download_objaverse_by_category(
            per_cat=args.per_cat,
            processes=args.processes,
            category_filter=selected,
        )
        total += n
        logger.info(f"Phase 1 done: {n} models extracted")

        if args.cars_only:
            logger.info("=" * 60)
            logger.info("PHASE 1b: BlendSwap vehicles download")
            logger.info("=" * 60)
            n_bs = download_blendswap_vehicles(
                max_pages=args.bs_pages,
                max_items=args.max_bs_items,
            )
            total += n_bs
            logger.info(f"Phase 1b done: {n_bs} vehicle models extracted")

    if run_all or args.blends:
        logger.info("=" * 60)
        logger.info("PHASE 2: Extract existing .blend files")
        logger.info("=" * 60)
        n = extract_existing_blends(
            max_blends=args.max_blends,
            cars_only=args.cars_only,
        )
        total += n
        logger.info(f"Phase 2 done: {n} blends extracted")

    if not args.no_rebuild:
        rebuild_pt_cache()

    logger.info("=" * 60)
    logger.info(f"TOTAL new training samples: {total}")
    logger.info("Training loop will pick up new data in the next scan cycle.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
