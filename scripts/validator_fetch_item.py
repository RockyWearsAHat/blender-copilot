#!/usr/bin/env python3
"""Fetch (materialize) the next unreviewed cache item for Blender validation.

Why this exists:
- Blender's Python should stay torch-free.
- But we still want to validate *exactly* what training uses:
  `data/processed/.mesh_cache/*.pt` items.

This script:
- Reads reviews from <work-dir>/reviews.jsonl (item_id set)
- Iterates cache .pt files to find the next unreviewed item
- Decodes mesh_tokens -> vertices/faces using MeshTokenizer
- Writes <work-dir>/items/<item_id>.json for Blender to load
- Prints a small JSON summary to stdout

Usage:
  ./.venv/bin/python scripts/validator_fetch_item.py \
    --cache-dir data/processed/.mesh_cache \
    --work-dir data/validation_queue_live

Fresh-scope only (newly regenerated/new pulls/generations):
    --fresh-only --fresh-hours 72

Resume after a particular item:
  --after-cache-pt /abs/path/to/cache.pt --after-item-index 123

Materialize a specific item:
  --cache-pt /abs/path/to/cache.pt --item-index 123
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import math
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MASTER_CACHE_DIR = PROJECT_ROOT / "data" / "master_cache"

if TYPE_CHECKING:
    from processing.mesh_tokenizer import MeshTokenizer


def _load_project_config(project_root: Path) -> dict[str, Any]:
    cfg_path = project_root / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}


def _guess_project_root(p: Path) -> Path:
    cur = p.resolve()
    if cur.is_file():
        cur = cur.parent
    for _ in range(6):
        if (cur / "config.yaml").exists() and (cur / "run.py").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return p.resolve() if p.exists() else Path.cwd().resolve()


def _build_tokenizer(project_root: Path) -> "MeshTokenizer":
    from processing.mesh_tokenizer import MeshTokenizer

    cfg = _load_project_config(project_root)
    tok = cfg.get("tokenization", {}) if isinstance(cfg, dict) else {}

    vocab_size = int(tok.get("vocab_size", 8192) or 8192)
    coord_range = tok.get("coord_range", (-1.0, 1.0))
    try:
        coord_range = (float(coord_range[0]), float(coord_range[1]))
    except Exception:
        coord_range = (-1.0, 1.0)

    max_faces = int(tok.get("max_faces", 4000) or 4000)
    use_vertex_indexed = bool(tok.get("use_vertex_indexed", False))
    max_vertices = int(tok.get("max_vertices", 4096) or 4096)

    return MeshTokenizer(
        vocab_size=vocab_size,
        coord_range=coord_range,
        max_faces=max_faces,
        use_vertex_indexed=use_vertex_indexed,
        max_vertices=max_vertices,
    )


def _infer_vocab_size_from_tokens(tokens: list[int], default_vocab: int) -> int:
    """Infer vocab size from observed token range.

    Many caches in this repo use vocab=1024 (see config.retrain_v1024_*).
    If we decode with the wrong vocab, meshes look like nonsense.
    """
    if not tokens:
        return int(default_vocab)
    try:
        m = int(max(tokens))
    except Exception:
        return int(default_vocab)
    # If tokens are already near the default vocab size, keep it.
    if m >= int(default_vocab) - 1:
        return int(default_vocab)

    # Smallest power-of-two vocab that can represent max token.
    # Clamp to a sane range.
    if m <= 0:
        return int(default_vocab)
    v = 2 ** int(math.ceil(math.log2(m + 1)))
    v = max(64, min(16384, int(v)))
    return v


def _item_id(cache_pt: Path, item_index: int) -> str:
    key = f"{cache_pt.resolve()}:{int(item_index)}".encode("utf-8")
    return hashlib.sha1(key).hexdigest()[:16]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if isinstance(v, torch.Tensor):
            return float(v.item())
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if isinstance(v, torch.Tensor):
            return int(v.item())
        return int(v)
    except Exception:
        return int(default)


def _load_reviews(work_dir: Path, *, min_ts: int | None = None) -> set[str]:
    reviewed: set[str] = set()
    p = work_dir / "reviews.jsonl"
    if not p.exists():
        return reviewed
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if min_ts is not None:
                ts = _safe_int(obj.get("ts"), 0)
                if ts <= 0 or ts < int(min_ts):
                    continue
            item_id = obj.get("item_id")
            if isinstance(item_id, str) and item_id:
                reviewed.add(item_id)
    except Exception:
        pass
    return reviewed


def _iter_cache_files(cache_dir: Path) -> list[Path]:
    files = sorted(cache_dir.glob("*.pt"))
    return [p for p in files if p.is_file()]


def _iter_cache_items(cache_pt: Path) -> list[dict[str, Any]]:
    obj = torch.load(cache_pt, map_location="cpu", weights_only=False)
    if isinstance(obj, list):
        return [it for it in obj if isinstance(it, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def _pick_tokens(item: dict[str, Any]) -> torch.Tensor | None:
    t = item.get("mesh_tokens")
    if isinstance(t, torch.Tensor):
        return t
    t = item.get("tokens")
    if isinstance(t, torch.Tensor):
        return t
    return None


def _first_non_empty_str(*values: Any) -> str:
    for value in values:
        if isinstance(value, str):
            s = value.strip()
            if s:
                return s
    return ""


def _infer_label(item: dict[str, Any]) -> str:
    label = _first_non_empty_str(item.get("label"))
    if label:
        return label

    ws = item.get("workflow_supervision")
    if isinstance(ws, dict):
        label = _first_non_empty_str(
            ws.get("target_instruction"),
            ws.get("initial_state_summary"),
        )
        if label:
            return label

    return ""


def _infer_data_source(item: dict[str, Any]) -> str:
    source = _first_non_empty_str(item.get("data_source"))
    if source:
        return source

    composition = item.get("composition")
    if isinstance(composition, dict):
        source = _first_non_empty_str(
            composition.get("dataset"),
            composition.get("source"),
            composition.get("category"),
        )
        if source:
            return source

    return "mesh_cache"


def _parse_fresh_since_epoch(*, fresh_since_epoch: float, fresh_hours: float) -> int | None:
    if float(fresh_since_epoch) > 0:
        return int(float(fresh_since_epoch))
    if float(fresh_hours) > 0:
        return int(time.time() - float(fresh_hours) * 3600.0)
    return None


def _item_timestamp_candidates(item: dict[str, Any]) -> list[int]:
    keys = (
        "generated_at",
        "created_at",
        "pull_ts",
        "pulled_at",
        "regen_ts",
        "regenerated_at",
        "human_validated_at",
    )
    vals: list[int] = []
    for k in keys:
        if k in item:
            ts = _safe_int(item.get(k), 0)
            if ts > 0:
                vals.append(ts)
    return vals


def _item_has_new_marker(item: dict[str, Any]) -> bool:
    bool_markers = (
        "is_generated",
        "is_regenerated",
        "is_new_pull",
        "fresh_pull",
        "new_generation",
    )
    for key in bool_markers:
        if bool(item.get(key, False)):
            return True

    sample_type = _first_non_empty_str(item.get("sample_type")).lower()
    if sample_type in {"generated", "generation", "new_pull", "regenerated"}:
        return True

    if _first_non_empty_str(item.get("pull_id")):
        return True

    return False


def _processed_source_path_for_item(item: dict[str, Any], project_root: Path) -> Path | None:
    mref = item.get("master_cache_ref")
    if not isinstance(mref, dict):
        return None
    src = _first_non_empty_str(mref.get("data_source"), item.get("data_source"))
    src_file = _first_non_empty_str(mref.get("source_file"))
    if not src or not src_file:
        return None
    p = (project_root / "data" / "processed" / src / src_file).resolve()
    if p.exists():
        return p
    return None


def _is_fresh_item(*,
                   item: dict[str, Any],
                   cache_pt: Path,
                   project_root: Path,
                   fresh_since_epoch: int | None,
                   cache_mtime_cache: dict[Path, int],
                   source_mtime_cache: dict[Path, int]) -> bool:
    # Marker-based freshness for generated/new-pull style samples.
    if _item_has_new_marker(item):
        return True

    if fresh_since_epoch is None:
        return False

    # Item-level timestamps.
    for ts in _item_timestamp_candidates(item):
        if ts >= int(fresh_since_epoch):
            return True

    # Cache file mtime (covers newly regenerated batches).
    c_mtime = cache_mtime_cache.get(cache_pt)
    if c_mtime is None:
        try:
            c_mtime = int(cache_pt.stat().st_mtime)
        except Exception:
            c_mtime = 0
        cache_mtime_cache[cache_pt] = c_mtime
    if c_mtime >= int(fresh_since_epoch):
        return True

    # Source JSON mtime (covers new pulls entering processed/).
    src_path = _processed_source_path_for_item(item, project_root)
    if src_path is not None:
        s_mtime = source_mtime_cache.get(src_path)
        if s_mtime is None:
            try:
                s_mtime = int(src_path.stat().st_mtime)
            except Exception:
                s_mtime = 0
            source_mtime_cache[src_path] = s_mtime
        if s_mtime >= int(fresh_since_epoch):
            return True

    return False


def _deep_json_safe(obj: Any) -> Any:
    """Recursively convert torch.Tensor / numpy values to JSON-safe types."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, dict):
        return {k: _deep_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_json_safe(x) for x in obj]
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return obj


def _extract_scene_context(item: dict[str, Any]) -> dict[str, Any]:
    sc = item.get("scene_context")
    if not isinstance(sc, dict):
        return {}
    return _deep_json_safe(sc)


def _decode_source_scene_from_master(master_entry: dict[str, Any]) -> dict[str, Any] | None:
    """Recover exact source scene JSON if embedded in master cache."""
    raw_scene = master_entry.get("source_json")
    if isinstance(raw_scene, dict):
        return _deep_json_safe(raw_scene)

    gz_payload = master_entry.get("source_json_gz")
    if gz_payload is None:
        return None

    try:
        if isinstance(gz_payload, torch.Tensor):
            gz_bytes = bytes(int(x) & 0xFF for x in gz_payload.flatten().tolist())
        elif isinstance(gz_payload, (bytes, bytearray)):
            gz_bytes = bytes(gz_payload)
        elif isinstance(gz_payload, list):
            gz_bytes = bytes(int(x) & 0xFF for x in gz_payload)
        else:
            return None
        data = json.loads(gzip.decompress(gz_bytes).decode("utf-8"))
        if isinstance(data, dict):
            return _deep_json_safe(data)
    except Exception:
        return None
    return None


def _load_exact_source_bytes(*,
                             master_entry: dict[str, Any],
                             data_source: str,
                             source_file: str) -> tuple[bytes | None, str]:
    """Load exact source JSON bytes from master entry or processed file."""
    gz_payload = master_entry.get("source_json_gz")
    if gz_payload is not None:
        try:
            if isinstance(gz_payload, torch.Tensor):
                gz_bytes = bytes(int(x) & 0xFF for x in gz_payload.flatten().tolist())
            elif isinstance(gz_payload, (bytes, bytearray)):
                gz_bytes = bytes(gz_payload)
            elif isinstance(gz_payload, list):
                gz_bytes = bytes(int(x) & 0xFF for x in gz_payload)
            else:
                gz_bytes = b""
            if gz_bytes:
                return gzip.decompress(gz_bytes), "master_cache"
        except Exception:
            pass

    if data_source and source_file:
        p = PROJECT_ROOT / "data" / "processed" / data_source / source_file
        try:
            if p.exists():
                return p.read_bytes(), "processed_json"
        except Exception:
            pass

    return None, "unavailable"


def _find_source_blend_path(*, data_source: str, source_file: str) -> str:
    """Best-effort lookup for the original .blend file behind a source JSON."""
    stem = Path(str(source_file or "")).stem
    if not stem:
        return ""

    blend_name = f"{stem}.blend"
    roots: list[Path] = []

    if data_source:
        roots.append(PROJECT_ROOT / "data" / "raw" / data_source)
    if data_source == "blender_official":
        roots.append(PROJECT_ROOT / "data" / "raw" / "blender_official")
        roots.append(PROJECT_ROOT / "data" / "raw" / "blender_official" / "models")

    for root in roots:
        try:
            if not root.exists():
                continue
            direct = root / blend_name
            if direct.exists():
                return str(direct.resolve())
            models_direct = root / "models" / blend_name
            if models_direct.exists():
                return str(models_direct.resolve())
            found = next(root.rglob(blend_name), None)
            if found is not None and found.exists():
                return str(found.resolve())
        except Exception:
            continue

    return ""


def _master_object_to_source_like(obj: dict[str, Any]) -> dict[str, Any]:
    """Convert flattened master-cache object back to source-like schema."""
    out: dict[str, Any] = {}

    # Rebuild mesh dict (matches extractor schema shape)
    mesh: dict[str, Any] = {}
    if "vertices" in obj:
        mesh["vertices"] = _deep_json_safe(obj.get("vertices"))
    if "faces" in obj:
        mesh["faces"] = _deep_json_safe(obj.get("faces"))
    if obj.get("normals") is not None:
        mesh["normals"] = _deep_json_safe(obj.get("normals"))
    if obj.get("face_material_indices") is not None:
        mesh["face_material_indices"] = _deep_json_safe(obj.get("face_material_indices"))
    if obj.get("face_smooth") is not None:
        mesh["face_smooth"] = _deep_json_safe(obj.get("face_smooth"))

    uv_layers = obj.get("uv_layers")
    if uv_layers:
        mesh["uv_layers"] = _deep_json_safe(uv_layers)

    vcol_layers = obj.get("vertex_color_layers")
    if vcol_layers:
        mesh["vertex_color_layers"] = _deep_json_safe(vcol_layers)

    if obj.get("normalization_center") is not None:
        mesh["normalization_center"] = _deep_json_safe(obj.get("normalization_center"))
    if obj.get("normalization_scale") is not None:
        mesh["normalization_scale"] = _deep_json_safe(obj.get("normalization_scale"))

    mesh_extra = obj.get("mesh_extra")
    if isinstance(mesh_extra, dict):
        for k, v in mesh_extra.items():
            if k not in mesh:
                mesh[k] = _deep_json_safe(v)

    # Preserve all other object keys as-is
    skip_keys = {
        "vertices", "faces", "normals", "face_material_indices", "face_smooth",
        "uv_layers", "vertex_color_layers", "mesh_extra",
        "normalization_center", "normalization_scale",
    }
    for k, v in obj.items():
        if k in skip_keys:
            continue
        out[k] = _deep_json_safe(v)

    if mesh:
        out["mesh"] = mesh
    return out


def _master_entry_to_source_like_scene(master_entry: dict[str, Any]) -> dict[str, Any]:
    """Build a source-like scene payload from a flattened master cache entry."""
    top_skip = {
        "_version", "_source_file", "_data_source", "_built_at",
        "objects", "summary", "animations", "material_names", "object_names",
        "per_object_labels", "source_extras",
    }
    scene: dict[str, Any] = {}

    for k, v in master_entry.items():
        if k in top_skip:
            continue
        scene[k] = _deep_json_safe(v)

    objects = master_entry.get("objects", [])
    if isinstance(objects, list):
        scene["objects"] = [
            _master_object_to_source_like(o)
            for o in objects
            if isinstance(o, dict)
        ]

    source_extras = master_entry.get("source_extras", {})
    if isinstance(source_extras, dict):
        for k, v in source_extras.items():
            if k not in scene:
                scene[k] = _deep_json_safe(v)
    return scene


def _load_master_object_from_ref(item: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    ref = item.get("master_cache_ref")
    if not isinstance(ref, dict):
        return None, None

    rel_path = ref.get("cache_rel_path")
    obj_index = ref.get("object_index")
    if not isinstance(rel_path, str) or not rel_path.strip():
        return None, None
    if not isinstance(obj_index, (int, float, str)):
        return None, None
    try:
        idx = int(obj_index)
    except Exception:
        return None, None
    if idx < 0:
        return None, None

    cache_path = MASTER_CACHE_DIR / rel_path
    if not cache_path.exists():
        return None, None

    try:
        master_entry = torch.load(cache_path, weights_only=False)
    except Exception:
        return None, None

    if not isinstance(master_entry, dict):
        return None, None
    objects = master_entry.get("objects", [])
    if not isinstance(objects, list) or idx >= len(objects):
        return None, None

    obj = objects[idx]
    if not isinstance(obj, dict):
        return None, None

    return obj, master_entry


def _extract_material_names(scene_context: dict[str, Any]) -> list[str]:
    out: list[str] = []
    mats = scene_context.get("materials")
    if not isinstance(mats, list):
        return out

    seen: set[str] = set()
    for m in mats:
        if not isinstance(m, dict):
            continue
        name = m.get("name")
        if not isinstance(name, str):
            continue
        clean = name.strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


def _build_scene_summary(scene_context: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(scene_context.keys())
    summary: dict[str, Any] = {
        "keys": keys,
        "has_materials": "materials" in scene_context,
        "has_images": "images" in scene_context,
        "has_file_context": "file_context" in scene_context,
    }
    mats = scene_context.get("materials")
    if isinstance(mats, list):
        summary["material_count"] = int(len(mats))
    images = scene_context.get("images")
    if isinstance(images, dict):
        summary["image_count"] = int(len(images))
    face_mats = scene_context.get("face_material_indices")
    if isinstance(face_mats, list):
        summary["face_material_index_count"] = int(len(face_mats))
    return summary


def _mesh_stats(verts: list[list[float]], faces: list[list[int]]) -> dict[str, Any]:
    face_count = int(len(faces))
    vertex_count = int(len(verts))
    if face_count <= 0 or vertex_count <= 0:
        return {
            "vertex_count": vertex_count,
            "face_count": face_count,
            "degenerate_face_count": 0,
            "degenerate_face_ratio": 0.0,
            "unique_vertex_count": 0,
            "unique_vertex_ratio": 0.0,
        }

    degenerate = 0
    for tri in faces:
        if not isinstance(tri, (list, tuple)) or len(tri) < 3:
            degenerate += 1
            continue
        try:
            a = verts[int(tri[0])]
            b = verts[int(tri[1])]
            c = verts[int(tri[2])]
            abx, aby, abz = (float(b[0] - a[0]), float(b[1] - a[1]), float(b[2] - a[2]))
            acx, acy, acz = (float(c[0] - a[0]), float(c[1] - a[1]), float(c[2] - a[2]))
            cx = aby * acz - abz * acy
            cy = abz * acx - abx * acz
            cz = abx * acy - aby * acx
            area2 = cx * cx + cy * cy + cz * cz
            if area2 <= 1e-12:
                degenerate += 1
        except Exception:
            degenerate += 1

    unique_vertices = len(
        {
            (
                round(float(v[0]), 5),
                round(float(v[1]), 5),
                round(float(v[2]), 5),
            )
            for v in verts
            if isinstance(v, (list, tuple)) and len(v) >= 3
        }
    )

    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "degenerate_face_count": int(degenerate),
        "degenerate_face_ratio": float(degenerate / max(1, face_count)),
        "unique_vertex_count": int(unique_vertices),
        "unique_vertex_ratio": float(unique_vertices / max(1, vertex_count)),
    }


def _weld_vertices(
    verts: list[list[float]],
    faces: list[list[int]],
    *,
    decimals: int = 5,
) -> tuple[list[list[float]], list[list[int]], dict[str, Any]]:
    """Merge duplicate vertices (quantization duplicates) and remap faces.

    MeshTokenizer.decode_tokens emits per-face vertices by design.
    For Blender validation we want a stitched manifold-like surface whenever
    coordinates match after quantization.
    """
    if not verts or not faces:
        return verts, faces, {
            "weld_applied": False,
            "vertex_count_before": int(len(verts)),
            "vertex_count_after": int(len(verts)),
            "merged_vertices": 0,
        }

    key_to_new: dict[tuple[float, float, float], int] = {}
    old_to_new: list[int] = [-1] * len(verts)
    welded_verts: list[list[float]] = []

    for i, v in enumerate(verts):
        if not isinstance(v, (list, tuple)) or len(v) < 3:
            continue
        key = (
            round(float(v[0]), decimals),
            round(float(v[1]), decimals),
            round(float(v[2]), decimals),
        )
        new_idx = key_to_new.get(key)
        if new_idx is None:
            new_idx = len(welded_verts)
            key_to_new[key] = new_idx
            welded_verts.append([float(v[0]), float(v[1]), float(v[2])])
        old_to_new[i] = new_idx

    welded_faces: list[list[int]] = []
    dropped = 0
    for tri in faces:
        if not isinstance(tri, (list, tuple)) or len(tri) < 3:
            dropped += 1
            continue
        try:
            a = old_to_new[int(tri[0])]
            b = old_to_new[int(tri[1])]
            c = old_to_new[int(tri[2])]
        except Exception:
            dropped += 1
            continue
        if a < 0 or b < 0 or c < 0:
            dropped += 1
            continue
        if len({a, b, c}) < 3:
            dropped += 1
            continue
        welded_faces.append([a, b, c])

    before = int(len(verts))
    after = int(len(welded_verts))
    return welded_verts, welded_faces, {
        "weld_applied": bool(after < before),
        "vertex_count_before": before,
        "vertex_count_after": after,
        "merged_vertices": int(max(0, before - after)),
        "dropped_faces": int(dropped),
    }


def _build_flags(*,
                label: str,
                quality_weight: float,
                human_verdict: str,
                stats: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if not label:
        flags.append("missing_label")
    if quality_weight <= 0.0:
        flags.append("quality_weight=0")
    if human_verdict == "reject":
        flags.append("human_rejected")

    face_count = int(stats.get("face_count", 0) or 0)
    deg_ratio = float(stats.get("degenerate_face_ratio", 0.0) or 0.0)
    if face_count < 8:
        flags.append("very_low_face_count")
    if deg_ratio > 0.15:
        flags.append("high_degenerate_faces")

    return flags


def _materialize(*, work_dir: Path, tokenizer: "MeshTokenizer", cache_pt: Path, item_index: int) -> dict[str, Any]:
    items = _iter_cache_items(cache_pt)
    if not (0 <= int(item_index) < len(items)):
        raise SystemExit(f"item_index out of range: {cache_pt.name}[{item_index}] len={len(items)}")

    it = items[int(item_index)]

    # IMPORTANT: preview geometry must match trainable geometry semantics.
    # Prefer explicit RAW_LOCAL item geometry when present (and not over-budget
    # full-res storage), otherwise decode mesh_tokens (the actual training path).
    # Do NOT take display geometry from master-cache object vertices because
    # those may be evaluated/baked scene outputs.
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    verts_raw: list[list[float]] = []
    faces_raw: list[list[int]] = []
    weld_info: dict[str, Any] = {}
    tokens: list[int] = []

    raw_v = it.get("raw_vertices")
    raw_f = it.get("raw_faces")
    item_geometry_space = _first_non_empty_str(it.get("geometry_space")).upper()
    item_geometry_is_baked = bool(it.get("geometry_is_baked", False))
    over_budget = bool(it.get("over_budget", False))
    used_raw = False

    master_obj, master_entry = _load_master_object_from_ref(it)
    if (
        raw_v is not None
        and raw_f is not None
        and item_geometry_space == "RAW_LOCAL"
        and not item_geometry_is_baked
        and not over_budget
    ):
        try:
            v_np = raw_v.numpy() if isinstance(raw_v, torch.Tensor) else raw_v
            f_np = raw_f.numpy() if isinstance(raw_f, torch.Tensor) else raw_f
            verts = [[float(x) for x in row] for row in v_np]
            faces = [[int(x) for x in row] for row in f_np]
            if len(verts) > 0 and len(faces) > 0:
                used_raw = True
        except Exception:
            pass

    if not used_raw:
        tokens_t = _pick_tokens(it)
        if tokens_t is None:
            raise SystemExit(f"cache item missing mesh_tokens: {cache_pt.name}[{item_index}]")

        tokens = [int(x) for x in tokens_t.flatten().tolist()]

        # Decode with the correct per-item vocab size.
        inferred_vocab = _infer_vocab_size_from_tokens(tokens, tokenizer.vocab_size)
        if inferred_vocab != tokenizer.vocab_size:
            tok_cls = type(tokenizer)
            tokenizer = tok_cls(
                vocab_size=inferred_vocab,
                coord_range=(float(tokenizer.coord_min), float(tokenizer.coord_max)),
                max_faces=tokenizer.max_faces,
                use_vertex_indexed=tokenizer.use_vertex_indexed,
                max_vertices=tokenizer.max_vertices,
            )

        verts_raw, faces_raw = tokenizer.decode_tokens(tokens)
        verts, faces, weld_info = _weld_vertices(verts_raw, faces_raw)

    weld_info = {} if used_raw else weld_info

    # Ensure tokens list is available for token_len even on raw path.
    if used_raw:
        tokens_t = _pick_tokens(it)
        tokens = [int(x) for x in tokens_t.flatten().tolist()] if tokens_t is not None else []

    item_id = _item_id(cache_pt, item_index)

    label = _infer_label(it)

    tags = it.get("user_tags")
    if not isinstance(tags, list):
        tags = []

    data_source = _infer_data_source(it)

    human_verdict = _first_non_empty_str(it.get("human_verdict")).lower()

    qw = it.get("quality_weight")
    try:
        raw_qw = qw.item() if isinstance(qw, torch.Tensor) else qw
        quality_weight = float(raw_qw) if raw_qw is not None else 0.0
    except Exception:
        quality_weight = 0.0

    stats = _mesh_stats(verts, faces)
    if used_raw:
        stats["raw_vertex_count"] = int(len(verts))
        stats["raw_face_count"] = int(len(faces))
        stats["weld_info"] = {"weld_applied": False, "used_raw_geometry": True}
        stats["geometry_source"] = "raw_vertices/raw_faces"
    else:
        stats["raw_vertex_count"] = int(len(verts_raw))
        stats["raw_face_count"] = int(len(faces_raw))
        stats["weld_info"] = weld_info
        stats["geometry_source"] = "decoded_tokens"
    flags = _build_flags(
        label=label,
        quality_weight=quality_weight,
        human_verdict=human_verdict,
        stats=stats,
    )
    is_trainable = bool(quality_weight > 0.0 and human_verdict != "reject")
    sample_type = _first_non_empty_str(it.get("sample_type")) or "object"
    scene_context = _extract_scene_context(it)
    if master_obj is not None and master_entry is not None:
        merged_scene = dict(scene_context)
        merged_scene.setdefault("materials", _deep_json_safe(master_obj.get("materials", [])))
        if "face_material_indices" not in merged_scene and master_obj.get("face_material_indices") is not None:
            merged_scene["face_material_indices"] = _deep_json_safe(master_obj.get("face_material_indices"))
        if "face_smooth" not in merged_scene and master_obj.get("face_smooth") is not None:
            merged_scene["face_smooth"] = _deep_json_safe(master_obj.get("face_smooth"))
        merged_scene.setdefault("transforms", _deep_json_safe(master_obj.get("transforms", {})))
        merged_scene.setdefault("uv_layers", _deep_json_safe(master_obj.get("uv_layers", {})))
        merged_scene.setdefault("vertex_color_layers", _deep_json_safe(master_obj.get("vertex_color_layers", {})))
        merged_scene.setdefault("modifiers", _deep_json_safe(master_obj.get("modifiers", [])))
        merged_scene.setdefault("shape_keys", _deep_json_safe(master_obj.get("shape_keys", [])))
        merged_scene.setdefault("object_context", _deep_json_safe(master_obj))
        merged_scene.setdefault(
            "file_context",
            _deep_json_safe(
                {
                    "source_file": master_entry.get("source_file"),
                    "source_format": master_entry.get("source_format"),
                    "label": master_entry.get("label"),
                    "metadata": master_entry.get("metadata", {}),
                    "world": master_entry.get("world", {}),
                    "render": master_entry.get("render", {}),
                    "fps": master_entry.get("fps"),
                    "frame_start": master_entry.get("frame_start"),
                    "frame_end": master_entry.get("frame_end"),
                    "collections": master_entry.get("collections", []),
                    "orphan_materials": master_entry.get("orphan_materials", []),
                    "orphan_actions": master_entry.get("orphan_actions", []),
                    "node_groups": master_entry.get("node_groups", []),
                    "source_extras": master_entry.get("source_extras", {}),
                    "summary": master_entry.get("summary", {}),
                }
            ),
        )
        scene_context = merged_scene
    material_names = _extract_material_names(scene_context)
    scene_summary = _build_scene_summary(scene_context)

    out_dir = work_dir / "items"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{item_id}.json"
    scene_path = out_dir / f"{item_id}.scene_context.json"

    payload = {
        "item_id": item_id,
        "cache_pt": str(cache_pt.resolve()),
        "item_index": int(item_index),
        "label": label,
        "tags": tags,
        "data_source": data_source,
        "quality_weight": quality_weight,
        "human_verdict": human_verdict,
        "is_trainable": is_trainable,
        "token_len": int(len(tokens)),
        "mesh_stats": stats,
        "flags": flags,
        "sample_type": sample_type,
        "material_names": material_names,
        "scene_context": scene_context,
        "scene_summary": scene_summary,
        "raw_item_keys": sorted(str(k) for k in it.keys()),
        "vertices": verts,
        "faces": faces,
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")

    scene_path.write_text(
        json.dumps(
            {
                "item_id": item_id,
                "cache_pt": str(cache_pt.resolve()),
                "item_index": int(item_index),
                "label": label,
                "sample_type": sample_type,
                "material_names": material_names,
                "scene_summary": scene_summary,
                "scene_context": scene_context,
            }
        ),
        encoding="utf-8",
    )

    return {
        "ok": True,
        "item_id": item_id,
        "item_json": str(out_path),
        "cache_pt": str(cache_pt.resolve()),
        "item_index": int(item_index),
        "label": label,
        "tags": tags,
        "data_source": data_source,
        "quality_weight": quality_weight,
        "human_verdict": human_verdict,
        "is_trainable": is_trainable,
        "token_len": int(len(tokens)),
        "mesh_stats": stats,
        "flags": flags,
        "sample_type": sample_type,
        "material_names": material_names,
        "scene_summary": scene_summary,
        "scene_context_keys": scene_summary.get("keys", []),
        "scene_json": str(scene_path),
    }


def _find_next(*,
              cache_dir: Path,
              reviewed: set[str],
              after_cache_pt: str,
              after_item_index: int,
              fresh_only: bool,
              fresh_since_epoch: int | None,
              project_root: Path) -> tuple[Path, int] | None:
    files = _iter_cache_files(cache_dir)
    after_cache_pt_res = ""
    if after_cache_pt:
        try:
            after_cache_pt_res = str(Path(after_cache_pt).resolve())
        except Exception:
            after_cache_pt_res = after_cache_pt

    def _is_preferred(item: dict[str, Any]) -> bool:
        sc = item.get("scene_context") if isinstance(item, dict) else None
        mats = sc.get("materials") if isinstance(sc, dict) else None
        has_materials = (
            isinstance(sc, dict)
            and isinstance(mats, list)
            and len(mats) > 0
        )
        over_budget = bool(item.get("over_budget", False)) if isinstance(item, dict) else False
        return has_materials and not over_budget

    started = not bool(after_cache_pt_res)
    fallback_candidate: tuple[Path, int] | None = None
    cache_mtime_cache: dict[Path, int] = {}
    source_mtime_cache: dict[Path, int] = {}
    for cache_pt in files:
        cache_res = str(cache_pt.resolve())
        if not started:
            if cache_res == after_cache_pt_res:
                started = True
            else:
                continue

        items = _iter_cache_items(cache_pt)
        start_i = 0
        if cache_res == after_cache_pt_res and after_item_index >= 0:
            start_i = int(after_item_index) + 1

        for i in range(start_i, len(items)):
            item_id = _item_id(cache_pt, i)
            if item_id in reviewed:
                continue
            item = items[i]
            if fresh_only and not _is_fresh_item(
                item=item,
                cache_pt=cache_pt,
                project_root=project_root,
                fresh_since_epoch=fresh_since_epoch,
                cache_mtime_cache=cache_mtime_cache,
                source_mtime_cache=source_mtime_cache,
            ):
                continue
            if _is_preferred(item):
                return cache_pt, i
            if fallback_candidate is None:
                fallback_candidate = (cache_pt, i)

    if fallback_candidate is not None:
        return fallback_candidate

    # If the provided "after" pointer wasn't found, fall back to start.
    if after_cache_pt_res and not started:
        return _find_next(
            cache_dir=cache_dir,
            reviewed=reviewed,
            after_cache_pt="",
            after_item_index=-1,
            fresh_only=fresh_only,
            fresh_since_epoch=fresh_since_epoch,
            project_root=project_root,
        )

    return None


def _reconstruct_scene(*, work_dir: Path, cache_pt: Path, item_index: int) -> dict[str, Any]:
    """Load ALL sibling objects from the same source file as the given item.

    Returns a JSON-serializable dict with every object's vertices, faces,
    materials, transforms, and labels — enough for Blender to reconstruct
    the full scene the training item was extracted from.
    """
    items = _iter_cache_items(cache_pt)
    if not (0 <= int(item_index) < len(items)):
        return {"ok": False, "error": f"item_index out of range: {cache_pt.name}[{item_index}]"}

    it = items[int(item_index)]
    ref = it.get("master_cache_ref")
    if not isinstance(ref, dict):
        return {"ok": False, "error": "Item has no master_cache_ref — cannot reconstruct scene"}

    rel_path = ref.get("cache_rel_path")
    if not isinstance(rel_path, str) or not rel_path.strip():
        return {"ok": False, "error": "master_cache_ref missing cache_rel_path"}

    master_path = MASTER_CACHE_DIR / rel_path
    if not master_path.exists():
        return {"ok": False, "error": f"Master cache file not found: {rel_path}"}

    try:
        master_entry = torch.load(master_path, weights_only=False)
    except Exception as e:
        return {"ok": False, "error": f"Failed to load master cache: {e}"}

    if not isinstance(master_entry, dict):
        return {"ok": False, "error": "Master cache entry is not a dict"}

    objects = master_entry.get("objects", [])
    if not isinstance(objects, list) or not objects:
        return {"ok": False, "error": "Master cache has no objects"}

    # Extract file-level metadata
    source_file = str(master_entry.get("source_file", master_entry.get("_source_file", "")))
    file_label = str(master_entry.get("label", ""))
    data_source = _first_non_empty_str(
        ref.get("data_source"),
        it.get("data_source"),
        master_entry.get("_data_source"),
    )
    metadata = master_entry.get("metadata", {})
    summary = master_entry.get("summary", {})
    exact_source_bytes, source_scene_origin = _load_exact_source_bytes(
        master_entry=master_entry,
        data_source=data_source,
        source_file=source_file,
    )
    source_scene_exact_sha1 = ""
    source_scene_exact_match_master = False
    source_scene: dict[str, Any] | None = None
    source_blend_path = _find_source_blend_path(
        data_source=data_source,
        source_file=source_file,
    )
    if exact_source_bytes:
        try:
            source_scene_exact_sha1 = hashlib.sha1(exact_source_bytes).hexdigest()
            source_scene = json.loads(exact_source_bytes.decode("utf-8"))
        except Exception:
            source_scene = None

    if source_scene is None:
        source_scene = (
            _decode_source_scene_from_master(master_entry)
            or _master_entry_to_source_like_scene(master_entry)
        )

    master_sha = _first_non_empty_str(
        master_entry.get("source_json_raw_sha1"),
        master_entry.get("source_json_sha1"),
    )
    if master_sha and source_scene_exact_sha1:
        source_scene_exact_match_master = (master_sha == source_scene_exact_sha1)

    scene_objects: list[dict[str, Any]] = []
    scene_cameras: list[dict[str, Any]] = []
    scene_lights: list[dict[str, Any]] = []
    current_object_index = int(ref.get("object_index", -1))

    for oi, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        obj_type = str(obj.get("type", "MESH"))
        obj_name = str(obj.get("name", f"Object_{oi}"))
        transforms = _deep_json_safe(obj.get("transforms", {}))

        # ── Cameras ──────────────────────────────────────────────────────────
        if obj_type == "CAMERA":
            cam_data = _deep_json_safe(obj.get("camera", {}))
            if cam_data or transforms:
                scene_cameras.append({
                    "name": obj_name,
                    "transforms": transforms,
                    "camera": cam_data,
                    "is_active": (obj_name == str(
                        master_entry.get("source_extras", {}).get("active_camera", "")
                    )),
                })
            continue

        # ── Lights ───────────────────────────────────────────────────────────
        if obj_type == "LIGHT":
            light_data = _deep_json_safe(obj.get("light", {}))
            if light_data or transforms:
                scene_lights.append({
                    "name": obj_name,
                    "transforms": transforms,
                    "light": light_data,
                })
            continue

        # ── Mesh objects ─────────────────────────────────────────────────────
        v_raw = obj.get("vertices")
        f_raw = obj.get("faces")
        inst = obj.get("instances")
        if v_raw is None or f_raw is None:
            # Allow instancing-only objects with no baked mesh
            if inst:
                scene_objects.append({
                    "object_index": int(oi),
                    "name": obj_name,
                    "label": str(obj.get("text_label", "")),
                    "is_current_item": (oi == current_object_index),
                    "vertex_count": 0,
                    "face_count": 0,
                    "vertices": [],
                    "faces": [],
                    "transforms": transforms,
                    "denorm_offset": [0.0, 0.0, 0.0],
                    "denorm_scale": 1.0,
                    "materials": _deep_json_safe(obj.get("materials", [])),
                    "instances": _deep_json_safe(inst),
                })
            continue

        try:
            v_np = v_raw.numpy() if isinstance(v_raw, torch.Tensor) else v_raw
            f_np = f_raw.numpy() if isinstance(f_raw, torch.Tensor) else f_raw
            verts = [[float(x) for x in row] for row in v_np]
            faces = [[int(x) for x in row] for row in f_np]
        except Exception:
            continue

        if not verts or not faces:
            # Geometry may be empty for instancers; keep if instance transforms exist
            if inst:
                scene_objects.append({
                    "object_index": int(oi),
                    "name": obj_name,
                    "label": str(obj.get("text_label", "")),
                    "is_current_item": (oi == current_object_index),
                    "vertex_count": 0,
                    "face_count": 0,
                    "vertices": [],
                    "faces": [],
                    "transforms": transforms,
                    "denorm_offset": [0.0, 0.0, 0.0],
                    "denorm_scale": 1.0,
                    "materials": _deep_json_safe(obj.get("materials", [])),
                    "instances": _deep_json_safe(inst),
                })
            continue

        # Extract per-object info
        obj_label = str(obj.get("text_label", ""))
        materials = _deep_json_safe(obj.get("materials", []))
        face_material_indices = _deep_json_safe(obj.get("face_material_indices"))
        face_smooth = _deep_json_safe(obj.get("face_smooth"))

        # Get normalization parameters.
        # blend_extractor nests them inside obj["mesh"]; build_master_cache may
        # hoist them to the top level.  Try both.
        _mesh_sub = obj.get("mesh") if isinstance(obj.get("mesh"), dict) else None
        norm_center = None
        norm_scale = None
        for _src in ([_mesh_sub] if _mesh_sub else []) + [obj]:
            if _src is None:
                continue
            _nc = _src.get("normalization_center")
            _ns = _src.get("normalization_scale")
            if _nc is not None and _ns is not None:
                norm_center = _nc
                norm_scale = _ns
                break

        # Compute denormalization parameters for the Blender operator.
        # We pass NORMALIZED vertices + denorm params so Blender can set
        # proper object location/scale (gives correct pivot/origin).
        denorm_offset = [0.0, 0.0, 0.0]
        denorm_scale = 1.0
        if norm_center is not None and norm_scale is not None:
            try:
                denorm_offset = [float(c) for c in norm_center]
                denorm_scale = float(norm_scale)
            except Exception:
                pass
        elif transforms:
            # Fallback for older data without normalization params:
            # location ≈ vertex centroid, max(dims)/2 ≈ normalization scale
            loc = transforms.get("location", [0, 0, 0])
            dims = _deep_json_safe(obj.get("dimensions", [0, 0, 0]))
            try:
                loc = [float(x) for x in loc]
                dims = [float(x) for x in dims]
                max_dim = max(dims) if dims else 0
                denorm_scale = max_dim / 2.0 if max_dim > 0.001 else 1.0
                denorm_offset = loc
            except Exception:
                pass

        scene_obj = {
            "object_index": int(oi),
            "name": obj_name,
            "label": obj_label,
            "is_current_item": (oi == current_object_index),
            "vertex_count": int(len(verts)),
            "face_count": int(len(faces)),
            "vertices": verts,   # keep normalized [-1,1]
            "faces": faces,
            "transforms": transforms,
            "denorm_offset": denorm_offset,
            "denorm_scale": denorm_scale,
            "materials": materials,
            "collections": _deep_json_safe(obj.get("collections", [])),
            "visible": bool(obj.get("visible", True)),
            "hide_viewport": bool(obj.get("hide_viewport", False)),
            "hide_render": bool(obj.get("hide_render", False)),
            "hide_select": bool(obj.get("hide_select", False)),
        }

        # Raw local-space mesh (pre-modifier) for higher-fidelity reconstruction.
        rv = obj.get("raw_vertices")
        rf = obj.get("raw_faces")
        if isinstance(rv, (list, tuple)) and isinstance(rf, (list, tuple)) and rv and rf:
            scene_obj["raw_vertices"] = _deep_json_safe(rv)
            scene_obj["raw_faces"] = _deep_json_safe(rf)
            scene_obj["geometry_space"] = "RAW_LOCAL"
        else:
            scene_obj["geometry_space"] = "EVALUATED_WORLD"

        if inst:
            scene_obj["instances"] = _deep_json_safe(inst)
        if face_material_indices is not None:
            scene_obj["face_material_indices"] = face_material_indices
        if face_smooth is not None:
            scene_obj["face_smooth"] = face_smooth
        # Modifier stack — needed for visual reconstruction
        mods = obj.get("modifiers")
        if mods:
            scene_obj["modifiers"] = _deep_json_safe(mods)
        # Rigid body physics — needed for physics scene reconstruction
        rb = obj.get("rigid_body")
        if rb:
            scene_obj["rigid_body"] = _deep_json_safe(rb)
        # UV layers — needed for proper texture mapping display
        uv = obj.get("uv_layers")
        if uv:
            scene_obj["uv_layers"] = _deep_json_safe(uv)

        # NOTE: raw_vertices are in local object space and are intentionally
        # NOT used for scene reconstruction — the evaluated vertices above are
        # already in world space and are the correct input for visual display.

        # Animation — keyframes + action name per object
        kf = obj.get("keyframes")
        if kf:
            scene_obj["keyframes"] = _deep_json_safe(kf)
        an = obj.get("action_name")
        if an:
            scene_obj["action_name"] = str(an)

        # Particle systems (hair, emitter, physics)
        ps = obj.get("particle_systems")
        if ps:
            scene_obj["particle_systems"] = _deep_json_safe(ps)

        # Object constraints
        cns = obj.get("constraints")
        if cns:
            scene_obj["constraints"] = _deep_json_safe(cns)

        # Drivers
        drv = obj.get("drivers")
        if drv:
            scene_obj["drivers"] = _deep_json_safe(drv)

        # Shape keys
        sk = obj.get("shape_keys")
        if sk:
            scene_obj["shape_keys"] = _deep_json_safe(sk)

        scene_objects.append(scene_obj)

    if not scene_objects and not scene_cameras and not scene_lights:
        return {"ok": False, "error": "No valid objects found in master cache entry"}

    # Extract world/environment and image data
    world_data = _deep_json_safe(master_entry.get("world", {}))
    render_data = _deep_json_safe(master_entry.get("render", {}))
    node_groups_data = _deep_json_safe(master_entry.get("node_groups", []))
    # Images: include image_data (full-res PNG) + thumbnail + metadata; skip raw float pixel arrays
    raw_images = master_entry.get("images", {})
    images_data: dict[str, Any] = {}
    if isinstance(raw_images, dict):
        for img_name, img_info in raw_images.items():
            if isinstance(img_info, dict):
                images_data[img_name] = {
                    k: v for k, v in img_info.items()
                    # skip only raw float pixel arrays (large/redundant with image_data PNG)
                    if k not in ("pixels", "pixel_data")
                }

    # Write scene JSON
    out_dir = work_dir / "scenes"
    out_dir.mkdir(parents=True, exist_ok=True)
    item_id = _item_id(cache_pt, item_index)
    scene_path = out_dir / f"{item_id}_scene.json"
    source_scene_exact_path = out_dir / f"{item_id}_source_exact.json"
    if exact_source_bytes:
        try:
            source_scene_exact_path.write_bytes(exact_source_bytes)
        except Exception:
            pass

    payload = {
        "ok": True,
        "item_id": item_id,
        "source_file": source_file,
        "data_source": data_source,
        "file_label": file_label,
        "metadata": _deep_json_safe(metadata) if isinstance(metadata, dict) else {},
        "summary": _deep_json_safe(summary) if isinstance(summary, dict) else {},
        "total_objects": int(len(scene_objects)),
        "current_object_index": current_object_index,
        "scene_json": str(scene_path),
        "objects": scene_objects,
        "objects_full": _deep_json_safe(source_scene.get("objects", [])) if isinstance(source_scene, dict) else [],
        "cameras": scene_cameras,
        "lights": scene_lights,
        "world": world_data,
        "render": render_data,
        "node_groups": node_groups_data,
        "images": images_data,
        "frame_start": int(master_entry.get("frame_start", 1)),
        "frame_end": int(master_entry.get("frame_end", 250)),
        "frame_current": int(
            (master_entry.get("source_extras", {}) or {}).get(
                "frame_current",
                master_entry.get("frame_start", 1),
            )
        ),
        "fps": int(master_entry.get("fps", 24)),
        "nla_tracks": _deep_json_safe(master_entry.get("nla_tracks", [])),
        # Lossless scene payload for exact downstream extraction.
        "source_scene": source_scene,
        "source_scene_origin": source_scene_origin,
        "source_blend_path": source_blend_path,
        "source_scene_exact_json": (str(source_scene_exact_path)
                                     if source_scene_exact_path.exists() else ""),
        "source_scene_exact_sha1": source_scene_exact_sha1,
        "source_scene_master_sha1": master_sha,
        "source_scene_exact_match_master": bool(source_scene_exact_match_master),
    }

    scene_path.write_text(json.dumps(payload), encoding="utf-8")

    # Return lightweight summary to stdout (no geometry)
    stdout_summary = {
        "ok": True,
        "item_id": item_id,
        "source_file": source_file,
        "data_source": data_source,
        "file_label": file_label,
        "total_objects": int(len(scene_objects)),
        "total_cameras": int(len(scene_cameras)),
        "total_lights": int(len(scene_lights)),
        "has_world": bool(world_data),
        "has_images": bool(images_data),
        "current_object_index": current_object_index,
        "scene_json": str(scene_path),
        "source_blend_path": source_blend_path,
        "source_scene_exact_json": (str(source_scene_exact_path)
                         if source_scene_exact_path.exists() else ""),
        "source_scene_exact_sha1": source_scene_exact_sha1,
        "source_scene_exact_match_master": bool(source_scene_exact_match_master),
        "objects": [
            {
                "object_index": o["object_index"],
                "name": o["name"],
                "label": o["label"],
                "is_current_item": o["is_current_item"],
                "vertex_count": o["vertex_count"],
                "face_count": o["face_count"],
            }
            for o in scene_objects
        ],
    }
    return stdout_summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)

    # Resume pointer (optional)
    p.add_argument("--after-cache-pt", type=str, default="")
    p.add_argument("--after-item-index", type=int, default=-1)

    # Exact item materialization (optional)
    p.add_argument("--cache-pt", type=Path, default=None)
    p.add_argument("--item-index", type=int, default=-1)

    # Fresh-scope filter (optional)
    p.add_argument("--fresh-only", action="store_true",
                   help="Only select newly regenerated/new-pull/generated items")
    p.add_argument("--fresh-hours", type=float, default=0.0,
                   help="Treat items as fresh if they are within this time window")
    p.add_argument("--fresh-since-epoch", type=float, default=0.0,
                   help="Unix epoch (seconds): treat items as fresh if newer than this")
    p.add_argument("--reconstruct-scene", action="store_true",
                   help="Reconstruct full scene (all sibling objects) for the given --cache-pt/--item-index")

    args = p.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if not cache_dir.exists():
        raise SystemExit(f"cache-dir not found: {cache_dir}")

    project_root = _guess_project_root(cache_dir)
    tokenizer = _build_tokenizer(project_root)

    fresh_since_epoch = _parse_fresh_since_epoch(
        fresh_since_epoch=float(args.fresh_since_epoch),
        fresh_hours=float(args.fresh_hours),
    )

    reviewed = _load_reviews(
        work_dir,
        min_ts=fresh_since_epoch if args.fresh_only else None,
    )

    if args.reconstruct_scene:
        if args.cache_pt is None or int(args.item_index) < 0:
            raise SystemExit("--reconstruct-scene requires --cache-pt and --item-index")
        out = _reconstruct_scene(
            work_dir=work_dir,
            cache_pt=Path(args.cache_pt).resolve(),
            item_index=int(args.item_index),
        )
        print(json.dumps(out))
        return

    if args.cache_pt is not None and int(args.item_index) >= 0:
        out = _materialize(work_dir=work_dir, tokenizer=tokenizer, cache_pt=Path(args.cache_pt).resolve(), item_index=int(args.item_index))
        print(json.dumps(out))
        return

    nxt = _find_next(
        cache_dir=cache_dir,
        reviewed=reviewed,
        after_cache_pt=args.after_cache_pt,
        after_item_index=int(args.after_item_index),
        fresh_only=bool(args.fresh_only),
        fresh_since_epoch=fresh_since_epoch,
        project_root=project_root,
    )

    if nxt is None:
        print(json.dumps({"ok": False, "done": True, "message": "no more unreviewed items"}))
        return

    cache_pt, item_index = nxt
    out = _materialize(work_dir=work_dir, tokenizer=tokenizer, cache_pt=cache_pt, item_index=item_index)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
