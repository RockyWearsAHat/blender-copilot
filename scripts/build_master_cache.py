#!/usr/bin/env python3
"""Build a LOSSLESS master cache from all source JSONs.

The master cache preserves ALL data from every source file with zero
truncation, zero filtering, and zero tokenization.  It converts slow-to-parse
JSON files (~20 GB) into efficient PyTorch .pt files for instant loading.

Architecture:
    data/processed/{source}/*.json   →   data/master_cache/{source}/{hash}.pt
                                         data/master_cache/index.pt

Each master cache entry stores per-object:
    - Full vertices (float32 tensor, ALL of them)
    - Full faces (int32 tensor, ALL of them)
    - Full normals (float32 tensor, if available)
    - Materials (complete node trees, dicts)
    - Face material indices (int32 tensor)
    - UV layers (per-loop float32 tensors)
    - Vertex colors (per-loop float32 tensors)
    - Face smooth flags (bool tensor)
    - Modifiers, shape keys, vertex groups, transforms
    - Pre-computed quality metrics

Plus scene-level data:
    - Scene label, metadata (name, description, tags, categories)
    - Images (base64 thumbnails)
    - Armatures, animation, world, render settings
    - Object hierarchy / collections

This is the SINGLE SOURCE OF TRUTH.  Training caches are derived from it.

Usage:
    python scripts/build_master_cache.py                # Build from all sources
    python scripts/build_master_cache.py --source objaverse  # Specific source
    python scripts/build_master_cache.py --dry-run      # Preview without writing
    python scripts/build_master_cache.py --force         # Rebuild existing entries
    python scripts/build_master_cache.py --stats         # Show stats about existing cache
"""

import argparse
import gc
import gzip
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
MASTER_CACHE_DIR = BASE / "data" / "master_cache"

SOURCE_DIRS = {
    "objaverse":        BASE / "data" / "processed" / "objaverse",
    "blender_official":  BASE / "data" / "processed" / "blender_official",
    "blendswap":        BASE / "data" / "processed" / "blendswap",
    "smutbase":         BASE / "data" / "processed" / "smutbase",
    "github":           BASE / "data" / "processed" / "github",
    "open3dlab":        BASE / "data" / "processed" / "open3dlab",
    "youtube":          BASE / "data" / "processed" / "youtube",
    "objaverse_xl":     BASE / "data" / "processed" / "objaverse_xl",
}

# Master cache version — bump when format changes
# v3: structured sections (animations, material_names, object_names,
#     per_object_labels, file_label/file_tags per object)
# v4: exact source JSON raw-byte preservation (gzip + raw SHA1)
# v5: compact scene deconstruction index for subset training queries
CACHE_VERSION = 5


# ── Quality metrics (pre-computed, stored in master cache) ──────────

def _compute_bbox(vertices: np.ndarray) -> dict:
    """Compute bounding box metrics."""
    if len(vertices) == 0:
        return {"min": [0, 0, 0], "max": [0, 0, 0], "dimensions": [0, 0, 0],
                "aspect_ratio": 1.0, "volume_estimate": 0.0}
    vmin = vertices.min(axis=0).tolist()
    vmax = vertices.max(axis=0).tolist()
    dims = [vmax[i] - vmin[i] for i in range(3)]
    sorted_dims = sorted(dims, reverse=True)
    aspect = sorted_dims[0] / max(sorted_dims[-1], 1e-6)
    vol = dims[0] * dims[1] * dims[2]
    return {
        "min": vmin, "max": vmax, "dimensions": dims,
        "aspect_ratio": round(aspect, 3),
        "volume_estimate": round(vol, 6),
    }


def _compute_mesh_quality(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """Pre-compute quality metrics for an object's mesh.

    Uses vectorized numpy operations for speed on large meshes.
    """
    n_verts = len(vertices)
    n_faces = len(faces)

    if n_faces == 0 or n_verts == 0:
        return {"face_count": 0, "vertex_count": 0, "quality_score": 0.0}

    # Clamp face indices to valid range
    faces_clamped = np.clip(faces, 0, n_verts - 1)
    ncols = faces_clamped.shape[1] if faces_clamped.ndim == 2 else 3

    # Edge lengths — vectorized
    avg_edge = 0.0
    if faces_clamped.ndim == 2 and ncols >= 2:
        all_edges = []
        for i in range(ncols):
            j = (i + 1) % ncols
            v0 = vertices[faces_clamped[:, i]]
            v1 = vertices[faces_clamped[:, j]]
            all_edges.append(np.linalg.norm(v1 - v0, axis=1))
        edge_lens = np.concatenate(all_edges)
        avg_edge = float(edge_lens.mean()) if len(edge_lens) > 0 else 0.0

    # Surface area estimate — vectorized (first 3 verts per face)
    surface_area = 0.0
    if faces_clamped.ndim == 2 and ncols >= 3:
        v0 = vertices[faces_clamped[:, 0]]
        v1 = vertices[faces_clamped[:, 1]]
        v2 = vertices[faces_clamped[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        surface_area = float(0.5 * np.linalg.norm(cross, axis=1).sum())

    bbox = _compute_bbox(vertices)

    # Connected components (union-find) — only for meshes under 50K faces
    n_components = -1  # -1 means "not computed"
    if n_faces <= 50000:
        parent = {}
        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        for face in faces:
            for i in range(1, len(face)):
                union(int(face[0]), int(face[i]))
        all_verts_set = set()
        for face in faces:
            all_verts_set.update(int(v) for v in face)
        n_components = len(set(find(v) for v in all_verts_set))

    # Quality heuristic (0-1 scale)
    score = 0.5
    if n_faces >= 50:
        score += 0.1
    if n_faces >= 500:
        score += 0.1
    if n_faces >= 2000:
        score += 0.1
    if n_components == 1:
        score += 0.1
    elif n_components > 1 and n_components <= 5:
        score += 0.05
    if avg_edge > 0 and avg_edge < 0.5:  # Reasonable edge lengths
        score += 0.05
    score = min(1.0, score)

    return {
        "face_count": n_faces,
        "vertex_count": n_verts,
        "avg_edge_length": round(avg_edge, 6),
        "surface_area": round(surface_area, 6),
        "connected_components": n_components,
        "bbox": bbox,
        "quality_score": round(score, 3),
    }


# ── Object conversion ──────────────────────────────────────────────

def _convert_object(obj: dict, compute_quality: bool = True) -> dict:
    """Convert a single object from JSON dict to master cache format.

    Stores geometry as tensors (fast), everything else as dicts/lists.
    ZERO data loss — every field from the source JSON is preserved.
    """
    mesh = obj.get("mesh", {})
    eval_verts = mesh.get("vertices", [])
    eval_faces = mesh.get("faces", [])
    raw_local_verts = obj.get("raw_vertices", [])
    raw_local_faces = obj.get("raw_faces", [])

    use_raw_local = bool(
        isinstance(raw_local_verts, list)
        and isinstance(raw_local_faces, list)
        and len(raw_local_verts) > 0
        and len(raw_local_faces) > 0
    )
    modifiers = obj.get("modifiers", [])
    has_nodes_modifier = any(
        isinstance(m, dict) and str(m.get("type", "")).upper() == "NODES"
        for m in (modifiers if isinstance(modifiers, list) else [])
    )
    geometry_is_baked = bool((not use_raw_local) and has_nodes_modifier)

    # Modeling geometry in master cache should represent original/pre-modifier
    # object-space mesh whenever available. Evaluated geometry remains recoverable
    # from source_json_gz for high-fidelity scene reconstruction workflows.
    model_verts = raw_local_verts if use_raw_local else eval_verts
    model_faces = raw_local_faces if use_raw_local else eval_faces

    item = {
        # Object identity
        "name": obj.get("name", ""),
        "resolved_name": obj.get("resolved_name", ""),
        "text_label": obj.get("text_label", ""),
        # File-level label context (set by blend_extractor per-object)
        "file_label": obj.get("file_label", ""),
        "file_tags": obj.get("file_tags", []),

        # Geometry — stored as tensors for fast loading
        "vertices": (torch.tensor(np.array(model_verts, dtype=np.float32))
                 if model_verts else torch.zeros(0, 3, dtype=torch.float32)),
        "faces": (torch.tensor(np.array(model_faces, dtype=np.int32))
              if model_faces else torch.zeros(0, 3, dtype=torch.int32)),
        "geometry_space": "RAW_LOCAL" if use_raw_local else "EVALUATED_WORLD",
        "geometry_is_baked": geometry_is_baked,
        "has_nodes_modifier": has_nodes_modifier,
        "raw_vertices": raw_local_verts if use_raw_local else [],
        "raw_faces": raw_local_faces if use_raw_local else [],

        # Normals
        "normals": (torch.tensor(np.array(mesh.get("normals", []), dtype=np.float32))
                    if mesh.get("normals") else None),

        # Face material indices
        "face_material_indices": (
            torch.tensor(np.array(mesh.get("face_material_indices", []), dtype=np.int32))
            if mesh.get("face_material_indices") else None),

        # Materials — full node tree dicts, stored as-is
        "materials": obj.get("materials", []),
        "material_quality": obj.get("material_quality", ""),

        # UV layers — per-loop coordinates  {layer_name: [[u,v], ...]}
        "uv_layers": {},
        # Vertex colors — per-loop RGBA  {layer_name: [[r,g,b,a], ...]}
        "vertex_color_layers": {},

        # Per-face smooth flags
        "face_smooth": (
            torch.tensor(mesh.get("face_smooth", []), dtype=torch.bool)
            if mesh.get("face_smooth") else None),

        # Modifiers, shape keys, vertex groups
        "modifiers": modifiers if isinstance(modifiers, list) else [],
        "shape_keys": obj.get("shape_keys", []),
        "vertex_groups": obj.get("vertex_groups", []),

        # Transforms
        "transforms": obj.get("transforms", {}),
        "bbox_min": obj.get("bbox_min", []),
        "bbox_max": obj.get("bbox_max", []),
        "dimensions": obj.get("dimensions", []),

        # Normalization params (from blend_extractor) for scene reconstruction
        "normalization_center": mesh.get("normalization_center"),
        "normalization_scale": mesh.get("normalization_scale"),

        # Hierarchy
        "parent": obj.get("parent", ""),
        "parent_type": obj.get("parent_type", ""),
        "parent_bone": obj.get("parent_bone", ""),

        # Object type (MESH, ARMATURE, CURVE, etc.)
        "type": obj.get("type", "MESH"),

        # Metadata from source
        "categories": obj.get("categories", ""),
        "description": obj.get("description", ""),
        "tags": obj.get("tags", []),
    }

    # UV layers — convert to tensors
    uv_data = mesh.get("uv_layers", {})
    if isinstance(uv_data, dict):
        for layer_name, coords in uv_data.items():
            if coords:
                item["uv_layers"][layer_name] = torch.tensor(
                    np.array(coords, dtype=np.float32))
    elif isinstance(uv_data, list):
        # Some extractors store as list of dicts
        for i, layer in enumerate(uv_data):
            if isinstance(layer, dict):
                name = layer.get("name", f"UVMap_{i}")
                coords = layer.get("data", layer.get("coords", []))
                if coords:
                    item["uv_layers"][name] = torch.tensor(
                        np.array(coords, dtype=np.float32))

    # Vertex color layers — convert to tensors
    vcol_data = mesh.get("vertex_color_layers", {})
    if isinstance(vcol_data, dict):
        for layer_name, colors in vcol_data.items():
            if colors:
                item["vertex_color_layers"][layer_name] = torch.tensor(
                    np.array(colors, dtype=np.float32))

    # Preserve any additional mesh keys not explicitly converted above.
    known_mesh_keys = {
        "vertices", "faces", "normals", "face_material_indices",
        "uv_layers", "vertex_color_layers", "face_smooth",
    }
    mesh_extra = {k: v for k, v in mesh.items() if k not in known_mesh_keys}
    if mesh_extra:
        item["mesh_extra"] = mesh_extra

    # Preserve any additional object-level keys not explicitly converted.
    known_obj_keys = {
        "name", "resolved_name", "text_label", "mesh", "materials",
        "material_quality", "modifiers", "shape_keys", "vertex_groups",
        "transforms", "bbox_min", "bbox_max", "dimensions", "parent",
        "parent_type", "parent_bone", "type", "categories", "description",
        "tags", "raw_vertices", "raw_faces",
    }
    for key, value in obj.items():
        if key in known_obj_keys:
            continue
        # Keep all extra payloads (e.g., armature/animation/custom metadata)
        item[key] = value

    # Pre-compute quality metrics
    if compute_quality and model_verts and model_faces:
        v_arr = np.array(model_verts, dtype=np.float32)
        f_arr = np.array(model_faces, dtype=np.int32)
        item["quality"] = _compute_mesh_quality(v_arr, f_arr)
    else:
        item["quality"] = {"face_count": len(model_faces), "vertex_count": len(model_verts),
                           "quality_score": 0.0}

    return item


# ── File conversion ─────────────────────────────────────────────────

def convert_source_file(filepath: Path, data_source: str,
                        compute_quality: bool = True) -> dict | None:
    """Convert a source JSON into a master cache entry.

    Returns a dict representing the entire file with all objects and
    scene-level data, or None if the file is unreadable.
    """
    try:
        source_json_bytes = filepath.read_bytes()
        data = json.loads(source_json_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, MemoryError, OSError) as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None

    if isinstance(data, dict):
        objects_raw = data.get("objects", [data])
        metadata = data.get("metadata", {})
    else:
        objects_raw = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
        metadata = {}

    # Preserve exact source JSON payload in compact form for lossless recovery.
    source_json_gz = b""
    source_json_size = 0
    source_json_sha1 = ""
    source_json_raw_sha1 = ""
    try:
        source_json_gz = gzip.compress(source_json_bytes, compresslevel=6)
        source_json_size = len(source_json_bytes)
        source_json_sha1 = hashlib.sha1(source_json_bytes).hexdigest()
        source_json_raw_sha1 = source_json_sha1
    except Exception:
        source_json_gz = b""
        source_json_size = 0
        source_json_sha1 = ""
        source_json_raw_sha1 = ""

    # Convert all objects
    objects = []
    for obj in objects_raw:
        if not isinstance(obj, dict):
            continue
        converted = _convert_object(obj, compute_quality=compute_quality)
        objects.append(converted)

    if not objects:
        logger.debug(f"No objects in {filepath}")
        return None

    # Compact deconstruction index for fast subset extraction from master cache.
    # This avoids touching .blend files and avoids loading/parsing everything
    # when a training subset only needs specific sections.
    scene_deconstruction_index = {
        "top_level_keys": [],
        "object_count": 0,
        "object_types": {},
        "object_name_to_index": {},
        "object_keys": [],
        "mesh_keys": [],
        "sections": {},
    }
    if isinstance(data, dict):
        scene_deconstruction_index["top_level_keys"] = sorted(str(k) for k in data.keys())
        objects_raw_idx = data.get("objects")
        objects_for_index = objects_raw_idx if isinstance(objects_raw_idx, list) else []

        obj_types: dict[str, int] = {}
        obj_name_to_index: dict[str, int] = {}
        obj_key_union: set[str] = set()
        mesh_key_union: set[str] = set()

        for oi, obj_raw in enumerate(objects_for_index):
            if not isinstance(obj_raw, dict):
                continue
            obj_type = str(obj_raw.get("type", "UNKNOWN"))
            obj_types[obj_type] = int(obj_types.get(obj_type, 0)) + 1

            obj_name = str(obj_raw.get("name", "")).strip()
            if obj_name and obj_name not in obj_name_to_index:
                obj_name_to_index[obj_name] = int(oi)

            obj_key_union.update(str(k) for k in obj_raw.keys())
            mesh_raw = obj_raw.get("mesh")
            if isinstance(mesh_raw, dict):
                mesh_key_union.update(str(k) for k in mesh_raw.keys())

        scene_deconstruction_index["object_count"] = int(len(objects_for_index))
        scene_deconstruction_index["object_types"] = obj_types
        scene_deconstruction_index["object_name_to_index"] = obj_name_to_index
        scene_deconstruction_index["object_keys"] = sorted(obj_key_union)
        scene_deconstruction_index["mesh_keys"] = sorted(mesh_key_union)
        scene_deconstruction_index["sections"] = {
            "has_world": bool(data.get("world")),
            "has_render": bool(data.get("render")),
            "has_images": bool(data.get("images")),
            "has_node_groups": bool(data.get("node_groups")),
            "has_collections": bool(data.get("collections")),
            "has_orphan_materials": bool(data.get("orphan_materials")),
            "has_orphan_actions": bool(data.get("orphan_actions")),
            "has_metadata": bool(data.get("metadata")),
        }

    # Scene-level data
    source_extras = {}
    has_animations = False
    has_armatures = False
    animation_data_collected = []  # Structured animation section
    if isinstance(data, dict):
        known_top_level = {
            "objects", "metadata", "source_file", "source_format", "label",
            "images", "world", "render", "fps", "frame_start", "frame_end",
            "collections", "orphan_materials", "orphan_actions", "node_groups",
        }
        source_extras = {k: v for k, v in data.items() if k not in known_top_level}

        orphan_actions = data.get("orphan_actions")
        has_animations = bool(orphan_actions)
        if orphan_actions:
            animation_data_collected.append({
                "type": "orphan_actions",
                "data": orphan_actions,
            })

        if not has_animations:
            for obj in objects:
                anim_payload = (
                    obj.get("animation_data")
                    or obj.get("animation")
                    or obj.get("actions")
                    or obj.get("nla_tracks")
                )
                if anim_payload:
                    has_animations = True
                    animation_data_collected.append({
                        "type": "object_animation",
                        "object_name": obj.get("name", ""),
                        "data": anim_payload,
                    })

        # Also collect per-object keyframes/actions where available
        for obj in objects:
            kf = obj.get("keyframes")
            act = obj.get("action_name")
            if kf or act:
                animation_data_collected.append({
                    "type": "keyframes",
                    "object_name": obj.get("name", ""),
                    "action_name": act or "",
                    "keyframe_count": len(kf) if isinstance(kf, list) else 0,
                })

        has_armatures = any(
            (o.get("type") == "ARMATURE") or bool(o.get("armature"))
            for o in objects
        )

    # Pre-compute structured fields used in both entry and summary
    _material_names = list({m.get("name", "") for o in objects
                            for m in o.get("materials", [])
                            if isinstance(m, dict) and m.get("name")})

    entry = {
        # Cache metadata
        "_version": CACHE_VERSION,
        "_source_file": str(filepath.name),
        "_data_source": data_source,
        "_built_at": time.time(),

        # Objects
        "objects": objects,

        # File-level metadata
        "source_file": data.get("source_file", str(filepath.name)) if isinstance(data, dict) else str(filepath.name),
        "source_format": data.get("source_format", "") if isinstance(data, dict) else "",
        "label": data.get("label", "") if isinstance(data, dict) else "",

        # Metadata (name, description, tags, categories, license, etc.)
        "metadata": metadata,

        # Lossless original extracted scene JSON, compressed for compact storage.
        "source_json_gz": source_json_gz,
        "source_json_size": int(source_json_size),
        "source_json_gz_size": int(len(source_json_gz)),
        "source_json_sha1": source_json_sha1,
        "source_json_raw_sha1": source_json_raw_sha1,

        # Compact index of the full deconstructed source scene.
        # Canonical full payload remains in source_json_gz.
        "scene_deconstruction_index": scene_deconstruction_index,

        # ── Structured metadata sections ──
        # These separate concerns so training data can be built intelligently.

        # Animation section: all animation data from the file, structured
        # (actions, keyframes, NLA tracks, armature animations)
        "animations": animation_data_collected,

        # Material catalogue: unique material names across all objects
        "material_names": _material_names,

        # Object name catalogue: all Blender object names in the file
        "object_names": [o.get("name", "") for o in objects],

        # Geometry label hints: per-object labels from the extractor
        # (these are the object's own Blender name, NOT the file label)
        "per_object_labels": [
            {
                "index": i,
                "name": o.get("name", ""),
                "text_label": o.get("text_label", ""),
                "file_label": o.get("file_label", ""),
                "material_names": [m.get("name", "") for m in o.get("materials", [])
                                   if isinstance(m, dict)],
                "face_count": o.get("quality", {}).get("face_count", 0),
            }
            for i, o in enumerate(objects)
        ],

        # Scene-level data
        "images": data.get("images", {}) if isinstance(data, dict) else {},
        "world": data.get("world", {}) if isinstance(data, dict) else {},
        "render": data.get("render", {}) if isinstance(data, dict) else {},
        "fps": data.get("fps", None) if isinstance(data, dict) else None,
        "frame_start": data.get("frame_start", None) if isinstance(data, dict) else None,
        "frame_end": data.get("frame_end", None) if isinstance(data, dict) else None,
        "collections": data.get("collections", []) if isinstance(data, dict) else [],
        "orphan_materials": data.get("orphan_materials", []) if isinstance(data, dict) else [],
        "orphan_actions": data.get("orphan_actions", []) if isinstance(data, dict) else [],
        "node_groups": data.get("node_groups", []) if isinstance(data, dict) else [],
        "source_extras": source_extras,

        # Summary statistics (for fast filtering without loading objects)
        "summary": {
            "n_objects": len(objects),
            "n_mesh_objects": sum(1 for o in objects
                                 if o["vertices"].numel() > 0),
            "total_faces": sum(o["quality"]["face_count"] for o in objects),
            "total_vertices": sum(o["quality"]["vertex_count"] for o in objects),
            "has_materials": any(bool(o["materials"]) for o in objects),
            "has_uvs": any(bool(o["uv_layers"]) for o in objects),
            "has_vertex_colors": any(bool(o["vertex_color_layers"]) for o in objects),
            "has_animations": has_animations,
            "has_armatures": has_armatures,
            "n_animations": len(animation_data_collected),
            "n_unique_materials": len(_material_names),
            "max_object_faces": max((o["quality"]["face_count"] for o in objects), default=0),
        },
    }

    return entry


def _file_hash(filepath: Path) -> str:
    """Deterministic hash for a source file path."""
    return hashlib.md5(str(filepath).encode()).hexdigest()[:16]


# ── Main build loop ─────────────────────────────────────────────────

def build_master_cache(sources: list[str] | None = None,
                       force: bool = False,
                       dry_run: bool = False,
                       compute_quality: bool = True) -> dict:
    """Build master cache from source JSONs.

    Returns summary statistics dict.
    """
    if sources is None:
        sources = list(SOURCE_DIRS.keys())

    stats = {
        "sources_processed": 0,
        "files_processed": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "total_objects": 0,
        "total_faces": 0,
        "objects_with_materials": 0,
        "objects_with_uvs": 0,
        "bytes_written": 0,
    }

    # Collect all source files
    all_files: list[tuple[Path, str]] = []
    for source_name in sources:
        src_dir = SOURCE_DIRS.get(source_name)
        if not src_dir or not src_dir.exists():
            logger.info(f"Source dir not found: {source_name} ({src_dir})")
            continue
        json_files = sorted(src_dir.glob("*.json"))
        logger.info(f"{source_name}: {len(json_files)} JSON files")
        for jf in json_files:
            all_files.append((jf, source_name))

    logger.info(f"Total source files to process: {len(all_files)}")

    if not all_files:
        logger.warning("No source files found!")
        return stats

    # Create output dirs
    if not dry_run:
        for source_name in sources:
            out_dir = MASTER_CACHE_DIR / source_name
            out_dir.mkdir(parents=True, exist_ok=True)

    # Index for fast lookups
    index_entries = []
    t0 = time.time()

    for file_idx, (filepath, source_name) in enumerate(all_files):
        fhash = _file_hash(filepath)
        out_path = MASTER_CACHE_DIR / source_name / f"{fhash}.pt"

        # Skip if already cached (unless --force)
        if out_path.exists() and not force:
            stats["files_skipped"] += 1
            # Load summary from existing for index
            try:
                existing = torch.load(out_path, weights_only=False)
                index_entries.append({
                    "hash": fhash,
                    "source": source_name,
                    "source_file": str(filepath.name),
                    "path": str(out_path.relative_to(MASTER_CACHE_DIR)),
                    "summary": existing.get("summary", {}),
                    "label": existing.get("label", ""),
                    "metadata_name": existing.get("metadata", {}).get("name", ""),
                })
            except Exception:
                pass
            continue

        # Convert
        entry = convert_source_file(filepath, source_name,
                                    compute_quality=compute_quality)
        if entry is None:
            stats["files_failed"] += 1
            continue

        stats["files_processed"] += 1
        stats["sources_processed"] = len(set(s for _, s in all_files[:file_idx + 1]))
        n_obj = entry["summary"]["n_objects"]
        n_faces = entry["summary"]["total_faces"]
        stats["total_objects"] += n_obj
        stats["total_faces"] += n_faces
        stats["objects_with_materials"] += sum(
            1 for o in entry["objects"] if o["materials"])
        stats["objects_with_uvs"] += sum(
            1 for o in entry["objects"] if o["uv_layers"])

        # Index entry
        index_entries.append({
            "hash": fhash,
            "source": source_name,
            "source_file": str(filepath.name),
            "path": str(out_path.relative_to(MASTER_CACHE_DIR)),
            "summary": entry["summary"],
            "label": entry.get("label", ""),
            "metadata_name": entry.get("metadata", {}).get("name", ""),
        })

        if dry_run:
            label = entry.get("label", "") or entry.get("metadata", {}).get("name", "")
            logger.info(
                f"[{file_idx + 1}/{len(all_files)}] {source_name}/{filepath.name}: "
                f"{n_obj} objects, {n_faces} faces, "
                f"mats={entry['summary']['has_materials']}, "
                f"label={label[:60]}"
            )
        else:
            # Save
            torch.save(entry, out_path)
            fsize = out_path.stat().st_size
            stats["bytes_written"] += fsize

            if (file_idx + 1) % 50 == 0 or file_idx == len(all_files) - 1:
                elapsed = time.time() - t0
                rate = (file_idx + 1) / elapsed
                eta = (len(all_files) - file_idx - 1) / rate
                logger.info(
                    f"[{file_idx + 1}/{len(all_files)}] {source_name}: "
                    f"{n_obj} objs, {n_faces} faces | "
                    f"{stats['bytes_written'] / 1024 / 1024:.0f} MB written | "
                    f"ETA {eta:.0f}s"
                )

        # Periodic GC for large files
        if n_faces > 100_000:
            gc.collect()

    # Save global index
    if not dry_run and index_entries:
        index_path = MASTER_CACHE_DIR / "index.pt"
        torch.save({
            "version": CACHE_VERSION,
            "built_at": time.time(),
            "entries": index_entries,
        }, index_path)
        logger.info(f"Index saved: {len(index_entries)} entries → {index_path}")

    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"Master cache build complete in {elapsed:.0f}s")
    logger.info(f"  Files processed: {stats['files_processed']}")
    logger.info(f"  Files skipped (existing): {stats['files_skipped']}")
    logger.info(f"  Files failed: {stats['files_failed']}")
    logger.info(f"  Total objects: {stats['total_objects']}")
    logger.info(f"  Total faces: {stats['total_faces']:,}")
    logger.info(f"  Objects with materials: {stats['objects_with_materials']}")
    logger.info(f"  Objects with UVs: {stats['objects_with_uvs']}")
    logger.info(f"  Bytes written: {stats['bytes_written'] / 1024 / 1024:.1f} MB")

    return stats


# ── Stats mode ──────────────────────────────────────────────────────

def show_stats():
    """Show statistics about the existing master cache."""
    index_path = MASTER_CACHE_DIR / "index.pt"
    if not index_path.exists():
        logger.info("No master cache index found. Run build first.")
        return

    index = torch.load(index_path, weights_only=False)
    entries = index["entries"]
    logger.info(f"Master Cache Index (v{index['version']})")
    logger.info(f"  Built: {time.ctime(index['built_at'])}")
    logger.info(f"  Total entries: {len(entries)}")

    # Per-source stats
    from collections import defaultdict
    by_source = defaultdict(lambda: {"files": 0, "objects": 0, "faces": 0,
                                     "has_mats": 0, "has_uvs": 0})
    for e in entries:
        s = by_source[e["source"]]
        s["files"] += 1
        summary = e.get("summary", {})
        s["objects"] += summary.get("n_objects", 0)
        s["faces"] += summary.get("total_faces", 0)
        s["has_mats"] += 1 if summary.get("has_materials") else 0
        s["has_uvs"] += 1 if summary.get("has_uvs") else 0

    logger.info(f"\n  {'Source':<20} {'Files':>6} {'Objects':>8} {'Faces':>12} {'Materials':>10} {'UVs':>6}")
    logger.info(f"  {'-'*62}")
    total_o = total_f = 0
    for src, s in sorted(by_source.items()):
        mat_pct = f"{100 * s['has_mats'] / max(1, s['files']):.0f}%"
        uv_pct = f"{100 * s['has_uvs'] / max(1, s['files']):.0f}%"
        logger.info(f"  {src:<20} {s['files']:>6} {s['objects']:>8} "
                     f"{s['faces']:>12,} {mat_pct:>10} {uv_pct:>6}")
        total_o += s["objects"]
        total_f += s["faces"]
    logger.info(f"  {'TOTAL':<20} {len(entries):>6} {total_o:>8} {total_f:>12,}")

    # Disk usage
    total_size = 0
    for source_dir in MASTER_CACHE_DIR.iterdir():
        if source_dir.is_dir():
            for f in source_dir.glob("*.pt"):
                total_size += f.stat().st_size
    total_size += index_path.stat().st_size
    logger.info(f"\n  Disk usage: {total_size / 1024 / 1024:.1f} MB")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build lossless master cache from source JSONs")
    parser.add_argument("--source", type=str, default=None,
                        help="Process only this source (e.g. 'objaverse')")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild existing cache entries")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files")
    parser.add_argument("--no-quality", action="store_true",
                        help="Skip quality metric computation (faster)")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics about existing master cache")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    sources = [args.source] if args.source else None

    build_master_cache(
        sources=sources,
        force=args.force,
        dry_run=args.dry_run,
        compute_quality=not args.no_quality,
    )


if __name__ == "__main__":
    main()
