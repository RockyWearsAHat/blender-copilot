#!/usr/bin/env python3
"""Build task-specific training caches from the lossless master cache.

This script reads from data/master_cache/ (built by build_master_cache.py)
and produces tokenized, filtered training data in data/training_cache/{task}/.

The master cache has ZERO data loss.  This script applies configurable:
    - Face count limits (tokenization truncation)
    - Quality filters
    - Language filters
    - Material requirements
    - Per-label caps
    - Source selection

Each training cache is fully reproducible from the master cache + config.

Usage:
    python scripts/build_training_cache.py                          # Default config
    python scripts/build_training_cache.py --config config.yaml     # Custom config
    python scripts/build_training_cache.py --task mesh_gen          # Named task
    python scripts/build_training_cache.py --max-faces 8000         # Override
    python scripts/build_training_cache.py --dry-run                # Preview
    python scripts/build_training_cache.py --stats                  # Show existing cache stats
"""

import argparse
import gc
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.mesh_tokenizer import MeshTokenizer
from processing.bpe_tokenizer import BPETokenizer
from processing.generate_synthetic import normalize_mesh
from processing.labeler_smart import generate_smart_label, compute_bbox_aspect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
MASTER_CACHE_DIR = BASE / "data" / "master_cache"
TRAINING_CACHE_DIR = BASE / "data" / "training_cache"


# ── Default training config ─────────────────────────────────────────

DEFAULT_CONFIG = {
    "task_name": "default",

    # Tokenization
    "vocab_size": 8192,
    "max_faces": 8000,
    "max_text_length": 256,

    # Filtering
    "min_faces": 50,
    "require_materials": False,
    "require_english": True,
    "min_quality_score": 0.0,

    # Caps
    "max_per_label": 50,
    "max_per_source_file": 10,

    # Sources (None = all)
    "sources": None,

    # Scene assembly
    "include_assembled_scenes": True,
    "max_scene_objects": 8,

    # Over-budget handling
    "keep_largest_component": False,  # LCC after truncation — too destructive
    "over_budget_strategy": "simplify_then_subsample",

    # Orientation normalization
    # IMPORTANT: default to preserving source orientation.
    # Many assets are already correctly Z-up, and heuristic axis swapping can
    # damage valid samples. Keep this off unless explicitly needed.
    "ensure_upright": False,

    # Source-specific axis correction
    # Objaverse GLB assets are often stored in Y-up coordinates.
    # Apply +90° X (Y->Z) during training-cache build so validator views
    # are consistent with Blender's Z-up world.
    "axis_correction_by_source": {
        "objaverse": "x_plus_90",
        "objaverse_xl": "x_plus_90",
    },

    # Scene decomposition (DISABLED — splits objects into useless fragments)
    "decompose_components": False,
    "min_component_faces": 20,
    "max_components_per_object": 16,
}


# ── Language detection ──────────────────────────────────────────────

_NON_LATIN_RE = re.compile(
    r'[\u0400-\u04ff\u0600-\u06ff\u0900-\u097f\u0e00-\u0e7f'
    r'\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff]'
)

_NON_ENGLISH_WORDS = {
    "kereta", "gerbong", "warna", "gradasi", "kaca", "bening", "kuning",
    "ban", "trotoar", "hijau", "biru", "ungu", "oranye", "merah",
    "putih", "hitam", "terowongan", "lokomotif", "bolong", "rel",
    "cabang", "lampu", "stasiun", "jalan", "tanjakan", "tanah", "kolam",
    "air", "awan", "gunung", "gedung", "alas", "masjid", "pohon",
    "masinis", "telur", "belang", "polkadot", "daun", "aspal", "nyala",
    "rumah", "pintu", "atap", "dinding", "lantai", "meja", "kursi",
    "roda", "sayap", "kuda", "ikan", "burung", "bunga", "batu",
    "pasir", "sungai", "laut", "hutan", "kebun", "sawah", "jembatan",
    "carbodymat", "coklat", "abu", "emas", "perak",
    "groupe", "noeud", "curva", "beton", "lumiere", "fenetre", "porte",
    "maison", "arbre", "voiture", "bateau",
    "metallkedja", "hus", "bil", "stol", "bord",
    "blanco", "brillante", "rojo", "azul", "verde", "amarillo", "negro",
    "casa", "coche", "ventana",
    "normale", "luce",
}

_GENERIC_LABELS = {
    "object", "mesh", "3d object", "3d mesh", "shape", "3d shape", "part",
    "piece", "simple object", "basic object", "simple shape", "detailed object",
    "multi object scene composition", "dummy", "base", "lattice",
}


def _is_non_english(label: str) -> bool:
    if not label:
        return True
    if _NON_LATIN_RE.search(label):
        return True
    clean = re.sub(r'[^a-z0-9\s]', ' ', label.lower())
    words = [w for w in clean.split() if w and len(w) > 1]
    if not words:
        return True
    non_eng = sum(1 for w in words if w in _NON_ENGLISH_WORDS)
    return non_eng / len(words) > 0.4


def _is_weak_label(label: str) -> bool:
    if not label:
        return True
    clean = re.sub(r'[^a-z0-9\s,]', ' ', label.lower()).strip()
    if clean in _GENERIC_LABELS:
        return True
    # Numeric-heavy (version numbers, IDs — but not model designations)
    # Threshold 0.5 lets through "colt m1911" (0.44) while catching "v123 456"
    alnum = re.sub(r'[^a-z0-9]', '', clean)
    if alnum and sum(c.isdigit() for c in alnum) / len(alnum) > 0.5:
        return True
    return False


# ── Z-order / Morton code ──────────────────────────────────────────

def _compute_z_order(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute z-order (Morton code) permutation for faces.

    Returns an array of face indices sorted by spatial Morton code.
    Uses vectorized numpy operations for speed on large meshes.
    """
    n_faces = len(faces)
    n_verts = len(vertices)

    if n_faces == 0:
        return np.array([], dtype=np.int64)

    # Vectorized face center computation
    if faces.ndim == 2:
        faces_clamped = np.clip(faces, 0, max(0, n_verts - 1))
        ncols = faces_clamped.shape[1]
        centers = np.zeros((n_faces, 3), dtype=np.float32)
        for col in range(ncols):
            centers += vertices[faces_clamped[:, col]]
        centers /= ncols
    else:
        # Fallback for ragged faces
        centers = np.zeros((n_faces, 3))
        for i, face in enumerate(faces):
            valid = [int(fi) for fi in face if int(fi) < n_verts]
            if valid:
                centers[i] = vertices[valid].mean(axis=0)

    c_min = centers.min(axis=0)
    c_max = centers.max(axis=0)
    c_range = c_max - c_min
    c_range[c_range < 1e-6] = 1.0
    norm_c = ((centers - c_min) / c_range * 1023).astype(np.int64)
    norm_c = np.clip(norm_c, 0, 1023)

    # Vectorized Morton encoding
    def spread_bits_vec(v):
        v = v & 0x3FF
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    morton = (spread_bits_vec(norm_c[:, 0])
              | (spread_bits_vec(norm_c[:, 1]) << 1)
              | (spread_bits_vec(norm_c[:, 2]) << 2))

    return np.argsort(morton)


def _largest_connected_component(vertices, faces):
    """Keep only the largest connected component after truncation.

    Returns:
        (new_verts, new_faces, kept_face_indices)
        kept_face_indices: indices into the INPUT faces array that survived.
    """
    if len(faces) == 0:
        return vertices, faces, np.array([], dtype=np.int64)
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

    comp_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        root = find(int(face[0]))
        comp_faces[root].append(fi)

    if len(comp_faces) <= 1:
        return vertices, faces, np.arange(len(faces), dtype=np.int64)

    largest = max(comp_faces, key=lambda r: len(comp_faces[r]))
    keep = sorted(comp_faces[largest])  # sorted for determinism

    v_arr = np.array(vertices) if not isinstance(vertices, np.ndarray) else vertices
    f_arr = np.array(faces) if not isinstance(faces, np.ndarray) else faces
    kept_faces = f_arr[keep]
    used = np.unique(kept_faces.ravel())
    remap = {int(old): new for new, old in enumerate(used)}
    new_verts = v_arr[used]
    new_faces = np.vectorize(lambda x: remap[x])(kept_faces)
    return new_verts, new_faces, np.array(keep, dtype=np.int64)


def _simplify_mesh_quadric(vertices: np.ndarray,
                           faces: np.ndarray,
                           target_faces: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Simplify mesh to target face count using trimesh quadric decimation.

    Returns (verts, faces) on success, else None.
    """
    if len(faces) <= target_faces:
        return vertices.astype(np.float32, copy=False), faces.astype(np.int32, copy=False)

    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        decimated = None
        for aggression in (None, 7, 10):
            try:
                kwargs = {"face_count": int(target_faces)}
                if aggression is not None:
                    kwargs["aggression"] = aggression
                candidate = mesh.simplify_quadric_decimation(**kwargs)
                if candidate is not None and len(candidate.faces) >= 2:
                    decimated = candidate
                    break
            except Exception:
                continue

        if decimated is None:
            return None

        out_v = np.asarray(decimated.vertices, dtype=np.float32)
        out_f = np.asarray(decimated.faces, dtype=np.int32)
        if len(out_f) < 2:
            return None
        return out_v, out_f
    except Exception:
        return None


def _face_centers(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    idx = np.clip(faces, 0, max(0, len(vertices) - 1))
    return vertices[idx].mean(axis=1).astype(np.float32, copy=False)


def _map_per_face_values_by_nearest_centers(values,
                                             old_vertices: np.ndarray,
                                             old_faces: np.ndarray,
                                             new_vertices: np.ndarray,
                                             new_faces: np.ndarray,
                                             default=0):
    """Remap per-face values from old faces to new faces via nearest face center."""
    if len(new_faces) == 0:
        return []

    val_list = values.numpy().tolist() if isinstance(values, torch.Tensor) else list(values)
    if len(old_faces) == 0:
        return [default] * len(new_faces)

    old_centers = _face_centers(old_vertices, old_faces)
    new_centers = _face_centers(new_vertices, new_faces)
    if len(old_centers) == 0:
        return [default] * len(new_faces)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(old_centers)
        _, nn = tree.query(new_centers, k=1)
        nn = np.asarray(nn, dtype=np.int64)
    except Exception:
        # Fallback: chunked brute-force nearest neighbor
        nn = np.zeros(len(new_centers), dtype=np.int64)
        chunk = 256
        for i in range(0, len(new_centers), chunk):
            c = new_centers[i:i + chunk]
            d2 = ((c[:, None, :] - old_centers[None, :, :]) ** 2).sum(axis=2)
            nn[i:i + len(c)] = np.argmin(d2, axis=1)

    out = []
    for src_i in nn:
        si = int(src_i)
        out.append(val_list[si] if 0 <= si < len(val_list) else default)
    return out


def _extract_all_components(vertices, faces, *, min_faces: int = 20):
    """Extract ALL connected components from a mesh.

    Returns a list of (verts, faces, kept_face_indices) tuples,
    sorted largest-first.  Components smaller than min_faces are dropped.
    """
    if len(faces) == 0:
        return []

    v_arr = np.array(vertices) if not isinstance(vertices, np.ndarray) else vertices
    f_arr = np.array(faces) if not isinstance(faces, np.ndarray) else faces

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

    for face in f_arr:
        for i in range(1, len(face)):
            union(int(face[0]), int(face[i]))

    comp_faces = defaultdict(list)
    for fi, face in enumerate(f_arr):
        root = find(int(face[0]))
        comp_faces[root].append(fi)

    # Sort components by face count (largest first)
    components = sorted(comp_faces.values(), key=len, reverse=True)

    results = []
    for face_indices in components:
        if len(face_indices) < min_faces:
            continue
        keep = sorted(face_indices)
        kept_faces = f_arr[keep]
        used = np.unique(kept_faces.ravel())
        remap = {int(old): new for new, old in enumerate(used)}
        new_verts = v_arr[used]
        new_faces = np.vectorize(lambda x: remap[x])(kept_faces)
        results.append((new_verts, new_faces, np.array(keep, dtype=np.int64)))

    return results


def _component_shape_hint(verts: np.ndarray) -> str:
    """Infer a rough shape description from bounding box proportions."""
    if len(verts) < 3:
        return ""
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    dims = bbox_max - bbox_min
    dims = np.clip(dims, 1e-6, None)
    # Sort dimensions to get aspect ratios
    s = np.sort(dims)
    if s[2] / s[0] < 1.5:
        return "compact"  # roughly cubic
    if s[0] / s[2] < 0.15:
        if s[1] / s[2] < 0.15:
            return "elongated"  # stick-like
        return "flat"  # plate/panel
    if s[0] / s[1] > 0.6 and s[2] / s[1] > 2.0:
        return "tall"
    return ""


def _label_component(parent_label: str, comp_idx: int, n_comps: int,
                     verts: np.ndarray, n_faces: int) -> str:
    """Generate a label for a decomposed component.

    Uses the parent object label as a base and adds component info.
    """
    # Strip any existing detail tags like "(high-detail)"
    base = re.sub(r'\s*\(.*?\)\s*$', '', parent_label).strip()

    shape = _component_shape_hint(verts)
    face_desc = "low-poly" if n_faces < 200 else ("medium-poly" if n_faces < 2000 else "detailed")

    parts = [base]
    if n_comps > 1:
        parts.append(f"component {comp_idx + 1}/{n_comps}")
    if shape:
        parts.append(shape)
    parts.append(face_desc)

    return ", ".join(parts)


# ── Label generation ────────────────────────────────────────────────

def _generate_label(obj: dict, file_metadata: dict,
                    file_label: str, is_single: bool,
                    sibling_names: list[str]) -> str:
    """Generate a clean English label for a training item.

    Uses per-object file_label/file_tags (set by blend_extractor) when
    available, falling back to file-level metadata for single-object files.
    """
    name = obj.get("name", "")
    mats = obj.get("materials", [])
    mat_names = [m.get("name", "") for m in mats if isinstance(m, dict)]
    mod_types = [m.get("type", "") for m in obj.get("modifiers", []) if isinstance(m, dict)]

    verts = obj["vertices"]
    faces = obj["faces"]
    n_verts = len(verts) if isinstance(verts, (list, np.ndarray)) else verts.shape[0]
    n_faces = len(faces) if isinstance(faces, (list, np.ndarray)) else faces.shape[0]

    v_list = verts.numpy().tolist() if isinstance(verts, torch.Tensor) else verts
    bbox_aspect = compute_bbox_aspect(v_list) if n_verts > 0 else None

    meta = file_metadata

    # Per-object file_label and file_tags (from blend_extractor v3+)
    obj_file_label = obj.get("file_label", "")
    obj_file_tags = obj.get("file_tags", [])

    # For single-object files, trust per-object file_label or fall back
    # to the master entry file_label.
    # For multi-object files, pass file_label only if available per-object.
    if is_single:
        effective_file_label = obj_file_label or file_label
        effective_meta_name = meta.get("name", "")
        effective_meta_desc = str(meta.get("description", ""))[:200]
        effective_meta_tags = obj_file_tags or meta.get("tags", [])
        effective_meta_cats = meta.get("categories", "")
    else:
        # Multi-object: do NOT promote file metadata to individual objects
        # unless the object's own file_label matches the file_label.
        effective_file_label = obj_file_label if obj_file_label else ""
        effective_meta_name = ""
        effective_meta_desc = ""
        effective_meta_tags = obj_file_tags if obj_file_tags else []
        effective_meta_cats = ""

    label = generate_smart_label(
        obj_name=name,
        material_names=mat_names,
        modifier_types=mod_types,
        num_faces=n_faces,
        num_verts=n_verts,
        bbox_aspect=bbox_aspect,
        file_label=effective_file_label,
        metadata_name=effective_meta_name,
        metadata_desc=effective_meta_desc,
        metadata_tags=effective_meta_tags,
        metadata_categories=effective_meta_cats,
        sibling_names=sibling_names,
    )
    return label


# ── Build scene context for training items ──────────────────────────

def _build_scene_context(obj: dict, master_entry: dict,
                         z_order: np.ndarray | None = None,
                         selected_face_indices: np.ndarray | None = None,
                         kept_indices: np.ndarray | None = None,
                         face_material_indices_override: list | None = None,
                         face_smooth_override: list | None = None,
                         include_face_arrays: bool = True) -> dict:
    """Build the scene_context dict that gets stored in training cache items.

    This preserves materials, UVs, vertex colors, etc. for the validator
    and rendering pipeline.

    When over-budget processing is applied, per-face arrays (FMI, face_smooth)
    are reordered to match z-order, truncated, and then filtered by the
    kept_indices from connected-component extraction so they match 1:1 with
    the final mesh faces.
    """
    sc = {}

    def _json_safe(val):
        if isinstance(val, torch.Tensor):
            return val.item() if val.numel() == 1 else val.numpy().tolist()
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, dict):
            return {k: _json_safe(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_json_safe(v) for v in val]
        return val

    # Materials
    if obj.get("materials"):
        sc["materials"] = obj["materials"]

    # --- helper: reorder + filter a per-face list to match final face order ---
    def _reorder_per_face(values, default=0):
        """Apply z_order reordering, truncation, and component filtering."""
        val_list = values.numpy().tolist() if isinstance(values, torch.Tensor) else list(values)
        if z_order is not None:
            if selected_face_indices is not None and len(selected_face_indices) > 0:
                reordered = [val_list[i] if i < len(val_list) else default
                             for i in selected_face_indices]
            else:
                reordered = [val_list[i] if i < len(val_list) else default
                             for i in z_order]
            # filter to faces kept after connected-component extraction
            if kept_indices is not None and len(kept_indices) > 0:
                reordered = [reordered[i] for i in kept_indices if i < len(reordered)]
            return reordered
        return val_list

    if include_face_arrays:
        # Face material indices
        if face_material_indices_override is not None:
            sc["face_material_indices"] = face_material_indices_override
        else:
            fmi = obj.get("face_material_indices")
            if fmi is not None and len(fmi) > 0:
                sc["face_material_indices"] = _reorder_per_face(fmi, default=0)

        # Face smooth
        if face_smooth_override is not None:
            sc["face_smooth"] = face_smooth_override
        elif obj.get("face_smooth") is not None:
            fs = obj["face_smooth"]
            sc["face_smooth"] = _reorder_per_face(fs, default=False)

    # UV layers
    if obj.get("uv_layers"):
        uv_dict = {}
        uv_layers = obj["uv_layers"]
        if isinstance(uv_layers, dict):
            for name, data in uv_layers.items():
                if isinstance(data, torch.Tensor):
                    uv_dict[name] = data.numpy().tolist()
                else:
                    uv_dict[name] = data
        elif isinstance(uv_layers, list):
            for i, layer in enumerate(uv_layers):
                if isinstance(layer, dict):
                    name = str(layer.get("name") or f"UVMap_{i}")
                    coords = layer.get("data", layer.get("coords", []))
                    uv_dict[name] = _json_safe(coords)
                else:
                    uv_dict[f"UVMap_{i}"] = _json_safe(layer)
        sc["uv_layers"] = uv_dict

    # Vertex colors
    if obj.get("vertex_color_layers"):
        vc_dict = {}
        vcols = obj["vertex_color_layers"]
        if isinstance(vcols, dict):
            for name, data in vcols.items():
                if isinstance(data, torch.Tensor):
                    vc_dict[name] = data.numpy().tolist()
                else:
                    vc_dict[name] = data
        elif isinstance(vcols, list):
            for i, layer in enumerate(vcols):
                if isinstance(layer, dict):
                    name = str(layer.get("name") or f"Color_{i}")
                    colors = layer.get("data", layer.get("colors", []))
                    vc_dict[name] = _json_safe(colors)
                else:
                    vc_dict[f"Color_{i}"] = _json_safe(layer)
        sc["vertex_color_layers"] = vc_dict

    # Modifiers
    if obj.get("modifiers"):
        sc["modifiers"] = obj["modifiers"]

    # Shape keys
    if obj.get("shape_keys"):
        sc["shape_keys"] = obj["shape_keys"]

    # UV maps (alternate storage)
    if obj.get("uv_maps"):
        sc["uv_maps"] = obj["uv_maps"]

    # Transforms
    if obj.get("transforms"):
        sc["transforms"] = obj["transforms"]

    # Images from the file
    if master_entry.get("images"):
        sc["images"] = master_entry["images"]

    # File-level context preserved from master cache for validator inspection.
    file_context = {
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
    sc["file_context"] = _json_safe(file_context)

    # Object-level passthrough context (includes armature/animation/custom extras).
    object_context = {}
    for key, value in obj.items():
        if key in {
            "vertices", "faces", "normals", "face_material_indices", "face_smooth"
        }:
            continue
        object_context[key] = _json_safe(value)
    sc["object_context"] = object_context

    return sc


# ── Orientation normalization ───────────────────────────────────────

def _ensure_upright(verts: np.ndarray) -> np.ndarray:
    """Detect and correct objects that are obviously lying sideways.

    Many imported meshes (especially from Y-up formats like glTF/OBJ)
    end up lying on their side after extraction.  This detects that and
    swaps the tallest horizontal axis with Z so the object stands upright.

    Only corrects SEVERELY sideways objects (Z < 20% of the tallest axis)
    to avoid breaking naturally flat things like terrain or tables.

    Returns:
        Corrected vertex array (may be the same object if no fix needed).
    """
    if len(verts) < 4:
        return verts

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    dims = bbox_max - bbox_min  # [dx, dy, dz]

    max_dim = dims.max()
    if max_dim < 1e-6:
        return verts

    z_dim = dims[2]
    z_ratio = z_dim / max_dim

    # If Z is at least 20% of the tallest axis, it's probably fine
    # (tables, terrain, coins, etc. are naturally flat)
    if z_ratio >= 0.20:
        return verts

    # Z is very short — object is likely lying flat/sideways.
    # Swap the tallest axis with Z.
    tallest = int(np.argmax(dims))
    if tallest == 2:
        return verts  # Z is already tallest (shouldn't happen given ratio check)

    rotated = verts.copy()
    if tallest == 1:
        # Y is tallest → swap Y and Z (common Y-up → Z-up fix)
        rotated[:, 1] = verts[:, 2]
        rotated[:, 2] = verts[:, 1]
    elif tallest == 0:
        # X is tallest → swap X and Z
        rotated[:, 0] = verts[:, 2]
        rotated[:, 2] = verts[:, 0]

    return rotated


def _apply_axis_correction(verts: np.ndarray, data_source: str, config: dict) -> np.ndarray:
    """Apply deterministic source-specific axis correction.

    Supported corrections:
      - x_plus_90: [x, y, z] -> [x, -z, y]
      - x_minus_90: [x, y, z] -> [x, z, -y]
    """
    mapping = config.get("axis_correction_by_source", {})
    if not isinstance(mapping, dict):
        return verts

    mode = mapping.get(data_source)
    if not isinstance(mode, str) or len(verts) == 0:
        return verts

    if mode == "x_plus_90":
        out = verts.copy()
        out[:, 1] = -verts[:, 2]
        out[:, 2] = verts[:, 1]
        return out
    if mode == "x_minus_90":
        out = verts.copy()
        out[:, 1] = verts[:, 2]
        out[:, 2] = -verts[:, 1]
        return out
    return verts


# ── Core: convert master cache object to training item ──────────────

def _object_to_training_item(
    obj: dict,
    master_entry: dict,
    label: str,
    data_source: str,
    object_index: int,
    master_cache_rel_path: str,
    mesh_tokenizer: MeshTokenizer,
    text_tokenizer: BPETokenizer,
    config: dict,
) -> dict | None:
    """Convert a master cache object into a training cache item.

    Returns None if the object doesn't meet training criteria.
    """
    max_faces = config["max_faces"]
    min_faces = config["min_faces"]
    max_text = config["max_text_length"]

    # Get geometry
    verts_t = obj["vertices"]
    faces_t = obj["faces"]
    if isinstance(verts_t, torch.Tensor):
        verts = verts_t.numpy()
        faces = faces_t.numpy()
    else:
        verts = np.array(verts_t, dtype=np.float32)
        faces = np.array(faces_t, dtype=np.int32)

    n_faces = len(faces)
    n_verts = len(verts)

    if n_faces < min_faces or n_verts < 4:
        return None

    # Quality filter
    quality = obj.get("quality", {})
    if quality.get("quality_score", 0) < config.get("min_quality_score", 0):
        return None

    # Material requirement
    if config.get("require_materials") and not obj.get("materials"):
        return None

    # Source-specific axis correction (e.g., Objaverse Y-up -> Blender Z-up)
    verts = _apply_axis_correction(verts, data_source, config)

    # Normalize to [-1, 1]
    try:
        verts_list = verts.tolist()
        verts_list = normalize_mesh(verts_list, target_range=(-1.0, 1.0))
        verts = np.array(verts_list, dtype=np.float32)
    except Exception:
        return None

    # Orientation (optional): fix objects lying sideways
    if config.get("ensure_upright", False):
        verts = _ensure_upright(verts)

    # Handle over-budget meshes
    original_faces = n_faces
    over_budget = n_faces > max_faces
    z_order = None
    selected_face_indices = None
    over_budget_strategy_used = None
    face_material_indices_override = None
    face_smooth_override = None

    kept_indices = None  # indices of faces kept after component extraction

    if over_budget:
        strategy = str(config.get("over_budget_strategy", "simplify_then_subsample")).lower()

        simplified = None
        if strategy in ("simplify_then_subsample", "simplify"):
            simplified = _simplify_mesh_quadric(verts, faces, max_faces)

        if simplified is not None:
            old_verts = verts
            old_faces = faces
            verts, faces = simplified
            over_budget_strategy_used = "quadric_simplify"

            # Remap per-face attributes to simplified faces so validator material
            # assignment remains aligned after topology changes.
            fmi = obj.get("face_material_indices")
            if fmi is not None and len(fmi) > 0:
                face_material_indices_override = _map_per_face_values_by_nearest_centers(
                    fmi, old_verts, old_faces, verts, faces, default=0
                )

            fs = obj.get("face_smooth")
            if fs is not None and len(fs) > 0:
                face_smooth_override = _map_per_face_values_by_nearest_centers(
                    fs, old_verts, old_faces, verts, faces, default=False
                )
        else:
            # Fallback: Z-order sort + uniform subsample to budget.
            # Uniform sampling across z-order preserves whole-object coverage and
            # avoids the "partial chunk" artifacts from prefix truncation.
            z_order = _compute_z_order(verts, faces)
            faces_ordered = faces[z_order]
            if len(faces_ordered) <= max_faces:
                pick = np.arange(len(faces_ordered), dtype=np.int64)
            else:
                pick = np.linspace(0, len(faces_ordered) - 1, num=max_faces, dtype=np.int64)
                pick = np.unique(pick)
                if len(pick) < max_faces:
                    extra = np.setdiff1d(np.arange(len(faces_ordered), dtype=np.int64), pick, assume_unique=False)
                    need = max_faces - len(pick)
                    if need > 0 and len(extra) > 0:
                        pick = np.concatenate([pick, extra[:need]])
                pick = np.sort(pick)
            truncated = faces_ordered[pick]
            selected_face_indices = z_order[pick]

            if config.get("keep_largest_component", False):
                # LCC extraction — can be destructive, disabled by default
                verts, truncated, kept_indices = _largest_connected_component(verts, truncated)
                faces = np.array(truncated) if not isinstance(truncated, np.ndarray) else truncated
            else:
                # Just keep the truncated faces; remap verts to remove orphans
                used_vids = np.unique(truncated.ravel())
                remap = np.full(len(verts), -1, dtype=np.int32)
                remap[used_vids] = np.arange(len(used_vids), dtype=np.int32)
                verts = verts[used_vids]
                faces = remap[truncated]

            over_budget_strategy_used = "zorder_uniform_subsample"

        n_faces = len(faces)
        if n_faces < min_faces:
            return None

        # Re-normalize so the truncated region fills [-1, 1]
        try:
            verts_list = verts.tolist() if isinstance(verts, np.ndarray) else verts
            verts_list = normalize_mesh(verts_list, target_range=(-1.0, 1.0))
            verts = np.array(verts_list, dtype=np.float32)
        except Exception:
            pass

    # Tokenize
    tokens = mesh_tokenizer.encode_mesh(verts.tolist(), faces.tolist())
    if not tokens or len(tokens) < 3:
        return None
    if tokens[0] != mesh_tokenizer.BOS or tokens[-1] != mesh_tokenizer.EOS:
        return None

    # Encode text
    text_ids, text_mask = text_tokenizer.encode_padded(label, max_text)

    # Quality weight
    qw = 0.5
    if n_faces > 100:
        qw += 0.2
    if n_faces > 500:
        qw += 0.1
    if obj.get("materials"):
        qw += 0.15
    if over_budget:
        qw += 0.1
    sample_weight = max(0.3, 0.3 + qw * 1.2)

    # Append detail tag for over-budget
    final_label = label
    if over_budget and original_faces > 2000:
        detail = "high-detail" if original_faces > 10000 else "detailed"
        final_label = f"{label} ({detail})"

    item = {
        "text_ids": torch.tensor(text_ids, dtype=torch.long),
        "text_mask": torch.tensor(text_mask, dtype=torch.float),
        "mesh_tokens": torch.tensor(tokens, dtype=torch.long),
        "quality_weight": torch.tensor(sample_weight, dtype=torch.float),
        "label": final_label,
        "data_source": data_source,
    }

    # Always store raw geometry (pre-quantization) so the validator can
    # display the true mesh without re-tokenization artifacts.
    item["raw_vertices"] = torch.tensor(verts, dtype=torch.float32)
    item["raw_faces"] = torch.tensor(np.array(faces, dtype=np.int32))

    if over_budget:
        item["over_budget"] = True
        item["original_face_count"] = original_faces
        if over_budget_strategy_used:
            item["over_budget_strategy"] = over_budget_strategy_used

    # Master cache reference for full-resolution access
    item["master_cache_ref"] = {
        "data_source": data_source,
        "source_file": master_entry.get("_source_file", ""),
        "cache_rel_path": master_cache_rel_path,
        "object_index": int(object_index),
    }

    # Scene context (materials, UVs, etc.) — with proper face-order mapping
    sc = _build_scene_context(obj, master_entry,
                              z_order=z_order,
                              selected_face_indices=selected_face_indices,
                              kept_indices=kept_indices,
                              face_material_indices_override=face_material_indices_override,
                              face_smooth_override=face_smooth_override)
    if sc:
        item["scene_context"] = sc

    return item


# ── Scene decomposition helper ──────────────────────────────────────

def _decompose_and_append(
    obj: dict,
    master_entry: dict,
    parent_label: str,
    data_source: str,
    mesh_tok: MeshTokenizer,
    text_tok: BPETokenizer,
    config: dict,
    items_out: list,
    seen_hashes: set,
    stats: dict,
) -> None:
    """Extract connected components from a mesh object and add each as
    a separate training item.  Modifies *items_out* in-place.

    Only produces additional items when >=2 components meet the minimum-
    face threshold — single-component meshes are already handled by the
    normal path.
    """
    min_comp_faces = config.get("min_component_faces", 20)
    max_comps = config.get("max_components_per_object", 16)
    min_faces = config.get("min_faces", 50)
    max_text = config.get("max_text_length", 256)

    # Raw geometry from the object
    verts_t = obj["vertices"]
    faces_t = obj["faces"]
    if isinstance(verts_t, torch.Tensor):
        verts = verts_t.numpy()
        faces = faces_t.numpy()
    else:
        verts = np.array(verts_t, dtype=np.float32)
        faces = np.array(faces_t, dtype=np.int32)

    if len(faces) < min_comp_faces * 2:
        return  # too few faces to ever produce >=2 usable components

    components = _extract_all_components(verts, faces, min_faces=min_comp_faces)

    # Only decompose if there are at least 2 qualifying components
    if len(components) < 2:
        return

    components = components[:max_comps]

    added = 0
    for ci, (c_verts, c_faces, _kept_fi) in enumerate(components):
        n_f = len(c_faces)
        if n_f < min_faces:
            continue

        # Normalize component to [-1, 1]
        try:
            c_verts_list = c_verts.tolist()
            c_verts_list = normalize_mesh(c_verts_list, target_range=(-1.0, 1.0))
            c_verts = np.array(c_verts_list, dtype=np.float32)
        except Exception:
            continue

        # Tokenize component
        tokens = mesh_tok.encode_mesh(c_verts.tolist(), c_faces.tolist())
        if not tokens or len(tokens) < 3:
            continue
        if tokens[0] != mesh_tok.BOS or tokens[-1] != mesh_tok.EOS:
            continue

        # Dedup
        tok_hash = hash(tuple(tokens))
        if tok_hash in seen_hashes:
            stats["items_rejected"]["duplicate"] += 1
            continue
        seen_hashes.add(tok_hash)

        # Label
        comp_label = _label_component(
            parent_label, ci, len(components), c_verts, n_f)

        text_ids, text_mask = text_tok.encode_padded(comp_label, max_text)

        # Quality weight (slightly lower than whole-object)
        qw = 0.4
        if n_f > 100:
            qw += 0.15
        if n_f > 500:
            qw += 0.1
        if obj.get("materials"):
            qw += 0.1
        sample_weight = max(0.3, 0.3 + qw * 1.0)

        item = {
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.float),
            "mesh_tokens": torch.tensor(tokens, dtype=torch.long),
            "quality_weight": torch.tensor(sample_weight, dtype=torch.float),
            "label": comp_label,
            "data_source": data_source,
            "raw_vertices": torch.tensor(c_verts, dtype=torch.float32),
            "raw_faces": torch.tensor(np.array(c_faces, dtype=np.int32)),
            "decomposed_component": True,
            "component_index": ci,
            "total_components": len(components),
            "master_cache_ref": {
                "data_source": data_source,
                "source_file": master_entry.get("_source_file", ""),
            },
        }

        # Build scene context for the component (limited — mainly materials
        # for the faces we kept, if any).  Skip for simplicity — the component
        # inherits the parent's material list but face-material mapping would
        # need per-face reindexing.  Components are primarily for geometry
        # training; materials can come from the whole-object items.

        items_out.append(item)
        added += 1

    if added > 0:
        stats.setdefault("components_created", 0)
        stats["components_created"] = stats.get("components_created", 0) + added


# ── Main build loop ─────────────────────────────────────────────────

def build_training_cache(config: dict, dry_run: bool = False) -> dict:
    """Build a training cache from the master cache using the given config."""
    task_name = config.get("task_name", "default")
    out_dir = TRAINING_CACHE_DIR / task_name
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Init tokenizers
    mesh_tok = MeshTokenizer(
        vocab_size=config.get("vocab_size", 8192),
        max_faces=config.get("max_faces", 8000),
    )
    bpe_path = BASE / "data" / "datasets" / "geometry" / "bpe_tokenizer"
    text_tok = BPETokenizer.load(bpe_path)

    # Load master cache index
    index_path = MASTER_CACHE_DIR / "index.pt"
    if not index_path.exists():
        logger.error("Master cache index not found. Run build_master_cache.py first.")
        return {}

    index = torch.load(index_path, weights_only=False)
    entries = index["entries"]
    logger.info(f"Master cache: {len(entries)} source files")

    # Filter by source
    allowed_sources = config.get("sources")
    if allowed_sources:
        entries = [e for e in entries if e["source"] in allowed_sources]
        logger.info(f"After source filter: {len(entries)} files")

    stats = {
        "files_processed": 0,
        "items_created": 0,
        "items_rejected": defaultdict(int),
        "label_counts": defaultdict(int),
        "source_counts": defaultdict(int),
        "total_faces": 0,
        "bytes_written": 0,
    }

    t0 = time.time()
    all_items = []  # Collect for batching into output .pt files
    batch_size = 50  # Items per output .pt file
    batch_idx = 0
    seen_token_hashes = set()

    for ei, entry_meta in enumerate(entries):
        cache_path = MASTER_CACHE_DIR / entry_meta["path"]
        if not cache_path.exists():
            continue

        try:
            master_entry = torch.load(cache_path, weights_only=False)
        except Exception as e:
            logger.warning(f"Failed to load {cache_path}: {e}")
            continue

        objects = master_entry.get("objects", [])
        metadata = master_entry.get("metadata", {})
        file_label = master_entry.get("label", "")
        data_source = master_entry.get("_data_source", entry_meta.get("source", "unknown"))
        is_single = len(objects) == 1

        # Cap objects processed per file to avoid spending minutes on
        # massive files with 5000+ objects when we only keep max_per_file.
        # Sample evenly rather than just taking first N.
        max_objs_to_process = config.get("max_objects_per_file", 500)
        if len(objects) > max_objs_to_process:
            step = len(objects) / max_objs_to_process
            indices = [int(i * step) for i in range(max_objs_to_process)]
            objects_to_process = [(idx, objects[idx]) for idx in indices]
            stats["items_rejected"]["file_too_large"] += len(objects) - max_objs_to_process
        else:
            objects_to_process = list(enumerate(objects))

        # Collect sibling names for labeling
        all_names = [o.get("name", "") for o in objects]

        file_items = []

        for oi, obj in objects_to_process:
            # Generate label
            sibling_names = [n for i, n in enumerate(all_names) if i != oi and n]
            label = _generate_label(obj, metadata, file_label, is_single, sibling_names)

            # Label quality gates
            if not label or _is_weak_label(label):
                stats["items_rejected"]["weak_label"] += 1
                continue
            if config.get("require_english", True) and _is_non_english(label):
                stats["items_rejected"]["non_english"] += 1
                continue

            # Convert to training item
            item = _object_to_training_item(
                obj, master_entry, label, data_source,
                oi, entry_meta.get("path", ""),
                mesh_tok, text_tok, config,
            )
            if item is None:
                stats["items_rejected"]["quality_filter"] += 1
                continue

            # Dedup by token hash
            token_hash = hash(tuple(item["mesh_tokens"].tolist()))
            if token_hash in seen_token_hashes:
                stats["items_rejected"]["duplicate"] += 1
                continue
            seen_token_hashes.add(token_hash)

            file_items.append(item)

        # Per-file cap
        max_per_file = config.get("max_per_source_file", 10)
        if len(file_items) > max_per_file:
            # Keep highest quality
            file_items.sort(
                key=lambda x: x["quality_weight"].item(), reverse=True)
            stats["items_rejected"]["file_cap"] += len(file_items) - max_per_file
            file_items = file_items[:max_per_file]

        # Per-label cap
        max_per_label = config.get("max_per_label", 50)
        for item in file_items:
            lbl = item["label"].lower().strip()
            if stats["label_counts"][lbl] >= max_per_label:
                stats["items_rejected"]["label_cap"] += 1
                continue

            stats["label_counts"][lbl] += 1
            stats["source_counts"][data_source] += 1
            stats["items_created"] += 1
            n_faces = (len(item["mesh_tokens"]) - 2) // 9
            stats["total_faces"] += n_faces
            if not dry_run:
                all_items.append(item)

        stats["files_processed"] += 1

        # Write batches
        if not dry_run and len(all_items) >= batch_size:
            batch_path = out_dir / f"batch_{batch_idx:05d}.pt"
            torch.save(all_items[:batch_size], batch_path)
            stats["bytes_written"] += batch_path.stat().st_size
            batch_idx += 1
            all_items = all_items[batch_size:]

        # Progress
        if (ei + 1) % 100 == 0 or ei == len(entries) - 1:
            elapsed = time.time() - t0
            logger.info(
                f"[{ei + 1}/{len(entries)}] "
                f"{stats['items_created']} items, "
                f"{stats['total_faces']:,} faces, "
                f"{elapsed:.0f}s"
            )

        # GC
        if master_entry.get("summary", {}).get("total_faces", 0) > 100_000:
            del master_entry
            gc.collect()

    # Write remaining items
    if not dry_run and all_items:
        batch_path = out_dir / f"batch_{batch_idx:05d}.pt"
        torch.save(all_items, batch_path)
        stats["bytes_written"] += batch_path.stat().st_size

    # Write config (include total_items for UI)
    if not dry_run:
        config_path = out_dir / "config.json"
        import json
        config_out = dict(config)
        config_out["total_items"] = stats["items_created"]
        config_out["batch_size"] = batch_size
        with open(config_path, "w") as f:
            json.dump(config_out, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"Training cache '{task_name}' built in {elapsed:.0f}s")
    logger.info(f"  Items created:  {stats['items_created']}")
    logger.info(f"  Total faces:    {stats['total_faces']:,}")
    logger.info(f"  Files processed: {stats['files_processed']}")
    logger.info(f"  Bytes written:  {stats['bytes_written'] / 1024 / 1024:.1f} MB")
    if stats.get("components_created"):
        logger.info(f"  Decomposed components: {stats['components_created']}")
    logger.info("\n  Rejections:")
    for reason, count in sorted(stats["items_rejected"].items()):
        logger.info(f"    {reason}: {count}")
    logger.info("\n  Source distribution:")
    for src, count in sorted(stats["source_counts"].items(),
                              key=lambda x: -x[1]):
        pct = 100 * count / max(1, stats["items_created"])
        logger.info(f"    {src}: {count} ({pct:.1f}%)")

    return stats


# ── Config loading ──────────────────────────────────────────────────

def load_config(config_path: str | None, overrides: dict) -> dict:
    """Load training cache config from file with CLI overrides."""
    config = dict(DEFAULT_CONFIG)

    if config_path:
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        # Look for training_cache section
        if "training_cache" in file_cfg:
            config.update(file_cfg["training_cache"])
        else:
            config.update(file_cfg)

    # CLI overrides
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    return config


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build training caches from master cache")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML file")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name (output dir name)")
    parser.add_argument("--max-faces", type=int, default=None)
    parser.add_argument("--min-faces", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--max-per-file", type=int, default=None,
                        dest="max_per_source_file")
    parser.add_argument("--require-materials", action="store_true", default=None)
    parser.add_argument("--source", type=str, default=None,
                        help="Comma-separated sources")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stats", action="store_true",
                        help="Show existing training cache stats")
    args = parser.parse_args()

    if args.stats:
        _show_stats()
        return

    overrides = {}
    if args.task:
        overrides["task_name"] = args.task
    if args.max_faces is not None:
        overrides["max_faces"] = args.max_faces
    if args.min_faces is not None:
        overrides["min_faces"] = args.min_faces
    if args.vocab_size is not None:
        overrides["vocab_size"] = args.vocab_size
    if args.max_per_label is not None:
        overrides["max_per_label"] = args.max_per_label
    if args.max_per_source_file is not None:
        overrides["max_per_source_file"] = args.max_per_source_file
    if args.require_materials is not None:
        overrides["require_materials"] = args.require_materials
    if args.source:
        overrides["sources"] = [s.strip() for s in args.source.split(",")]

    config = load_config(args.config, overrides)
    logger.info(f"Config: {config}")

    build_training_cache(config, dry_run=args.dry_run)


def _show_stats():
    """Show stats about existing training caches."""
    if not TRAINING_CACHE_DIR.exists():
        logger.info("No training caches found.")
        return
    for task_dir in sorted(TRAINING_CACHE_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        pt_files = list(task_dir.glob("*.pt"))
        total_items = 0
        total_size = sum(f.stat().st_size for f in pt_files)
        for f in pt_files:
            try:
                items = torch.load(f, weights_only=False)
                if isinstance(items, list):
                    total_items += len(items)
            except Exception:
                pass
        config_path = task_dir / "config.json"
        cfg = {}
        if config_path.exists():
            import json
            cfg = json.load(open(config_path))
        logger.info(
            f"  {task_dir.name}: {total_items} items, "
            f"{len(pt_files)} files, "
            f"{total_size / 1024 / 1024:.1f} MB, "
            f"max_faces={cfg.get('max_faces', '?')}"
        )


if __name__ == "__main__":
    main()
