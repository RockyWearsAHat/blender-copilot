"""Rebuild mesh cache: decimate + relabel ALL training data.

This script processes all 1973+ cached .pt files AND all source JSON files
to create high-quality single-object training samples with descriptive labels.

Key improvements over current cache:
1. DECIMATE meshes >2000 faces to fit training budget (currently 70% wasted)
2. RELABEL using object names, material names, mesh properties (not vague 'small 3d mesh')
3. SINGLE OBJECT focus — multi-object files produce separate cache entries per object
4. GEOMETRY ANALYSIS — aspect ratio, symmetry, face count inform the label
5. SOURCE ATTRIBUTION — each sample records which data source it came from
6. LABEL FREQUENCY CAP — optional max samples per label to prevent overfitting

By default this script is INCREMENTAL: source files that already have a
.pt cache entry are skipped. Pass --force-rebuild to reprocess everything.

Usage:
    python scripts/rebuild_cache.py                        # Incremental (new files only)
    python scripts/rebuild_cache.py --force-rebuild        # Reprocess all source files
    python scripts/rebuild_cache.py --dry-run              # Preview without writing
    python scripts/rebuild_cache.py --source-only          # Only process source JSONs
    python scripts/rebuild_cache.py --relabel-only         # Only fix labels in existing cache
    python scripts/rebuild_cache.py --max-per-label 100    # Cap samples per label (recommended)
    python scripts/rebuild_cache.py --cap-existing         # Cap labels in already-built cache
    python scripts/rebuild_cache.py --fix-attribution      # Back-fill data_source on old cache
"""
import argparse
import gc
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.mesh_tokenizer import MeshTokenizer
from processing.bpe_tokenizer import BPETokenizer
from processing.generate_synthetic import normalize_mesh
from processing.labeler_smart import generate_smart_label, compute_bbox_aspect
from processing.qwen_client import (
    qwen_label_text, build_label_context, warm_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent


def _qwen_label(obj_name, mat_names, file_label, meta_name,
                meta_desc, meta_tags, num_faces, timeout=30):
    """Call Qwen text model to generate a clean English label from all context.

    Delegates to the shared qwen_client which:
    - Keeps the model warm to avoid cold-boot overhead
    - Filters animation/behavioral tags via build_label_context()
    - Uses the _PHYSICAL_TAGS allowlist so only geometry-relevant tags
      reach the LLM prompt.

    Returns a 2-8 word English label string, or None on failure.
    """
    # Build structured context — this strips animation/behavioral tags
    context = build_label_context(
        obj_name=obj_name,
        material_names=mat_names or [],
        file_label=file_label or "",
        metadata_name=meta_name or "",
        metadata_desc=meta_desc or "",
        metadata_tags=meta_tags or [],
        num_faces=num_faces,
    )

    return qwen_label_text(context, timeout=timeout)
CACHE_DIR = BASE / "data" / "processed" / ".mesh_cache"
SOURCE_DIRS = [
    BASE / "data" / "processed" / "objaverse",
    BASE / "data" / "processed" / "blender_official",
    BASE / "data" / "processed" / "blendswap",
    BASE / "data" / "processed" / "smutbase",
    BASE / "data" / "processed" / "github",
    BASE / "data" / "processed" / "open3dlab",
    BASE / "data" / "processed" / "youtube",
]


def _load_face_budget_from_config() -> int:
    cfg_path = BASE / "config.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        cfg = {}
    token_cfg = cfg.get("tokenization", {}) if isinstance(cfg, dict) else {}
    unified_cfg = cfg.get("unified", {}) if isinstance(cfg, dict) else {}
    geom_cfg = unified_cfg.get("geometry", {}) if isinstance(unified_cfg, dict) else {}

    token_faces = int(token_cfg.get("max_faces", 0) or 0)
    max_seq = int(geom_cfg.get("max_seq_length", 36_002) or 36_002)
    seq_faces = max(0, (max_seq - 2) // 9)

    default_faces = max(4_000, token_faces, seq_faces)
    env_faces = int(os.environ.get("TRAIN_MAX_FACES", default_faces))
    return max(256, env_faces)


MAX_FACES = _load_face_budget_from_config()
TARGET_DECIMATE = max(256, min(MAX_FACES - 64, int(MAX_FACES * 0.95)))


# ── Quality thresholds ────────────────────────────────────────────────
MIN_FACES_THRESHOLD = 50     # Skip objects with fewer faces (fragments)
MAX_ITEMS_PER_FILE = 10      # Cap output per source file (keep best)

_GENERIC_PART_LABELS = {
    "object", "mesh", "3d object", "3d mesh", "shape", "3d shape", "part", "piece",
    "simple object", "basic object", "simple shape", "detailed object",
    "multi object scene composition",  # generic fallback
    "dummy", "base", "lattice",  # useless
}
_PRIMITIVE_WORDS = {
    "cube", "box", "sphere", "cylinder", "cone", "plane", "grid", "torus",
    "icosphere", "circle", "disc", "disk", "bezier", "curve", "nurbs", "path",
    "primitive",
}
# Material/color words that don't describe what a 3D object IS.
# Labels consisting entirely of these are weak supervision.
_MATERIAL_COLOR_WORDS = {
    "metal", "wood", "stone", "glass", "plastic", "rubber", "fabric",
    "concrete", "brick", "ceramic", "leather", "steel", "iron",
    "copper", "brass", "chrome", "aluminum", "aluminium",
    "gold", "silver", "bronze",
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "grey", "gray", "brown", "cyan", "magenta",
    "dark", "light", "bright", "shiny", "matte", "glossy",
    "dry", "wet", "rough", "smooth",
    "point", "vert", "vertex", "edge", "face",  # Blender internals
    "toon", "cel", "outline", "emission", "transparent",
    "normale", "normal", "luce", "ral",
    "irradiancevolume",
}

# Regex patterns for garbage labels — Source engine entities, file paths,
# non-English text, known gibberish.
_GARBAGE_LABEL_PATTERNS = [
    re.compile(r'^(prop[_ ]static|func[_ ]detail|worldspawn)\b', re.I),
    re.compile(r'^bp house \d+', re.I),
    re.compile(r'materials/', re.I),           # file path fragments
    re.compile(r'~\w+/', re.I),                # ~username/ paths
    re.compile(r'toolsnodraw', re.I),          # Source engine
    re.compile(r'ansazirconoceno', re.I),       # gibberish
    re.compile(r'metallkedja', re.I),           # Swedish
    re.compile(r'\bgroupe\b', re.I),            # French
    re.compile(r'daun.*hijau', re.I),           # Indonesian
    re.compile(r'h[ìí]nh\s+tr[ụu]', re.I),    # Vietnamese "cylinder"
    re.compile(r'blanco\s+brillante', re.I),    # Spanish
    re.compile(r'^(curva|curvas|beton|noeud)$', re.I),  # non-English single words
]
# Non-Latin scripts → non-English label
_NON_LATIN_RE = re.compile(r'[\u0400-\u04ff\u0600-\u06ff\u0900-\u097f\u0e00-\u0e7f\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff]')

# Common non-English words (primarily Indonesian Blender assets)
_NON_ENGLISH_WORDS = {
    # Indonesian
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
    # French
    "groupe", "noeud", "curva", "beton", "lumiere", "fenetre", "porte",
    "maison", "arbre", "voiture", "bateau",
    # Swedish
    "metallkedja", "hus", "bil", "stol", "bord",
    # Spanish
    "blanco", "brillante", "rojo", "azul", "verde", "amarillo", "negro",
    "casa", "coche", "ventana",
    # Italian
    "normale", "luce",
}


def _is_non_english_label(label: str) -> bool:
    """Detect non-English labels using word frequency analysis."""
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


def _is_semantically_weak_label(label: str) -> bool:
    if not label:
        return True
    clean = re.sub(r'[^a-z0-9\s,]', ' ', label.lower())
    clean = re.sub(r'\s+', ' ', clean).strip(' ,')
    if not clean:
        return True
    if clean in _GENERIC_PART_LABELS:
        return True

    tokens = [t for t in re.split(r'[\s,]+', clean) if t]
    if not tokens:
        return True

    primitive_tokens = [t for t in tokens if t in _PRIMITIVE_WORDS]
    if len(tokens) <= 2 and len(primitive_tokens) == len(tokens):
        return True

    # labels like "cube cell", "point cube", "room cube" are weak proxies
    if len(tokens) == 2 and (tokens[0] in _PRIMITIVE_WORDS or tokens[1] in _PRIMITIVE_WORDS):
        return True

    # Labels consisting entirely of material/color words are not object labels
    material_tokens = [t for t in tokens if t in _MATERIAL_COLOR_WORDS]
    if len(material_tokens) == len(tokens):
        return True

    # numeric-heavy / id-like labels are low quality supervision
    alnum = re.sub(r'[^a-z0-9]', '', clean)
    if not alnum:
        return True
    digit_ratio = sum(c.isdigit() for c in alnum) / max(1, len(alnum))
    if digit_ratio > 0.35:
        return True

    # Non-Latin characters (Cyrillic, CJK, Arabic, etc.) → non-English
    if _NON_LATIN_RE.search(label):
        return True

    # Known garbage patterns (source engine, file paths, non-English, gibberish)
    for pat in _GARBAGE_LABEL_PATTERNS:
        if pat.search(label):
            return True

    return False


# ── Decimation ───────────────────────────────────────────────────────

def _count_connected_components(faces_arr):
    """Count connected components in a face array via union-find.
    
    High component count relative to face count = shattered mesh = garbage.
    """
    if len(faces_arr) == 0:
        return 0
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
    for face in faces_arr:
        for i in range(1, len(face)):
            union(int(face[0]), int(face[i]))
    all_verts = set()
    for face in faces_arr:
        all_verts.update(int(v) for v in face)
    return len(set(find(v) for v in all_verts))


def _largest_connected_component(verts, faces):
    """Return (verts, faces) of only the largest connected component.
    
    After z-order truncation, over-budget meshes may fragment into multiple
    disconnected pieces.  Keeping only the largest component ensures the
    training mesh is a single connected surface — not shredded fragments.
    
    Uses union-find on vertex indices, then keeps the component with the
    most faces.  Vertex indices are compacted so the returned mesh is clean.
    """
    if len(faces) == 0:
        return verts, faces
    
    # Union-find
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
    
    # Group faces by component root
    from collections import defaultdict
    comp_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        root = find(int(face[0]))
        comp_faces[root].append(fi)
    
    if len(comp_faces) <= 1:
        return verts, faces  # Already single component
    
    # Keep the largest component (most faces)
    largest_root = max(comp_faces, key=lambda r: len(comp_faces[r]))
    keep_face_indices = comp_faces[largest_root]
    
    # Extract faces and compact vertices
    v_arr = np.array(verts) if not isinstance(verts, np.ndarray) else verts
    f_arr = np.array(faces) if not isinstance(faces, np.ndarray) else faces
    kept_faces = f_arr[keep_face_indices]
    
    # Remap vertex indices
    used_verts = np.unique(kept_faces.ravel())
    old_to_new = {int(old): new for new, old in enumerate(used_verts)}
    new_verts = v_arr[used_verts]
    new_faces = np.vectorize(lambda x: old_to_new[x])(kept_faces)
    
    return new_verts.tolist(), new_faces.tolist()


def decimate_mesh(verts, faces, target_faces):
    """DEPRECATED — no longer used. We let the tokenizer truncate in z-order instead.
    
    Kept for reference. Decimate mesh using trimesh quadric decimation.
    
    Returns (verts, faces, success).
    
    NEVER falls back to random face sampling — that produces disconnected
    triangle soup that poisons training data.  If proper decimation fails,
    returns success=False so the caller can skip the mesh.
    """
    try:
        import trimesh
        v_arr = np.array(verts, dtype=np.float64)
        f_arr = np.array(faces, dtype=np.int64)
        if f_arr.ndim != 2 or f_arr.shape[1] != 3:
            return verts, faces, False
        
        mesh = trimesh.Trimesh(vertices=v_arr, faces=f_arr, process=False)
        
        for aggression in [None, 7, 10]:
            try:
                kwargs = {"aggression": aggression} if aggression else {}
                d = mesh.simplify_quadric_decimation(face_count=target_faces, **kwargs)
                if len(d.faces) <= int(target_faces * 1.1) and len(d.faces) >= 4:
                    # Quality check: reject if decimation shattered the mesh
                    n_comp = _count_connected_components(d.faces)
                    comp_ratio = n_comp / max(len(d.faces), 1)
                    if comp_ratio > 0.15:
                        logger.debug(f"  Decimation shattered mesh: {n_comp} components / "
                                     f"{len(d.faces)} faces = {comp_ratio:.2f} — trying higher aggression")
                        continue
                    return d.vertices.tolist(), d.faces.tolist(), True
            except Exception:
                continue
        
        # Retry with mesh cleanup (merge verts, remove degenerates) then decimate
        try:
            mesh_clean = trimesh.Trimesh(vertices=v_arr, faces=f_arr, process=True)
            mesh_clean.merge_vertices()
            mesh_clean.remove_degenerate_faces()
            if len(mesh_clean.faces) <= target_faces:
                return mesh_clean.vertices.tolist(), mesh_clean.faces.tolist(), True
            d = mesh_clean.simplify_quadric_decimation(face_count=target_faces, aggression=7)
            if len(d.faces) <= int(target_faces * 1.1):
                n_comp = _count_connected_components(d.faces)
                comp_ratio = n_comp / max(len(d.faces), 1)
                if comp_ratio <= 0.15:
                    return d.vertices.tolist(), d.faces.tolist(), True
                logger.debug(f"  Cleaned decimation still shattered: {comp_ratio:.2f}")
        except Exception:
            pass
        
        # NO random face sampling fallback — skip this mesh entirely.
        # Random sampling produces disconnected triangle soup that
        # teaches the model to generate garbage geometry.
        logger.debug(f"  Decimation failed for {len(faces)}-face mesh — skipping (no random fallback)")
        return verts, faces, False
        
    except Exception as e:
        logger.warning(f"Decimation failed ({len(faces)} faces): {e}")
        return verts, faces, False


def _merge_meshes(meshes):
    """Merge a list of (verts, faces) meshes into one mesh."""
    out_verts = []
    out_faces = []
    vert_offset = 0
    for verts, faces in meshes:
        if not verts or not faces:
            continue
        out_verts.extend(verts)
        out_faces.extend([[idx + vert_offset for idx in face] for face in faces])
        vert_offset += len(verts)
    return out_verts, out_faces


def _build_scene_label(file_label: str, metadata_name: str, metadata_categories) -> str:
    """Build scene/composition label for multi-object files.
    
    Returns empty string if no usable English label can be constructed.
    """
    for candidate in (metadata_name, file_label):
        if candidate:
            clean = re.sub(r'[^a-zA-Z0-9\s,_-]', ' ', str(candidate)).lower()
            clean = re.sub(r'[_-]+', ' ', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()
            if clean and not _is_semantically_weak_label(clean) and not _is_non_english_label(clean):
                return clean[:100]

    if metadata_categories:
        cat_text = str(metadata_categories).lower().strip()
        if cat_text and not _is_non_english_label(cat_text):
            return f"{cat_text} scene"[:100]

    return ""  # Return empty instead of generic label — caller should skip


# ── Source JSON Processing ───────────────────────────────────────────

def process_source_file(filepath, mesh_tokenizer, text_tokenizer,
                        max_text_length=256, dry_run=False,
                        data_source="unknown"):
    """Process a source JSON into individual object cache items.

    Each object in the file becomes a SEPARATE cache entry with its own label.
    ``data_source`` is stored on every item for per-source quality tracking.
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, MemoryError) as e:
        logger.debug(f"Failed to load {filepath}: {e}")
        return []
    
    if isinstance(data, dict):
        objects = data.get('objects', [data])
        metadata = data.get('metadata', {})
        data_ref = data  # Keep reference for scene context extraction
    else:
        objects = [data]
        metadata = {}
        data_ref = {}
    
    meta_name = metadata.get('name', '')
    meta_desc = str(metadata.get('description', ''))[:200]
    meta_tags = metadata.get('tags', [])
    meta_cats = metadata.get('categories', metadata.get('category', ''))
    
    # File-level label from filename — clean up common junk patterns
    file_label = Path(filepath).stem
    # Clean hex UIDs from Objaverse
    if re.match(r'^[0-9a-f]{32}$', file_label):
        file_label = ''
    # Clean BlendSwap numeric ID prefixes (e.g. "4732_popular_s12_barcelona")
    file_label = re.sub(r'^\d+[_\s]+popular[_\s]*', '', file_label)
    file_label = re.sub(r'^\d+[_\s]+', '', file_label)
    
    items = []
    max_f = MAX_FACES
    
    # Track seen token hashes for deduplication within this file
    seen_token_hashes = set()
    
    # Collect ALL object names for sibling context in labeling
    all_obj_names = [o.get('name', '') or '' for o in objects]
    
    is_single_object = len(objects) == 1

    scene_mesh_parts = []

    for idx, obj in enumerate(objects):
        mesh = obj.get('mesh', {})
        verts = mesh.get('vertices', [])
        faces = mesh.get('faces', [])
        
        if not verts or not faces or len(faces) < MIN_FACES_THRESHOLD or len(verts) < 4:
            continue
        
        # Get object-level info
        obj_name = obj.get('name', '') or ''
        raw_mats = obj.get('materials', [])
        mod_types = [m.get('type', '') or '' for m in obj.get('modifiers', [])]

        # Geometry provenance: prefer original/raw geometry for modeler training.
        raw_verts = obj.get('raw_vertices', [])
        raw_faces = obj.get('raw_faces', [])
        has_raw_geometry = bool(
            isinstance(raw_verts, list) and isinstance(raw_faces, list)
            and len(raw_verts) > 0 and len(raw_faces) > 0
        )
        has_nodes_modifier = any(str(t).upper() == 'NODES' for t in mod_types)

        # If this object is driven by Geometry Nodes but has no original/raw mesh,
        # the evaluated mesh is effectively baked scene-builder output, not
        # modeling supervision. Skip it from modeler cache.
        if has_nodes_modifier and not has_raw_geometry:
            continue

        if has_raw_geometry:
            verts = raw_verts
            faces = raw_faces
        
        # Only use material names from PROPERLY APPLIED materials.
        # A proper material has a real node tree (use_nodes=True, has shader nodes).
        # Materials that are just empty slots or Blender defaults ("simple" type,
        # use_nodes=False) provide no real info and pollute labels.
        mat_names = []
        has_real_materials = False
        for m in raw_mats:
            name = m.get('name', '') or ''
            nodes = m.get('nodes', [])
            use_nodes = m.get('use_nodes', True)  # Default true for objaverse format
            mat_type = m.get('type', '')
            
            # A material is "real" if it has shader nodes beyond just Material Output,
            # or has base_color that isn't default grey, or has use_nodes=True with nodes.
            is_real = False
            if nodes and len(nodes) > 1:
                # Has actual shader nodes (BSDF, textures, etc.)
                is_real = True
            elif use_nodes and nodes:
                is_real = True
            elif m.get('base_color'):
                bc = m.get('base_color', [])
                # Check if it's not just default grey [0.4, 0.4, 0.4, 1.0] or [0.5, 0.5, 0.5, 1.0]
                if isinstance(bc, (list, tuple)) and len(bc) >= 3:
                    r, g, b = float(bc[0]), float(bc[1]), float(bc[2])
                    is_default_grey = (abs(r - g) < 0.01 and abs(g - b) < 0.01 
                                       and 0.3 < r < 0.6)
                    if not is_default_grey:
                        is_real = True
            elif mat_type == 'simple' and not use_nodes:
                is_real = False
            
            if name:
                mat_names.append(name)
            if is_real:
                has_real_materials = True
        
        # Skip tiny objects that are likely just helpers/fragments
        if len(faces) < MIN_FACES_THRESHOLD:
            continue
        
        # Try lossless cleanup first (merge dups, remove degenerates).
        # If still over budget, let the tokenizer naturally truncate to max_faces.
        # Z-order sorting ensures the truncated portion is spatially coherent —
        # much better than decimation which destroys mesh topology.
        original_faces = len(faces)
        over_budget = False
        if len(faces) > max_f:
            try:
                import trimesh
                v_arr = np.array(verts, dtype=np.float64)
                f_arr = np.array(faces, dtype=np.int64)
                mesh_clean = trimesh.Trimesh(vertices=v_arr, faces=f_arr, process=True)
                mesh_clean.merge_vertices()
                mesh_clean.update_faces(mesh_clean.nondegenerate_faces())
                mesh_clean.update_faces(mesh_clean.unique_faces())
                verts = mesh_clean.vertices.tolist()
                faces = mesh_clean.faces.tolist()
                if len(faces) <= max_f:
                    logger.debug(f"  Lossless cleanup: {original_faces} → {len(faces)} faces")
                else:
                    over_budget = True
                    logger.debug(f"  Over budget: {len(faces)} faces (cleanup from {original_faces}). "
                                 f"Tokenizer will truncate to {max_f} in z-order.")
            except Exception:
                over_budget = len(faces) > max_f
                if over_budget:
                    logger.debug(f"  Over budget: {len(faces)} faces (cleanup failed). "
                                 f"Tokenizer will truncate to {max_f} in z-order.")
        if len(faces) < MIN_FACES_THRESHOLD:
            continue
        
        # For over-budget meshes: pre-truncate in z-order, then keep only
        # the largest connected component.  This prevents the shredded-mesh
        # problem where z-order truncation slices through the surface and
        # produces disconnected fragments.
        if over_budget and len(faces) > max_f:
            v_arr = np.array(verts, dtype=np.float64)
            f_arr = faces  # list of lists
            
            # Replicate tokenizer's z-order sorting
            centers = []
            for face in f_arr:
                valid = [fi for fi in face if fi < len(v_arr)]
                if valid:
                    center = v_arr[valid].mean(axis=0)
                else:
                    center = np.zeros(3)
                centers.append(center)
            centers = np.array(centers)
            c_min = centers.min(axis=0)
            c_max = centers.max(axis=0)
            c_range = c_max - c_min
            c_range[c_range < 1e-6] = 1.0
            normalized_c = ((centers - c_min) / c_range * 1023).astype(int)
            normalized_c = np.clip(normalized_c, 0, 1023)
            
            from processing.mesh_tokenizer import MeshTokenizer
            morton_codes = [MeshTokenizer._morton_encode_3d(c[0], c[1], c[2])
                           for c in normalized_c]
            order = np.argsort(morton_codes)
            truncated_faces = [f_arr[i] for i in order[:max_f]]
            
            # Keep only the largest connected component
            trunc_verts, trunc_faces = _largest_connected_component(verts, truncated_faces)
            if len(trunc_faces) >= 4:
                pre_trunc = len(truncated_faces)
                verts = trunc_verts
                faces = trunc_faces
                logger.debug(f"  Z-order truncation: {original_faces} → {pre_trunc} faces, "
                             f"largest component: {len(faces)} faces")
            # else: fall through with original verts/faces — tokenizer
            #       will truncate but the mesh may be fragmented
        
        if len(faces) < MIN_FACES_THRESHOLD:
            continue
        
        # Normalize to [-1, 1]
        try:
            bbox_aspect = compute_bbox_aspect(verts)
            verts = normalize_mesh(verts, target_range=(-1.0, 1.0))
        except Exception:
            continue
        
        # Tokenize — tokenizer naturally truncates to max_faces in z-order
        tokens = mesh_tokenizer.encode_mesh(verts, faces)
        if not tokens or tokens[0] != mesh_tokenizer.BOS or tokens[-1] != mesh_tokenizer.EOS:
            continue
        
        # Deduplicate: skip if we already have a mesh with identical tokens
        # This removes hundreds of instanced objects (bricks, leaves, etc.)
        token_hash = hash(tuple(tokens))
        if token_hash in seen_token_hashes:
            continue
        seen_token_hashes.add(token_hash)
        
        # Generate smart label (programmatic baseline)
        # Only pass material names to labeler if the object has REAL materials.
        # Fake/default materials would pollute labels with useless names.
        label_mat_names = mat_names if has_real_materials else []
        sibling_names = [n for i, n in enumerate(all_obj_names) if i != idx and n]
        label = generate_smart_label(
            obj_name=obj_name,
            material_names=label_mat_names,
            modifier_types=mod_types,
            num_faces=len(faces),
            num_verts=len(verts),
            bbox_aspect=bbox_aspect,
            file_label=file_label if is_single_object else '',
            metadata_name=meta_name if is_single_object else '',
            metadata_desc=meta_desc if is_single_object else '',
            metadata_tags=meta_tags if is_single_object else [],
            metadata_categories=meta_cats if is_single_object else '',
            sibling_names=sibling_names,
        )
        
        # Upgrade to Qwen label only for single-object files.
        # For multi-object scenes, scene-level terms (e.g. "cafe") can leak
        # and incorrectly relabel every part object.
        if is_single_object:
            qwen_lbl = _qwen_label(
                obj_name=obj_name,
                mat_names=label_mat_names,
                file_label=file_label,
                meta_name=meta_name,
                meta_desc=meta_desc,
                meta_tags=meta_tags,
                num_faces=len(faces),
            )
            if qwen_lbl:
                label = qwen_lbl
        
        # Multi-object part supervision quality gate:
        # skip labels that are too generic/primitive-heavy; these poison prompt alignment.
        if (not is_single_object) and _is_semantically_weak_label(label):
            continue

        # Universal label quality gate — applies to ALL items.
        # Empty labels, material-only labels ("metal", "wood dry"), and
        # generic descriptions ("simple 3d shape") are not trainable.
        if not label or _is_semantically_weak_label(label):
            continue

        # Non-English label gate — skip labels in Indonesian, French, etc.
        # These provide no usable English text↔geometry supervision.
        if _is_non_english_label(label):
            continue

        if not is_single_object:
            scene_mesh_parts.append((verts, faces))
        
        if over_budget and original_faces > 2000:
            detail = "high-detail" if original_faces > 10000 else "detailed"
            label = f"{label} ({detail})"
        
        # Encode text
        text_ids, text_mask = text_tokenizer.encode_padded(label, max_text_length)
        
        # Quality weight based on mesh properties
        quality = 0.5
        if len(faces) > 100:
            quality += 0.2
        if len(faces) > 500:
            quality += 0.1
        if mat_names and any(isinstance(m, str) and m.lower() not in ('material', '') for m in mat_names):
            quality += 0.15
        if over_budget:
            quality += 0.1  # Was originally high-detail
        sample_weight = max(0.3, 0.3 + quality * 1.2)
        
        item = {
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.float),
            "mesh_tokens": torch.tensor(tokens, dtype=torch.long),
            "quality_weight": torch.tensor(sample_weight, dtype=torch.float),
            "label": label,          # Store for debugging / audit
            "data_source": data_source,  # Which repo this came from
            "geometry_space": ("RAW_LOCAL" if has_raw_geometry else "EVALUATED_WORLD"),
            "geometry_is_baked": bool((not has_raw_geometry) and has_nodes_modifier),
            "has_nodes_modifier": bool(has_nodes_modifier),
        }
        
        # Over-budget meshes: store raw geometry for future retokenization
        # at higher max_faces (e.g. on H100). The mesh_tokens contain a
        # z-order-truncated version that IS trainable now on M3.
        if over_budget:
            item["over_budget"] = True
            item["original_face_count"] = original_faces
            # Store full cleaned mesh (all faces, not truncated) for future
            # retokenization at higher max_faces on better hardware.
            item["raw_vertices"] = torch.tensor(
                np.array(verts, dtype=np.float32))
            item["raw_faces"] = torch.tensor(
                np.array(faces, dtype=np.int32))
        
        # ── Preserve rich scene context for LLM training data gen ──
        scene_context = {}
        
        # Materials (full node trees with shader data — needed by renderer)
        if obj.get('materials'):
            scene_context['materials'] = obj['materials']
        
        # Per-face material slot assignments — reordered to match z-order
        # so the validator can align them with the decoded mesh.
        face_mat_idx = mesh.get('face_material_indices', [])
        if face_mat_idx:
            # Compute z-order (same as tokenizer) to reorder face_mat_idx
            try:
                v_arr = np.array(verts, dtype=np.float64)
                centers = []
                for face in faces:
                    valid = [fi for fi in face if fi < len(v_arr)]
                    center = v_arr[valid].mean(axis=0) if valid else np.zeros(3)
                    centers.append(center)
                centers = np.array(centers)
                c_min, c_max = centers.min(axis=0), centers.max(axis=0)
                c_range = c_max - c_min
                c_range[c_range < 1e-6] = 1.0
                norm_c = ((centers - c_min) / c_range * 1023).astype(int)
                norm_c = np.clip(norm_c, 0, 1023)
                morton_codes = [MeshTokenizer._morton_encode_3d(c[0], c[1], c[2])
                               for c in norm_c]
                z_order = np.argsort(morton_codes)
                # Reorder face_mat_idx to match z-order (capped to face count)
                reordered = [face_mat_idx[i] if i < len(face_mat_idx) else 0
                            for i in z_order[:len(faces)]]
                scene_context['face_material_indices'] = reordered
            except Exception:
                scene_context['face_material_indices'] = face_mat_idx
        
        # Per-loop UV coordinates (keyed by UV layer name)
        uv_layers = mesh.get('uv_layers', {})
        if uv_layers:
            scene_context['uv_layers'] = uv_layers
        
        # Per-loop vertex colors (keyed by layer name)
        vcol_layers = mesh.get('vertex_color_layers', {})
        if vcol_layers:
            scene_context['vertex_color_layers'] = vcol_layers
        
        # Per-face smooth shading flags
        face_smooth = mesh.get('face_smooth', [])
        if face_smooth:
            scene_context['face_smooth'] = face_smooth
        
        # Scene-level image thumbnails (base64 JPEG, referenced by material nodes by name)
        images = data_ref.get('images', {}) if isinstance(data_ref, dict) else {}
        if images:
            scene_context['images'] = images
        
        # Modifiers
        if obj.get('modifiers'):
            scene_context['modifiers'] = obj['modifiers']
        
        # Shape keys
        if obj.get('shape_keys'):
            scene_context['shape_keys'] = obj['shape_keys']
        
        # UV maps
        if obj.get('uv_maps'):
            scene_context['uv_maps'] = obj['uv_maps']
        
        # Vertex groups
        if obj.get('vertex_groups'):
            scene_context['vertex_groups'] = obj['vertex_groups']
        
        # Vertex colors
        if obj.get('vertex_color_layers'):
            scene_context['vertex_color_layers'] = obj['vertex_color_layers']
        
        # Object transforms
        if obj.get('transforms'):
            scene_context['transforms'] = obj['transforms']
        
        # Parent info
        if obj.get('parent'):
            scene_context['parent'] = obj['parent']
            if obj.get('parent_type'):
                scene_context['parent_type'] = obj['parent_type']
            if obj.get('parent_bone'):
                scene_context['parent_bone'] = obj['parent_bone']
        
        # File-level context (stored once per file, shared by all objects)
        file_context = {}
        if isinstance(data_ref, dict):
            # Armatures in the same file
            armatures = [o for o in data_ref.get('objects', [])
                        if o.get('type') == 'ARMATURE']
            if armatures:
                arm_summaries = []
                for arm_obj in armatures:
                    arm_data = arm_obj.get('armature', {})
                    arm_summaries.append({
                        'name': arm_obj.get('name', ''),
                        'bone_count': arm_data.get('bone_count', 0),
                        'bones': [b.get('name', '') for b in arm_data.get('bones', [])[:50]],
                    })
                file_context['armatures'] = arm_summaries
            
            # Orphan materials
            if data_ref.get('orphan_materials'):
                file_context['orphan_materials'] = data_ref['orphan_materials']
            
            # Animation actions
            if data_ref.get('orphan_actions'):
                act_summaries = []
                for act in data_ref['orphan_actions']:
                    act_summaries.append({
                        'name': act.get('name', ''),
                        'frame_range': act.get('frame_range', []),
                        'fcurve_count': act.get('fcurve_count', 0),
                    })
                file_context['actions'] = act_summaries
            
            # Scene timing
            if data_ref.get('fps'):
                file_context['fps'] = data_ref['fps']
            if data_ref.get('frame_start') is not None:
                file_context['frame_range'] = [
                    data_ref.get('frame_start', 0),
                    data_ref.get('frame_end', 0)]
            
            # Render settings
            if data_ref.get('render'):
                file_context['render'] = data_ref['render']
            
            # World/environment
            if data_ref.get('world'):
                file_context['world'] = data_ref['world']
        
        if file_context:
            scene_context['file_context'] = file_context
        
        if scene_context:
            item['scene_context'] = scene_context
        
        items.append(item)
        
        if dry_run:
            faces_str = f"{original_faces}->{len(tokens)//9}" if over_budget else str(len(faces))
            logger.info(f"  [{faces_str} faces] {label}")
    
    # Add one composition sample for multi-object scenes so the model learns
    # whole-assembly relationships in addition to isolated parts.
    if not is_single_object and len(scene_mesh_parts) >= 2:
        # Keep top objects by face count to stay in budget.
        scene_mesh_parts = sorted(scene_mesh_parts, key=lambda x: len(x[1]), reverse=True)[:8]
        sc_verts, sc_faces = _merge_meshes(scene_mesh_parts)
        if sc_verts and sc_faces and len(sc_faces) >= 4:
            sc_original_faces = len(sc_faces)
            sc_over_budget = False
            if len(sc_faces) > max_f:
                # Try lossless cleanup first, then let tokenizer truncate
                try:
                    import trimesh
                    sc_mesh = trimesh.Trimesh(
                        vertices=np.array(sc_verts, dtype=np.float64),
                        faces=np.array(sc_faces, dtype=np.int64),
                        process=True)
                    sc_mesh.merge_vertices()
                    sc_mesh.update_faces(sc_mesh.nondegenerate_faces())
                    sc_mesh.update_faces(sc_mesh.unique_faces())
                    sc_verts = sc_mesh.vertices.tolist()
                    sc_faces = sc_mesh.faces.tolist()
                    if len(sc_faces) > max_f:
                        sc_over_budget = True
                        logger.debug(f"  Scene composition over budget: {len(sc_faces)} faces. "
                                     f"Tokenizer will truncate to {max_f} in z-order.")
                except Exception:
                    sc_over_budget = len(sc_faces) > max_f
            if sc_verts and sc_faces and len(sc_faces) >= 4:
                try:
                    sc_bbox = compute_bbox_aspect(sc_verts)
                    sc_verts_n = normalize_mesh(sc_verts, target_range=(-1.0, 1.0))
                    sc_tokens = mesh_tokenizer.encode_mesh(sc_verts_n, sc_faces)
                    max_tokens = max_f * 9 + 2
                    # Tokenizer truncates to max_faces in z-order — tokens are always valid
                    if sc_tokens and sc_tokens[0] == mesh_tokenizer.BOS and sc_tokens[-1] == mesh_tokenizer.EOS:
                        scene_label = _build_scene_label(file_label, meta_name, meta_cats)
                        if scene_label:
                            text_ids, text_mask = text_tokenizer.encode_padded(scene_label, max_text_length)
                            sc_quality = 0.7
                            if len(sc_faces) > 500:
                                sc_quality += 0.2
                            if sc_over_budget:
                                sc_quality += 0.1  # High-detail scene
                            scene_item = {
                                "text_ids": torch.tensor(text_ids, dtype=torch.long),
                                "text_mask": torch.tensor(text_mask, dtype=torch.float),
                                "mesh_tokens": torch.tensor(sc_tokens, dtype=torch.long),
                                "quality_weight": torch.tensor(max(0.4, min(2.0, sc_quality)), dtype=torch.float),
                                "label": scene_label,
                                "data_source": data_source,
                                "sample_type": "scene_composition",
                                "composition_object_count": len(scene_mesh_parts),
                            }
                            if sc_over_budget:
                                scene_item["over_budget"] = True
                                scene_item["original_face_count"] = sc_original_faces
                            if isinstance(data_ref, dict):
                                scene_item["scene_context"] = {
                                    "file_metadata": data_ref.get("metadata", {}),
                                    "object_count": len(objects),
                                }
                            items.append(scene_item)
                except Exception:
                    pass

    del data_ref, objects
    gc.collect()
    
    # ── Per-file cap: keep at most MAX_ITEMS_PER_FILE items, sorted by
    # face count (higher = more detailed = better training signal).
    if len(items) > MAX_ITEMS_PER_FILE:
        # Score items: prefer high face count, real materials, non-generic labels
        def _item_sort_key(it):
            mt = it.get("mesh_tokens")
            n_faces = (len(mt) - 2) // 9 if mt is not None and len(mt) > 2 else 0
            has_mats = 1 if it.get("scene_context", {}).get("materials") else 0
            is_comp = 1 if it.get("sample_type") == "scene_composition" else 0
            return (is_comp, has_mats, n_faces)  # Compositions first, then by quality
        items.sort(key=_item_sort_key, reverse=True)
        logger.debug(f"  Per-file cap: {len(items)} -> {MAX_ITEMS_PER_FILE} items")
        items = items[:MAX_ITEMS_PER_FILE]
    
    return items


def relabel_cached_item(item, text_tokenizer, max_text_length=256):
    """Fix the label of an existing cached item using mesh analysis."""
    tokens = item["mesh_tokens"]
    mesh_len = len(tokens)
    num_faces = (mesh_len - 2) // 9 if mesh_len > 2 else 0
    
    # Decode current label
    mask_len = int(item["text_mask"].sum().item())
    current_ids = item["text_ids"][:mask_len].tolist()
    current_label = text_tokenizer.decode(current_ids)
    
    # Check if label is vague
    vague_patterns = [
        'small 3d mesh', 'medium 3d mesh', 'high-poly small',
        'low-poly small', 'scene root', 'material 3d mesh',
        'small material', 'medium-poly small',
    ]
    is_vague = any(p in current_label.lower() for p in vague_patterns)
    
    # Also check for pure numeric/hex labels
    clean = re.sub(r'[^a-z0-9 ]', '', current_label.lower()).strip()
    if re.match(r'^[\d\s]+$', clean) or re.match(r'^[0-9a-f]{8,}$', clean):
        is_vague = True
    
    if not is_vague:
        return None  # Label is fine, skip
    
    # Generate geometry-based label
    geo_desc = []
    if num_faces < 50:
        geo_desc.append('low-poly')
    elif num_faces < 200:
        geo_desc.append('simple')
    elif num_faces > 1500:
        geo_desc.append('detailed')
    
    # Analyze the mesh tokens for shape info
    coord_tokens = [t for t in tokens.tolist() if t >= 4]
    if coord_tokens:
        from processing.mesh_tokenizer import MeshTokenizer
        mt = MeshTokenizer(vocab_size=8192)
        coords = [mt.dequantize_token(t) for t in coord_tokens]
        xs = coords[0::3]
        ys = coords[1::3]
        zs = coords[2::3]
        
        x_range = max(xs) - min(xs) if xs else 0
        y_range = max(ys) - min(ys) if ys else 0
        z_range = max(zs) - min(zs) if zs else 0
        
        max_r = max(x_range, y_range, z_range)
        min_r = min(x_range, y_range, z_range) + 0.001
        
        if max_r / min_r > 3:
            if z_range == max_r:
                geo_desc.append('tall')
            elif z_range == min_r:
                geo_desc.append('flat')
            else:
                geo_desc.append('elongated')
        
        # Check symmetry (X or Y axis)
        if xs:
            x_arr = np.array(xs)
            sym_x = abs(x_arr.mean()) < 0.05  # Centered on X
            if sym_x:
                geo_desc.append('symmetric')
    
    if not geo_desc:
        geo_desc.append('3d')
    geo_desc.append('object')
    
    new_label = ' '.join(geo_desc)
    new_ids, new_mask = text_tokenizer.encode_padded(new_label, max_text_length)
    
    return {
        "text_ids": torch.tensor(new_ids, dtype=torch.long),
        "text_mask": torch.tensor(new_mask, dtype=torch.float),
        "old_label": current_label,
        "new_label": new_label,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rebuild mesh training cache")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--source-only", action="store_true", help="Only process source JSONs")
    parser.add_argument("--relabel-only", action="store_true", help="Only fix labels in cache")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Reprocess all source files even if already cached (default: skip cached)")
    parser.add_argument("--max-files", type=int, default=0, help="Limit source files to process")
    parser.add_argument("--max-per-label", type=int, default=100,
                        help="Cap number of samples per unique label (default: 100)")
    parser.add_argument("--cap-existing", action="store_true",
                        help="Apply --max-per-label cap to already-built cache (no source rebuild)")
    parser.add_argument("--fix-attribution", action="store_true",
                        help="Back-fill data_source field on old cache entries that lack it")
    args = parser.parse_args()
    
    # Load tokenizers
    bpe_path = BASE / "data" / "datasets" / "geometry" / "bpe_tokenizer"
    if not bpe_path.exists():
        logger.error(f"BPE tokenizer not found at {bpe_path}")
        sys.exit(1)
    
    text_tokenizer = BPETokenizer.load(bpe_path)
    mesh_tokenizer = MeshTokenizer(vocab_size=8192, coord_range=(-1.0, 1.0), max_faces=MAX_FACES)
    
    logger.info(f"Mesh tokenizer: {mesh_tokenizer.vocab_size} vocab, max {MAX_FACES} faces")
    logger.info(f"Text tokenizer: {text_tokenizer.vocab_size} vocab")

    # Warm the Qwen model once so all subsequent _qwen_label calls are fast
    try:
        warm_model()
        logger.info("Qwen model warmed up for labeling")
    except Exception as e:
        logger.warning(f"Could not warm Qwen model (will use cold calls): {e}")
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "source_files": 0,
        "source_objects": 0,
        "decimated": 0,
        "relabeled": 0,
        "skipped": 0,
        "total_items": 0,
        "vague_fixed": 0,
        "label_cap_removed": 0,
        "attribution_fixed": 0,
    }
    
    # ── Phase 1: Rebuild from source JSONs ──
    if not args.relabel_only:
        logger.info("\n=== Phase 1: Processing source JSON files ===")
        
        # Build source-tagged file list: (filepath, source_name)
        all_source_files = []  # list of (Path, str)
        for src_dir in SOURCE_DIRS:
            if src_dir.exists():
                jsons = sorted(src_dir.glob("*.json"))
                source_name = src_dir.name  # e.g. 'objaverse', 'blendswap'
                all_source_files.extend((p, source_name) for p in jsons)
                logger.info(f"  {src_dir.name}: {len(jsons)} files")
        
        logger.info(f"Total source files: {len(all_source_files)}")
        
        if args.max_files:
            all_source_files = all_source_files[:args.max_files]
        
        skipped_cached = 0
        for i, (filepath, source_name) in enumerate(all_source_files):
            cache_key = hashlib.md5(str(filepath).encode()).hexdigest()[:16]
            cache_path = CACHE_DIR / f"{cache_key}.pt"

            # Skip already-cached files unless --force-rebuild is set.
            # This makes incremental updates fast: only new source files
            # are processed, existing cache entries are left untouched.
            if not args.force_rebuild and cache_path.exists() and cache_path.stat().st_size > 200:
                skipped_cached += 1
                continue

            size_mb = filepath.stat().st_size / 1048576
            if size_mb > 2000:
                logger.info(f"  Skipping {filepath.name} ({size_mb:.0f}MB — over 2GB JSON limit)")
                stats["skipped"] += 1
                continue
            
            items = process_source_file(
                str(filepath), mesh_tokenizer, text_tokenizer,
                dry_run=args.dry_run,
                data_source=source_name)
            
            if items:
                stats["source_objects"] += len(items)
                # Keep label and scene_context in .pt for training data gen
                
                if not args.dry_run:
                    torch.save(items, cache_path)
                
                # Count decimated
                for item in items:
                    if len(item["mesh_tokens"]) > TARGET_DECIMATE * 9:
                        stats["decimated"] += 1
            
            stats["source_files"] += 1
            
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1}/{len(all_source_files)} files, "
                           f"{stats['source_objects']} objects so far")
            
            gc.collect()

        logger.info(f"  Skipped (already cached): {skipped_cached} files")
        stats["skipped_cached"] = skipped_cached

    # ── Phase 1.4: Back-fill data_source on existing cache entries ──
    if args.fix_attribution or (not args.relabel_only and not args.cap_existing):
        # Build a mapping of cache_key -> source_name from SOURCE_DIRS
        logger.info("\n=== Phase 1.4: Back-filling data_source attribution ===")
        key_to_source = {}
        for src_dir in SOURCE_DIRS:
            if src_dir.exists():
                for p in src_dir.glob("*.json"):
                    k = hashlib.md5(str(p).encode()).hexdigest()[:16]
                    key_to_source[k] = src_dir.name

        attribution_fixed = 0
        for cache_path in sorted(CACHE_DIR.glob("*.pt")):
            cache_key = cache_path.stem
            source_name = key_to_source.get(cache_key, "unknown")
            if source_name == "unknown":
                continue  # Nothing to back-fill
            try:
                cached = torch.load(cache_path, weights_only=False)
                if not cached:
                    continue
                if isinstance(cached, dict):
                    cached_items = [cached]
                elif isinstance(cached, list):
                    cached_items = cached
                else:
                    continue
                needs_save = False
                for item in cached_items:
                    if item.get("data_source", "unknown") == "unknown":
                        item["data_source"] = source_name
                        attribution_fixed += 1
                        needs_save = True
                if needs_save and not args.dry_run:
                    torch.save(cached, cache_path)
            except Exception as e:
                logger.debug(f"  attribution back-fill error {cache_path.name}: {e}")
                continue

        logger.info(f"  Items attribution fixed: {attribution_fixed}")
        stats["attribution_fixed"] = attribution_fixed

    # ── Phase 1.5: Cross-file deduplication + empty label removal ──
    if not args.relabel_only:
        logger.info("\n=== Phase 1.5: Cross-file dedup + cleanup ===")
        cache_files_dedup = sorted(CACHE_DIR.glob("*.pt"))
        
        global_hashes = {}  # hash -> (file_path, index, token_len)
        dupes_removed = 0
        empties_removed = 0
        files_modified = 0
        
        for cache_path in cache_files_dedup:
            try:
                cached = torch.load(cache_path, weights_only=False)
                if not cached:
                    continue
                if isinstance(cached, dict):
                    cached_items = [cached]
                elif isinstance(cached, list):
                    cached_items = cached
                else:
                    continue
            except Exception:
                continue
            
            original_len = len(cached_items)
            cleaned = []
            for item in cached_items:
                mt = item.get("mesh_tokens")
                label = item.get("label", "").strip()
                
                # Remove empty labels
                if not label or len(label) < 2:
                    empties_removed += 1
                    continue
                
                # Cross-file dedup: hash the full mesh token sequence
                if mt is not None and isinstance(mt, torch.Tensor) and len(mt) > 0:
                    # Use a robust hash of the mesh tokens
                    token_tuple = tuple(mt.tolist())
                    token_hash = hash(token_tuple)
                    
                    if token_hash in global_hashes:
                        # Keep the one with the longer/better label
                        existing_label = global_hashes[token_hash][2]
                        if len(label) > len(existing_label):
                            # This one has a better label — keep it, mark old for removal
                            global_hashes[token_hash] = (str(cache_path), len(cleaned), label)
                            cleaned.append(item)
                        else:
                            dupes_removed += 1
                            continue
                    else:
                        global_hashes[token_hash] = (str(cache_path), len(cleaned), label)
                        cleaned.append(item)
                else:
                    empties_removed += 1
                    continue
            
            if len(cleaned) != original_len:
                files_modified += 1
                if cleaned and not args.dry_run:
                    if isinstance(cached, dict):
                        torch.save(cleaned[0], cache_path)
                    else:
                        torch.save(cleaned, cache_path)
                elif not cleaned and not args.dry_run:
                    cache_path.unlink(missing_ok=True)
        
        logger.info(f"  Cross-file duplicates removed: {dupes_removed}")
        logger.info(f"  Empty/invalid samples removed: {empties_removed}")
        logger.info(f"  Cache files modified: {files_modified}")
        stats["dupes_removed"] = dupes_removed
        stats["empties_removed"] = empties_removed

    # ── Phase 1.6: Label frequency capping ──
    should_cap = (args.max_per_label > 0) or args.cap_existing
    effective_cap = args.max_per_label if args.max_per_label > 0 else 100
    if should_cap and not args.source_only:
        logger.info(f"\n=== Phase 1.6: Label frequency cap (max {effective_cap} per label) ===")
        # First pass: count label occurrences across all cache files
        from collections import Counter
        label_counts: Counter = Counter()
        label_cap_removed = 0

        all_cache = sorted(CACHE_DIR.glob("*.pt"))
        # Collect all labels first
        for cp in all_cache:
            try:
                cached = torch.load(cp, weights_only=False)
                if isinstance(cached, dict):
                    cached_items = [cached]
                elif isinstance(cached, list):
                    cached_items = cached
                else:
                    continue
                for item in cached_items:
                    lbl = item.get("label", "").strip().lower()
                    if lbl:
                        label_counts[lbl] += 1
            except Exception:
                continue

        over_cap = {lbl: cnt for lbl, cnt in label_counts.items() if cnt > effective_cap}
        if over_cap:
            logger.info(f"  Labels over cap ({effective_cap}): {len(over_cap)}")
            for lbl, cnt in sorted(over_cap.items(), key=lambda x: -x[1])[:10]:
                logger.info(f"    [{cnt}x] \"{lbl[:70]}\"")

            # Second pass: prune excess samples
            seen_label_counts: Counter = Counter()
            for cp in all_cache:
                try:
                    cached = torch.load(cp, weights_only=False)
                    if isinstance(cached, dict):
                        cached_items = [cached]
                    elif isinstance(cached, list):
                        cached_items = cached
                    else:
                        continue
                except Exception:
                    continue
                original_len = len(cached_items)
                kept = []
                for item in cached_items:
                    lbl = item.get("label", "").strip().lower()
                    if lbl in over_cap:
                        if seen_label_counts[lbl] < effective_cap:
                            kept.append(item)
                            seen_label_counts[lbl] += 1
                        else:
                            label_cap_removed += 1
                    else:
                        kept.append(item)
                if len(kept) != original_len and not args.dry_run:
                    if kept:
                        if isinstance(cached, dict):
                            torch.save(kept[0], cp)
                        else:
                            torch.save(kept, cp)
                    else:
                        cp.unlink()
            logger.info(f"  Samples removed by label cap: {label_cap_removed}")
        else:
            logger.info(f"  No labels exceed cap of {effective_cap} — nothing to do")
        stats["label_cap_removed"] = label_cap_removed

    # ── Phase 2: Fix vague labels in existing cache ──
    if not args.source_only:
        logger.info("\n=== Phase 2: Fixing vague labels in cache ===")
        
        cache_files = sorted(CACHE_DIR.glob("*.pt"))
        logger.info(f"Cache files to check: {len(cache_files)}")
        
        for i, cache_path in enumerate(cache_files):
            try:
                cached = torch.load(cache_path, weights_only=False)
                if not cached:
                    continue
                
                modified = False
                for item in cached:
                    fix = relabel_cached_item(item, text_tokenizer)
                    if fix:
                        item["text_ids"] = fix["text_ids"]
                        item["text_mask"] = fix["text_mask"]
                        modified = True
                        stats["vague_fixed"] += 1
                        if args.dry_run:
                            logger.info(f"  RELABEL: {fix['old_label']!r} -> {fix['new_label']!r}")
                
                if modified and not args.dry_run:
                    torch.save(cached, cache_path)
                
                stats["total_items"] += len(cached)
                
            except Exception as e:
                logger.debug(f"Error processing {cache_path.name}: {e}")
                continue
            
            if (i + 1) % 200 == 0:
                logger.info(f"  Checked {i+1}/{len(cache_files)} cache files, "
                           f"{stats['vague_fixed']} labels fixed")
    
    # ── Summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"REBUILD COMPLETE {'(DRY RUN)' if args.dry_run else ''}")
    logger.info(f"{'='*60}")
    logger.info(f"Source files processed: {stats['source_files']}")
    logger.info(f"Source files skipped (already cached): {stats.get('skipped_cached', 0)}")
    logger.info(f"Objects extracted: {stats['source_objects']}")
    logger.info(f"Meshes decimated: {stats['decimated']}")
    logger.info(f"Files skipped (too large): {stats['skipped']}")
    logger.info(f"Cross-file duplicates removed: {stats.get('dupes_removed', 0)}")
    logger.info(f"Empty/invalid removed: {stats.get('empties_removed', 0)}")
    logger.info(f"Attribution back-filled: {stats.get('attribution_fixed', 0)}")
    logger.info(f"Label cap removed: {stats.get('label_cap_removed', 0)}")
    logger.info(f"Vague labels fixed: {stats['vague_fixed']}")
    logger.info(f"Total cache items: {stats['total_items']}")
    
    # Count final stats
    if not args.dry_run:
        final_count = len(list(CACHE_DIR.glob("*.pt")))
        trainable = 0
        for p in CACHE_DIR.glob("*.pt"):
            try:
                items = torch.load(p, weights_only=False)
                trainable += sum(1 for it in items if len(it["mesh_tokens"]) <= MAX_FACES * 9 + 2)
            except Exception:
                continue
        logger.info(f"Final cache files: {final_count}")
        logger.info(f"Total trainable samples: {trainable}")
        logger.info(f"All under {MAX_FACES} faces ({MAX_FACES * 9 + 2} tokens)")


if __name__ == "__main__":
    main()
