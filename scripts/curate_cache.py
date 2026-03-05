#!/usr/bin/env python3
"""Curate the mesh cache: strict quality filtering + material integration.

This script post-processes the existing .mesh_cache to produce a curated
dataset that contains only HIGH-QUALITY training items. It follows the
philosophy: "100 PERFECT items > 1000 moderately good items."

Curation rules (in order):
1. HARD REJECT: <50 faces (useless fragments, road segments, bolts)
2. HARD REJECT: Non-English labels (Indonesian, Swedish, French, etc.)
3. HARD REJECT: Weak/generic labels ("object", "mesh", "simple shape")
4. HARD REJECT: Shattered meshes (high disconnected component ratio)
5. PER-FILE CAP: Max 10 items per source .pt file (keep by quality score)
6. PER-LABEL CAP: Max 5 items globally sharing the same label
7. MATERIAL ENRICHMENT: Integrate material names into labels

Does NOT delete data — works on a copy. Original cache remains in
.mesh_cache_backup (created automatically).

Usage:
    python scripts/curate_cache.py --dry-run          # Preview only
    python scripts/curate_cache.py --apply             # Curate in-place
    python scripts/curate_cache.py --apply --min-faces 100  # Stricter threshold
"""

import argparse
import collections
import gc
import hashlib
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Force unbuffered logging when output is redirected
for handler in logging.root.handlers:
    if hasattr(handler, 'stream'):
        handler.stream = sys.stderr
sys.stderr = open(sys.stderr.fileno(), 'w', buffering=1)  # line-buffered

BASE = Path(__file__).parent.parent
CACHE_DIR = BASE / "data" / "processed" / ".mesh_cache"
BACKUP_DIR = BASE / "data" / "processed" / ".mesh_cache_backup"

# ── Non-English word detection ───────────────────────────────────────
# Common non-English words found in our training data (primarily Indonesian
# Blender assets, plus some French, Swedish, Spanish, German).
# A label is flagged as non-English if >50% of its substantive words
# appear in this list.

_NON_ENGLISH_WORDS = {
    # Indonesian (largest source of non-English labels)
    "kereta", "gerbong", "warna", "gradasi", "kaca", "bening", "kuning",
    "ban", "trotoar", "hijau", "biru", "ungu", "oranye", "merah", "pink",
    "putih", "hitam", "terowongan", "lokomotif", "bolong", "rel", "cabang",
    "lampu", "stasiun", "jalan", "tanjakan", "tanah", "kolam", "air",
    "awan", "gunung", "gedung", "alas", "masjid", "pohon", "masinis",
    "telur", "belang", "polkadot", "daun", "aspal", "nyala", "rumah",
    "pintu", "atap", "dinding", "lantai", "meja", "kursi", "roda",
    "sayap", "kuda", "ikan", "burung", "bunga", "batu", "pasir",
    "sungai", "laut", "hutan", "kebun", "sawah", "jembatan", "gerbang",
    "menara", "pagar", "tiang", "papan", "kotak", "botol", "ember",
    "cangkir", "piring", "sendok", "garpu", "pisau", "gunting",
    "tongkat", "tali", "rantai", "bola", "bendera", "tenda", "payung",
    "sepeda", "mobil", "bis", "kapal", "pesawat", "helikopter",
    "coklat", "abu", "emas", "perak", "polka", "dot",
    "lowpoly", "highpoly",  # Not non-English per se but common in Indonesian context
    # Indonesian/Malay compound words
    "carbodymat",
    
    # French
    "groupe", "noeud", "curva", "curvas", "beton", "lumiere", "fenetre",
    "porte", "mur", "toit", "sol", "plafond", "escalier", "chaise",
    "maison", "arbre", "voiture", "bateau",
    
    # Swedish
    "metallkedja", "hus", "bil", "stol", "bord", "dörr", "fönster",
    
    # Spanish
    "blanco", "brillante", "rojo", "azul", "verde", "amarillo", "negro",
    "casa", "coche", "mesa", "silla", "puerta", "ventana",
    
    # German
    "straße", "haus", "tisch", "stuhl", "fenster", "wand",
    
    # Italian
    "normale", "luce", "ral",
    
    # Vietnamese (Latin script portions)
    "hình", "trụ", "vòng", "tròn", "thép", "thạch",
    
    # Swedish (additional)
    "stearinljus", "ljuslåga", "trä", "sten", "glas", "ljus",
    "vägg", "golv", "tak", "möbel", "lampa",
    
    # French (additional)
    "toiture", "tuile", "ardoise", "charpente", "gouttière",
    
    # Dutch
    "vetrook", "steen", "hout", "goud", "zilver", "groen",
    
    # Russian transliterated
    "dom", "stena", "okno",
}

# Words that should NOT count as English content in a label
_NON_CONTENT_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "it", "by", "with", "from", "as", "but", "not", "no", "so",
    "up", "out", "if", "my", "his", "her", "its", "our", "new", "old",
    "high", "low", "big", "small", "very", "more", "most", "just", "all",
    "detail", "detailed", "simple", "basic", "complex",
}

# Minimal set of COMMON English 3D modeling words — label must contain
# at least ONE of these to be considered useful English supervision.
_ENGLISH_3D_WORDS = {
    # Objects/furniture
    "chair", "table", "desk", "bench", "sofa", "couch", "bed", "wardrobe",
    "cabinet", "shelf", "drawer", "closet", "bookshelf", "stool", "ottoman",
    # Architecture
    "house", "building", "wall", "floor", "roof", "door", "window", "stairs",
    "staircase", "column", "pillar", "arch", "bridge", "tower", "fence",
    "gate", "chimney", "balcony", "porch", "garage", "barn", "shed",
    "church", "castle", "temple", "mosque", "cathedral", "palace",
    # Vehicles
    "car", "truck", "bus", "van", "taxi", "ambulance", "train", "locomotive",
    "wagon", "boat", "ship", "submarine", "plane", "aircraft", "helicopter",
    "motorcycle", "bicycle", "scooter", "skateboard",
    # Nature
    "tree", "bush", "flower", "grass", "leaf", "leaves", "rock", "stone",
    "mountain", "hill", "cliff", "cave", "island", "river", "lake",
    "ocean", "beach", "sand", "cloud", "sky", "ground", "terrain",
    # Characters
    "head", "face", "body", "arm", "hand", "leg", "foot", "skull",
    "bone", "skeleton", "robot", "creature", "monster", "dragon",
    "human", "man", "woman", "child", "soldier", "knight", "warrior",
    # Animals
    "horse", "dog", "cat", "bird", "fish", "snake", "bear", "wolf",
    "deer", "rabbit", "mouse", "elephant", "lion", "tiger",
    # Weapons/tools
    "sword", "shield", "weapon", "gun", "pistol", "rifle", "knife",
    "axe", "hammer", "wrench", "screwdriver", "saw",
    # Electronics
    "phone", "computer", "screen", "monitor", "keyboard", "speaker",
    "camera", "lamp", "light", "lantern", "chandelier",
    # Food/kitchen
    "cup", "mug", "bowl", "plate", "bottle", "glass", "jar", "pot", "pan",
    "fork", "spoon", "kettle", "teapot", "vase",
    # Containers
    "box", "crate", "barrel", "chest", "bag", "basket", "bucket",
    # Mechanical
    "gear", "wheel", "engine", "motor", "pipe", "tube", "valve", "pump",
    "machine", "crane", "conveyor", "piston", "spring", "bolt", "screw",
    # Everyday
    "book", "pen", "pencil", "paper", "clock", "watch", "mirror",
    "picture", "frame", "candle", "key", "lock", "chain", "rope",
    "sign", "flag", "banner", "pole", "post", "rail", "track",
    # Shapes (useful for geometry)
    "cube", "sphere", "cylinder", "cone", "pyramid", "torus", "ring",
    "disk", "panel", "slab", "block", "beam", "rod", "bar",
    # Materials (as object descriptors)
    "wooden", "metal", "metallic", "glass", "stone", "concrete", "brick",
    "plastic", "ceramic", "leather", "fabric",
    # Actions/descriptors that indicate English
    "low", "poly", "stylized", "realistic", "ornate", "decorative",
    "modern", "ancient", "medieval", "futuristic", "industrial",
    "round", "square", "flat", "tall", "long", "wide", "thin",
    "open", "closed", "broken", "damaged", "rusty",
    # Misc
    "corridor", "tunnel", "road", "street", "path", "sidewalk", "parking",
    "city", "village", "park", "garden", "playground", "stadium",
    "forest", "jungle", "desert", "swamp", "farm", "field",
    "store", "shop", "market", "restaurant", "cafe", "bar", "hotel",
    "hospital", "school", "office", "factory", "warehouse",
    "antenna", "satellite", "radar", "turbine", "windmill",
    "fountain", "statue", "monument", "obelisk", "tombstone",
    "tire", "bumper", "hood", "trunk", "fender", "spoiler",
    "helmet", "armor", "crown", "ring", "necklace", "bracelet",
    "carpet", "rug", "curtain", "pillow", "blanket", "towel",
    "traffic", "signal", "hydrant", "mailbox", "bench",
    "trash", "bin", "dumpster", "container",
}

# Regex for non-Latin characters (Cyrillic, CJK, Arabic, Korean, Devanagari, Thai)
# and Vietnamese diacritical marks (ạ, ả, ã, ắ, ằ, ặ, etc.)
_NON_LATIN_RE = re.compile(
    r'[\u0400-\u04ff\u0600-\u06ff\u0900-\u097f\u0e00-\u0e7f'
    r'\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff'
    r'\u1ea0-\u1ef9]'  # Vietnamese extended Latin
)


def is_non_english_label(label: str) -> bool:
    """Detect non-English labels using word frequency analysis."""
    if not label:
        return True
    
    # Non-Latin scripts are definitely non-English
    if _NON_LATIN_RE.search(label):
        return True
    
    # Clean and tokenize
    clean = re.sub(r'[^a-z0-9\s]', ' ', label.lower())
    words = [w for w in clean.split() if w and len(w) > 1 and w not in _NON_CONTENT_WORDS]
    
    if not words:
        return True  # No substantive words
    
    # Check non-English word ratio
    non_english_count = sum(1 for w in words if w in _NON_ENGLISH_WORDS)
    non_english_ratio = non_english_count / len(words)
    
    if non_english_ratio > 0.4:
        return True
    
    # Check if at least one word is a recognizable English 3D term
    has_english_3d_word = any(w in _ENGLISH_3D_WORDS for w in words)
    
    # If no English 3D words AND some non-English words, likely non-English
    if not has_english_3d_word and non_english_count > 0:
        return True
    
    return False


# ── Weak label detection (same logic as rebuild_cache.py) ────────────

_GENERIC_LABELS = {
    "object", "mesh", "3d object", "3d mesh", "shape", "3d shape", "part",
    "piece", "simple object", "basic object", "simple shape",
    "detailed object", "multi object scene composition",
    "dummy", "base", "lattice",
}

_MATERIAL_ONLY_WORDS = {
    "metal", "wood", "stone", "glass", "plastic", "rubber", "fabric",
    "concrete", "brick", "ceramic", "leather", "steel", "iron",
    "copper", "brass", "chrome", "aluminum", "aluminium",
    "gold", "silver", "bronze",
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "grey", "gray", "brown", "cyan", "magenta",
    "dark", "light", "bright", "shiny", "matte", "glossy",
    "dry", "wet", "rough", "smooth",
}


def is_weak_label(label: str) -> bool:
    """Check if label is too generic/weak for meaningful supervision."""
    if not label:
        return True
    clean = re.sub(r'[^a-z0-9\s,]', ' ', label.lower())
    clean = re.sub(r'\s+', ' ', clean).strip(' ,')
    if not clean or len(clean) < 3:
        return True
    if clean in _GENERIC_LABELS:
        return True
    
    tokens = [t for t in re.split(r'[\s,]+', clean) if t]
    if not tokens:
        return True
    
    # Labels consisting entirely of material/color words
    material_tokens = [t for t in tokens if t in _MATERIAL_ONLY_WORDS]
    if len(material_tokens) == len(tokens):
        return True
    
    # Numeric-heavy labels
    alnum = re.sub(r'[^a-z0-9]', '', clean)
    if not alnum:
        return True
    digit_ratio = sum(c.isdigit() for c in alnum) / max(1, len(alnum))
    if digit_ratio > 0.35:
        return True
    
    return False


# ── Mesh quality analysis ────────────────────────────────────────────

def estimate_connected_components_from_tokens(mesh_tokens):
    """Fast estimation of connected components from mesh tokens.
    
    Uses numpy array operations instead of Python dicts for speed.
    Each 9-token face encodes 3 vertices * 3 coords. Faces sharing
    identical vertex coordinates are connected.
    """
    if not hasattr(mesh_tokens, 'numpy'):
        tokens = np.array(mesh_tokens, dtype=np.int32)
    else:
        tokens = mesh_tokens.numpy().astype(np.int32)
    
    # Skip BOS/EOS
    if len(tokens) < 11:
        return 1, 1
    
    face_tokens = tokens[1:-1]
    n_faces = len(face_tokens) // 9
    
    if n_faces < 2:
        return 1, n_faces
    
    # For very large meshes, sample to keep it fast
    MAX_FACES_CHECK = 1000
    if n_faces > MAX_FACES_CHECK:
        # Sample evenly from the token stream (z-ordered, so spatial sampling)
        step = n_faces // MAX_FACES_CHECK
        face_indices = list(range(0, n_faces, step))[:MAX_FACES_CHECK]
    else:
        face_indices = list(range(n_faces))
    
    n_check = len(face_indices)
    
    # Reshape to (n_check, 3, 3) — face, vertex, coord
    face_data = np.zeros((n_check, 3, 3), dtype=np.int32)
    for i, fi in enumerate(face_indices):
        base = fi * 9
        for j in range(3):
            vbase = base + j * 3
            if vbase + 2 < len(face_tokens):
                face_data[i, j] = face_tokens[vbase:vbase+3]
    
    # Create vertex keys by packing 3 ints into a single int64
    # key = x * 10000^2 + y * 10000 + z (works for vocab up to 10000)
    vert_keys = (face_data[:, :, 0].astype(np.int64) * 100000000 +
                 face_data[:, :, 1].astype(np.int64) * 10000 +
                 face_data[:, :, 2].astype(np.int64))  # shape (n_check, 3)
    
    # Union-find using numpy
    parent = np.arange(n_check, dtype=np.int32)
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    # Build vertex -> first face mapping
    vert_to_face = {}
    for fi in range(n_check):
        for vi in range(3):
            key = int(vert_keys[fi, vi])
            if key in vert_to_face:
                # Union this face with the first face sharing this vertex
                ra, rb = find(fi), find(vert_to_face[key])
                if ra != rb:
                    parent[ra] = rb
            else:
                vert_to_face[key] = fi
    
    # Count components
    roots = set()
    for i in range(n_check):
        roots.add(find(i))
    
    return len(roots), n_faces


def compute_quality_score(item, n_faces, n_components, has_materials):
    """Compute a 0-1 quality score for ranking items."""
    score = 0.0
    
    # Face count (more = better geometry, up to a point)
    if n_faces >= 500:
        score += 0.3
    elif n_faces >= 200:
        score += 0.2
    elif n_faces >= 100:
        score += 0.15
    elif n_faces >= 50:
        score += 0.05
    
    # Connectivity (fewer components = more solid mesh)
    if n_components <= 1:
        score += 0.3
    elif n_components <= 3:
        score += 0.2
    elif n_components <= 5:
        score += 0.1
    
    # Has real materials
    if has_materials:
        score += 0.2
    
    # Label quality
    label = item.get('label', '')
    label_words = len(label.split())
    if label_words >= 2:
        score += 0.1
    if label_words >= 3:
        score += 0.1
    
    return min(1.0, score)


def extract_material_descriptor(item):
    """Extract useful material description from scene_context."""
    sc = item.get('scene_context', {})
    if not sc:
        return None
    
    mats = sc.get('materials', [])
    if not mats:
        return None
    
    # Extract material names, clean them up
    mat_names = []
    for m in mats:
        if isinstance(m, dict):
            name = m.get('name', '')
        elif isinstance(m, str):
            name = m
        else:
            continue
        if not name:
            continue
        # Clean up Blender material names
        # Remove .001, .002 suffixes
        clean = re.sub(r'\.\d{3}$', '', name)
        # Remove known non-English material names
        clean_lower = clean.lower().strip()
        if clean_lower and clean_lower not in ('material', 'default', ''):
            mat_names.append(clean)
    
    if not mat_names:
        return None
    
    # Take unique names (max 3)
    seen = set()
    unique = []
    for n in mat_names:
        key = n.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(n)
        if len(unique) >= 3:
            break
    
    return ", ".join(unique)


def curate_cache(dry_run=True, min_faces=50, max_per_file=10, max_per_label=5, skip_cc=False):
    """Main curation pipeline."""
    
    if not CACHE_DIR.exists():
        logger.error(f"Cache directory not found: {CACHE_DIR}")
        return
    
    cache_files = sorted(CACHE_DIR.glob("*.pt"))
    logger.info(f"Cache files to process: {len(cache_files)}")
    
    # ── Stats tracking ──
    stats = {
        "total_input": 0,
        "rejected_tiny": 0,         # <min_faces
        "rejected_non_english": 0,   # Non-English labels
        "rejected_weak_label": 0,    # Generic/weak labels
        "rejected_shattered": 0,     # High component ratio
        "rejected_per_file_cap": 0,  # Exceeded per-file limit
        "rejected_per_label_cap": 0, # Exceeded per-label limit
        "labels_enriched": 0,        # Material info added to label
        "total_output": 0,
        "files_modified": 0,
        "files_emptied": 0,
    }
    
    # ── Phase 1: Score and filter per file ──
    logger.info("\n=== Phase 1: Per-file quality filtering ===")
    
    # We'll collect all surviving items with their metadata for global dedup
    all_surviving = []  # list of (file_path, item, quality_score, face_count)
    
    for fi, cache_path in enumerate(cache_files):
        try:
            cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        except Exception as e:
            logger.debug(f"Failed to load {cache_path.name}: {e}")
            continue
        
        if isinstance(cached, dict):
            items = [cached]
        elif isinstance(cached, list):
            items = cached
        else:
            continue
        
        stats["total_input"] += len(items)
        
        # Score each item
        scored_items = []
        for item in items:
            mt = item.get('mesh_tokens')
            if mt is None or (hasattr(mt, '__len__') and len(mt) < 11):
                stats["rejected_tiny"] += 1
                continue
            
            tl = len(mt) if hasattr(mt, '__len__') else 0
            n_faces = (tl - 2) // 9 if tl > 2 else 0
            label = item.get('label', '')
            
            # ── Hard reject: tiny meshes ──
            if n_faces < min_faces:
                stats["rejected_tiny"] += 1
                continue
            
            # ── Hard reject: non-English labels ──
            if is_non_english_label(label):
                stats["rejected_non_english"] += 1
                continue
            
            # ── Hard reject: weak/generic labels ──
            if is_weak_label(label):
                stats["rejected_weak_label"] += 1
                continue
            
            # ── Hard reject: shattered meshes (skip for large meshes — fast path) ──
            if not skip_cc and n_faces < 500:
                n_components, n_face_check = estimate_connected_components_from_tokens(mt)
                if n_face_check > 20:
                    component_ratio = n_components / n_face_check
                    if component_ratio > 0.3:  # >30% of faces are disconnected
                        stats["rejected_shattered"] += 1
                        continue
            else:
                n_components = 1  # assume large meshes are solid
            
            # ── Material enrichment ──
            sc = item.get('scene_context', {})
            mats = sc.get('materials', []) if sc else []
            has_materials = bool(mats)
            
            mat_desc = extract_material_descriptor(item)
            if mat_desc and mat_desc.lower() not in label.lower():
                # Don't duplicate material info already in label
                words_in_label = set(label.lower().split())
                mat_words = set(mat_desc.lower().split())
                # Only add if <50% overlap
                overlap = len(words_in_label & mat_words)
                if overlap < len(mat_words) * 0.5:
                    enriched_label = f"{label}, {mat_desc}" if label else mat_desc
                    # Cap total label length
                    if len(enriched_label) <= 120:
                        item['label_original'] = label
                        item['label'] = enriched_label
                        stats["labels_enriched"] += 1
            
            quality = compute_quality_score(item, n_faces, n_components, has_materials)
            scored_items.append((item, quality, n_faces))
        
        # ── Per-file cap ──
        if len(scored_items) > max_per_file:
            # Sort by quality descending, keep top N
            scored_items.sort(key=lambda x: (-x[1], -x[2]))
            rejected_count = len(scored_items) - max_per_file
            stats["rejected_per_file_cap"] += rejected_count
            scored_items = scored_items[:max_per_file]
        
        for item, quality, n_faces in scored_items:
            all_surviving.append((str(cache_path), item, quality, n_faces))
        
        if (fi + 1) % 50 == 0:
            logger.info(f"  Processed {fi+1}/{len(cache_files)} files, "
                        f"{len(all_surviving)} items surviving so far")
            # Write progress to a separate file for monitoring
            with open('/tmp/curate_progress.txt', 'w') as pf:
                pf.write(f"{fi+1}/{len(cache_files)} files, {len(all_surviving)} surviving\n")
        
        del cached, items
        if (fi + 1) % 50 == 0:
            gc.collect()
    
    logger.info(f"  After per-file filtering: {len(all_surviving)} items")
    
    # ── Phase 2: Global per-label cap ──
    logger.info("\n=== Phase 2: Global per-label deduplication ===")
    
    label_counts = collections.Counter()
    final_items_by_file = collections.defaultdict(list)  # file_path -> [item, ...]
    
    # Sort by quality descending so we keep the best items per label
    all_surviving.sort(key=lambda x: (-x[2], -x[3]))  # quality desc, faces desc
    
    for file_path, item, quality, n_faces in all_surviving:
        label = item.get('label', '').strip().lower()
        # Normalize label for dedup (remove detail/high-detail suffixes)
        norm_label = re.sub(r'\s*\((high-)?detail(ed)?\)\s*', '', label).strip()
        
        if label_counts[norm_label] >= max_per_label:
            stats["rejected_per_label_cap"] += 1
            continue
        
        label_counts[norm_label] += 1
        final_items_by_file[file_path].append(item)
    
    total_final = sum(len(items) for items in final_items_by_file.values())
    logger.info(f"  After per-label cap: {total_final} items")
    logger.info(f"  Unique labels: {len(label_counts)}")
    
    # ── Phase 3: Write curated cache ──
    if dry_run:
        logger.info("\n=== DRY RUN — no files modified ===")
    else:
        logger.info("\n=== Phase 3: Writing curated cache ===")
        
        # Backup first (if not already backed up)
        if not BACKUP_DIR.exists():
            logger.info(f"  Creating backup at {BACKUP_DIR}")
            shutil.copytree(CACHE_DIR, BACKUP_DIR)
            logger.info("  Backup complete")
        else:
            logger.info(f"  Backup already exists at {BACKUP_DIR}")
    
    for cache_path in cache_files:
        path_str = str(cache_path)
        
        if path_str in final_items_by_file:
            new_items = final_items_by_file[path_str]
            
            if not dry_run:
                # Re-encode text_ids for items with enriched labels
                # We need the text tokenizer for this
                for item in new_items:
                    if 'label_original' in item:
                        # Label was enriched — need to re-encode text_ids
                        try:
                            _re_encode_text(item)
                        except Exception:
                            # If re-encoding fails, revert to original label
                            item['label'] = item.pop('label_original')
                
                torch.save(new_items, cache_path)
                stats["files_modified"] += 1
        else:
            # No items survived — empty this file
            if not dry_run:
                cache_path.unlink(missing_ok=True)
                stats["files_emptied"] += 1
    
    stats["total_output"] = total_final
    
    # ── Summary ──
    logger.info(f"\n{'='*60}")
    logger.info(f"CURATION {'PREVIEW' if dry_run else 'COMPLETE'}")
    logger.info(f"{'='*60}")
    logger.info(f"Input items:              {stats['total_input']:,}")
    logger.info(f"Rejected (tiny <{min_faces}f):    {stats['rejected_tiny']:,}")
    logger.info(f"Rejected (non-English):   {stats['rejected_non_english']:,}")
    logger.info(f"Rejected (weak label):    {stats['rejected_weak_label']:,}")
    logger.info(f"Rejected (shattered):     {stats['rejected_shattered']:,}")
    logger.info(f"Rejected (per-file cap):  {stats['rejected_per_file_cap']:,}")
    logger.info(f"Rejected (per-label cap): {stats['rejected_per_label_cap']:,}")
    logger.info(f"Labels enriched w/ mats:  {stats['labels_enriched']:,}")
    logger.info(f"Output items:             {stats['total_output']:,}")
    logger.info(f"Reduction:                {stats['total_input'] - stats['total_output']:,} "
                f"({(1 - stats['total_output']/max(1, stats['total_input']))*100:.1f}%)")
    
    if not dry_run:
        logger.info(f"Files modified:           {stats['files_modified']:,}")
        logger.info(f"Files emptied:            {stats['files_emptied']:,}")
    
    # Show top labels
    logger.info(f"\nTop 20 surviving labels:")
    for label, count in label_counts.most_common(20):
        logger.info(f"  [{count}x] {label[:70]}")
    
    # Face count distribution of survivors
    face_buckets = collections.Counter()
    for _, items in final_items_by_file.items():
        for item in items:
            mt = item.get('mesh_tokens')
            if mt is None:
                continue
            tl = len(mt) if hasattr(mt, '__len__') else 0
            fc = (tl - 2) // 9 if tl > 2 else 0
            if fc < 100:
                face_buckets['50-100'] += 1
            elif fc < 500:
                face_buckets['100-500'] += 1
            elif fc < 1000:
                face_buckets['500-1000'] += 1
            elif fc < 5000:
                face_buckets['1000-5000'] += 1
            else:
                face_buckets['5000+'] += 1
    
    logger.info(f"\nFace count distribution of surviving items:")
    for bucket in ['50-100', '100-500', '500-1000', '1000-5000', '5000+']:
        count = face_buckets.get(bucket, 0)
        pct = count / max(1, total_final) * 100
        bar = '#' * int(pct / 2)
        logger.info(f"  {bucket:>12s}: {count:6d} ({pct:5.1f}%) {bar}")
    
    # Write final progress
    with open('/tmp/curate_progress.txt', 'w') as pf:
        pf.write(f"DONE: {stats['total_output']} items from {stats['total_input']} input\n")
        pf.write(f"tiny={stats['rejected_tiny']} non_en={stats['rejected_non_english']} "
                 f"weak={stats['rejected_weak_label']} shattered={stats['rejected_shattered']} "
                 f"file_cap={stats['rejected_per_file_cap']} label_cap={stats['rejected_per_label_cap']}\n")
    
    return stats


# Text re-encoding for enriched labels
_text_tokenizer = None

def _get_text_tokenizer():
    global _text_tokenizer
    if _text_tokenizer is None:
        sys.path.insert(0, str(BASE))
        from processing.bpe_tokenizer import BPETokenizer
        bpe_path = BASE / "data" / "datasets" / "geometry" / "bpe_tokenizer"
        _text_tokenizer = BPETokenizer.load(bpe_path)
    return _text_tokenizer


def _re_encode_text(item, max_text_length=256):
    """Re-encode text_ids/text_mask after label enrichment."""
    tokenizer = _get_text_tokenizer()
    label = item.get('label', '')
    text_ids, text_mask = tokenizer.encode_padded(label, max_text_length)
    item['text_ids'] = torch.tensor(text_ids, dtype=torch.long)
    item['text_mask'] = torch.tensor(text_mask, dtype=torch.float)


def main():
    parser = argparse.ArgumentParser(description="Curate mesh training cache for quality")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview curation without modifying files")
    parser.add_argument("--apply", action="store_true",
                        help="Apply curation (modifies cache in-place, creates backup)")
    parser.add_argument("--min-faces", type=int, default=50,
                        help="Minimum face count (default: 50)")
    parser.add_argument("--max-per-file", type=int, default=10,
                        help="Max items per source .pt file (default: 10)")
    parser.add_argument("--max-per-label", type=int, default=5,
                        help="Max items globally per unique label (default: 5)")
    parser.add_argument("--skip-cc", action="store_true",
                        help="Skip connected component check (faster)")
    args = parser.parse_args()
    
    if not args.apply and not args.dry_run:
        logger.info("Neither --dry-run nor --apply specified. Running as dry-run.")
        args.dry_run = True
    
    if args.apply:
        args.dry_run = False
    
    curate_cache(
        dry_run=args.dry_run,
        min_faces=args.min_faces,
        max_per_file=args.max_per_file,
        max_per_label=args.max_per_label,
        skip_cc=args.skip_cc,
    )


if __name__ == "__main__":
    main()
