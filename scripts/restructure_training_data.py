#!/usr/bin/env python3
"""Restructure training data: quality-gate, deduplicate, balance, enrich.

This is the MASTER script for training data restructuring. It:

1. QUALITY GATE — Assigns every sample a quality tier (gold/silver/bronze/reject)
   based on label quality, mesh integrity, token validity, and geometry metrics.
2. DEDUPLICATE — Caps samples per label to prevent overfitting on scene parts.
3. BALANCE SOURCES — Prevents any single source from dominating.
4. ENRICH — Back-fills material data from source JSONs into cache items.
5. EXTRACT MATERIALS — Builds materials_train.jsonl from real material node trees
   found in source data (replacing the current mostly-synthetic file).
6. OUTPUT — Writes a clean restructured cache + quality report.

The original cache is preserved as a backup. Training loads from the
restructured cache which is guaranteed to be high-quality.

Usage:
    python scripts/restructure_training_data.py --dry-run      # Report only
    python scripts/restructure_training_data.py --apply         # Restructure
    python scripts/restructure_training_data.py --apply --aggressive  # Strict quality
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
CACHE_DIR = BASE / "data" / "processed" / ".mesh_cache"
BACKUP_DIR = BASE / "data" / "processed" / ".mesh_cache_backup"
OUTPUT_DIR = BASE / "data" / "processed" / ".mesh_cache"  # Overwrite in place
MATERIALS_OUT = BASE / "data" / "datasets" / "geometry" / "materials_train.jsonl"
SOURCE_DIRS = [
    BASE / "data" / "processed" / "objaverse",
    BASE / "data" / "processed" / "blender_official",
    BASE / "data" / "processed" / "blendswap",
    BASE / "data" / "processed" / "smutbase",
    BASE / "data" / "processed" / "github",
    BASE / "data" / "processed" / "open3dlab",
    BASE / "data" / "processed" / "youtube",
]

# ── Quality Classification Constants ─────────────────────────────────

# Labels that provide zero useful supervision
REJECT_LABELS = {
    "object", "mesh", "thing", "model", "untitled", "cube", "sphere",
    "cylinder", "plane", "default", "empty", "null", "none", "test",
    "shape", "3d object", "3d mesh", "3d shape", "small 3d mesh",
    "simple object", "basic object", "part", "piece", "dummy", "base",
    "lattice", "lp", "group", "point", "vert", "vertex", "edge", "face",
}

# Generic scene-level labels that don't describe a specific object
WEAK_SCENE_LABELS = {
    "multi object scene composition", "scene", "composition",
    "scene composition", "interior", "exterior", "room",
}

# Non-Latin character detection (Cyrillic, CJK, Arabic, etc.)
_NON_LATIN_RE = re.compile(r'[\u0400-\u04ff\u0600-\u06ff\u3000-\u9fff\uac00-\ud7af]')

# Extended diacritics: Vietnamese, Portuguese, Spanish accented chars that don't
# appear in standard English. We flag labels containing these.
# Covers: àáâãäåæçèéêëìíîïðñòóôõöùúûüý + Vietnamese marks ạắằẳẵặ...
_DIACRITICS_RE = re.compile(r'[àáâãäåæçèéêëìíîïðñòóôõöùúûüýạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]', re.IGNORECASE)

# Material-only words (don't describe the 3D object itself)
_MATERIAL_WORDS = {
    "metal", "wood", "stone", "glass", "plastic", "rubber", "fabric",
    "concrete", "brick", "ceramic", "leather", "steel", "iron",
    "copper", "brass", "chrome", "aluminum", "aluminium",
    "gold", "silver", "bronze", "toon", "cel", "outline", "emission",
    "transparent", "glossy", "matte", "shiny", "rough", "smooth",
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "white", "black", "grey", "gray", "brown", "cyan", "magenta",
    "dark", "light", "bright", "dry", "wet",
}

# Source engine / game entities
_GARBAGE_PATTERNS = [
    re.compile(r'^(prop[_ ]static|func[_ ]detail|worldspawn)', re.I),
    re.compile(r'materials/', re.I),
    re.compile(r'~\w+/', re.I),
    re.compile(r'toolsnodraw', re.I),
    re.compile(r'^bp house \d+', re.I),
    re.compile(r'irradiancevolume', re.I),
]

# NSFW / adult content terms
_NSFW_RE = re.compile(
    r'\b(vagina|penis|breast|nipple|genital|nsfw|nude|naked|cock|futa|hentai|'
    r'erotic|boob|butt|sexy|xxx|porn|fetish|d[i1]ck|p[u0]ssy|cum|tit[sy]?|'
    r'domina|dominatrix|stripper|bordello|brothel|orgasm|ejacul|sperm|'
    r'physbreast|physbutt|cleavage)\b',
    re.I,
)

# Asset library naming prefixes (e.g. "sm trv ...", "wv tudor ...", "b2 fractal ...")
_ASSET_PREFIX_RE = re.compile(
    r'^(sm|wv|trv|sd|b2|bp|mi|lp|hp)\s',
    re.I,
)

# ASCII-only foreign words (Indonesian, Spanish, Portuguese, Italian, etc.)
_ASCII_FOREIGN = {
    # Indonesian
    "jalan", "rumah", "pohon", "batu", "atap", "tanah", "pintu", "dinding",
    "lantai", "meja", "kolam", "hijau", "kuning", "merah", "putih", "hitam",
    "abu", "coklat", "biru", "daun", "kecil", "besar", "tinggi", "rendah",
    "belang", "dalam", "luar", "atas", "bawah", "depan", "samping",
    # Spanish / Portuguese
    "puertas", "cocina", "cemento", "balcao", "ventana", "techo", "suelo",
    "pared", "calle", "silla", "puerta", "cuarto", "habitacion", "cocido",
    "calça", "cinto", "fivela", "armario", "inferior", "interno",
    "tercerpiso", "stucoo", "stuco", "botón", "cinturón", "maçaneta",
    "prateleira", "negro", "blanco", "rojo", "azul", "verde", "morado",
    "amarillo", "marron", "gris", "naranja", "rosa",
    # Italian
    "giallo", "rosso", "bianco", "nero", "grigio", "marrone",
    "interno", "esterno",
    # French
    "fleur", "noeud", "maison", "porte", "fenetre", "toit", "plancher",
    # Misc (from prior audit)
    "cueca", "atrax", "ryukin", "fintail", "finback", "imagen",
    "alumnio", "erba", "vetro", "beton", "groupe", "curva",
    "curvas", "luce", "normale", "oggetti", "telo", "mravenec",
    "copertura", "gomma", "ringhiera", "griglia", "maniglia", "finestra",
    "lampada", "pavimento", "soffitto", "parete", "scrivania",
    "metallkedja", "marcoventana", "vetrook", "vetroemission",
    "roomoff", "lighty", "stearinljus",
}

# Blender-style material name pattern: "mtlSomething", "matBodySkin", etc.
_MATERIAL_NAME_RE = re.compile(
    r'\b(mtl|mat)[a-z]{3,}',
    re.I,
)

# Minimal English vocabulary for label validation.
# A valid English label should contain at least ONE of these words.
_ENGLISH_VOCAB = {
    # Articles / prepositions (only count when combined with nouns)
    # Common 3D object nouns
    "table", "chair", "lamp", "light", "car", "truck", "bus", "vehicle",
    "tree", "plant", "flower", "grass", "leaf", "bush", "log", "stump",
    "house", "building", "tower", "bridge", "road", "street", "path",
    "wall", "floor", "roof", "door", "window", "gate", "fence", "railing",
    "stairs", "stair", "step", "ladder", "elevator",
    "pipe", "wire", "cable", "bolt", "screw", "nail", "nut", "hinge",
    "box", "crate", "barrel", "bottle", "jar", "cup", "mug", "bowl",
    "plate", "tray", "basket", "bucket", "bin", "can", "container",
    "desk", "shelf", "cabinet", "drawer", "bed", "sofa", "couch", "bench",
    "toilet", "sink", "bath", "shower", "faucet", "mirror", "towel",
    "helmet", "sword", "shield", "armor", "weapon", "gun", "rifle",
    "blade", "axe", "hammer", "wrench", "tool", "saw", "drill",
    "wheel", "gear", "handle", "knob", "lever", "button", "switch",
    "screen", "monitor", "keyboard", "computer", "phone", "camera",
    "book", "paper", "pen", "pencil", "clock", "watch",
    "pillar", "column", "arch", "beam", "bar", "rod", "tube", "ring",
    "panel", "frame", "board", "sign", "banner", "flag",
    "rock", "stone", "boulder", "crystal", "gem", "sand",
    "fire", "flame", "smoke", "cloud", "rain", "snow", "ice",
    "eye", "head", "hand", "arm", "leg", "foot", "body", "bone", "skull",
    "face", "hair", "teeth", "tooth", "horn", "tail", "wing", "claw",
    "fish", "bird", "cat", "dog", "horse", "dragon", "monster", "creature",
    "angel", "demon", "knight", "soldier", "warrior", "wizard",
    "robot", "mech", "spaceship", "ship", "boat", "airplane", "plane",
    "train", "tank", "engine", "motor", "reactor", "generator",
    "antenna", "dish", "radar", "satellite",
    "pillow", "curtain", "carpet", "rug", "blanket", "cushion",
    "tile", "brick", "block", "slab", "plank", "board",
    "lamp", "chandelier", "lantern", "candle", "torch",
    "coin", "key", "lock", "chain", "rope", "belt",
    "hat", "boot", "glove", "cape", "robe", "dress", "shirt", "pants",
    "ring", "crown", "necklace", "bracelet", "earring",
    "cake", "bread", "fruit", "apple", "pizza", "food", "meat", "cheese",
    "pot", "pan", "stove", "oven", "grill", "fridge", "microwave",
    "guitar", "piano", "drum", "violin", "trumpet", "speaker",
    "painting", "sculpture", "statue", "vase", "trophy",
    "traffic", "hydrant", "mailbox", "dumpster", "trashcan",
    "ventilator", "fan", "heater", "radiator", "vent", "duct",
    "antenna", "pole", "post", "beam", "girder", "truss",
    "wheel", "tire", "bumper", "hood", "trunk", "fender",
    "sci", "scifi", "fantasy", "medieval", "modern", "futuristic",
    "industrial", "military", "urban", "rural", "ancient", "gothic",
    "ornament", "decoration", "accessory", "prop", "furniture",
    "concrete", "wooden", "metallic", "rusty", "broken", "damaged",
    "abandoned", "old", "new", "small", "large", "big", "tiny", "huge",
    "tall", "short", "flat", "round", "square", "thin", "thick",
    "low", "high", "poly", "detail", "simple", "complex",
    "left", "right", "top", "bottom", "front", "back", "side",
    "inner", "outer", "upper", "lower", "middle", "center",
    "halloween", "christmas", "pirate", "zombie", "vampire",
    "goblin", "elf", "dwarf", "troll", "orc", "skeleton",
    "mushroom", "corn", "pumpkin", "cactus", "bamboo", "vine",
    "sewer", "drain", "manhole", "grate", "vent",
    "roof", "chimney", "balcony", "porch", "terrace", "deck",
    "garden", "park", "forest", "mountain", "cave", "dungeon",
    "castle", "fortress", "temple", "church", "cathedral",
    "tavern", "shop", "store", "market", "warehouse", "factory",
    "apartment", "office", "hospital", "school", "prison",
    "highway", "tunnel", "subway", "railway", "airport",
    "base", "platform", "support", "foundation", "pedestal",
    "tongue", "nose", "ear", "jaw", "chin", "neck", "shoulder",
    "chest", "back", "hip", "knee", "ankle", "wrist", "finger",
    "paw", "fin", "beak", "feather", "scale", "fur",
    "alien", "android", "cyborg", "mutant",
    "tank", "cannon", "turret", "missile", "grenade", "bomb",
    "cockpit", "cabin", "cargo", "deck", "hull", "mast", "sail",
    "propeller", "rotor", "wing", "fuselage", "landing",
    "telescope", "microscope", "lens", "laser",
    "battery", "circuit", "chip", "wire", "plug", "socket",
    "tombstone", "coffin", "grave", "cross", "altar",
    "fountain", "well", "pool", "pond", "river", "waterfall",
    "bridge", "pier", "dock", "harbor", "lighthouse",
    "telescope", "binoculars", "compass", "map",
    "tent", "campfire", "sleeping", "backpack",
    "cart", "wagon", "chariot", "carriage", "sled",
    "cage", "nest", "hive", "web", "cocoon",
    "medal", "badge", "emblem", "crest", "seal",
    "scroll", "tablet", "tome", "grimoire",
    "potion", "flask", "vial", "cauldron", "chalice",
    "lava", "magma", "geyser", "volcano",
    "arch", "doorway", "corridor", "hallway", "passage",
    "curtain", "drape", "tapestry", "banner",
    "chandelier", "sconce", "brazier",
    "pebble", "gravel", "cobblestone", "flagstone",
    "antenna", "satellite", "transponder",
    "cog", "piston", "valve", "gauge", "dial", "meter",
    "crane", "pulley", "winch", "hoist",
    "anvil", "forge", "bellows", "kiln",
    "barrel", "keg", "cask", "chest",
    "throne", "lectern", "podium", "stage",
    # People / characters
    "woman", "man", "girl", "boy", "child", "person", "figure", "character",
    "sorceress", "witch", "mage", "cleric", "rogue", "archer", "ranger",
    "king", "queen", "prince", "princess", "lord", "lady",
    "peasant", "merchant", "blacksmith", "farmer", "guard", "thief",
    "samurai", "ninja", "monk", "paladin", "barbarian", "druid",
    # Animals extended
    "wolf", "bear", "deer", "fox", "rabbit", "rat", "mouse", "bat",
    "snake", "spider", "beetle", "butterfly", "moth", "ant", "bee", "wasp",
    "frog", "toad", "lizard", "turtle", "tortoise", "crocodile",
    "whale", "shark", "dolphin", "octopus", "crab", "lobster", "jellyfish",
    "eagle", "hawk", "owl", "crow", "raven", "parrot", "penguin", "swan",
    "lion", "tiger", "elephant", "giraffe", "gorilla", "monkey",
    "pig", "cow", "sheep", "goat", "chicken", "duck", "turkey",
    "worm", "slug", "snail", "caterpillar",
    # Vehicles extended
    "motorcycle", "bicycle", "scooter", "helicopter", "rocket", "shuttle",
    "submarine", "yacht", "canoe", "kayak", "raft",
    "tractor", "bulldozer", "excavator", "forklift", "trailer",
    # Common descriptors
    "handpainted", "stylized", "realistic", "cartoon", "anime",
    "lowpoly", "highpoly", "retro", "vintage", "steampunk", "cyberpunk",
    "apocalypse", "apocalyptic", "postwar", "dystopian", "utopian",
    "frozen", "burning", "floating", "flying", "underwater",
    "overgrown", "ruined", "sunken", "haunted", "enchanted",
    "miniature", "giant", "colossal", "enormous",
    "modular", "stackable", "collapsible", "portable",
    # Nature
    "cliff", "valley", "hill", "dune", "beach", "island", "swamp", "marsh",
    "coral", "reef", "seaweed", "kelp", "moss", "lichen", "fern",
    "oak", "pine", "birch", "willow", "maple", "palm", "redwood",
    "rose", "daisy", "tulip", "sunflower", "lily", "orchid",
    # Building parts
    "gutter", "downspout", "shingle", "siding", "baseboard",
    "molding", "cornice", "frieze", "capital", "plinth",
    "awning", "canopy", "gazebo", "pergola", "arbor",
    "fireplace", "mantle", "hearth", "flue",
    # Misc objects
    "umbrella", "parasol", "broom", "mop", "shovel", "pickaxe",
    "compass", "sextant", "spyglass", "lantern",
    "suitcase", "luggage", "purse", "wallet",
    "newspaper", "magazine", "envelope", "stamp",
    "scissors", "knife", "fork", "spoon", "chopstick",
    "canteen", "thermos", "cooler", "icebox",
    "trampoline", "swing", "slide", "seesaw",
    "surfboard", "skateboard", "snowboard", "ski",
    "balloon", "kite", "parachute", "glider",
    "cigarette", "cigar", "matchbox", "lighter",
    "typewriter", "radio", "television", "projector",
    "microscope", "beaker", "flask", "test",
    "dagger", "spear", "bow", "arrow", "crossbow", "catapult",
    "staff", "wand", "orb", "amulet", "talisman",
    "headband", "visor", "mask", "goggles", "glasses",
    "shed", "barn", "silo", "mill", "windmill", "watermill",
    "pier", "jetty", "wharf", "boardwalk",
    "skyline", "cityscape", "rooftop", "penthouse",
    "corridor", "attic", "basement", "cellar",
    "cupboard", "wardrobe", "bookshelf", "nightstand",
    "texture", "pattern", "decal", "sticker",
    "connector", "joint", "bracket", "clamp", "rivet",
    "chimney", "smokestack", "exhaust", "intake", "outlet",
    "hovercar", "starship", "freighter", "cruiser", "fighter",
    "ray", "portal", "hologram", "forcefield",
    # Food / kitchen
    "burger", "sandwich", "hotdog", "taco", "burrito", "sushi",
    "donut", "cookie", "pie", "ice", "cream", "candy",
    "coffee", "tea", "wine", "beer", "cocktail",
    "kettle", "teapot", "blender", "mixer", "toaster",
    # Sports / recreation
    "ball", "bat", "racket", "goal", "net", "hoop",
    "trophy", "medal", "podium", "stadium", "arena",
    # Music
    "flute", "harp", "cello", "saxophone", "banjo", "accordion",
    "microphone", "amplifier", "headphones",
    # Tech
    "server", "router", "modem", "switch", "hub",
    "printer", "scanner", "copier", "shredder",
    "usb", "hdmi", "cable", "adapter",
    # Parts / mechanical
    "spring", "coil", "bushing", "bearing", "shaft", "axle",
    "cam", "flywheel", "crankshaft", "clutch", "brake",
    "bumper", "grille", "spoiler", "diffuser",
    "exhaust", "muffler", "catalytic",
    # Interior design
    "rug", "mat", "ottoman", "armchair", "recliner", "stool",
    "buffet", "credenza", "hutch", "sideboard",
    "vanity", "dresser", "console", "island",
    "sconce", "pendant", "spotlight", "floodlight",
    # Containers
    "sack", "pouch", "crate", "pallet", "drum",
    "tank", "reservoir", "cistern", "trough",
    # Fabric / textile
    "cloth", "linen", "silk", "velvet", "canvas", "burlap",
    "ribbon", "lace", "thread", "yarn", "spool",
    # Geology
    "stalactite", "stalagmite", "geode", "fossil",
    "obsidian", "marble", "granite", "limestone", "sandstone",
}


# ── Quality Scoring Functions ────────────────────────────────────────

def score_label_quality(label: str) -> tuple[float, str]:
    """Score a label's quality from 0.0 (garbage) to 1.0 (excellent).

    Returns (score, reason).
    """
    if not label or not label.strip():
        return 0.0, "empty"

    clean = label.strip().lower()

    # Outright reject
    if clean in REJECT_LABELS:
        return 0.0, f"reject_label: {clean}"

    # NSFW / adult content — always reject
    if _NSFW_RE.search(clean):
        return 0.0, "nsfw"

    # Weak scene labels
    if clean in WEAK_SCENE_LABELS:
        return 0.1, f"weak_scene: {clean}"

    # Non-Latin characters (non-English)
    if _NON_LATIN_RE.search(label):
        return 0.05, "non_latin"

    # Extended diacritics (Vietnamese, Portuguese, Spanish, etc.)
    if _DIACRITICS_RE.search(label):
        return 0.1, "foreign_diacritics"

    words = set(re.split(r'[\s,._-]+', clean))
    words = {w for w in words if w}  # remove empty

    # Known foreign words (expanded)
    foreign_hits = words & _ASCII_FOREIGN
    if foreign_hits:
        foreign_ratio = len(foreign_hits) / max(1, len(words))
        if foreign_ratio >= 0.3:
            return 0.1, f"foreign: {foreign_hits}"

    # Asset library prefix patterns (sm trv ..., wv tudor ..., b2 ...)
    if _ASSET_PREFIX_RE.match(clean):
        return 0.15, "asset_prefix"

    # Material-name-as-label (mtlBodySkin, matFloor, etc.)
    if _MATERIAL_NAME_RE.search(clean):
        # Only reject if majority of label is material name
        non_mat_words = [w for w in words if not re.match(r'^(mtl|mat)', w, re.I)]
        if len(non_mat_words) < len(words) * 0.5:
            return 0.1, "material_name_label"

    # Garbage patterns
    for pat in _GARBAGE_PATTERNS:
        if pat.search(label):
            return 0.0, "garbage_pattern"

    # Material-only labels (all words are material descriptors)
    meaningful = [w for w in words if w and w not in _MATERIAL_WORDS and len(w) > 1]
    if not meaningful:
        return 0.15, "material_only"

    # Numeric-heavy
    alnum = re.sub(r'[^a-z0-9]', '', clean)
    if alnum:
        digit_ratio = sum(c.isdigit() for c in alnum) / len(alnum)
        if digit_ratio > 0.4:
            return 0.1, "numeric_heavy"

    # Too short (1-2 chars)
    if len(clean) < 3:
        return 0.1, "too_short"

    # ── English vocabulary check ──
    # A good label should contain at least one recognizable English word
    # that describes geometry/objects. This catches random Blender names.
    alpha_words = {w for w in words if w.isalpha() and len(w) > 2}

    # Check both exact match AND simple plural stripping (cats -> cat, boxes -> box)
    english_hits = alpha_words & _ENGLISH_VOCAB
    if not english_hits:
        for w in alpha_words:
            stem = w
            if stem.endswith("es") and len(stem) > 4:
                stem = stem[:-2]
            elif stem.endswith("s") and len(stem) > 3:
                stem = stem[:-1]
            if stem in _ENGLISH_VOCAB:
                english_hits.add(w)
    has_english_content = len(english_hits) > 0

    # Also accept (high-detail) / (low-detail) objaverse tags as signal
    has_detail_tag = "(high-detail)" in clean or "(low-detail)" in clean

    word_count = len([w for w in words if w and len(w) > 1])

    if has_english_content:
        # Excellent: 2-8 descriptive English words
        if 2 <= word_count <= 8 and len(english_hits) >= 2:
            return 1.0, "excellent"
        # Good: at least one English word
        if word_count >= 1:
            return 0.7, "good"
    elif has_detail_tag and word_count >= 2:
        # Objaverse with detail tag — accept at 0.5
        return 0.5, "objaverse_tagged"

    # No recognizable English — likely Blender internal name
    if word_count >= 2 and not has_english_content:
        return 0.12, "no_english_words"

    # Single-word label with no English match
    if word_count == 1 and not has_english_content:
        return 0.1, "single_unknown_word"

    return 0.4, "mediocre"


def score_mesh_quality(sample: dict) -> tuple[float, str]:
    """Score mesh/token quality from 0.0 to 1.0."""
    tokens = sample.get("mesh_tokens")
    if tokens is None:
        return 0.0, "no_tokens"

    tl = len(tokens) if hasattr(tokens, "__len__") else 0
    if tl < 2:
        return 0.0, "empty_tokens"

    # Check for NaN/Inf
    if isinstance(tokens, torch.Tensor):
        if torch.isnan(tokens.float()).any() or torch.isinf(tokens.float()).any():
            return 0.0, "nan_in_tokens"

    # Very short sequences (< 20 tokens = < 2 faces) are usually degenerate
    if tl < 20:
        return 0.1, "too_few_tokens"

    # Over-budget truncated meshes lose quality from the truncation
    if sample.get("over_budget"):
        orig = sample.get("original_face_count", 0)
        current_faces = (tl - 2) // 9
        if orig > 0 and current_faces > 0:
            retention = current_faces / orig
            if retention < 0.1:
                return 0.2, f"heavily_truncated ({retention:.1%})"
            elif retention < 0.3:
                return 0.4, f"truncated ({retention:.1%})"
            else:
                return 0.6, f"mildly_truncated ({retention:.1%})"
        return 0.4, "over_budget"

    # Token length scoring (prefer moderate complexity)
    face_count = (tl - 2) // 9
    if face_count < 4:
        return 0.2, "trivial_mesh"
    elif face_count < 50:
        return 0.6, "simple_mesh"
    elif face_count <= 2000:
        return 1.0, "good_complexity"
    elif face_count <= 8000:
        return 0.9, "high_complexity"
    else:
        return 0.8, "very_high_complexity"


def compute_quality_tier(label_score: float, mesh_score: float) -> str:
    """Assign a quality tier based on combined scores."""
    combined = label_score * 0.5 + mesh_score * 0.5

    if combined >= 0.7 and label_score >= 0.5 and mesh_score >= 0.5:
        return "gold"
    elif combined >= 0.4 and label_score >= 0.2 and mesh_score >= 0.2:
        return "silver"
    elif combined >= 0.15 and mesh_score >= 0.1:
        return "bronze"
    else:
        return "reject"


# ── Material Extraction ──────────────────────────────────────────────

def extract_materials_from_sources() -> list[dict]:
    """Scan all source JSONs and extract real material node trees.

    Returns a list of material dicts in the format expected by
    MaterialStream: {"text": "...", "node_tree": {"nodes": [...], "links": [...]}}
    """
    materials = []
    seen_hashes = set()

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            continue
        source_name = source_dir.name
        json_files = sorted(source_dir.glob("*.json"))

        for jf in json_files:
            if jf.name.endswith(".meta.json"):
                continue
            try:
                with open(jf) as f:
                    data = json.load(f)
            except Exception:
                continue

            objects = data.get("objects", [data]) if isinstance(data, dict) else [data]
            file_label = data.get("text_label", jf.stem)

            for obj in objects:
                obj_name = obj.get("name", "")
                raw_mats = obj.get("materials", [])
                if not raw_mats:
                    continue

                for mat in raw_mats:
                    nodes = mat.get("nodes", [])
                    links = mat.get("links", [])
                    mat_name = mat.get("name", "")

                    # Skip empty/placeholder materials
                    if not nodes or len(nodes) < 2:
                        continue

                    # Must have at least one shader node (BSDF, etc.)
                    shader_types = {
                        "BSDF_PRINCIPLED", "BSDF_DIFFUSE", "BSDF_GLOSSY",
                        "BSDF_GLASS", "BSDF_TRANSPARENT", "BSDF_REFRACTION",
                        "BSDF_VELVET", "BSDF_TOON", "BSDF_ANISOTROPIC",
                        "EMISSION", "SUBSURFACE_SCATTERING", "MIX_SHADER",
                        "ADD_SHADER",
                        # Also handle Blender 4.x names
                        "ShaderNodeBsdfPrincipled", "ShaderNodeBsdfDiffuse",
                        "ShaderNodeBsdfGlossy", "ShaderNodeBsdfGlass",
                        "ShaderNodeEmission", "ShaderNodeMixShader",
                        "ShaderNodeBsdfAnisotropic",
                    }
                    has_shader = any(
                        n.get("type", "") in shader_types
                        for n in nodes
                    )
                    if not has_shader:
                        continue

                    # Dedup by node tree hash
                    tree_str = json.dumps(
                        {"nodes": nodes, "links": links},
                        sort_keys=True, default=str
                    )
                    tree_hash = hashlib.md5(tree_str.encode()).hexdigest()[:16]
                    if tree_hash in seen_hashes:
                        continue
                    seen_hashes.add(tree_hash)

                    # Build descriptive text label for the material
                    label_parts = []
                    if mat_name and mat_name.lower() not in ("material", ""):
                        clean_name = re.sub(r'[^a-zA-Z0-9\s]', ' ', mat_name)
                        clean_name = re.sub(r'\s+', ' ', clean_name).strip().lower()
                        if clean_name and len(clean_name) > 2:
                            label_parts.append(clean_name)
                    if obj_name and obj_name.lower() not in ("object", "mesh", ""):
                        clean_obj = re.sub(r'[^a-zA-Z0-9\s]', ' ', obj_name)
                        clean_obj = re.sub(r'\s+', ' ', clean_obj).strip().lower()
                        if clean_obj and len(clean_obj) > 2:
                            label_parts.append(clean_obj)

                    if label_parts:
                        text = " ".join(label_parts[:3]) + " material"
                    else:
                        # Infer from node types
                        if any("GLASS" in n.get("type", "") for n in nodes):
                            text = "glass material"
                        elif any("EMISSION" in n.get("type", "") for n in nodes):
                            text = "emissive material"
                        elif any("METALLIC" in str(n) for n in nodes):
                            text = "metallic material"
                        else:
                            text = "principled material"

                    materials.append({
                        "text": text[:200],
                        "node_tree": {"nodes": nodes, "links": links},
                        "source": source_name,
                        "source_file": jf.name,
                        "complexity": len(nodes),
                    })

    return materials


# ── Main Restructuring Logic ─────────────────────────────────────────

def load_all_cache_items() -> list[tuple[Path, int, dict]]:
    """Load all cache items. Returns [(file_path, index_in_file, sample_dict)]."""
    pt_files = sorted(CACHE_DIR.glob("*.pt"))
    all_items = []
    for f in pt_files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
            samples = data if isinstance(data, list) else [data]
            for i, s in enumerate(samples):
                all_items.append((f, i, s))
        except Exception as e:
            logger.warning(f"Failed to load {f.name}: {e}")
    return all_items


def restructure(
    max_per_label: int = 10,
    max_source_pct: float = 0.75,
    min_label_score: float = 0.15,
    min_mesh_score: float = 0.10,
    aggressive: bool = False,
    dry_run: bool = True,
    no_materials: bool = False,
) -> dict:
    """Run the full restructuring pipeline.

    Returns a quality report dict.
    """
    if aggressive:
        min_label_score = 0.4
        min_mesh_score = 0.4
        max_per_label = 5

    logger.info("=" * 60)
    logger.info("TRAINING DATA RESTRUCTURING")
    logger.info("=" * 60)

    # ── Phase 1: Load & Score ────────────────────────────────────
    logger.info("\n[1/6] Loading all cache items...")
    all_items = load_all_cache_items()
    logger.info(f"  Loaded {len(all_items)} samples from {len(set(f for f,_,_ in all_items))} files")

    logger.info("\n[2/6] Scoring quality...")
    scored_items = []
    tier_counts = Counter()
    reject_reasons = Counter()
    label_scores_dist = []
    mesh_scores_dist = []

    for fpath, idx, sample in all_items:
        label = sample.get("label", "")
        l_score, l_reason = score_label_quality(label)
        m_score, m_reason = score_mesh_quality(sample)
        tier = compute_quality_tier(l_score, m_score)

        label_scores_dist.append(l_score)
        mesh_scores_dist.append(m_score)
        tier_counts[tier] += 1

        if tier == "reject":
            reject_reasons[f"L:{l_reason} M:{m_reason}"] += 1
            continue

        if l_score < min_label_score:
            reject_reasons[f"label_below_threshold: {l_reason}"] += 1
            continue
        if m_score < min_mesh_score:
            reject_reasons[f"mesh_below_threshold: {m_reason}"] += 1
            continue

        scored_items.append({
            "fpath": fpath,
            "idx": idx,
            "sample": sample,
            "label_score": l_score,
            "mesh_score": m_score,
            "tier": tier,
            "label": label.strip().lower(),
            "source": sample.get("data_source", "unknown"),
        })

    logger.info(f"  Quality tiers: {dict(tier_counts)}")
    logger.info(f"  After quality gate: {len(scored_items)} / {len(all_items)} retained")
    logger.info(f"  Rejection reasons:")
    for reason, cnt in reject_reasons.most_common(20):
        logger.info(f"    [{cnt}x] {reason}")

    # ── Phase 2: Deduplicate by label ────────────────────────────
    logger.info(f"\n[3/6] Deduplicating (max {max_per_label} per label)...")

    # Sort by quality (best first) so we keep the best samples per label
    scored_items.sort(key=lambda x: -(x["label_score"] * 0.5 + x["mesh_score"] * 0.5))

    label_counts = Counter()
    deduped = []
    dedup_removed = 0

    for item in scored_items:
        lbl = item["label"]
        if label_counts[lbl] >= max_per_label:
            dedup_removed += 1
            continue
        label_counts[lbl] += 1
        deduped.append(item)

    logger.info(f"  Removed {dedup_removed} excess duplicates")
    logger.info(f"  After dedup: {len(deduped)} samples, {len(label_counts)} unique labels")

    # ── Phase 3: Source Balancing ─────────────────────────────────
    logger.info(f"\n[4/6] Balancing sources (max {max_source_pct:.0%} per source)...")

    source_counts = Counter(item["source"] for item in deduped)
    source_trimmed = Counter()

    # Sort within each source by quality (best first)
    by_source = defaultdict(list)
    for item in deduped:
        by_source[item["source"]].append(item)
    for src in by_source:
        by_source[src].sort(key=lambda x: -(x["label_score"] * 0.5 + x["mesh_score"] * 0.5))

    # Closed-form balance: for any source exceeding max_source_pct,
    # compute target size based on remaining sources to achieve exact ratio.
    # This avoids the spiral-down problem of iterative trimming.
    over_cap = {s: items for s, items in by_source.items()
                if len(items) / len(deduped) > max_source_pct}
    under_cap = {s: items for s, items in by_source.items()
                 if s not in over_cap}
    rest_total = sum(len(v) for v in under_cap.values())

    if over_cap and rest_total > 0:
        # Each over-cap source gets capped so final_pct = max_source_pct
        # target = rest_total * max_source_pct / (1 - max_source_pct * n_over)
        # For simplicity, cap each independently:
        for src, items in over_cap.items():
            target = int(rest_total * max_source_pct / (1 - max_source_pct))
            if len(items) > target:
                source_trimmed[src] = len(items) - target
                by_source[src] = items[:target]

    balanced = []
    for items in by_source.values():
        balanced.extend(items)

    logger.info(f"  Source counts before balance: {dict(source_counts)}")
    if source_trimmed:
        logger.info(f"  Trimmed: {dict(source_trimmed)}")
    final_source_counts = Counter(item["source"] for item in balanced)
    logger.info(f"  After balance: {dict(final_source_counts)}")
    logger.info(f"  Total: {len(balanced)} samples")

    # ── Phase 4: Compute final tier distribution ──────────────────
    logger.info(f"\n[5/6] Final tier distribution:")
    final_tiers = Counter(item["tier"] for item in balanced)
    for tier in ["gold", "silver", "bronze"]:
        cnt = final_tiers.get(tier, 0)
        pct = 100 * cnt / max(1, len(balanced))
        logger.info(f"  {tier}: {cnt} ({pct:.1f}%)")

    # ── Phase 5: Extract materials ────────────────────────────────
    if no_materials:
        logger.info(f"\n[6/6] Skipping material extraction (--no-materials)")
        materials = []
    else:
        logger.info(f"\n[6/6] Extracting materials from source files...")
        materials = extract_materials_from_sources()
        logger.info(f"  Found {len(materials)} unique real materials")

    complexity_counts = Counter()
    for m in materials:
        c = m["complexity"]
        if c <= 3:
            complexity_counts["simple (≤3 nodes)"] += 1
        elif c <= 10:
            complexity_counts["medium (4-10 nodes)"] += 1
        elif c <= 25:
            complexity_counts["complex (11-25 nodes)"] += 1
        else:
            complexity_counts["very complex (>25 nodes)"] += 1
    logger.info(f"  Material complexity: {dict(complexity_counts)}")

    # ── Build quality report ──────────────────────────────────────
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "original_samples": len(all_items),
        "after_quality_gate": len(scored_items),
        "after_dedup": len(deduped),
        "after_balance": len(balanced),
        "tier_distribution": dict(final_tiers),
        "source_distribution": dict(final_source_counts),
        "unique_labels": len(set(item["label"] for item in balanced)),
        "materials_extracted": len(materials),
        "material_complexity": dict(complexity_counts),
        "quality_settings": {
            "max_per_label": max_per_label,
            "max_source_pct": max_source_pct,
            "min_label_score": min_label_score,
            "min_mesh_score": min_mesh_score,
            "aggressive": aggressive,
        },
    }

    # ── Apply if not dry run ──────────────────────────────────────
    if dry_run:
        logger.info("\n[DRY RUN] No changes written. Use --apply to restructure.")
    else:
        _apply_restructure(balanced, materials, report)

    return report


def _apply_restructure(
    balanced: list[dict],
    materials: list[dict],
    report: dict,
):
    """Write the restructured cache, materials, and report to disk."""
    logger.info("\n=== APPLYING RESTRUCTURE ===")

    # Backup original cache
    if CACHE_DIR.exists() and not BACKUP_DIR.exists():
        logger.info(f"Backing up original cache to {BACKUP_DIR}...")
        shutil.copytree(CACHE_DIR, BACKUP_DIR)
        logger.info("  Backup complete")
    elif BACKUP_DIR.exists():
        logger.info("  Backup already exists, skipping")

    # Group items by original file path for efficient re-writing
    by_file = defaultdict(list)
    for item in balanced:
        by_file[item["fpath"]].append(item)

    # Clear existing cache (except non-.pt files like fingerprints)
    existing_pt = list(CACHE_DIR.glob("*.pt"))
    for f in existing_pt:
        f.unlink()
    logger.info(f"  Cleared {len(existing_pt)} old .pt files")

    # Write restructured cache
    # Group balanced items into reasonably-sized .pt files
    # Use content-hash filenames for stable naming
    written_files = 0
    written_samples = 0

    for fpath, items in by_file.items():
        cache_items = []
        for item in items:
            sample = item["sample"]
            # Enrich with quality metadata
            sample["quality_tier"] = item["tier"]
            sample["label_quality_score"] = item["label_score"]
            sample["mesh_quality_score"] = item["mesh_score"]
            # Ensure quality_weight reflects our scoring
            combined = item["label_score"] * 0.5 + item["mesh_score"] * 0.5
            sample["quality_weight"] = torch.tensor(
                max(0.5, min(2.0, combined * 1.5)), dtype=torch.float32
            )
            cache_items.append(sample)

        if cache_items:
            # Use the original filename
            out_path = CACHE_DIR / fpath.name
            torch.save(cache_items, out_path)
            written_files += 1
            written_samples += len(cache_items)

    logger.info(f"  Wrote {written_samples} samples across {written_files} .pt files")

    # Write extracted materials
    if materials:
        MATERIALS_OUT.parent.mkdir(parents=True, exist_ok=True)
        # Keep existing synthetic materials and prepend real ones
        existing_mats = []
        if MATERIALS_OUT.exists():
            with open(MATERIALS_OUT) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            m = json.loads(line)
                            # Mark existing ones as synthetic if they lack source
                            if "source" not in m:
                                m["source"] = "synthetic"
                            existing_mats.append(m)
                        except Exception:
                            pass

        # Write: real materials first, then existing
        with open(MATERIALS_OUT, "w") as f:
            for m in materials:
                f.write(json.dumps(m, default=str) + "\n")
            # Keep synthetic materials that don't overlap with real ones
            real_texts = {m["text"].lower() for m in materials}
            synth_kept = 0
            for m in existing_mats:
                if m["text"].lower() not in real_texts:
                    f.write(json.dumps(m, default=str) + "\n")
                    synth_kept += 1

        logger.info(f"  Wrote {len(materials)} real + {synth_kept} synthetic materials to {MATERIALS_OUT}")

    # Write quality report
    report_path = BASE / "data" / "training_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"  Quality report: {report_path}")

    logger.info("\n=== RESTRUCTURE COMPLETE ===")
    logger.info(f"  Samples: {report['original_samples']} → {written_samples}")
    logger.info(f"  Materials: {len(materials)} real ({MATERIALS_OUT})")
    logger.info(f"  Backup: {BACKUP_DIR}")


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Restructure training data")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write changes (default: dry-run)")
    parser.add_argument("--aggressive", action="store_true",
                        help="Strict quality thresholds (gold+silver only)")
    parser.add_argument("--max-per-label", type=int, default=10,
                        help="Max samples per label (default: 10)")
    parser.add_argument("--max-source-pct", type=float, default=0.75,
                        help="Max fraction for any single source (default: 0.75)")
    parser.add_argument("--min-label-score", type=float, default=0.15,
                        help="Min label quality score to keep (default: 0.15)")
    parser.add_argument("--min-mesh-score", type=float, default=0.10,
                        help="Min mesh quality score to keep (default: 0.10)")
    parser.add_argument("--no-materials", action="store_true",
                        help="Skip material extraction (faster for testing)")

    args = parser.parse_args()

    report = restructure(
        max_per_label=args.max_per_label,
        max_source_pct=args.max_source_pct,
        min_label_score=args.min_label_score,
        min_mesh_score=args.min_mesh_score,
        aggressive=args.aggressive,
        dry_run=not args.apply,
        no_materials=args.no_materials,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESTRUCTURE SUMMARY")
    print("=" * 60)
    print(f"Original samples:     {report['original_samples']}")
    print(f"After quality gate:   {report['after_quality_gate']}")
    print(f"After deduplication:  {report['after_dedup']}")
    print(f"After source balance: {report['after_balance']}")
    print(f"Unique labels:        {report['unique_labels']}")
    print(f"Materials extracted:  {report['materials_extracted']}")
    print(f"\nTier distribution:")
    for tier, cnt in sorted(report['tier_distribution'].items()):
        pct = 100 * cnt / max(1, report['after_balance'])
        print(f"  {tier}: {cnt} ({pct:.1f}%)")
    print(f"\nSource distribution:")
    for src, cnt in sorted(report['source_distribution'].items(),
                           key=lambda x: -x[1]):
        pct = 100 * cnt / max(1, report['after_balance'])
        print(f"  {src}: {cnt} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
