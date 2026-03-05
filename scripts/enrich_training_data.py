#!/usr/bin/env python3
"""
ENRICH training data — quality scoring, label cleanup & weight assignment.

ZERO samples are removed. Every sample gets:
  - quality_tier  (gold / silver / bronze / iron)
  - label_quality_score  (0.0–1.0)
  - mesh_quality_score   (0.0–1.0)
  - quality_weight       (0.2–1.5 — used as loss multiplier during training)

Iron-tier samples still train — they just contribute less gradient.

Usage:
    python scripts/enrich_training_data.py              # dry-run
    python scripts/enrich_training_data.py --apply      # write in-place
    python scripts/enrich_training_data.py --apply --extract-materials
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
CACHE_DIR = BASE / "data" / "processed" / ".mesh_cache"
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


# ═════════════════════════════════════════════════════════════════════
#  Label quality scoring — reusable from old restructure, BUT now
#  scores feed into quality_weight, NOT rejection decisions.
# ═════════════════════════════════════════════════════════════════════

# Labels providing minimal useful text supervision (get lowest weight)
WEAK_LABELS = {
    "object", "mesh", "thing", "model", "untitled", "cube", "sphere",
    "cylinder", "plane", "default", "empty", "null", "none", "test",
    "shape", "3d object", "3d mesh", "3d shape", "small 3d mesh",
    "simple object", "basic object", "part", "piece", "dummy", "base",
    "lattice", "lp", "group", "point", "vert", "vertex", "edge", "face",
}

WEAK_SCENE_LABELS = {
    "multi object scene composition", "scene", "composition",
    "scene composition", "interior", "exterior", "room",
}

_NON_LATIN_RE = re.compile(
    r'[\u0400-\u04ff\u0600-\u06ff\u3000-\u9fff\uac00-\ud7af]'
)
_DIACRITICS_RE = re.compile(
    r'[àáâãäåæçèéêëìíîïðñòóôõöùúûüýạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]',
    re.IGNORECASE,
)

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

_GARBAGE_PATTERNS = [
    re.compile(r'^(prop[_ ]static|func[_ ]detail|worldspawn)', re.I),
    re.compile(r'materials/', re.I),
    re.compile(r'~\w+/', re.I),
    re.compile(r'toolsnodraw', re.I),
    re.compile(r'^bp house \d+', re.I),
    re.compile(r'irradiancevolume', re.I),
]

_NSFW_RE = re.compile(
    r'\b(vagina|penis|breast|nipple|genital|nsfw|nude|naked|cock|futa|hentai|'
    r'erotic|boob|butt|sexy|xxx|porn|fetish|d[i1]ck|p[u0]ssy|cum|tit[sy]?|'
    r'domina|dominatrix|stripper|bordello|brothel|orgasm|ejacul|sperm|'
    r'physbreast|physbutt|cleavage)\b',
    re.I,
)

_ASSET_PREFIX_RE = re.compile(r'^(sm|wv|trv|sd|b2|bp|mi|lp|hp)\s', re.I)

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
    # Misc
    "cueca", "atrax", "ryukin", "fintail", "finback", "imagen",
    "alumnio", "erba", "vetro", "beton", "groupe", "curva",
    "curvas", "luce", "normale", "oggetti", "telo", "mravenec",
    "copertura", "gomma", "ringhiera", "griglia", "maniglia", "finestra",
    "lampada", "pavimento", "soffitto", "parete", "scrivania",
    "metallkedja", "marcoventana", "vetrook", "vetroemission",
    "roomoff", "lighty", "stearinljus",
}

_MATERIAL_NAME_RE = re.compile(r'\b(mtl|mat)[a-z]{3,}', re.I)

# Large English vocabulary for label validation
_ENGLISH_VOCAB = {
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
    "chandelier", "lantern", "candle", "torch",
    "coin", "key", "lock", "chain", "rope", "belt",
    "hat", "boot", "glove", "cape", "robe", "dress", "shirt", "pants",
    "crown", "necklace", "bracelet", "earring",
    "cake", "bread", "fruit", "apple", "pizza", "food", "meat", "cheese",
    "pot", "pan", "stove", "oven", "grill", "fridge", "microwave",
    "guitar", "piano", "drum", "violin", "trumpet", "speaker",
    "painting", "sculpture", "statue", "vase", "trophy",
    "traffic", "hydrant", "mailbox", "dumpster", "trashcan",
    "ventilator", "fan", "heater", "radiator", "vent", "duct",
    "pole", "post", "girder", "truss",
    "tire", "bumper", "hood", "trunk", "fender",
    "sci", "scifi", "fantasy", "medieval", "modern", "futuristic",
    "industrial", "military", "urban", "rural", "ancient", "gothic",
    "ornament", "decoration", "accessory", "prop", "furniture",
    "wooden", "metallic", "rusty", "broken", "damaged",
    "abandoned", "old", "new", "small", "large", "big", "tiny", "huge",
    "tall", "short", "flat", "round", "square", "thin", "thick",
    "low", "high", "poly", "detail", "simple", "complex",
    "left", "right", "top", "bottom", "front", "back", "side",
    "inner", "outer", "upper", "lower", "middle", "center",
    "halloween", "christmas", "pirate", "zombie", "vampire",
    "goblin", "elf", "dwarf", "troll", "orc", "skeleton",
    "mushroom", "corn", "pumpkin", "cactus", "bamboo", "vine",
    "sewer", "drain", "manhole", "grate",
    "chimney", "balcony", "porch", "terrace", "deck",
    "garden", "park", "forest", "mountain", "cave", "dungeon",
    "castle", "fortress", "temple", "church", "cathedral",
    "tavern", "shop", "store", "market", "warehouse", "factory",
    "apartment", "office", "hospital", "school", "prison",
    "highway", "tunnel", "subway", "railway", "airport",
    "platform", "support", "foundation", "pedestal",
    "tongue", "nose", "ear", "jaw", "chin", "neck", "shoulder",
    "chest", "hip", "knee", "ankle", "wrist", "finger",
    "paw", "fin", "beak", "feather", "scale", "fur",
    "alien", "android", "cyborg", "mutant",
    "cannon", "turret", "missile", "grenade", "bomb",
    "cockpit", "cabin", "cargo", "hull", "mast", "sail",
    "propeller", "rotor", "fuselage", "landing",
    "telescope", "microscope", "lens", "laser",
    "battery", "circuit", "chip", "plug", "socket",
    "tombstone", "coffin", "grave", "cross", "altar",
    "fountain", "well", "pool", "pond", "river", "waterfall",
    "pier", "dock", "harbor", "lighthouse",
    "binoculars", "compass", "map",
    "tent", "campfire", "sleeping", "backpack",
    "cart", "wagon", "chariot", "carriage", "sled",
    "cage", "nest", "hive", "web", "cocoon",
    "medal", "badge", "emblem", "crest", "seal",
    "scroll", "tablet", "tome", "grimoire",
    "potion", "flask", "vial", "cauldron", "chalice",
    "lava", "magma", "geyser", "volcano",
    "doorway", "corridor", "hallway", "passage",
    "drape", "tapestry", "sconce", "brazier",
    "pebble", "gravel", "cobblestone", "flagstone",
    "transponder", "cog", "piston", "valve", "gauge", "dial", "meter",
    "crane", "pulley", "winch", "hoist",
    "anvil", "forge", "bellows", "kiln",
    "keg", "cask", "throne", "lectern", "podium", "stage",
    "woman", "man", "girl", "boy", "child", "person", "figure", "character",
    "sorceress", "witch", "mage", "cleric", "rogue", "archer", "ranger",
    "king", "queen", "prince", "princess", "lord", "lady",
    "peasant", "merchant", "blacksmith", "farmer", "guard", "thief",
    "samurai", "ninja", "monk", "paladin", "barbarian", "druid",
    "wolf", "bear", "deer", "fox", "rabbit", "rat", "mouse", "bat",
    "snake", "spider", "beetle", "butterfly", "moth", "ant", "bee", "wasp",
    "frog", "toad", "lizard", "turtle", "tortoise", "crocodile",
    "whale", "shark", "dolphin", "octopus", "crab", "lobster", "jellyfish",
    "eagle", "hawk", "owl", "crow", "raven", "parrot", "penguin", "swan",
    "lion", "tiger", "elephant", "giraffe", "gorilla", "monkey",
    "pig", "cow", "sheep", "goat", "chicken", "duck", "turkey",
    "worm", "slug", "snail", "caterpillar",
    "motorcycle", "bicycle", "scooter", "helicopter", "rocket", "shuttle",
    "submarine", "yacht", "canoe", "kayak", "raft",
    "tractor", "bulldozer", "excavator", "forklift", "trailer",
    "handpainted", "stylized", "realistic", "cartoon", "anime",
    "lowpoly", "highpoly", "retro", "vintage", "steampunk", "cyberpunk",
    "apocalypse", "apocalyptic", "postwar", "dystopian", "utopian",
    "frozen", "burning", "floating", "flying", "underwater",
    "overgrown", "ruined", "sunken", "haunted", "enchanted",
    "miniature", "giant", "colossal", "enormous",
    "modular", "stackable", "collapsible", "portable",
    "cliff", "valley", "hill", "dune", "beach", "island", "swamp", "marsh",
    "coral", "reef", "seaweed", "kelp", "moss", "lichen", "fern",
    "oak", "pine", "birch", "willow", "maple", "palm", "redwood",
    "rose", "daisy", "tulip", "sunflower", "lily", "orchid",
    "gutter", "downspout", "shingle", "siding", "baseboard",
    "molding", "cornice", "frieze", "capital", "plinth",
    "awning", "canopy", "gazebo", "pergola", "arbor",
    "fireplace", "mantle", "hearth", "flue",
    "umbrella", "parasol", "broom", "mop", "shovel", "pickaxe",
    "sextant", "spyglass",
    "suitcase", "luggage", "purse", "wallet",
    "newspaper", "magazine", "envelope", "stamp",
    "scissors", "knife", "fork", "spoon", "chopstick",
    "canteen", "thermos", "cooler", "icebox",
    "trampoline", "swing", "slide", "seesaw",
    "surfboard", "skateboard", "snowboard", "ski",
    "balloon", "kite", "parachute", "glider",
    "cigarette", "cigar", "matchbox", "lighter",
    "typewriter", "radio", "television", "projector",
    "beaker", "dagger", "spear", "bow", "arrow", "crossbow", "catapult",
    "staff", "wand", "orb", "amulet", "talisman",
    "headband", "visor", "mask", "goggles", "glasses",
    "shed", "barn", "silo", "mill", "windmill", "watermill",
    "jetty", "wharf", "boardwalk",
    "skyline", "cityscape", "rooftop", "penthouse",
    "attic", "basement", "cellar",
    "cupboard", "wardrobe", "bookshelf", "nightstand",
    "texture", "pattern", "decal", "sticker",
    "connector", "joint", "bracket", "clamp", "rivet",
    "smokestack", "exhaust", "intake", "outlet",
    "hovercar", "starship", "freighter", "cruiser", "fighter",
    "ray", "portal", "hologram", "forcefield",
    "burger", "sandwich", "hotdog", "taco", "burrito", "sushi",
    "donut", "cookie", "pie", "cream", "candy",
    "coffee", "tea", "wine", "beer", "cocktail",
    "kettle", "teapot", "blender", "mixer", "toaster",
    "ball", "racket", "goal", "net", "hoop",
    "stadium", "arena",
    "flute", "harp", "cello", "saxophone", "banjo", "accordion",
    "microphone", "amplifier", "headphones",
    "server", "router", "modem", "hub",
    "printer", "scanner", "copier", "shredder",
    "usb", "hdmi", "adapter",
    "spring", "coil", "bushing", "bearing", "shaft", "axle",
    "cam", "flywheel", "crankshaft", "clutch", "brake",
    "grille", "spoiler", "diffuser",
    "muffler", "catalytic",
    "mat", "ottoman", "armchair", "recliner", "stool",
    "buffet", "credenza", "hutch", "sideboard",
    "vanity", "dresser", "console",
    "pendant", "spotlight", "floodlight",
    "sack", "pouch", "pallet",
    "reservoir", "cistern", "trough",
    "cloth", "linen", "silk", "velvet", "canvas", "burlap",
    "ribbon", "lace", "thread", "yarn", "spool",
    "stalactite", "stalagmite", "geode", "fossil",
    "obsidian", "marble", "granite", "limestone", "sandstone",
}


# ═════════════════════════════════════════════════════════════════════
#  Scoring functions
# ═════════════════════════════════════════════════════════════════════

def score_label_quality(label: str) -> tuple[float, str]:
    """Score a label 0.0–1.0. Every sample is kept; score informs weight."""
    if not label or not label.strip():
        return 0.05, "empty"

    clean = label.strip().lower()

    # Weak generic labels — still valid geometry, just poor text supervision
    if clean in WEAK_LABELS:
        return 0.1, "weak_label"

    # NSFW — keep the sample, just note it for metadata
    if _NSFW_RE.search(clean):
        return 0.5, "nsfw"

    # Weak scene labels
    if clean in WEAK_SCENE_LABELS:
        return 0.15, "weak_scene"

    # Non-Latin characters
    if _NON_LATIN_RE.search(label):
        return 0.15, "non_latin"

    # Extended diacritics
    if _DIACRITICS_RE.search(label):
        return 0.2, "foreign_diacritics"

    words = set(re.split(r'[\s,._-]+', clean))
    words = {w for w in words if w}

    # Foreign ASCII words
    foreign_hits = words & _ASCII_FOREIGN
    if foreign_hits:
        foreign_ratio = len(foreign_hits) / max(1, len(words))
        if foreign_ratio >= 0.3:
            return 0.2, f"foreign({len(foreign_hits)}w)"

    # Asset prefix
    if _ASSET_PREFIX_RE.match(clean):
        return 0.2, "asset_prefix"

    # Material-name-as-label
    if _MATERIAL_NAME_RE.search(clean):
        non_mat = [w for w in words if not re.match(r'^(mtl|mat)', w, re.I)]
        if len(non_mat) < len(words) * 0.5:
            return 0.2, "material_name_label"

    # Garbage patterns (source engine, etc.)
    for pat in _GARBAGE_PATTERNS:
        if pat.search(label):
            return 0.1, "garbage_pattern"

    # Material-only labels
    meaningful = [w for w in words if w and w not in _MATERIAL_WORDS and len(w) > 1]
    if not meaningful:
        return 0.2, "material_only"

    # Numeric-heavy
    alnum = re.sub(r'[^a-z0-9]', '', clean)
    if alnum:
        digit_ratio = sum(c.isdigit() for c in alnum) / len(alnum)
        if digit_ratio > 0.4:
            return 0.15, "numeric_heavy"

    # Very short
    if len(clean) < 3:
        return 0.15, "too_short"

    # ── English vocabulary check ──
    alpha_words = {w for w in words if w.isalpha() and len(w) > 2}
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

    has_english = len(english_hits) > 0
    has_detail_tag = "(high-detail)" in clean or "(low-detail)" in clean
    word_count = len([w for w in words if w and len(w) > 1])

    if has_english:
        if 2 <= word_count <= 8 and len(english_hits) >= 2:
            return 1.0, "excellent"
        if word_count >= 1:
            return 0.7, "good"
    elif has_detail_tag and word_count >= 2:
        return 0.5, "objaverse_tagged"

    # No English — likely Blender internal name
    if word_count >= 2 and not has_english:
        return 0.25, "no_english_words"

    if word_count == 1 and not has_english:
        return 0.2, "single_unknown_word"

    return 0.4, "mediocre"


def score_mesh_quality(sample: dict) -> tuple[float, str]:
    """Score mesh/token quality 0.0–1.0."""
    tokens = sample.get("mesh_tokens")
    if tokens is None:
        return 0.05, "no_tokens"

    tl = len(tokens) if hasattr(tokens, "__len__") else 0
    if tl < 2:
        return 0.05, "empty_tokens"

    if isinstance(tokens, torch.Tensor):
        if torch.isnan(tokens.float()).any() or torch.isinf(tokens.float()).any():
            return 0.05, "nan_in_tokens"

    # Very short = simple but still valid geometry
    if tl < 20:
        return 0.2, "tiny_mesh"

    # Over-budget / truncated
    if sample.get("over_budget"):
        orig = sample.get("original_face_count", 0)
        current_faces = (tl - 2) // 9
        if orig > 0 and current_faces > 0:
            retention = current_faces / orig
            if retention < 0.1:
                return 0.3, f"heavily_truncated({retention:.0%})"
            elif retention < 0.3:
                return 0.5, f"truncated({retention:.0%})"
            else:
                return 0.7, f"mildly_truncated({retention:.0%})"
        return 0.5, "over_budget"

    face_count = (tl - 2) // 9
    if face_count < 4:
        return 0.3, "trivial_mesh"
    elif face_count < 50:
        return 0.7, "simple_mesh"
    elif face_count <= 2000:
        return 1.0, "good_complexity"
    elif face_count <= 8000:
        return 0.9, "high_complexity"
    else:
        return 0.85, "very_high_complexity"


def compute_quality_weight(label_score: float, mesh_score: float) -> float:
    """Compute quality_weight used as loss multiplier.

    Range:  0.2  (poor label + poor mesh)  →  1.5  (excellent both)
    The mesh score matters more than the label for geometry training.
    """
    # Weighted combination: 35% label, 65% mesh
    combined = label_score * 0.35 + mesh_score * 0.65
    # Map 0.0–1.0 → 0.2–1.5
    weight = 0.2 + combined * 1.3
    return round(min(1.5, max(0.2, weight)), 3)


def compute_quality_tier(label_score: float, mesh_score: float) -> str:
    """Assign tier for reporting. ALL tiers are kept."""
    combined = label_score * 0.35 + mesh_score * 0.65
    if combined >= 0.7 and label_score >= 0.5 and mesh_score >= 0.5:
        return "gold"
    elif combined >= 0.45 and mesh_score >= 0.3:
        return "silver"
    elif combined >= 0.25 and mesh_score >= 0.15:
        return "bronze"
    else:
        return "iron"  # lowest tier but STILL KEPT


# ═════════════════════════════════════════════════════════════════════
#  Material extraction (unchanged — already proved out)
# ═════════════════════════════════════════════════════════════════════

def extract_materials_from_sources() -> list[dict]:
    """Scan all source JSONs and extract real material node trees."""
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

            for obj in objects:
                obj_name = obj.get("name", "")
                raw_mats = obj.get("materials", [])
                if not raw_mats:
                    continue

                for mat in raw_mats:
                    nodes = mat.get("nodes", [])
                    links = mat.get("links", [])
                    mat_name = mat.get("name", "")

                    if not nodes or len(nodes) < 2:
                        continue

                    shader_types = {
                        "BSDF_PRINCIPLED", "BSDF_DIFFUSE", "BSDF_GLOSSY",
                        "BSDF_GLASS", "BSDF_TRANSPARENT", "BSDF_REFRACTION",
                        "BSDF_VELVET", "BSDF_TOON", "BSDF_ANISOTROPIC",
                        "EMISSION", "SUBSURFACE_SCATTERING", "MIX_SHADER",
                        "ADD_SHADER",
                        "ShaderNodeBsdfPrincipled", "ShaderNodeBsdfDiffuse",
                        "ShaderNodeBsdfGlossy", "ShaderNodeBsdfGlass",
                        "ShaderNodeEmission", "ShaderNodeMixShader",
                        "ShaderNodeBsdfAnisotropic",
                    }
                    has_shader = any(
                        n.get("type", "") in shader_types for n in nodes
                    )
                    if not has_shader:
                        continue

                    tree_str = json.dumps(
                        {"nodes": nodes, "links": links},
                        sort_keys=True, default=str,
                    )
                    tree_hash = hashlib.md5(tree_str.encode()).hexdigest()[:16]
                    if tree_hash in seen_hashes:
                        continue
                    seen_hashes.add(tree_hash)

                    label_parts = []
                    if mat_name and mat_name.lower() not in ("material", ""):
                        cn = re.sub(r'[^a-zA-Z0-9\s]', ' ', mat_name)
                        cn = re.sub(r'\s+', ' ', cn).strip().lower()
                        if cn and len(cn) > 2:
                            label_parts.append(cn)
                    if obj_name and obj_name.lower() not in ("object", "mesh", ""):
                        co = re.sub(r'[^a-zA-Z0-9\s]', ' ', obj_name)
                        co = re.sub(r'\s+', ' ', co).strip().lower()
                        if co and len(co) > 2:
                            label_parts.append(co)

                    if label_parts:
                        text = " ".join(label_parts[:3]) + " material"
                    else:
                        if any("GLASS" in n.get("type", "") for n in nodes):
                            text = "glass material"
                        elif any("EMISSION" in n.get("type", "") for n in nodes):
                            text = "emissive material"
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


# ═════════════════════════════════════════════════════════════════════
#  Main enrichment logic — ZERO SAMPLES REMOVED
#  Streams one .pt file at a time to avoid loading 8GB into memory.
# ═════════════════════════════════════════════════════════════════════

def enrich(
    dry_run: bool = True,
    extract_materials: bool = False,
) -> dict:
    """Score every sample and assign quality_weight. No samples removed.

    Processes one .pt file at a time to keep memory bounded.
    Returns quality report dict.
    """
    logger.info("=" * 60)
    logger.info("TRAINING DATA ENRICHMENT (zero-removal)")
    logger.info("=" * 60)

    pt_files = sorted(CACHE_DIR.glob("*.pt"))
    logger.info(f"\n[1/3] Found {len(pt_files)} .pt files to process")

    # ── Stream, score, optionally write ─────────────────────────
    logger.info(f"\n[2/3] Scoring & enriching all samples (streaming)...")
    tier_counts = Counter()
    label_reason_counts = Counter()
    mesh_reason_counts = Counter()
    source_counts = Counter()
    unique_labels = set()
    weight_sum = 0.0
    weight_min = 999.0
    weight_max = 0.0
    label_score_sum = 0.0
    mesh_score_sum = 0.0
    total_samples = 0
    total_files = 0
    failed_files = 0

    for fi, fpath in enumerate(pt_files):
        try:
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            samples = data if isinstance(data, list) else [data]
        except Exception as e:
            logger.warning(f"  Failed to load {fpath.name}: {e}")
            failed_files += 1
            continue

        modified = False
        for sample in samples:
            label = sample.get("label", "")
            l_score, l_reason = score_label_quality(label)
            m_score, m_reason = score_mesh_quality(sample)
            tier = compute_quality_tier(l_score, m_score)
            qw = compute_quality_weight(l_score, m_score)

            # Enrich
            sample["quality_tier"] = tier
            sample["label_quality_score"] = l_score
            sample["mesh_quality_score"] = m_score
            sample["quality_weight"] = torch.tensor(qw, dtype=torch.float32)
            modified = True

            # Stats
            tier_counts[tier] += 1
            label_reason_counts[l_reason] += 1
            mesh_reason_counts[m_reason] += 1
            source_counts[sample.get("data_source", "unknown")] += 1
            unique_labels.add(label.strip().lower())
            weight_sum += qw
            weight_min = min(weight_min, qw)
            weight_max = max(weight_max, qw)
            label_score_sum += l_score
            mesh_score_sum += m_score
            total_samples += 1

        # Write back immediately if not dry run
        if not dry_run and modified:
            torch.save(samples, fpath)

        total_files += 1
        # Progress every 100 files
        if (fi + 1) % 100 == 0:
            logger.info(f"    ... processed {fi + 1}/{len(pt_files)} files "
                        f"({total_samples} samples so far)")

        # Free memory
        del data, samples

    logger.info(f"  Processed {total_files} files, {total_samples} total samples")
    if failed_files:
        logger.warning(f"  {failed_files} files failed to load")

    avg_weight = weight_sum / max(1, total_samples)
    avg_label = label_score_sum / max(1, total_samples)
    avg_mesh = mesh_score_sum / max(1, total_samples)

    logger.info(f"\n  Tier distribution:")
    for tier in ["gold", "silver", "bronze", "iron"]:
        cnt = tier_counts.get(tier, 0)
        pct = 100 * cnt / max(1, total_samples)
        logger.info(f"    {tier:8s}: {cnt:6d} ({pct:5.1f}%)")

    logger.info(f"\n  quality_weight: min={weight_min:.3f} "
                f"avg={avg_weight:.3f} max={weight_max:.3f}")
    logger.info(f"  avg label_score={avg_label:.3f}  "
                f"avg mesh_score={avg_mesh:.3f}")

    logger.info(f"\n  Label quality reasons (top 15):")
    for reason, cnt in label_reason_counts.most_common(15):
        logger.info(f"    [{cnt:5d}] {reason}")

    logger.info(f"\n  Mesh quality reasons (top 10):")
    for reason, cnt in mesh_reason_counts.most_common(10):
        logger.info(f"    [{cnt:5d}] {reason}")

    logger.info(f"\n  Source distribution:")
    for src, cnt in source_counts.most_common():
        pct = 100 * cnt / max(1, total_samples)
        logger.info(f"    {src:20s}: {cnt:6d} ({pct:5.1f}%)")

    # ── Materials ────────────────────────────────────────────────
    materials = []
    if extract_materials:
        logger.info(f"\n[3/3] Extracting materials from source files...")
        materials = extract_materials_from_sources()
        logger.info(f"  Found {len(materials)} unique real materials")

        # Write materials
        MATERIALS_OUT.parent.mkdir(parents=True, exist_ok=True)
        existing_synth = []
        if MATERIALS_OUT.exists():
            with open(MATERIALS_OUT) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            m = json.loads(line)
                            if m.get("source") == "synthetic":
                                existing_synth.append(m)
                        except Exception:
                            pass

        with open(MATERIALS_OUT, "w") as f:
            for m in materials:
                f.write(json.dumps(m, default=str) + "\n")
            for m in existing_synth:
                f.write(json.dumps(m, default=str) + "\n")

        logger.info(f"  Wrote {len(materials)} real + {len(existing_synth)} "
                    f"synthetic materials to {MATERIALS_OUT}")
    else:
        logger.info(f"\n[3/3] Skipping material extraction "
                    f"(use --extract-materials)")

    # ── Report ───────────────────────────────────────────────────
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "approach": "enrich-only (zero samples removed)",
        "total_samples": total_samples,
        "total_files": total_files,
        "failed_files": failed_files,
        "samples_removed": 0,
        "tier_distribution": dict(tier_counts),
        "source_distribution": dict(source_counts),
        "unique_labels": len(unique_labels),
        "quality_weight_stats": {
            "min": round(weight_min, 3),
            "avg": round(avg_weight, 3),
            "max": round(weight_max, 3),
        },
        "avg_label_score": round(avg_label, 3),
        "avg_mesh_score": round(avg_mesh, 3),
        "label_reasons": dict(label_reason_counts.most_common(30)),
        "mesh_reasons": dict(mesh_reason_counts.most_common(20)),
        "materials_extracted": len(materials),
    }

    # Write quality report
    report_path = BASE / "data" / "training_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    if dry_run:
        logger.info(f"\n[DRY RUN] Scores computed but NOT written. "
                    f"Use --apply to enrich in-place.")
    else:
        logger.info(f"\n=== ENRICHMENT APPLIED ===")
        logger.info(f"  {total_samples} samples enriched across "
                    f"{total_files} files (100% preserved)")
    logger.info(f"  Report: {report_path}")

    return report


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Enrich training data with quality scores (zero data removal)"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write enrichment in-place (default: dry-run)",
    )
    parser.add_argument(
        "--extract-materials", action="store_true",
        help="Also re-extract materials from source JSONs",
    )
    args = parser.parse_args()

    report = enrich(
        dry_run=not args.apply,
        extract_materials=args.extract_materials,
    )

    # Summary
    print("\n" + "=" * 60)
    print("ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"Total samples:  {report['total_samples']}")
    print(f"Total files:    {report['total_files']}")
    print(f"Samples removed: {report['samples_removed']}")
    print(f"Unique labels:  {report['unique_labels']}")
    print(f"Materials:      {report['materials_extracted']}")
    print(f"\nQuality weights: "
          f"min={report['quality_weight_stats']['min']:.3f} "
          f"avg={report['quality_weight_stats']['avg']:.3f} "
          f"max={report['quality_weight_stats']['max']:.3f}")
    print(f"\nTier distribution (ALL kept):")
    for tier in ["gold", "silver", "bronze", "iron"]:
        cnt = report["tier_distribution"].get(tier, 0)
        pct = 100 * cnt / max(1, report["total_samples"])
        print(f"  {tier:8s}: {cnt:6d} ({pct:5.1f}%)")
    print(f"\nSource distribution:")
    for src, cnt in sorted(report["source_distribution"].items(),
                           key=lambda x: -x[1]):
        pct = 100 * cnt / max(1, report["total_samples"])
        print(f"  {src:20s}: {cnt:6d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
