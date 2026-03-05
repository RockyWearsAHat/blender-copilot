"""Lightweight prompt semantic expansion for stronger text grounding.

This module injects compact, deterministic semantic hints based on common
object/theme words in a prompt. It is intentionally small and local so it can
run in both training and inference without extra dependencies.
"""

from __future__ import annotations

import random
import re
from typing import Iterable


_THEME_MAP: dict[str, tuple[str, ...]] = {
    # ── VEHICLES / AUTOMOTIVE ──────────────────────────────────────────────
    "car":          ("vehicle", "automotive", "wheels", "body", "transport"),
    "vehicle":      ("automotive", "transport", "machine"),
    "automobile":   ("car", "vehicle", "automotive"),
    "truck":        ("vehicle", "automotive", "cargo", "wheels"),
    "van":          ("vehicle", "automotive", "cargo"),
    "bus":          ("vehicle", "automotive", "transport", "large"),
    "motorcycle":   ("vehicle", "automotive", "wheels", "sport"),
    "bike":         ("vehicle", "wheels", "transport"),
    "sports":       ("aerodynamic", "performance", "automotive"),
    "sport":        ("aerodynamic", "performance", "automotive"),
    "lamborghini":  ("car", "vehicle", "automotive", "sports", "aerodynamic"),
    "huracan":      ("car", "vehicle", "automotive", "sports", "aerodynamic"),
    "ferrari":      ("car", "vehicle", "automotive", "sports", "aerodynamic"),
    "porsche":      ("car", "vehicle", "automotive", "sports", "aerodynamic"),
    "mustang":      ("car", "vehicle", "automotive", "sports"),
    "sedan":        ("car", "vehicle", "automotive"),
    "coupe":        ("car", "vehicle", "automotive", "sports"),
    "suv":          ("car", "vehicle", "automotive", "large"),
    "pickup":       ("truck", "vehicle", "automotive", "cargo"),
    "race":         ("automotive", "sports", "aerodynamic", "performance"),
    "supercar":     ("car", "vehicle", "automotive", "sports", "aerodynamic"),
    "dragster":     ("car", "vehicle", "automotive", "sports", "aerodynamic"),

    # ── AIRCRAFT / WATERCRAFT ──────────────────────────────────────────────
    "plane":        ("vehicle", "aerodynamic", "aircraft", "transport"),
    "airplane":     ("vehicle", "aerodynamic", "aircraft", "transport"),
    "aircraft":     ("vehicle", "aerodynamic", "transport"),
    "jet":          ("vehicle", "aerodynamic", "aircraft", "fast"),
    "fighter":      ("vehicle", "aerodynamic", "aircraft", "military"),
    "helicopter":   ("vehicle", "aircraft", "rotor", "transport"),
    "drone":        ("vehicle", "aircraft", "mechanical", "rotor"),
    "rocket":       ("vehicle", "spacecraft", "aerodynamic", "sci-fi"),
    "spaceship":    ("vehicle", "sci-fi", "hard-surface", "mechanical"),
    "spacecraft":   ("vehicle", "sci-fi", "hard-surface", "mechanical"),
    "ship":         ("vehicle", "transport", "hard-surface"),
    "boat":         ("vehicle", "transport", "water"),
    "submarine":    ("vehicle", "transport", "water", "mechanical"),
    "yacht":        ("vehicle", "transport", "water", "luxury"),
    "sailboat":     ("vehicle", "transport", "water"),

    # ── LANDSCAPE / TERRAIN / NATURE ──────────────────────────────────────
    "terrain":      ("landscape", "ground", "organic", "outdoor"),
    "landscape":    ("terrain", "outdoor", "organic", "nature"),
    "hill":         ("landscape", "elevation", "organic", "terrain"),
    "hills":        ("landscape", "elevation", "organic", "terrain"),
    "mountain":     ("landscape", "rock", "elevation", "outdoor"),
    "mountains":    ("landscape", "rock", "elevation", "outdoor"),
    "valley":       ("landscape", "terrain", "elevation", "outdoor"),
    "cliff":        ("landscape", "rock", "elevation", "outdoor"),
    "canyon":       ("landscape", "terrain", "rock", "outdoor"),
    "island":       ("landscape", "terrain", "water", "outdoor"),
    "beach":        ("landscape", "terrain", "sand", "outdoor"),
    "desert":       ("landscape", "terrain", "sand", "outdoor"),
    "ground":       ("landscape", "terrain", "outdoor"),
    "grass":        ("organic", "nature", "ground", "landscape"),
    "grassy":       ("organic", "nature", "ground", "landscape"),
    "field":        ("landscape", "nature", "outdoor", "organic"),
    "meadow":       ("landscape", "nature", "outdoor", "organic"),
    "swamp":        ("landscape", "terrain", "water", "organic"),
    "coast":        ("landscape", "terrain", "water", "outdoor"),

    # ── FLORA / PLANTS ────────────────────────────────────────────────────
    "tree":         ("organic", "nature", "branching", "outdoor"),
    "forest":       ("nature", "organic", "outdoor", "trees"),
    "jungle":       ("nature", "organic", "outdoor", "tropical"),
    "bush":         ("organic", "nature", "outdoor"),
    "flower":       ("organic", "nature", "outdoor"),
    "plant":        ("organic", "nature", "outdoor"),
    "cactus":       ("organic", "nature", "desert", "outdoor"),
    "mushroom":     ("organic", "nature", "fungi"),
    "coral":        ("organic", "nature", "water"),
    "vine":         ("organic", "nature", "branching"),
    "leaf":         ("organic", "nature", "flat"),
    "seaweed":      ("organic", "nature", "water"),

    # ── ROCKS / GEOLOGICAL ───────────────────────────────────────────────
    "rock":         ("terrain", "organic", "natural", "landscape"),
    "stone":        ("terrain", "organic", "natural", "landscape"),
    "crystal":      ("mineral", "geometric", "hard-surface"),
    "gem":          ("mineral", "geometric", "hard-surface"),
    "cave":         ("landscape", "terrain", "rock", "organic"),
    "boulder":      ("rock", "terrain", "natural"),
    "pebble":       ("rock", "terrain", "small"),

    # ── ARCHITECTURE / BUILDINGS ──────────────────────────────────────────
    "house":        ("architecture", "building", "structure", "hard-surface"),
    "building":     ("architecture", "structure", "hard-surface"),
    "skyscraper":   ("architecture", "building", "tall", "urban"),
    "tower":        ("architecture", "structure", "tall", "hard-surface"),
    "castle":       ("architecture", "medieval", "structure", "hard-surface"),
    "cathedral":    ("architecture", "building", "religious", "ornate"),
    "temple":       ("architecture", "building", "religious", "hard-surface"),
    "church":       ("architecture", "building", "religious"),
    "bridge":       ("architecture", "structure", "hard-surface", "transport"),
    "tunnel":       ("architecture", "structure", "underground"),
    "dam":          ("architecture", "structure", "large"),
    "wall":         ("architecture", "structure", "hard-surface"),
    "arch":         ("architecture", "structure", "hard-surface"),
    "stadium":      ("architecture", "building", "large", "sports"),
    "cabin":        ("architecture", "building", "small", "outdoor"),
    "cottage":      ("architecture", "building", "small"),
    "lighthouse":   ("architecture", "building", "tall", "coastal"),
    "barn":         ("architecture", "building", "rural"),
    "garage":       ("architecture", "building", "automotive"),
    "apartment":    ("architecture", "building", "urban"),
    "ruins":        ("architecture", "structure", "aged"),
    "pyramid":      ("architecture", "structure", "ancient"),
    "corridor":     ("architecture", "interior", "structure"),
    "doorway":      ("architecture", "structure", "opening"),
    "column":       ("architecture", "structure", "hard-surface"),

    # ── FURNITURE / INTERIOR ─────────────────────────────────────────────
    "chair":        ("furniture", "interior", "hard-surface", "functional"),
    "table":        ("furniture", "interior", "hard-surface", "functional"),
    "sofa":         ("furniture", "interior", "soft", "functional"),
    "couch":        ("furniture", "interior", "soft", "functional"),
    "desk":         ("furniture", "interior", "hard-surface", "functional"),
    "bookshelf":    ("furniture", "interior", "hard-surface"),
    "cabinet":      ("furniture", "interior", "hard-surface"),
    "wardrobe":     ("furniture", "interior", "hard-surface"),
    "bed":          ("furniture", "interior", "soft", "functional"),
    "bench":        ("furniture", "interior", "hard-surface", "outdoor"),
    "stool":        ("furniture", "interior", "hard-surface"),
    "lamp":         ("furniture", "interior", "light", "functional"),
    "shelf":        ("furniture", "interior", "hard-surface"),
    "drawer":       ("furniture", "interior", "hard-surface"),
    "armchair":     ("furniture", "interior", "soft", "functional"),

    # ── CHARACTERS / CREATURES / ANIMALS ─────────────────────────────────
    "character":    ("character", "organic", "humanoid"),
    "human":        ("character", "humanoid", "organic"),
    "person":       ("character", "humanoid", "organic"),
    "figure":       ("character", "humanoid", "organic"),
    "warrior":      ("character", "humanoid", "armor", "weapon"),
    "knight":       ("character", "humanoid", "armor", "medieval"),
    "soldier":      ("character", "humanoid", "armor", "military"),
    "wizard":       ("character", "humanoid", "fantasy"),
    "monster":      ("creature", "character", "organic", "fantasy"),
    "creature":     ("organic", "character", "fantasy"),
    "alien":        ("creature", "character", "sci-fi", "organic"),
    "dragon":       ("creature", "fantasy", "organic", "wings"),
    "dinosaur":     ("creature", "organic", "large"),
    "animal":       ("creature", "organic", "nature"),
    "elephant":     ("animal", "creature", "organic", "large"),
    "horse":        ("animal", "creature", "organic"),
    "dog":          ("animal", "creature", "organic"),
    "cat":          ("animal", "creature", "organic"),
    "bird":         ("animal", "creature", "organic", "wings"),
    "eagle":        ("animal", "creature", "organic", "wings"),
    "crow":         ("animal", "creature", "organic", "wings"),
    "fish":         ("animal", "creature", "water", "organic"),
    "shark":        ("animal", "creature", "water", "organic"),
    "snake":        ("animal", "creature", "organic", "elongated"),
    "spider":       ("animal", "creature", "organic", "insect"),
    "insect":       ("animal", "creature", "organic", "small"),
    "bee":          ("animal", "insect", "organic", "wings"),
    "wolf":         ("animal", "creature", "organic"),
    "bear":         ("animal", "creature", "organic", "large"),
    "lion":         ("animal", "creature", "organic"),
    "gorilla":      ("animal", "creature", "organic"),
    "robot":        ("mechanical", "hard-surface", "machine", "industrial"),
    "mech":         ("mechanical", "hard-surface", "machine", "industrial"),
    "android":      ("mechanical", "character", "humanoid", "sci-fi"),
    "cyborg":       ("mechanical", "character", "humanoid", "sci-fi"),

    # ── WEAPONS / PROPS ──────────────────────────────────────────────────
    "sword":        ("weapon", "medieval", "hard-surface"),
    "axe":          ("weapon", "medieval", "hard-surface"),
    "gun":          ("weapon", "military", "hard-surface", "mechanical"),
    "shield":       ("armor", "medieval", "hard-surface"),
    "armor":        ("hard-surface", "medieval", "character"),
    "hammer":       ("weapon", "tool", "hard-surface"),
    "bow":          ("weapon", "medieval"),
    "staff":        ("weapon", "fantasy"),
    "knife":        ("weapon", "tool", "hard-surface"),
    "spear":        ("weapon", "medieval", "hard-surface"),
    "wand":         ("weapon", "fantasy", "magic"),

    # ── FOOD / ORGANIC OBJECTS ───────────────────────────────────────────
    "food":         ("organic", "edible", "scene"),
    "food_prop":    ("organic", "edible", "prop"),
    "fruit":        ("organic", "food", "round"),
    "apple":        ("fruit", "organic", "food"),
    "bread":        ("food", "organic"),
    "donut":        ("food", "organic", "round"),
    "cake":         ("food", "organic"),
    "cup":          ("container", "functional", "interior"),
    "bottle":       ("container", "functional", "hard-surface"),
    "bowl":         ("container", "functional", "interior"),
    "plate":        ("container", "functional", "tableware"),
    "vase":         ("container", "decorative", "interior"),
    "barrel":       ("container", "hard-surface"),
    "candle":       ("decorative", "interior", "light"),
    "flask":        ("container", "science", "hard-surface"),

    # ── SCI-FI / MECHANICAL / INDUSTRIAL ─────────────────────────────────
    "scifi":        ("sci-fi", "mechanical", "hard-surface"),
    "cyberpunk":    ("sci-fi", "urban", "mechanical", "dark"),
    "futuristic":   ("sci-fi", "hard-surface", "mechanical"),
    "laser":        ("sci-fi", "mechanical", "weapon"),
    "turret":       ("weapon", "mechanical", "hard-surface"),
    "cannon":       ("weapon", "mechanical", "hard-surface"),
    "engine":       ("mechanical", "machine", "industrial"),
    "gear":         ("mechanical", "machine", "functional"),
    "pipe":         ("mechanical", "industrial", "hard-surface"),
    "antenna":      ("mechanical", "tall", "hard-surface"),
    "satellite":    ("mechanical", "spacecraft", "sci-fi"),
    "station":      ("sci-fi", "architecture", "mechanical"),

    # ── ABSTRACT / GEOMETRIC ─────────────────────────────────────────────
    "abstract":     ("geometric", "artistic", "procedural"),
    "geometric":    ("abstract", "hard-surface", "procedural"),
    "organic":      ("nature", "flowing", "irregular"),
    "low-poly":     ("geometric", "stylized", "game-ready"),
    "stylized":     ("artistic", "game-ready"),
    "pattern":      ("geometric", "abstract", "procedural"),
    "fractal":      ("abstract", "procedural", "geometric"),

    # ── MATERIAL / FINISH HINTS ──────────────────────────────────────────
    "matte":        ("surface", "material", "finish"),
    "glossy":       ("surface", "material", "reflective"),
    "metallic":     ("surface", "material", "hard-surface"),
    "wooden":       ("material", "organic", "hard-surface"),
    "stone":        ("material", "terrain", "hard-surface"),
    "glass":        ("material", "transparent", "hard-surface"),
    "ceramic":      ("material", "hard-surface"),
    "plastic":      ("material", "hard-surface"),
    "leather":      ("material", "organic", "surface"),
    "fabric":       ("material", "soft", "surface"),
    "black":        ("dark", "material", "finish"),
    "shiny":        ("surface", "material", "reflective"),
    "rusty":        ("material", "aged", "metallic"),
    "ancient":      ("aged", "material", "historic"),
    "carved":       ("material", "hard-surface", "detailed"),
}


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", (text or "").lower())


def infer_semantic_hints(text: str, *, max_hints: int = 8) -> list[str]:
    words = _tokenize_words(text)
    hints: list[str] = []
    seen = set(words)
    for word in words:
        for hint in _THEME_MAP.get(word, ()):
            if hint in seen:
                continue
            seen.add(hint)
            hints.append(hint)
            if len(hints) >= max_hints:
                return hints
    return hints


def enrich_prompt_text(
    text: str,
    *,
    max_hints: int = 8,
    stochastic: bool = False,
    keep_prob: float = 0.7,
    rng: random.Random | None = None,
) -> str:
    """Append compact implied-theme hints to prompt text.

    In training, use stochastic=True to vary which hints are kept. In inference,
    keep deterministic defaults.
    """
    base = (text or "").strip()
    if not base:
        return base
    hints = infer_semantic_hints(base, max_hints=max_hints)
    if not hints:
        return base

    if stochastic:
        rr = rng if rng is not None else random
        filtered = [h for h in hints if rr.random() < float(keep_prob)]
        if filtered:
            hints = filtered

    return f"{base} | implied: {' '.join(hints)}"


def expand_prompt_set(prompts: Iterable[str], *, max_hints: int = 8) -> list[str]:
    out: list[str] = []
    for prompt in prompts:
        out.append(prompt)
        out.append(enrich_prompt_text(prompt, max_hints=max_hints))
    return out
