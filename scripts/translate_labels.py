#!/usr/bin/env python3
"""Batch-translate foreign words in training labels using Qwen via Ollama.

Scans all source JSON files, collects unique words that may be foreign,
sends them to Qwen for translation, and updates data/label_translations.json.

Usage:
    python scripts/translate_labels.py              # Preview only
    python scripts/translate_labels.py --apply      # Update the cache file
    python scripts/translate_labels.py --model qwen2.5:14b  # Use specific model
"""

import argparse
import json
import re
import sys
import glob
import os
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

TRANSLATIONS_PATH = BASE / "data" / "label_translations.json"

# Common English words that should NOT be translated
_ENGLISH_WORDS = frozenset({
    # Colors
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'white', 'black',
    'brown', 'grey', 'gray', 'pink', 'gold', 'silver', 'dark', 'light',
    # Materials
    'wood', 'metal', 'glass', 'stone', 'iron', 'steel', 'copper', 'chrome',
    'plastic', 'leather', 'fabric', 'rubber', 'concrete', 'brick', 'marble',
    'paper', 'cloth', 'skin', 'hair', 'fur', 'wool', 'silk', 'cotton',
    'paint', 'rust', 'dirt', 'dust', 'mud', 'sand', 'grass', 'moss',
    # Objects
    'wall', 'floor', 'roof', 'door', 'window', 'table', 'chair', 'lamp',
    'car', 'wheel', 'tire', 'seat', 'body', 'head', 'eye', 'face', 'hand',
    'leg', 'arm', 'foot', 'bone', 'tooth', 'wing', 'tail', 'horn',
    'tree', 'leaf', 'leaves', 'bush', 'flower', 'plant', 'branch', 'trunk',
    'rock', 'mountain', 'ground', 'water', 'fire', 'ice', 'snow',
    'box', 'tube', 'pipe', 'wire', 'rope', 'chain', 'ring', 'disc',
    'bolt', 'screw', 'nail', 'hook', 'spring', 'gear', 'belt',
    'book', 'page', 'pen', 'pencil', 'brush', 'frame', 'panel',
    'light', 'spot', 'area', 'sun', 'moon', 'star', 'sky', 'cloud',
    # Geometry
    'cube', 'sphere', 'cylinder', 'plane', 'cone', 'torus', 'circle',
    'grid', 'curve', 'path', 'edge', 'vertex', 'face', 'mesh',
    # Parts
    'top', 'bottom', 'front', 'back', 'left', 'right', 'side', 'center',
    'inner', 'outer', 'upper', 'lower', 'mid', 'base', 'cap', 'tip',
    # Actions/states
    'smooth', 'rough', 'flat', 'round', 'sharp', 'thick', 'thin',
    'big', 'small', 'tall', 'short', 'wide', 'narrow', 'long',
    'old', 'new', 'clean', 'dirty', 'wet', 'dry', 'hot', 'cold',
    # Common 3D terms
    'emission', 'emissive', 'glossy', 'matte', 'satin', 'specular',
    'diffuse', 'normal', 'bump', 'displacement', 'ambient', 'occlusion',
    'transparency', 'translucent', 'reflection', 'refraction',
    # Common short words
    'the', 'and', 'for', 'with', 'from', 'into', 'onto', 'over',
    'set', 'mix', 'add', 'cut', 'end', 'out', 'off', 'low', 'high',
    'bay', 'bar', 'beam', 'pole', 'post', 'rod', 'log', 'plank',
    'hay', 'oak', 'ash', 'elm', 'pine', 'cedar', 'birch', 'maple',
    'pier', 'dock', 'barn', 'shed', 'hut', 'tent', 'cage', 'den',
})


def load_existing_translations():
    """Load the current translation cache."""
    if TRANSLATIONS_PATH.exists():
        return json.loads(TRANSLATIONS_PATH.read_text())
    return {}


def collect_unique_words():
    """Scan all source JSONs and collect unique words from object/material/meta names."""
    words = Counter()
    source_dirs = [
        BASE / "data" / "processed" / d
        for d in ("blender_official", "blendswap", "objaverse", "smutbase",
                  "github", "open3dlab", "youtube")
    ]

    for src_dir in source_dirs:
        if not src_dir.exists():
            continue
        for f in sorted(src_dir.glob("*.json")):
            if f.stat().st_size > 50_000_000:
                continue
            try:
                data = json.load(open(f))
            except Exception:
                continue

            # File-level metadata (Objaverse name, tags, categories)
            for meta_key in ("name", "label", "title", "description"):
                meta_val = data.get(meta_key, "") or ""
                for w in re.sub(r'[._\-\d]+', ' ', meta_val.lower()).split():
                    if len(w) >= 3:
                        words[w] += 1
            for tag in data.get("tags", []):
                if isinstance(tag, str):
                    for w in re.sub(r'[._\-\d]+', ' ', tag.lower()).split():
                        if len(w) >= 3:
                            words[w] += 1
            for cat in data.get("categories", []):
                if isinstance(cat, str):
                    for w in re.sub(r'[._\-\d]+', ' ', cat.lower()).split():
                        if len(w) >= 3:
                            words[w] += 1

            for obj in data.get('objects', []):
                # Object names
                name = obj.get('name', '') or ''
                for w in re.sub(r'[._\-\d]+', ' ', name.lower()).split():
                    if len(w) >= 3:
                        words[w] += 1
                # Material names
                for mat in obj.get('materials', []):
                    mname = mat.get('name', '') or ''
                    for w in re.sub(r'[._\-\d]+', ' ', mname.lower()).split():
                        if len(w) >= 3:
                            words[w] += 1

    return words


def find_untranslated_foreign(words, existing):
    """Find words that look foreign (not in English word list, not already translated)."""
    candidates = []
    for word, count in words.most_common():
        if word in existing:
            continue
        if word in _ENGLISH_WORDS:
            continue
        if count < 2:  # Only bother with words that appear multiple times
            continue
        candidates.append((word, count))
    return candidates


def batch_translate_with_qwen(words, model="qwen2.5:14b"):
    """Send a batch of words to Qwen for translation."""
    try:
        import requests
    except ImportError:
        print("ERROR: requests library not available")
        return {}

    word_list = [w for w, _ in words[:200]]  # Cap at 200 words per batch
    prompt = f"""I have a list of words extracted from 3D model files (Blender .blend files) that may be in various languages (German, French, Italian, Polish, Portuguese, Spanish, Japanese, etc.) or may be English.

For each word, tell me:
1. If it's a non-English word, translate it to English
2. If it's already English or a proper noun/brand name, respond with "ENGLISH"
3. If it's a technical/made-up term with no translation, respond with "SKIP"

Respond ONLY with a JSON object mapping each word to its translation, "ENGLISH", or "SKIP". No other text.

Words: {json.dumps(word_list)}"""

    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.1}},
            timeout=120,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")

        # Extract JSON from response
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            translations = {}
            for word, trans in result.items():
                word = word.lower().strip()
                trans = trans.strip()
                if trans not in ("ENGLISH", "SKIP", "english", "skip", ""):
                    translations[word] = trans.lower()
            return translations
    except Exception as e:
        print(f"Qwen translation failed: {e}")

    return {}


def main():
    parser = argparse.ArgumentParser(description="Translate foreign words in labels via Qwen")
    parser.add_argument("--apply", action="store_true", help="Save translations to cache file")
    parser.add_argument("--model", default="qwen2.5:14b", help="Ollama model to use")
    args = parser.parse_args()

    print("Loading existing translations...")
    existing = load_existing_translations()
    print(f"  {len(existing)} cached translations")

    print("Scanning source files for unique words...")
    words = collect_unique_words()
    print(f"  {len(words)} unique words found")

    print("Finding untranslated candidates...")
    candidates = find_untranslated_foreign(words, existing)
    print(f"  {len(candidates)} candidates for translation")

    if not candidates:
        print("No new words to translate!")
        return

    print(f"\nTop 50 candidates (word: count):")
    for word, count in candidates[:50]:
        print(f"  {count:5d}x  {word}")

    print(f"\nSending {min(len(candidates), 200)} words to Qwen ({args.model})...")
    new_translations = batch_translate_with_qwen(candidates, model=args.model)
    print(f"  Got {len(new_translations)} translations")

    if new_translations:
        print("\nNew translations:")
        for word, trans in sorted(new_translations.items()):
            print(f"  {word:20s} → {trans}")

    if args.apply and new_translations:
        existing.update(new_translations)
        TRANSLATIONS_PATH.write_text(json.dumps(existing, indent=4, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(existing)} total translations to {TRANSLATIONS_PATH}")
    elif new_translations:
        print("\nDry run — use --apply to save translations")


if __name__ == "__main__":
    main()
