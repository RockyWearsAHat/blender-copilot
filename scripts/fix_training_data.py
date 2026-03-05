#!/usr/bin/env python3
"""Fix training data: filter bad samples, rewrite labels with LLM, rebuild cache.

This script does three things:
  1. FILTER: Remove samples with mesh_tokens at max length (8102) — these are
     truncated complex meshes the model can never learn to reproduce
  2. RELABEL: Use local Ollama LLM to rewrite garbage labels into natural prompts
  3. REBUILD: Save cleaned data back to cache, ready for training

Usage:
    python scripts/fix_training_data.py                    # dry-run (report only)
    python scripts/fix_training_data.py --apply            # actually fix data
    python scripts/fix_training_data.py --apply --relabel  # fix + LLM relabeling
    python scripts/fix_training_data.py --max-tokens 4000  # custom threshold
"""

import argparse
import glob
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

import requests
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/processed/.mesh_cache")
BACKUP_DIR = Path("data/processed/.mesh_cache_backup")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5vl:32b"


def load_bpe_tokenizer():
    """Load BPE tokenizer for decoding text IDs back to strings."""
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.Load("data/datasets/geometry/bpe_tokenizer/tokenizer.model")
        return sp
    except Exception as e:
        logger.error(f"Cannot load BPE tokenizer: {e}")
        return None


def decode_label(sp, text_ids, text_mask):
    """Decode text_ids back to a string label."""
    text_len = int(text_mask.sum().item())
    ids = text_ids[:text_len].tolist()
    try:
        return sp.DecodeIds(ids)
    except Exception:
        return "[decode-error]"


def is_garbage_label(label: str) -> bool:
    """Check if a label is low-quality garbage that needs rewriting."""
    low = label.lower().strip()

    garbage_patterns = [
        "small 3d mesh",
        "high-poly small",
        "medium-poly small",
        "low-poly small 3d",
        "material 3d mesh",
        "(detailed mesh)",
        "(high-detail mesh)",
        "scene root material",
        "lambert",
        "material0",
        "material 0",
        "wire 1 material",
    ]
    for pat in garbage_patterns:
        if pat in low:
            return True

    if len(low) < 3:
        return True

    if low.replace(" ", "").isdigit():
        return True

    words = low.split()
    if len(words) == 1 and not words[0].isalpha():
        return True

    return False


def rewrite_label_with_llm(original_label: str, mesh_token_count: int) -> str:
    """Use Ollama LLM to rewrite a garbage label into a natural prompt."""
    size_hint = "simple" if mesh_token_count < 1000 else "medium-complexity" if mesh_token_count < 4000 else "detailed"

    prompt = f"""You are rewriting 3D model labels into natural text-to-3D prompts.

Original label: "{original_label}"
Mesh complexity: {size_hint} ({mesh_token_count} tokens)

Rules:
- Write a SHORT natural English prompt (3-10 words) that a user would type to generate this 3D object
- If the original label contains a recognizable object name, keep it
- If the label is total garbage (random IDs, material names), infer the most likely object from any clues
- Do NOT include technical terms like "mesh", "3d", "poly", "material", "detailed"
- Examples of good outputs: "a wooden chair", "red sports car", "stone castle tower", "ceramic coffee mug"
- Output ONLY the prompt, nothing else

Rewritten prompt:"""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 30},
        }, timeout=30)
        if resp.status_code == 200:
            result = resp.json().get("response", "").strip().strip('"').strip("'")
            if 3 <= len(result) <= 80:
                return result
    except Exception as e:
        logger.debug(f"LLM relabel failed: {e}")

    return original_label


def encode_label(sp, label: str, max_length: int = 256):
    """Encode a text label back to text_ids and text_mask tensors."""
    ids = sp.EncodeAsIds(label)
    ids = ids[:max_length]
    mask = [1.0] * len(ids)
    ids += [0] * (max_length - len(ids))
    mask += [0.0] * (max_length - len(mask))
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.float32),
    )


def main():
    parser = argparse.ArgumentParser(description="Fix training data quality")
    parser.add_argument("--apply", action="store_true", help="Actually modify files (default: dry-run)")
    parser.add_argument("--relabel", action="store_true", help="Use LLM to rewrite garbage labels")
    parser.add_argument("--max-tokens", type=int, default=4000,
                        help="Max mesh tokens to keep (default: 4000, removes ~87%% of data)")
    parser.add_argument("--backup", action="store_true", default=True,
                        help="Backup cache before modifying")
    args = parser.parse_args()

    sp = load_bpe_tokenizer()
    if sp is None:
        return

    cache_files = sorted(glob.glob(str(CACHE_DIR / "*.pt")))
    logger.info(f"Found {len(cache_files)} cache files")

    stats = {
        "total": len(cache_files),
        "kept": 0,
        "filtered_too_long": 0,
        "filtered_too_short": 0,
        "relabeled": 0,
        "label_ok": 0,
        "label_garbage": 0,
    }
    kept_files = []
    actions = []

    for f in cache_files:
        try:
            d = torch.load(f, weights_only=False)
            item = d[0] if isinstance(d, list) else d
        except Exception:
            stats["filtered_too_short"] += 1
            actions.append(("DELETE", f, "corrupt file"))
            continue

        mesh_tokens = item["mesh_tokens"]
        mesh_len = mesh_tokens.numel()

        if mesh_len > args.max_tokens:
            stats["filtered_too_long"] += 1
            actions.append(("DELETE", f, f"too long: {mesh_len} tokens"))
            continue

        if mesh_len < 20:
            stats["filtered_too_short"] += 1
            actions.append(("DELETE", f, f"too short: {mesh_len} tokens"))
            continue

        label = decode_label(sp, item["text_ids"], item["text_mask"])

        if is_garbage_label(label):
            stats["label_garbage"] += 1
            if args.relabel:
                new_label = rewrite_label_with_llm(label, mesh_len)
                if new_label != label:
                    stats["relabeled"] += 1
                    actions.append(("RELABEL", f, f"{label!r} -> {new_label!r}"))
                    if args.apply:
                        new_ids, new_mask = encode_label(sp, new_label)
                        item["text_ids"] = new_ids
                        item["text_mask"] = new_mask
                        save_data = [item] if isinstance(d, list) else item
                        torch.save(save_data, f)
                else:
                    actions.append(("GARBAGE_KEPT", f, f"LLM couldn't improve: {label!r}"))
            else:
                actions.append(("GARBAGE", f, f"garbage label: {label!r}"))
        else:
            stats["label_ok"] += 1

        stats["kept"] += 1
        kept_files.append(f)

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS ({'DRY RUN' if not args.apply else 'APPLIED'}):")
    logger.info(f"{'='*60}")
    logger.info(f"  Total cache files:     {stats['total']}")
    logger.info(f"  Kept:                  {stats['kept']} ({100*stats['kept']/max(1,stats['total']):.1f}%)")
    logger.info(f"  Filtered (too long):   {stats['filtered_too_long']} (>{args.max_tokens} tokens)")
    logger.info(f"  Filtered (too short):  {stats['filtered_too_short']}")
    logger.info(f"  Labels OK:             {stats['label_ok']}")
    logger.info(f"  Labels garbage:        {stats['label_garbage']}")
    if args.relabel:
        logger.info(f"  Labels rewritten:      {stats['relabeled']}")

    if not args.apply:
        logger.info(f"\nDry run — no files modified. Run with --apply to execute.")
        logger.info(f"\nSample actions:")
        for action_type, path, reason in actions[:30]:
            logger.info(f"  [{action_type}] {Path(path).name}: {reason}")
        return

    if args.backup and CACHE_DIR.exists():
        logger.info(f"\nBacking up cache to {BACKUP_DIR}...")
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(CACHE_DIR, BACKUP_DIR)
        logger.info("Backup complete")

    deleted = 0
    for action_type, path, reason in actions:
        if action_type == "DELETE":
            try:
                os.remove(path)
                deleted += 1
            except Exception:
                pass

    logger.info(f"\nDeleted {deleted} files")
    logger.info(f"Remaining: {stats['kept']} cache files")
    logger.info(f"\nDone. Restart training to use cleaned data.")


if __name__ == "__main__":
    main()
