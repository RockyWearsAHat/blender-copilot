#!/usr/bin/env python3
"""Regenerate geometry val/test splits for the *current* mesh tokenizer.

Problem this solves:
- Older `data/datasets/geometry/val.jsonl` files may contain mesh token IDs from
  an 8192-vocab tokenizer.
- When running with a 1024-vocab tokenizer, those IDs are out of range and will
  crash evaluation (or silently disable it).

This script regenerates `val.jsonl` and `test.jsonl` from synthetic primitives
(using `processing.generate_synthetic`) so that validation is always compatible
with the current `tokenization.vocab_size`.

It will back up incompatible existing files to:
- val_legacy_vocab<maxToken+1>.jsonl
- test_legacy_vocab<maxToken+1>.jsonl

Usage:
  .venv/bin/python scripts/regenerate_geometry_splits.py \
    --config config.retrain_v1024_focal.yaml \
    --num-val 512 --num-test 256
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


# Allow running as a standalone script from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "PyYAML is required to read config files. Install with: pip install pyyaml"
        ) from e
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _max_token_in_jsonl(path: Path, max_lines: int | None = None) -> int:
    max_tok = -1
    try:
        with path.open("r", encoding="utf-8") as f:
            for i, ln in enumerate(f):
                if max_lines is not None and i >= max_lines:
                    break
                ln = ln.strip()
                if not ln:
                    continue
                ex = json.loads(ln)
                toks = ex.get("tokens") or []
                if toks:
                    try:
                        max_tok = max(max_tok, int(max(toks)))
                    except Exception:
                        pass
    except FileNotFoundError:
        return -1
    return max_tok


def _backup_if_incompatible(path: Path, vocab_size: int) -> None:
    if not path.exists():
        return
    max_tok = _max_token_in_jsonl(path, max_lines=500)
    if max_tok < 0:
        return
    if max_tok < vocab_size:
        return

    # Best-effort guess of legacy vocab size (tokens are 0..vocab-1).
    legacy_vocab = max_tok + 1
    backup_path = path.with_name(
        f"{path.stem}_legacy_vocab{legacy_vocab}{path.suffix}"
    )
    if backup_path.exists():
        # Don't overwrite an existing backup.
        return

    path.rename(backup_path)


def _write_split(
    out_path: Path,
    *,
    tokenizer,
    num_examples: int,
    max_mesh_tokens: int,
    seed: int,
) -> None:
    from processing.generate_synthetic import (
        SHAPE_SPECS,
        COMPOSITE_SPECS,
        apply_rotation,
        generate_label,
        normalize_mesh,
    )

    rng = random.Random(seed)
    all_specs: dict[str, Any] = {**SHAPE_SPECS, **COMPOSITE_SPECS}
    keys = list(all_specs.keys())

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    n_written = 0

    with tmp_path.open("w", encoding="utf-8") as f:
        # Keep going until we successfully write num_examples.
        attempts = 0
        while n_written < num_examples:
            attempts += 1
            if attempts > num_examples * 50:
                raise RuntimeError(
                    f"Too many failed attempts generating {out_path.name} "
                    f"({n_written}/{num_examples} written)."
                )

            shape_key = rng.choice(keys)
            spec = all_specs[shape_key]
            params = spec["params"]()
            verts, faces = spec["generator"](params)
            if len(verts) < 3 or len(faces) < 1:
                continue

            # Deterministic but varied rotations (avoid axis-aligned token collapse).
            if rng.random() < 0.7:
                verts = apply_rotation(
                    verts,
                    angle_deg=rng.uniform(0.0, 360.0),
                    axis=rng.choice(["x", "y", "z"]),
                )

            verts = normalize_mesh(verts, target_range=(-1.0, 1.0))

            tokens = tokenizer.encode_mesh(verts, faces)
            if len(tokens) < 3 or len(tokens) > max_mesh_tokens:
                continue

            label = generate_label(shape_key, params)

            ex = {
                "text": label,
                "tokens": tokens,
                "num_vertices": int(len(verts)),
                "num_faces": int(len(faces)),
                "source": "synthetic",
            }
            f.write(json.dumps(ex) + "\n")
            n_written += 1

    tmp_path.replace(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (used for vocab_size, coord range, etc).",
    )
    ap.add_argument(
        "--output-dir",
        default="data/datasets/geometry",
        help="Geometry dataset directory containing val.jsonl/test.jsonl.",
    )
    ap.add_argument("--num-val", type=int, default=512)
    ap.add_argument("--num-test", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()

    config = _load_yaml(Path(args.config))
    tok_cfg = config.get("tokenization", {}) or {}
    train_cfg = config.get("training", {}) or {}

    vocab_size = int(tok_cfg.get("vocab_size", 8192))
    coord_range = tuple(tok_cfg.get("coordinate_range", (-1.0, 1.0)))
    max_faces = int(tok_cfg.get("max_faces", 400))

    max_mesh_tokens = int(
        config.get("unified", {})
        .get("geometry", {})
        .get("max_seq_length", 3602)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from processing.mesh_tokenizer import MeshTokenizer

    tokenizer = MeshTokenizer(
        vocab_size=vocab_size,
        coord_range=coord_range,
        max_faces=max_faces,
    )

    # Back up old incompatible splits.
    _backup_if_incompatible(out_dir / "val.jsonl", vocab_size)
    _backup_if_incompatible(out_dir / "test.jsonl", vocab_size)

    # Generate new splits.
    _write_split(
        out_dir / "val.jsonl",
        tokenizer=tokenizer,
        num_examples=int(args.num_val),
        max_mesh_tokens=max_mesh_tokens,
        seed=int(args.seed),
    )
    _write_split(
        out_dir / "test.jsonl",
        tokenizer=tokenizer,
        num_examples=int(args.num_test),
        max_mesh_tokens=max_mesh_tokens,
        seed=int(args.seed) + 1,
    )

    # Sanity check.
    mx_val = _max_token_in_jsonl(out_dir / "val.jsonl")
    mx_test = _max_token_in_jsonl(out_dir / "test.jsonl")
    print(f"Wrote val.jsonl:  max_token={mx_val}  vocab_size={vocab_size}")
    print(f"Wrote test.jsonl: max_token={mx_test}  vocab_size={vocab_size}")

    if mx_val >= vocab_size or mx_test >= vocab_size:
        raise SystemExit("ERROR: regenerated splits still contain out-of-range tokens")

    # Optional: note eval cadence (just for user clarity).
    eval_every = train_cfg.get("eval_every")
    if eval_every is not None:
        print(f"Note: training eval_every={eval_every} (best.pt updates only on eval steps)")


if __name__ == "__main__":
    main()
