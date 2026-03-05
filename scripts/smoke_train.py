#!/usr/bin/env python3
"""Fast training smoke-test.

Goal:
- Validate that model + loss + optimizer run end-to-end on the target device
  (especially MPS) without NaNs/Inf/OOM.
- Does NOT depend on datasets or Blender; uses synthetic random tokens.

Usage:
  python scripts/smoke_train.py --config config.m3_mps_quick.yaml --device auto --steps 5

Notes:
- Uses the same forward path as training/train_unified.py (geometry only).
- Keeps tensors small by default; override --seq-len if desired.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _detect_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke-test a few training steps")
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--steps", type=int, default=5, help="Optimizer steps (not micro-steps)")
    p.add_argument("--grad-accum", type=int, default=4, help="Micro-steps per optimizer step")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=0, help="Override mesh seq length")
    args = p.parse_args()

    _set_seed(args.seed)
    device = _detect_device(args.device)

    cfg = yaml.safe_load(args.config.read_text())
    uni = cfg.get("unified", {})
    geo = uni.get("geometry", {})
    train = cfg.get("training", {})

    text_vocab = int(uni.get("text_vocab_size", 8000))
    mesh_vocab = int(geo.get("mesh_vocab_size", 8192))
    text_len = int(uni.get("text_max_length", 192))
    seq_len = int(args.seq_len or geo.get("max_seq_length", 1024))

    # Keep it reasonably small even if config is huge.
    seq_len = max(32, min(seq_len, 8192))
    text_len = max(8, min(text_len, 256))

    from models.unified import UnifiedBlenderModel

    model = UnifiedBlenderModel(cfg).to(device)
    model.train()

    lr = float(train.get("learning_rate", 1e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(train.get("weight_decay", 0.01)),
    )

    # Random tokens; avoid 0 since training ignores 0 as PAD.
    def rand_tokens(shape: tuple[int, ...], vocab: int) -> torch.Tensor:
        return torch.randint(1, max(2, vocab), shape, device=device, dtype=torch.long)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    total_micro = args.steps * args.grad_accum
    for micro in range(total_micro):
        text_ids = rand_tokens((args.batch_size, text_len), text_vocab)
        text_mask = torch.ones((args.batch_size, text_len), device=device, dtype=torch.float32)
        mesh_tokens = rand_tokens((args.batch_size, seq_len), mesh_vocab)

        inp = mesh_tokens[:, :-1]
        tgt = mesh_tokens[:, 1:]

        logits = model.forward_geometry(text_ids, text_mask, inp)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at micro-step {micro}: {loss.item()}")

        (loss / args.grad_accum).backward()

        if (micro + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            opt_step = (micro + 1) // args.grad_accum
            print(f"opt_step={opt_step} loss={loss.item():.4f} device={device}")

    print("OK: smoke train completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
