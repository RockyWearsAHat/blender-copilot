#!/usr/bin/env python3
"""Fast eval: imitation accuracy against the synthetic teacher.

This is the quickest way to see if training is working.

Usage:
  /path/to/python scripts/eval_imitation.py --ckpt checkpoints/policy_goal/latest.pt --device auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.policy_transformer import PolicyTransformer
from policy.dataset import SyntheticImitationStream


def _detect_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--batches", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = _detect_device(args.device)
    payload = torch.load(args.ckpt, map_location="cpu")
    cfg = payload.get("config", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    model = PolicyTransformer(
        state_dim=int(model_cfg.get("state_dim", 12)),
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        action_type_vocab=int(model_cfg.get("action_type_vocab", 11)),
        action_param_vocab=int(model_cfg.get("action_param_vocab", 32)),
        max_seq_len=int(model_cfg.get("max_seq_len", 128)),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    stream = SyntheticImitationStream(
        seed=int(args.seed),
        seq_len=int(data_cfg.get("seq_len", 64)),
        batch_size=int(data_cfg.get("batch_size", 32)),
        state_dim=int(model_cfg.get("state_dim", 12)),
    )
    it = iter(stream)

    correct_type = 0
    correct_param = 0
    total = 0

    with torch.no_grad():
        for _ in range(int(args.batches)):
            batch = next(it)
            states = batch.states.to(device=device, dtype=torch.float32)
            a_type = batch.action_type.to(device=device)
            a_param = batch.action_param.to(device=device)

            type_logits, param_logits = model(states)
            pred_type = type_logits.argmax(dim=-1)
            pred_param = param_logits.argmax(dim=-1)

            correct_type += int((pred_type == a_type).sum().item())
            correct_param += int((pred_param == a_param).sum().item())
            total += int(a_type.numel())

    print(
        {
            "type_acc": correct_type / max(1, total),
            "param_acc": correct_param / max(1, total),
            "tokens": total,
            "device": device.type,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
