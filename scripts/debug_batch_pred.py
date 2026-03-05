#!/usr/bin/env python3
"""Debug helper: compare target vs predicted action_type distribution on a batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.policy_transformer import PolicyTransformer
from policy.dataset import SyntheticImitationStream
from policy.goal_sampler import ListGoalSampler, load_goal_records_from_geometry_jsonl


def _dist(arr: np.ndarray) -> dict[int, int]:
    u, c = np.unique(arr, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--goals", default="data/datasets/prompt_goals_core_balanced.jsonl")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = ckpt.get("config", {}).get("model", {})

    model = PolicyTransformer(
        state_dim=int(mcfg.get("state_dim", 28)),
        hidden_size=int(mcfg.get("hidden_size", 256)),
        num_layers=int(mcfg.get("num_layers", 4)),
        num_heads=int(mcfg.get("num_heads", 8)),
        dropout=float(mcfg.get("dropout", 0.1)),
        action_type_vocab=int(mcfg.get("action_type_vocab", 12)),
        action_param_vocab=int(mcfg.get("action_param_vocab", 32)),
        max_seq_len=int(mcfg.get("max_seq_len", 64)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    recs = load_goal_records_from_geometry_jsonl(
        path=Path(args.goals),
        max_records=100000,
        min_vertices=4,
        max_vertices=20000,
    )

    stream = SyntheticImitationStream(
        seed=int(args.seed),
        seq_len=64,
        batch_size=int(args.batch),
        state_dim=int(mcfg.get("state_dim", 28)),
        goal_sampler=ListGoalSampler(recs),
        mask_numeric_goal_prob=1.0,
        trajectory_mode="forward",
        reverse_prob=0.2,
    )

    batch = next(iter(stream))

    x = batch.states[:, :1, :].to(dtype=torch.float32)
    with torch.no_grad():
        type_logits, _ = model(x)
        pred = type_logits.argmax(dim=-1).squeeze(1).cpu().numpy()

    tgt = batch.action_type[:, 0].cpu().numpy()

    print("tgt", _dist(tgt))
    print("pred", _dist(pred))
    print("acc", float((pred == tgt).mean()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
