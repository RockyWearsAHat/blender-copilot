#!/usr/bin/env python3
"""Debug helper: overfit a single fixed batch and report action-type accuracy.

If this can't overfit, something is wrong with the training signal or model wiring.
"""

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
    p.add_argument("--goals", default="data/datasets/prompt_goals_core_balanced.jsonl")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seq", type=int, default=64)
    args = p.parse_args()

    device = torch.device(args.device)

    recs = load_goal_records_from_geometry_jsonl(
        path=Path(args.goals),
        max_records=100000,
        min_vertices=4,
        max_vertices=20000,
    )
    stream = SyntheticImitationStream(
        seed=123,
        seq_len=int(args.seq),
        batch_size=int(args.batch),
        state_dim=28,
        goal_sampler=ListGoalSampler(recs),
        mask_numeric_goal_prob=1.0,
        trajectory_mode="forward",
        reverse_prob=0.2,
    )

    batch = next(iter(stream))
    states = batch.states.to(device=device, dtype=torch.float32)
    a_type = batch.action_type.to(device=device)

    all_tgt = batch.action_type.cpu().numpy().reshape(-1)
    print("tgt_all_tokens", _dist(all_tgt))

    model = PolicyTransformer(
        state_dim=28,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        dropout=0.0,
        action_type_vocab=12,
        action_param_vocab=32,
        max_seq_len=max(64, int(args.seq)),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0)
    loss_fn = torch.nn.CrossEntropyLoss()

    def eval_first_step():
        model.eval()
        with torch.no_grad():
            tl, _ = model(states[:, :1, :])
            pred = tl.argmax(dim=-1).squeeze(1)
            tgt = a_type[:, 0]
            acc = (pred == tgt).float().mean().item()
            return acc, pred.detach().cpu().numpy(), tgt.detach().cpu().numpy()

    acc0, p0, t0 = eval_first_step()
    print("init acc", acc0)
    print("init tgt", _dist(t0))
    print("init pred", _dist(p0))

    model.train()
    for i in range(1, int(args.steps) + 1):
        opt.zero_grad(set_to_none=True)
        tl, _ = model(states)
        bsz, seq_len = a_type.shape
        loss = loss_fn(tl.reshape(bsz * seq_len, -1), a_type.reshape(-1))
        loss.backward()
        opt.step()
        if i % 50 == 0:
            acc, _, _ = eval_first_step()
            print("step", i, "loss", float(loss.item()), "acc", acc)
            model.train()

    acc1, p1, t1 = eval_first_step()
    print("final acc", acc1)
    print("final pred", _dist(p1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
