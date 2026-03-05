#!/usr/bin/env python3
"""Generate an action plan (JSON) from a trained policy checkpoint.

This runs in the regular Python venv (PyTorch available). The plan is then
executed inside Blender separately (so Blender doesn't need torch).

Usage:
  /path/to/python scripts/generate_policy_plan.py \
    --ckpt checkpoints/policy_goal/latest.pt \
    --steps 64 \
    --out data/eval/plan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.policy_transformer import PolicyTransformer
from policy.actions import PARAM_BINS
from policy.env import MeshStatsEnv
from policy.state import Goal, normalize_goal, normalize_state_vector


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
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal-vertices", type=int, default=1500)
    p.add_argument("--goal-symmetry", type=float, default=0.7)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = _detect_device(args.device)
    payload = torch.load(args.ckpt, map_location="cpu")
    cfg = payload.get("config", {})
    model_cfg = cfg.get("model", {})

    model = PolicyTransformer(
        state_dim=int(model_cfg.get("state_dim", 12)),
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        action_type_vocab=int(model_cfg.get("action_type_vocab", 11)),
        action_param_vocab=int(model_cfg.get("action_param_vocab", PARAM_BINS)),
        max_seq_len=int(model_cfg.get("max_seq_len", 128)),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    rng = np.random.default_rng(args.seed)
    env = MeshStatsEnv(seed=args.seed)

    goal = Goal(target_vertex_count=int(args.goal_vertices), target_symmetry=float(args.goal_symmetry))
    gvec = normalize_goal(goal)

    st = env.reset()
    steps: list[dict] = []

    for t in range(int(args.steps)):
        vec = normalize_state_vector(st.as_vector())
        feat = np.concatenate([vec, gvec], axis=0).astype(np.float32, copy=False)
        states = torch.from_numpy(feat[None, None, :]).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            type_logits, param_logits = model(states)

        a_type = int(type_logits[0, 0].argmax().item())
        a_param = int(param_logits[0, 0].argmax().item())

        steps.append({"action_type": a_type, "param": a_param})

        # advance toy env state so the plan is at least self-consistent
        sr = env.step(st, action=_ActionShim(a_type, a_param), goal=goal)
        st = sr.state

    plan = {
        "seed": int(args.seed),
        "goal": {"target_vertex_count": int(goal.target_vertex_count), "target_symmetry": float(goal.target_symmetry)},
        "steps": steps,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2))
    print(f"OK: wrote plan to {args.out}")
    return 0


class _ActionShim:
    def __init__(self, action_type: int, param: int):
        self.action_type = int(action_type)
        self.param = int(param)

    def clamp(self):
        a = int(self.action_type)
        p = int(self.param)
        if a < 0:
            a = 0
        if p < 0:
            p = 0
        if p >= PARAM_BINS:
            p = PARAM_BINS - 1
        return _ActionShim(a, p)


if __name__ == "__main__":
    raise SystemExit(main())
