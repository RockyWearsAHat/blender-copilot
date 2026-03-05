#!/usr/bin/env python3
"""Quick evaluation for the architecture-compliant policy model.

Loads `checkpoints/policy/latest.pt` and rolls out greedy actions in the
compact stats environment.

This is a sanity check (not a full Blender/topology evaluation).
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


def _detect_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    from models.policy_transformer import PolicyTransformer
    from policy.actions import Action
    from policy.env import MeshStatsEnv
    from policy.state import Goal, normalize_goal, normalize_state_vector

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/policy/latest.pt"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = _detect_device(args.device)
    payload = torch.load(args.ckpt, map_location="cpu")
    cfg = payload.get("config", {})

    model_cfg = cfg.get("model", {})
    model = PolicyTransformer(
        state_dim=int(model_cfg.get("state_dim", 10)),
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

    rng = np.random.default_rng(args.seed)
    env = MeshStatsEnv(seed=args.seed)

    done_count = 0
    total_reward = 0.0

    for _ in range(args.episodes):
        goal = Goal(
            target_vertex_count=int(rng.integers(200, 5000)),
            target_symmetry=float(rng.uniform(0.4, 0.9)),
        )
        st = env.reset()
        ep_reward = 0.0
        done = False

        for t in range(args.horizon):
            vec = normalize_state_vector(st.as_vector())
            gvec = normalize_goal(goal)
            feat = np.concatenate([vec, gvec], axis=0).astype(np.float32, copy=False)
            states = torch.from_numpy(feat[None, None, :]).to(device=device, dtype=torch.float32)  # (1,1,S)
            with torch.no_grad():
                type_logits, param_logits = model(states)
                a_type = int(type_logits[0, 0].argmax().item())
                a_param = int(param_logits[0, 0].argmax().item())

            act = Action(a_type, a_param)
            sr = env.step(st, act, goal)
            st = sr.state
            ep_reward += float(sr.reward)
            if sr.done:
                done = True
                break

        total_reward += ep_reward
        done_count += int(done)

    print(
        {
            "episodes": args.episodes,
            "done_rate": done_count / max(1, args.episodes),
            "avg_reward": total_reward / max(1, args.episodes),
            "device": device.type,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
