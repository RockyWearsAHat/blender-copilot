#!/usr/bin/env python3
"""Quickly inspect first-step action-type probabilities for prompts.

This is a lightweight debugging helper for prompt-conditioning.
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
from policy.prompt_parser import parse_prompt_to_goal
from policy.state import Goal, MeshState, normalize_goal, normalize_state_vector


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", action="append", default=[])
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    mcfg = ckpt.get("config", {}).get("model", {})

    model = PolicyTransformer(
        state_dim=int(mcfg.get("state_dim", 28)),
        hidden_size=int(mcfg.get("hidden_size", 256)),
        num_layers=int(mcfg.get("num_layers", 4)),
        num_heads=int(mcfg.get("num_heads", 8)),
        dropout=float(mcfg.get("dropout", 0.1)),
        action_type_vocab=int(mcfg.get("action_type_vocab", 11)),
        action_param_vocab=int(mcfg.get("action_param_vocab", 32)),
        max_seq_len=int(mcfg.get("max_seq_len", 64)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    st = MeshState(8, 6, 12, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0)
    vec = normalize_state_vector(st.as_vector())

    prompts = list(args.prompt) or [
        "the default cube",
        "a cube",
        "low poly lamborghini",
        "a low poly lamborghini",
    ]

    for text in prompts:
        feats = parse_prompt_to_goal(text).goal.text_features
        goal = Goal(target_vertex_count=0, target_symmetry=0.0, text_features=feats)
        feat = np.concatenate([vec, normalize_goal(goal)], axis=0).astype(np.float32)
        x = torch.from_numpy(feat).view(1, 1, -1)

        with torch.no_grad():
            tl, _ = model(x)
            probs = torch.softmax(tl[0, 0], dim=-1)
            v, i = torch.topk(probs, k=min(5, probs.numel()))

        top5 = list(zip(i.tolist(), [float(z) for z in v]))
        print(text, top5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
