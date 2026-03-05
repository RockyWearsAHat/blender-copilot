#!/usr/bin/env python3
"""Probe checkpoints without running Blender.

Prints the greedy (argmax) first action predicted by a checkpoint for a given
prompt and a canonical mesh-stat state. This is useful for quickly finding a
"good" checkpoint after an unstable training run.

Examples:
    # Probe a single prompt
    ./.venv/bin/python scripts/probe_checkpoint_first_action.py \
        --prompt "a cylinder" \
        --ckpt-glob "checkpoints/policy_goal/step_*.pt"

    # Probe a broad prompt suite
    ./.venv/bin/python scripts/probe_checkpoint_first_action.py \
        --prompts-file scripts/prompt_suites/broad_smoke.txt \
        --ckpt-glob "checkpoints/policy_goal/step_*.pt"

    # Probe N random prompts sampled from your geometry dataset
    ./.venv/bin/python scripts/probe_checkpoint_first_action.py \
        --sample-geometry-jsonl data/datasets/geometry/train.jsonl \
        --sample-n 64 \
        --ckpt-glob "checkpoints/policy_goal/step_*.pt"

Notes:
- This does NOT execute Blender and does not validate action applicability.
- Checkpoints with incompatible model/state dimensions are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
from collections import Counter

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _canonical_cube_state_vec() -> np.ndarray:
    from policy.state import MeshState, normalize_state_vector

    cube = MeshState(
        vertex_count=8,
        face_count=6,
        edge_count=12,
        bbox_x=1.0,
        bbox_y=1.0,
        bbox_z=1.0,
        avg_edge_length=1.0,
        manifold_flag=1.0,
        symmetry_score=1.0,
        selected_face_count=0,
    )
    return normalize_state_vector(cube.as_vector())


def _goal_vec_from_prompt(prompt: str) -> np.ndarray:
    from policy.prompt_parser import parse_prompt_to_goal
    from policy.state import normalize_goal

    goal = parse_prompt_to_goal(prompt).goal
    return normalize_goal(goal)


def _load_model_from_ckpt(ckpt_path: Path):
    from models.policy_transformer import PolicyTransformer

    payload = torch.load(ckpt_path, map_location="cpu")
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
    )
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return payload, model


def _predict_first_action(ckpt_path: Path, prompt: str) -> tuple[str, int, int]:
    from policy.actions import ActionType

    payload, model = _load_model_from_ckpt(ckpt_path)

    state_vec = _canonical_cube_state_vec()
    goal_vec = _goal_vec_from_prompt(prompt)
    feat = np.concatenate([state_vec, goal_vec], axis=0).astype(np.float32, copy=False)
    states = torch.from_numpy(feat[None, None, :])  # (B=1, T=1, D)

    with torch.no_grad():
        type_logits, param_logits = model(states)

    a_type = int(torch.argmax(type_logits[0, 0]).item())
    a_param = int(torch.argmax(param_logits[0, 0]).item())
    step = int(payload.get("step", -1))
    return ActionType(a_type).name, a_param, step


def _load_prompts_from_file(path: Path) -> list[str]:
    prompts: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        prompts.append(s)
    return prompts


def _sample_prompts_from_geometry_jsonl(path: Path, *, n: int, seed: int) -> list[str]:
    import numpy as _np

    rng = _np.random.default_rng(int(seed))
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
    if not texts:
        return []
    if n <= 0:
        return texts
    idx = rng.choice(len(texts), size=min(int(n), len(texts)), replace=False)
    return [texts[int(i)] for i in idx]


def _entropy_from_counts(counts: Counter[str]) -> float:
    total = float(sum(counts.values()))
    if total <= 0:
        return 0.0
    import math

    h = 0.0
    for c in counts.values():
        p = float(c) / total
        if p > 0:
            h -= p * math.log(p)
    return float(h)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--prompt",
        type=str,
        action="append",
        default=None,
        help="Prompt to probe (repeatable). If omitted, use --prompts-file and/or --sample-geometry-jsonl.",
    )
    p.add_argument("--prompts-file", type=Path, default=None, help="Text file with one prompt per line (# comments allowed).")
    p.add_argument("--sample-geometry-jsonl", type=Path, default=None, help="Sample prompts from a geometry jsonl (uses the 'text' field).")
    p.add_argument("--sample-n", type=int, default=0, help="Number of prompts to sample from --sample-geometry-jsonl (0 = all).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ckpt-glob",
        type=str,
        default="checkpoints/policy_goal/step_*.pt",
        help="Glob of checkpoint paths to probe.",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional limit on number of checkpoints probed (0 = no limit).")
    p.add_argument(
        "--collapse-actions",
        type=str,
        default="SELECT_RANDOM_FACE,NOOP",
        help="Comma-separated action names considered 'collapse' in this probe.",
    )
    args = p.parse_args()

    prompts: list[str] = []
    if args.prompt:
        prompts.extend([s.strip() for s in args.prompt if isinstance(s, str) and s.strip()])
    if args.prompts_file is not None:
        prompts.extend(_load_prompts_from_file(Path(args.prompts_file)))
    if args.sample_geometry_jsonl is not None:
        prompts.extend(
            _sample_prompts_from_geometry_jsonl(
                Path(args.sample_geometry_jsonl),
                n=int(args.sample_n),
                seed=int(args.seed),
            )
        )

    # De-dupe while preserving order.
    seen: set[str] = set()
    prompts = [s for s in prompts if not (s in seen or seen.add(s))]
    if not prompts:
        raise SystemExit("No prompts provided. Use --prompt, --prompts-file, and/or --sample-geometry-jsonl.")

    ckpts = sorted(Path().glob(args.ckpt_glob))
    if args.limit and args.limit > 0:
        ckpts = ckpts[: int(args.limit)]

    if not ckpts:
        raise SystemExit(f"No checkpoints matched: {args.ckpt_glob}")

    collapse_set = {s.strip() for s in str(args.collapse_actions).split(",") if s.strip()}
    print(f"prompts={len(prompts)} ckpt_glob={args.ckpt_glob!r} collapse={sorted(collapse_set)}")
    for ckpt in ckpts:
        counts: Counter[str] = Counter()
        steps: set[int] = set()
        errors = 0
        for pr in prompts:
            try:
                name, _param, step = _predict_first_action(ckpt, pr)
            except Exception:
                errors += 1
                continue
            counts[str(name)] += 1
            steps.add(int(step))

        n_ok = int(sum(counts.values()))
        if n_ok == 0:
            print(f"{ckpt}: SKIP (no successful predictions, errors={errors})")
            continue

        collapse = int(sum(counts[a] for a in collapse_set))
        collapse_rate = float(collapse) / float(n_ok)
        top_action, top_count = counts.most_common(1)[0]
        ent = _entropy_from_counts(counts)
        step_str = str(sorted(steps)[-1]) if steps else "-1"

        print(
            f"{ckpt} (step={step_str}): ok={n_ok} err={errors} "
            f"top={top_action}:{top_count} collapse_rate={collapse_rate:.3f} entropy={ent:.3f} counts={dict(counts)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
