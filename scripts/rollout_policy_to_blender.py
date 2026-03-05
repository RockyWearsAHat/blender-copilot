#!/usr/bin/env python3
"""End-to-end: policy checkpoint -> plan.json -> Blender execution -> exported mesh.

Runs torch inference in the venv, then calls Blender headless to execute the
action plan with bpy.

Usage:
  /path/to/python scripts/rollout_policy_to_blender.py \
    --ckpt checkpoints/policy_goal/latest.pt \
    --steps 64 \
    --out-dir data/eval/rollouts/policy_demo

Notes:
- Blender executable defaults to macOS app install path.
- This is the "see results" bridge: produces a .blend + .obj you can open.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal-vertices", type=int, default=1500)
    p.add_argument("--goal-symmetry", type=float, default=0.7)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--blender", type=str, default=DEFAULT_BLENDER)
    p.add_argument("--apply-modifiers", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan.json"

    gen = PROJECT_ROOT / "scripts" / "generate_policy_plan.py"
    exe = PROJECT_ROOT / "processing" / "execute_policy_plan.py"

    # 1) Generate plan
    cmd_gen = [
        sys.executable,
        str(gen),
        "--ckpt",
        str(args.ckpt),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--goal-vertices",
        str(args.goal_vertices),
        "--goal-symmetry",
        str(args.goal_symmetry),
        "--device",
        str(args.device),
        "--out",
        str(plan_path),
    ]
    subprocess.check_call(cmd_gen, cwd=str(PROJECT_ROOT))

    # 2) Execute in Blender
    cmd_blender = [
        str(args.blender),
        "--background",
        "--python",
        str(exe),
        "--",
        "--plan",
        str(plan_path),
        "--out-dir",
        str(out_dir),
    ]
    if args.apply_modifiers:
        cmd_blender.append("--apply-modifiers")

    subprocess.check_call(cmd_blender, cwd=str(PROJECT_ROOT))
    print(f"OK: rollout complete at {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
