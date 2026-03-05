#!/usr/bin/env python3
"""Batch-generate deterministic mountain action traces.

Produces forward traces that follow the non-destructive mountain workflow.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate mountain traces in Blender")
    p.add_argument("--blender", type=str, default=DEFAULT_BLENDER)
    p.add_argument("--out-dir", type=Path, default=Path("data/datasets/mountain_traces"))
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stylized-ratio", type=float, default=0.7)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def _style_for_index(i: int, stylized_ratio: float) -> str:
    cutoff = max(0.0, min(1.0, float(stylized_ratio)))
    # Deterministic alternation with controllable ratio.
    stride = 100
    return "stylized" if (i % stride) < int(cutoff * stride) else "retro"


def main() -> int:
    args = _parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    worker = PROJECT_ROOT / "processing" / "mountain_trace_worker.py"

    total = max(0, int(args.n))
    for i in range(total):
        seed_i = int(args.seed) + i
        style = _style_for_index(i, float(args.stylized_ratio))
        trace_id = f"mountain_{style}_{seed_i:06d}"
        out_dir = out_root / trace_id
        trace_path = out_dir / "trace.jsonl"

        if args.skip_existing and trace_path.exists():
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(args.blender),
            "--background",
            "--python",
            str(worker),
            "--",
            "--out-dir",
            str(out_dir),
            "--seed",
            str(seed_i),
            "--style",
            style,
        ]

        print(f"[{i+1}/{total}] {trace_id}", flush=True)
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            print(f"ERROR: Blender executable not found: {args.blender}", file=sys.stderr)
            return 1

        if proc.returncode != 0:
            print(f"  FAIL({proc.returncode}): {trace_id}")
            if proc.stdout:
                print("  stdout:\n" + proc.stdout)
            if proc.stderr:
                print("  stderr:\n" + proc.stderr)
            continue

        if not trace_path.exists():
            print(f"  FAIL(no trace): {trace_id}")
            if proc.stdout:
                print("  stdout:\n" + proc.stdout)
            if proc.stderr:
                print("  stderr:\n" + proc.stderr)
            continue

    print("OK: mountain trace generation finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
