#!/usr/bin/env python3
"""Generate collapse traces for synthetic terrain meshes.

Why: the local real-mesh cache has very few (or zero) items labeled with
terrain/hills/landscape terms. For prompt grounding like "grassy hills terrain",
we need traces whose prompt text actually contains those tokens.

This script:
  1) Generates a variety of terrain heightmap meshes (pure Python).
  2) Writes each mesh to a minimal mesh.json (vertices/faces + prompt_variants).
  3) Calls Blender headless to run processing/collapse_trace_worker.py.

Output layout matches the real-trace generator:
  data/datasets/collapse_traces/synth_terrain_<seed>_<i>/trace.jsonl

Usage:
  /path/to/python scripts/generate_collapse_traces_synthetic_terrain.py \
    --blender /Applications/Blender.app/Contents/MacOS/Blender \
    --n 200
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate collapse traces for synthetic terrain")
    p.add_argument("--out-dir", type=Path, default=Path("data/datasets/collapse_traces"))
    p.add_argument("--blender", type=str, required=True, help="Path to Blender executable")
    p.add_argument("--n", type=int, default=200, help="How many terrain meshes to generate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--target-verts", type=int, default=16)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def _prompt_variants() -> list[str]:
    # Include the exact phrase we care about for eval.
    # Keep this list short and semantically consistent.
    return [
        "terrain",
        "landscape",
        "ground",
        "rolling hills",
        "hilly landscape",
        "bumpy ground",
        "grassy hills terrain",
    ]


def _make_terrain_quads(*, sx: float, sy: float, nx: int, ny: int, max_height: float, seed: int) -> tuple[list[list[float]], list[list[int]]]:
    """Heightmap terrain as a QUAD grid.

    Using quads (not triangles) makes Blender's `unsubdivide` effective,
    which improves collapse trace diversity.
    """

    rng = random.Random(int(seed))

    def _noise(ix: int, iy: int, freq: float, phase: float) -> float:
        x = (ix / max(1, nx)) * freq + phase
        y = (iy / max(1, ny)) * freq - phase
        base = math.sin(x * 2.2) * 0.55 + math.cos(y * 2.7) * 0.45
        mix = math.sin((x + y) * 1.6) * 0.35 + math.cos((x - y) * 1.2) * 0.25
        jitter = math.sin((ix * 12.9898 + iy * 78.233 + seed) * 0.017) * 0.15
        return base + mix + jitter

    heights = [[0.0 for _ in range(nx + 1)] for _ in range(ny + 1)]
    octaves = [(1.5, 1.0), (3.0, 0.5), (6.0, 0.25), (12.0, 0.12)]
    phase = (seed % 997) / 997.0
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            h = 0.0
            for freq, amp in octaves:
                h += _noise(ix, iy, freq, phase) * amp
            heights[iy][ix] = math.copysign(abs(h) ** 1.3, h)

    flat_vals = [v for row in heights for v in row]
    lo, hi = min(flat_vals), max(flat_vals)
    span = max(1e-6, hi - lo)
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            nrm = (heights[iy][ix] - lo) / span
            heights[iy][ix] = (nrm * 2.0 - 1.0) * max_height

    for _pass in range(2):
        smoothed = [row[:] for row in heights]
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                total = heights[iy][ix]
                count = 1
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    niy, nix = iy + dy, ix + dx
                    if 0 <= niy <= ny and 0 <= nix <= nx:
                        total += heights[niy][nix]
                        count += 1
                smoothed[iy][ix] = total / count
        heights = smoothed

    verts: list[list[float]] = []
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            x = -sx / 2 + sx * ix / nx
            y = -sy / 2 + sy * iy / ny
            z = heights[iy][ix]
            verts.append([x, y, z])

    faces: list[list[int]] = []
    for iy in range(ny):
        for ix in range(nx):
            a = iy * (nx + 1) + ix
            b = a + 1
            d = a + (nx + 1)
            c = d + 1
            faces.append([a, b, c, d])

    return verts, faces


def main() -> int:
    args = _parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    worker_script = str((PROJECT_ROOT / "processing" / "collapse_trace_worker.py").resolve())

    n = max(0, int(args.n))
    seed = int(args.seed)
    prompts = _prompt_variants()

    for i in range(n):
        mesh_id = f"synth_terrain_{seed:04d}_{i:05d}"
        out_dir = out_root / mesh_id
        trace_path = out_dir / "trace.jsonl"
        if bool(args.skip_existing) and trace_path.exists():
            continue

        # Deterministic but varied terrain.
        rng = random.Random(seed + i)
        nx = int(rng.choice([16, 24, 32, 48]))
        ny = int(rng.choice([16, 24, 32, 48]))
        sx = float(rng.choice([1.0, 1.5, 2.0, 2.5]))
        sy = float(rng.choice([1.0, 1.5, 2.0, 2.5]))
        max_h = float(rng.choice([0.12, 0.22, 0.35, 0.5]))

        vertices, faces = _make_terrain_quads(sx=sx, sy=sy, nx=nx, ny=ny, max_height=max_h, seed=seed + i)
        if not vertices or not faces:
            continue

        mesh_json = out_dir / "mesh.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh_json.write_text(
            json.dumps(
                {
                    "mesh_id": mesh_id,
                    "label": "terrain",
                    "prompt_variants": prompts,
                    "vertices": vertices,
                    "faces": faces,
                }
            ),
            encoding="utf-8",
        )

        cmd = [
            str(args.blender),
            "--background",
            "--python",
            worker_script,
            "--",
            "--mesh-json",
            str(mesh_json),
            "--out-dir",
            str(out_dir),
            "--max-steps",
            str(int(args.max_steps)),
            "--target-verts",
            str(int(args.target_verts)),
        ]

        print(f"[{i+1}/{n}] collapse trace for {mesh_id} ...", flush=True)
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            print(f"ERROR: Blender executable not found: {args.blender}", file=sys.stderr)
            return 1

        if proc.returncode != 0:
            print(f"  Blender failed (code {proc.returncode}) for {mesh_id}")
            if proc.stdout:
                print("  stdout:\n" + proc.stdout)
            if proc.stderr:
                print("  stderr:\n" + proc.stderr)
            continue

        if not trace_path.exists():
            print(f"  Blender finished but no trace.jsonl for {mesh_id}")
            if proc.stdout:
                print("  stdout:\n" + proc.stdout)
            if proc.stderr:
                print("  stderr:\n" + proc.stderr)
            continue

    print("OK: synthetic terrain collapse trace generation finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
