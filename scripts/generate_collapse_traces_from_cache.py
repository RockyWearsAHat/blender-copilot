#!/usr/bin/env python3
"""Generate collapse traces for real web-scraped meshes from the cache.

Pipeline:
  1. Load `.pt` samples from data/processed/.mesh_cache (objaverse_xl, blendswap, ...).
  2. Decode `mesh_tokens` back into vertices/faces using MeshTokenizer.
  3. Export a minimal mesh JSON (vertices, faces, label, mesh_id).
  4. Call Blender in background with processing/collapse_trace_worker.py
     using `--mesh-json` to build the mesh and record a collapse trace
     (modifier removal + unsubdivide/dissolve/merge).

Result: for each cache sample we get
  out_dir/<mesh_id>/trace.jsonl
which can be turned into imitation (state, action) pairs for the policy.

This is an offline data-generation script; it does NOT run during training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate collapse traces from mesh cache")
    p.add_argument("--cache-dir", type=Path, default=Path("data/processed/.mesh_cache"))
    p.add_argument("--out-dir", type=Path, default=Path("data/datasets/collapse_traces"))
    p.add_argument("--blender", type=str, required=True, help="Path to Blender executable")
    p.add_argument("--max-files", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--target-verts", type=int, default=8)
    p.add_argument("--timeout-s", type=int, default=180, help="Per-mesh Blender timeout (seconds). Skips meshes that hang.")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--prompts-per-mesh", type=int, default=12, help="How many prompt variants to store in mesh.json")
    # Mesh tokenizer config (should match cache pre-processing)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--coord-min", type=float, default=-1.0)
    p.add_argument("--coord-max", type=float, default=1.0)
    p.add_argument("--max-faces", type=int, default=4000)
    return p.parse_args()


def _make_tokenizer(args: argparse.Namespace):
    from processing.mesh_tokenizer import MeshTokenizer

    return MeshTokenizer(
        vocab_size=int(args.vocab_size),
        coord_range=(float(args.coord_min), float(args.coord_max)),
        max_faces=int(args.max_faces),
    )


def _decode_vertices_faces(sample: dict, tok):
    mesh_tokens = sample.get("mesh_tokens")
    if mesh_tokens is None:
        return None, None
    if isinstance(mesh_tokens, torch.Tensor):
        tokens = mesh_tokens
    else:
        try:
            tokens = torch.as_tensor(mesh_tokens, dtype=torch.long)
        except Exception:
            return None, None
    from scripts.render_cache import decode_mesh_tokens

    vertices, faces = decode_mesh_tokens(tokens, tok)
    return vertices, faces


def main() -> int:
    args = _parse_args()
    from policy.prompt_augment import PromptAugmentConfig, make_prompt_variants

    cache_dir = Path(args.cache_dir)
    out_root = Path(args.out_dir)
    blender = args.blender
    timeout_s = int(args.timeout_s)

    if not cache_dir.is_dir():
        print(f"ERROR: cache-dir does not exist: {cache_dir}", file=sys.stderr)
        return 1

    tok = _make_tokenizer(args)

    pt_files = sorted(cache_dir.glob("*.pt"))[: int(args.max_files)]
    print(f"Found {len(pt_files)} cache files (limit {args.max_files})")

    for i, pt_path in enumerate(pt_files, 1):
        mesh_id = pt_path.stem
        out_dir = out_root / mesh_id
        trace_path = out_dir / "trace.jsonl"
        if args.skip_existing and trace_path.exists():
            continue

        try:
            sample = torch.load(pt_path, map_location="cpu")
        except Exception as e:
            print(f"[skip] {mesh_id}: failed to load .pt: {e}")
            continue

        if isinstance(sample, list):
            if not sample:
                continue
            sample = sample[0]
        if not isinstance(sample, dict):
            continue

        vertices, faces = _decode_vertices_faces(sample, tok)
        if not vertices or not faces:
            print(f"[skip] {mesh_id}: decode produced empty mesh")
            continue

        label = str(sample.get("label") or "")
        mesh_json = out_dir / "mesh.json"
        prompts_per = max(0, int(args.prompts_per_mesh))
        data = {
            "mesh_id": mesh_id,
            "label": label,
            "prompt_variants": make_prompt_variants(label, PromptAugmentConfig(max_variants=prompts_per)),
            "vertices": vertices,
            "faces": faces,
        }
        mesh_json.parent.mkdir(parents=True, exist_ok=True)
        mesh_json.write_text(json.dumps(data), encoding="utf-8")

        worker_script = str((PROJECT_ROOT / "processing" / "collapse_trace_worker.py").resolve())
        cmd = [
            blender,
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
        print(f"[{i}/{len(pt_files)}] collapse trace for {mesh_id} ...", flush=True)
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=(float(timeout_s) if timeout_s > 0 else None),
            )
        except FileNotFoundError:
            print(f"ERROR: Blender executable not found: {blender}", file=sys.stderr)
            return 1
        except subprocess.TimeoutExpired:
            print(f"  [timeout] Blender exceeded {timeout_s}s for {mesh_id}; skipping")
            continue

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

    print("OK: collapse trace generation finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
