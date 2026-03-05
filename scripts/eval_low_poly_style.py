#!/usr/bin/env python3
"""Evaluate low-poly style behavior from closed-loop Blender rollouts.

This script targets style semantics in the plan:
- stylized low poly: mostly flat/faceted, triangulation-friendly
- retro/PS1 low poly: smoother shading with controlled face budgets
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.test_suite import load_low_poly_style_suite


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate low-poly style prompts in Blender")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--blender", type=str, default=DEFAULT_BLENDER)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suite", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("data/eval/low_poly_style"))
    p.add_argument("--skip-rollout", action="store_true", help="Use existing rollout outputs if present")
    return p.parse_args()


def _run_rollout(*, ckpt: Path, blender: str, device: str, steps: int, seed: int, prompt: str, out_dir: Path) -> int:
    script = PROJECT_ROOT / "scripts" / "rollout_policy_closed_loop.py"
    cmd = [
        sys.executable,
        str(script),
        "--ckpt",
        str(ckpt),
        "--out-dir",
        str(out_dir),
        "--steps",
        str(int(steps)),
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
        "--blender",
        str(blender),
        "--prompt",
        str(prompt),
        "--low-poly-bias",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"rollout failed for prompt={prompt!r} (code={proc.returncode})", file=sys.stderr)
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
    return int(proc.returncode)


def _load_stats(out_dir: Path) -> dict:
    f = out_dir / "stats_final.json"
    if not f.exists():
        return {}
    payload = json.loads(f.read_text())
    if isinstance(payload, dict) and "stats" in payload and isinstance(payload["stats"], dict):
        return payload["stats"]
    return payload if isinstance(payload, dict) else {}


def _score_case(stats: dict, expected: dict) -> dict:
    checks: dict[str, dict] = {}
    face_count = int(stats.get("face_count", 0))
    manifold = float(stats.get("manifold_flag", 0.0))
    smooth_frac = float(stats.get("shade_smooth_fraction", 0.0))
    tri_ratio = float(stats.get("triangulated_ratio", 0.0))

    if "min_faces" in expected:
        ok = face_count >= int(expected["min_faces"])
        checks["min_faces"] = {"passed": bool(ok), "detail": f"{face_count} >= {int(expected['min_faces'])}"}
    if "max_faces" in expected:
        ok = face_count <= int(expected["max_faces"])
        checks["max_faces"] = {"passed": bool(ok), "detail": f"{face_count} <= {int(expected['max_faces'])}"}

    checks["manifold"] = {"passed": bool(manifold >= 0.95), "detail": f"manifold={manifold:.3f}"}

    hint = str(expected.get("style_hint", "")).strip().lower()
    if hint == "stylized":
        checks["style_smoothness"] = {
            "passed": bool(smooth_frac <= 0.25),
            "detail": f"smooth_fraction={smooth_frac:.3f} <= 0.25",
        }
        checks["style_faceting"] = {
            "passed": bool(tri_ratio >= 0.60),
            "detail": f"triangulated_ratio={tri_ratio:.3f} >= 0.60",
        }
    elif hint == "retro":
        checks["style_smoothness"] = {
            "passed": bool(smooth_frac >= 0.55),
            "detail": f"smooth_fraction={smooth_frac:.3f} >= 0.55",
        }

    passed = all(bool(item.get("passed", False)) for item in checks.values()) if checks else False
    return {"passed": bool(passed), "checks": checks, "stats": stats}


def main() -> int:
    args = _parse_args()
    suite = load_low_poly_style_suite(args.suite)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i, case in enumerate(suite):
        case_id = str(case.get("id", f"case_{i:03d}"))
        prompt = str(case.get("prompt", "")).strip()
        expected = case.get("expected", {}) or {}
        run_dir = out_root / case_id

        if not args.skip_rollout:
            code = _run_rollout(
                ckpt=args.ckpt,
                blender=args.blender,
                device=args.device,
                steps=int(args.steps),
                seed=int(args.seed) + i,
                prompt=prompt,
                out_dir=run_dir,
            )
            if code != 0:
                results.append({"id": case_id, "prompt": prompt, "generated": False, "error": f"rollout_failed_{code}"})
                continue

        stats = _load_stats(run_dir)
        if not stats:
            results.append({"id": case_id, "prompt": prompt, "generated": False, "error": "missing_stats"})
            continue

        score = _score_case(stats, expected)
        results.append(
            {
                "id": case_id,
                "prompt": prompt,
                "generated": True,
                "expected": expected,
                "style_hint": expected.get("style_hint"),
                **score,
            }
        )

    total = len(results)
    generated = sum(1 for r in results if r.get("generated"))
    passed = sum(1 for r in results if r.get("generated") and r.get("passed"))
    summary = {
        "total": total,
        "generated": generated,
        "generation_rate": float(generated / max(1, total)),
        "passed": passed,
        "pass_rate": float(passed / max(1, total)),
    }

    payload = {"summary": summary, "results": results}
    out_file = out_root / "results.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_file), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
