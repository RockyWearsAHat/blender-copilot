#!/usr/bin/env python3
"""Closed-loop rollout: policy chooses actions from real Blender state each step.

This is the fastest *meaningful* path to visible mesh improvement:
- Blender stays alive (one process)
- Policy inference happens in the venv (torch available)
- Each step uses Blender-extracted mesh stats as the next state

Usage:
  /path/to/python scripts/rollout_policy_closed_loop.py \
    --ckpt checkpoints/policy_goal/latest.pt \
    --out-dir data/eval/rollouts/closed_loop_demo \
    --steps 64
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from policy.state import MeshState


def _detect_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _stats_to_state(stats: dict) -> "MeshState":
    from policy.state import MeshState
    bb = stats.get("bounding_box", {})
    return MeshState(
        vertex_count=int(stats.get("vertex_count", 0)),
        face_count=int(stats.get("face_count", 0)),
        edge_count=int(stats.get("edge_count", 0)),
        bbox_x=float(bb.get("x", 0.0)),
        bbox_y=float(bb.get("y", 0.0)),
        bbox_z=float(bb.get("z", 0.0)),
        avg_edge_length=float(stats.get("avg_edge_length", 0.0)),
        manifold_flag=float(stats.get("manifold_flag", 0.0)),
        symmetry_score=float(stats.get("symmetry_score", 0.0)),
        selected_face_count=int(stats.get("selected_face_count", 0)),
    )


def _wait_json(path: Path, timeout_s: int = 120, poll_ms: int = 25) -> dict:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if path.exists():
            try:
                txt = path.read_text()
                if not txt.strip():
                    time.sleep(poll_ms / 1000.0)
                    continue
                return json.loads(txt)
            except json.JSONDecodeError:
                # Writer may be mid-write; retry.
                time.sleep(poll_ms / 1000.0)
                continue
        time.sleep(poll_ms / 1000.0)
    raise TimeoutError(f"Timeout waiting for {path}")


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _sample_from_logits(
    logits_1d: torch.Tensor,
    *,
    gen: torch.Generator,
    temperature: float,
    top_k: int,
    deterministic: bool,
) -> int:
    """Sample an index from logits with optional temperature + top-k.

    Uses CPU sampling for determinism across devices.
    """
    logits_cpu = logits_1d.detach().float().cpu()

    if deterministic or temperature <= 0:
        return int(torch.argmax(logits_cpu).item())

    logits_cpu = logits_cpu / float(temperature)

    if int(top_k) > 0 and int(top_k) < logits_cpu.numel():
        k = int(top_k)
        vals, idx = torch.topk(logits_cpu, k)
        probs = torch.softmax(vals, dim=-1)
        choice = int(torch.multinomial(probs, 1, generator=gen).item())
        return int(idx[choice].item())

    probs = torch.softmax(logits_cpu, dim=-1)
    return int(torch.multinomial(probs, 1, generator=gen).item())


def _apply_low_poly_bias(
    *,
    type_logits_1d: torch.Tensor,
    param_logits_1d: torch.Tensor,
    stats: dict,
    goal,
    low_poly_bias: bool,
    poly_budget_mult: float,
    poly_budget_min_verts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a lightweight low-poly inductive bias at inference time.

    This does not modify model weights. It only prevents known topology
    explosion actions when we're already over budget.
    """
    if not low_poly_bias:
        return type_logits_1d, param_logits_1d

    try:
        from policy.actions import ActionType
    except Exception:
        return type_logits_1d, param_logits_1d

    try:
        v = int(stats.get("vertex_count", 0))
    except Exception:
        v = 0

    try:
        target_v = int(getattr(goal, "target_vertex_count", 0) or 0)
    except Exception:
        target_v = 0

    base_budget = int(poly_budget_min_verts)
    if target_v > 0:
        budget = max(base_budget, int(round(float(target_v) * float(poly_budget_mult))))
    else:
        budget = base_budget

    tl = type_logits_1d.clone()
    pl = param_logits_1d.clone()

    if v >= budget:
        # Forbid known topology explosion actions once we're over budget.
        try:
            tl[int(ActionType.SUBDIVIDE)] = -1e9
        except Exception:
            pass
        try:
            tl[int(ActionType.BEVEL)] = -1e9
        except Exception:
            pass
        try:
            tl[int(ActionType.APPLY_MODIFIER)] = -1e9
        except Exception:
            pass

    # In the worker, APPLY_MODIFIER chooses kind by (param % 3):
    #   0->SUBSURF, 1->BEVEL, 2->SOLIDIFY
    # Mask all SUBSURF-aligned bins when mesh is moderately large.
    subsurf_block_v = max(1024, budget // 2)
    if v >= subsurf_block_v:
        try:
            for idx in range(int(pl.numel())):
                if idx % 3 == 0:
                    pl[idx] = -1e9
        except Exception:
            pass

    return tl, pl


def _canonicalize_state_dict(sd: dict) -> dict:
    """Make state_dict compatible across torch.compile/non-compile saves."""
    if not isinstance(sd, dict) or not sd:
        return sd
    # torch.compile commonly prefixes parameter keys with `_orig_mod.`
    # when saving the compiled wrapper. Strip that prefix if present.
    if any(str(k).startswith("_orig_mod.") for k in sd.keys()):
        out: dict = {}
        for k, v in sd.items():
            ks = str(k)
            if ks.startswith("_orig_mod."):
                ks = ks[len("_orig_mod.") :]
            out[ks] = v
        return out
    return sd


def _extract_policy_state_dict(payload: dict, ckpt_path: Path) -> dict:
    """Extract PolicyTransformer weights from checkpoint payload.

    Supports legacy and newer key names while rejecting incompatible
    checkpoint families (for example unified multi-task checkpoints).
    """
    model_type = str(payload.get("model_type", "")).strip().lower()
    if model_type and model_type != "policy":
        if model_type == "unified":
            raise RuntimeError(
                f"Checkpoint '{ckpt_path}' is model_type=unified and is not compatible with policy rollout. "
                "Use a policy checkpoint such as checkpoints/policy_goal/latest.pt"
            )
        raise RuntimeError(
            f"Checkpoint '{ckpt_path}' is model_type={model_type!r}, expected policy"
        )

    for key in ("model", "model_state_dict", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict) and candidate:
            return _canonicalize_state_dict(candidate)

    raise RuntimeError(
        f"Checkpoint '{ckpt_path}' does not contain policy weights "
        "(expected one of: model, model_state_dict, state_dict)"
    )


def _write_obj(path: Path, vertices: list, faces: list) -> None:
    """Write a simple OBJ mesh file from vertex/face arrays."""
    lines: list[str] = []
    for v in vertices:
        lines.append(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}")
    for face in faces:
        # OBJ is 1-indexed.
        idx = [int(i) + 1 for i in face]
        lines.append("f " + " ".join(str(i) for i in idx))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_unified_direct_generation(args, payload: dict, out_dir: Path, device: torch.device) -> int:
    """Run direct unified text->mesh generation and export mesh.obj.

    This path allows Blender addon rollout UI to use the latest unified
    training checkpoints directly, without policy closed-loop action rollout.
    """
    from inference.server import (
        ModelState,
        _merge_duplicate_vertices,
        _recalculate_normals_consistent,
        _sample_mesh_tokens,
        text_to_tokens,
    )

    state = ModelState()
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}
    state.load(str(args.ckpt), cfg, device_str=str(device))

    prompt_text = args.prompt or "mesh"
    model = state.model
    tokenizer = state.tokenizer
    model_type = state.model_type
    dev = state.device
    text_tok = state.text_tokenizer

    if model is None or tokenizer is None or dev is None:
        raise RuntimeError("Unified model did not initialize for rollout")

    geo_dec = model.geometry_decoder
    max_seq = getattr(geo_dec, "max_seq_length", 16202)
    effective_max_faces = min(int(args.max_faces), max(16, (max_seq - 2) // 9))
    max_tokens = effective_max_faces * 9 + 2

    text_max = getattr(model.text_encoder, "max_length", 256)
    if hasattr(model.text_encoder, "pos_embed"):
        text_max = model.text_encoder.pos_embed.num_embeddings

    text_ids, text_mask = text_to_tokens(prompt_text, max_length=text_max, text_tokenizer=text_tok)
    text_ids = text_ids.to(dev)
    text_mask = text_mask.to(dev)

    attempts = [
        (float(args.temperature), int(args.top_k), 0.90),
        (max(0.85, float(args.temperature)), max(32, int(args.top_k)), 0.93),
        (1.0, max(64, int(args.top_k)), 0.95),
    ]
    best_vertices = []
    best_faces = []
    best_token_count = 0

    for temp_i, topk_i, topp_i in attempts:
        tokens = _sample_mesh_tokens(
            model,
            model_type,
            text_ids,
            text_mask,
            max_tokens=max_tokens,
            temperature=temp_i,
            top_k=topk_i,
            top_p=topp_i,
            cfg_scale=3.5,
        )
        token_list = tokens[0].detach().cpu().tolist() if tokens is not None else []
        if not token_list:
            continue
        vertices, faces = tokenizer.decode_tokens(token_list)
        vertices, faces = _merge_duplicate_vertices(vertices, faces)
        faces = _recalculate_normals_consistent(vertices, faces)
        if len(faces) > len(best_faces):
            best_vertices = vertices
            best_faces = faces
            best_token_count = len(token_list)

    vertices = best_vertices
    faces = best_faces
    if not vertices or not faces:
        raise RuntimeError("Unified generation produced empty geometry")

    obj_path = out_dir / "mesh.obj"
    _write_obj(obj_path, vertices, faces)

    stats = {
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
    }
    payload_out = {
        "out_dir": str(out_dir),
        "final": {"stats": stats},
        "mode": "unified_direct",
        "checkpoint": str(args.ckpt),
        "token_count": int(best_token_count),
    }
    print(json.dumps(payload_out, indent=2))
    return 0


def main() -> int:
    from models.policy_transformer import PolicyTransformer
    from policy.prompt_parser import parse_prompt_to_goal
    from policy.state import Goal, normalize_goal, normalize_state_vector

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--blender", type=str, default=DEFAULT_BLENDER)
    p.add_argument("--apply-modifiers", action="store_true")
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional free-text request. Deterministically parsed into a numeric goal.",
    )
    p.add_argument("--goal-vertices", type=int, default=None)
    p.add_argument("--goal-symmetry", type=float, default=None)
    p.add_argument("--deterministic", action="store_true", help="Use greedy argmax actions (no sampling).")
    p.add_argument("--temperature", type=float, default=1.0, help="Action sampling temperature (higher = more varied).")
    p.add_argument("--top-k", type=int, default=0, help="Top-k sampling for actions (0 = disabled).")
    p.add_argument("--max-verts", type=int, default=250_000, help="Safety cap: stop rollout if mesh exceeds this many verts.")
    p.add_argument("--max-faces", type=int, default=250_000, help="Safety cap: stop rollout if mesh exceeds this many faces.")
    p.add_argument("--low-poly-bias", action="store_true", help="Enable inference-time low-poly masking (opt-in).")
    p.add_argument(
        "--poly-budget-mult",
        type=float,
        default=2.0,
        help="(low-poly-bias) Mask growth ops when verts exceed target * this multiplier.",
    )
    p.add_argument(
        "--poly-budget-min-verts",
        type=int,
        default=2048,
        help="(low-poly-bias) Minimum vertex budget used by low-poly bias.",
    )
    p.add_argument(
        "--unsafe-no-geometry-guards",
        action="store_true",
        help="Disable Blender-side geometry guards (may freeze/crash Blender).",
    )
    p.add_argument(
        "--legacy-default-goals",
        action="store_true",
        help="Legacy behaviour: if the prompt has no numeric constraints, fall back to default goal scalars (older checkpoints may rely on this).",
    )
    p.add_argument(
        "--score-reconstruction",
        action="store_true",
        help="If a prompt is provided, compute reference-based reconstruction metrics (Chamfer/F-score) vs geometry dataset reference and write reconstruction_metrics.json.",
    )
    p.add_argument(
        "--reference-geometry-dir",
        type=Path,
        default=Path("data/datasets/geometry"),
        help="Directory containing train/val/test.jsonl with tokenized reference meshes.",
    )
    p.add_argument(
        "--reference-splits",
        type=str,
        default="train,val,test",
        help="Comma-separated dataset splits to search for references.",
    )
    p.add_argument(
        "--score-weight-reconstruction",
        type=float,
        default=0.85,
        help="Weight for reconstruction fidelity in total score (default matches legacy behavior).",
    )
    p.add_argument(
        "--score-weight-trajectory",
        type=float,
        default=0.15,
        help="Weight for trajectory/workflow score in total score (default matches legacy behavior).",
    )
    args = p.parse_args()

    # Default behavior: if the user provided a prompt, emit reconstruction
    # metrics unless explicitly disabled (currently only via omitting prompt).
    args.score_reconstruction = bool(args.score_reconstruction) or bool(args.prompt)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _detect_device(args.device)

    payload = torch.load(args.ckpt, map_location="cpu")
    cfg = payload.get("config", {})

    model_type = str(payload.get("model_type", "")).strip().lower()
    if model_type == "unified":
        return _run_unified_direct_generation(args, payload, out_dir, device)

    model_cfg = cfg.get("model", {})

    model = PolicyTransformer(
        state_dim=int(model_cfg.get("state_dim", 12)),
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        num_layers=int(model_cfg.get("num_layers", 4)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        action_type_vocab=int(model_cfg.get("action_type_vocab", 11)),
        action_param_vocab=int(model_cfg.get("action_param_vocab", 32)),
        max_seq_len=int(model_cfg.get("max_seq_len", 128)),
    ).to(device)
    model_sd = _extract_policy_state_dict(payload, args.ckpt)
    model.load_state_dict(model_sd, strict=True)
    model.eval()

    # Build a Goal.
    # Prompt text is encoded only as a compact fingerprint (Goal.text_features).
    # Numeric targets are ONLY set when the user explicitly typed them or
    # passed overrides. If absent, targets stay 0 ("unconstrained").
    if args.prompt:
        parsed = parse_prompt_to_goal(args.prompt)
        goal = parsed.goal
        if args.goal_vertices is not None:
            goal.target_vertex_count = int(args.goal_vertices)
        if args.goal_symmetry is not None:
            goal.target_symmetry = float(args.goal_symmetry)

        # Optional compatibility mode for older checkpoints.
        if bool(args.legacy_default_goals):
            if args.goal_vertices is None and int(goal.target_vertex_count) == 0:
                goal.target_vertex_count = 1500
            if args.goal_symmetry is None and float(goal.target_symmetry) == 0.0:
                goal.target_symmetry = 0.7
    else:
        goal = Goal(
            target_vertex_count=int(args.goal_vertices) if args.goal_vertices is not None else 0,
            target_symmetry=float(args.goal_symmetry) if args.goal_symmetry is not None else 0.0,
        )

        if bool(args.legacy_default_goals):
            if args.goal_vertices is None and int(goal.target_vertex_count) == 0:
                goal.target_vertex_count = 1500
            if args.goal_symmetry is None and float(goal.target_symmetry) == 0.0:
                goal.target_symmetry = 0.7

    # Reproducible stochastic sampling (when enabled)
    torch.manual_seed(int(args.seed))
    gen = torch.Generator(device="cpu").manual_seed(int(args.seed))

    gvec = normalize_goal(goal)

    worker = PROJECT_ROOT / "processing" / "blender_policy_worker.py"
    cmd = [
        str(args.blender),
        "--background",
        "--python",
        str(worker),
        "--",
        "--work-dir",
        str(out_dir),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--max-verts",
        str(int(args.max_verts)),
        "--max-faces",
        str(int(args.max_faces)),
    ]
    if args.apply_modifiers:
        cmd.append("--apply-modifiers")
    if bool(args.unsafe_no_geometry_guards):
        cmd.append("--unsafe-no-geometry-guards")

    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))

    try:
        ready = _wait_json(out_dir / "ready.json", timeout_s=120)
        stats = ready.get("stats", {})

        def _too_big(s: dict) -> bool:
            try:
                v = int(s.get("vertex_count", 0))
                f = int(s.get("face_count", 0))
            except Exception:
                return False
            return (int(args.max_verts) > 0 and v >= int(args.max_verts)) or (int(args.max_faces) > 0 and f >= int(args.max_faces))

        if _too_big(stats):
            # Worker may early-stop immediately; just wait for final files.
            _wait_json(out_dir / "stats_final.json", timeout_s=120)
            final = json.loads((out_dir / "stats_final.json").read_text())
            print(json.dumps({"out_dir": str(out_dir), "final": final, "early_stop": True}, indent=2))
            return 0

        for i in range(int(args.steps)):
            st = _stats_to_state(stats)
            vec = normalize_state_vector(st.as_vector())
            feat = np.concatenate([vec, gvec], axis=0).astype(np.float32, copy=False)
            states = torch.from_numpy(feat[None, None, :]).to(device=device, dtype=torch.float32)

            with torch.no_grad():
                type_logits, param_logits = model(states)

            tl_cpu = type_logits[0, 0].detach().float().cpu()
            pl_cpu = param_logits[0, 0].detach().float().cpu()
            tl_cpu, pl_cpu = _apply_low_poly_bias(
                type_logits_1d=tl_cpu,
                param_logits_1d=pl_cpu,
                stats=stats,
                goal=goal,
                low_poly_bias=bool(args.low_poly_bias),
                poly_budget_mult=float(args.poly_budget_mult),
                poly_budget_min_verts=int(args.poly_budget_min_verts),
            )

            a_type = _sample_from_logits(
                tl_cpu,
                gen=gen,
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                deterministic=bool(args.deterministic),
            )
            a_param = _sample_from_logits(
                pl_cpu,
                gen=gen,
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                deterministic=bool(args.deterministic),
            )

            _write_json_atomic(
                out_dir / f"action_{i:04d}.json",
                {"action_type": int(a_type), "param": int(a_param)},
            )

            state_obj = _wait_json(out_dir / f"state_{i:04d}.json", timeout_s=120)
            stats = state_obj.get("stats", {})

            if _too_big(stats):
                # Stop producing actions; worker will export and exit.
                break

        final = _wait_json(out_dir / "stats_final.json", timeout_s=120)
        early = (out_dir / "early_stop.json")
        payload = {"out_dir": str(out_dir), "final": final}
        if early.exists():
            try:
                payload["early_stop"] = json.loads(early.read_text())
            except Exception:
                payload["early_stop"] = True

        # Optional: reference-based reconstruction scoring.
        # This is *evaluation only*; it does not alter Blender semantics.
        if args.prompt and bool(args.score_reconstruction):
            try:
                from evaluation.prompt_reference import find_reference_for_prompt
                from evaluation.reconstruction_scoring import (
                    compute_reconstruction_metrics,
                    score_rollout_trajectory,
                )

                w_recon = float(args.score_weight_reconstruction)
                w_traj = float(args.score_weight_trajectory)
                denom = w_recon + w_traj
                if denom <= 0:
                    w_recon, w_traj = 0.0, 1.0
                    denom = 1.0
                w_recon /= denom
                w_traj /= denom

                splits = tuple(s.strip() for s in str(args.reference_splits).split(",") if s.strip())
                ref = find_reference_for_prompt(
                    args.prompt,
                    geometry_dir=Path(args.reference_geometry_dir),
                    splits=splits or ("train", "val", "test"),
                )

                expected = None
                expected_reason = ""
                if ref is not None:
                    from evaluation.reconstruction_scoring import infer_expected_base_primitives_from_source

                    expected, expected_reason = infer_expected_base_primitives_from_source(ref.source)

                traj = score_rollout_trajectory(
                    out_dir,
                    expected_base_primitives=expected,
                    expected_base_reason=str(expected_reason),
                )
                score_payload: dict = {
                    "prompt": str(args.prompt),
                    "reference_found": bool(ref is not None),
                    "trajectory": {"path_score": float(traj.path_score), "breakdown": traj.breakdown},
                }

                obj_path = out_dir / "mesh.obj"
                if ref is not None and obj_path.exists():
                    metrics = compute_reconstruction_metrics(
                        obj_path=obj_path,
                        reference_tokens=ref.tokens,
                        n_surface_points=4096,
                        normalize=True,
                    )
                    score_payload["reference"] = {
                        "prompt": ref.prompt,
                        "split": ref.split,
                        "source": ref.source,
                        "num_vertices": ref.num_vertices,
                        "num_faces": ref.num_faces,
                    }
                    score_payload["metrics"] = metrics

                    # Scalar score dominated by reconstruction fidelity.
                    # Use F-score@0.02 as primary, with a small trajectory term.
                    try:
                        f02 = float(metrics.get("f_score_002", {}).get("f_score", 0.0))
                    except Exception:
                        f02 = 0.0
                    score_payload["score"] = {
                        "reconstruction_f_score_002": float(f02),
                        "trajectory_path_score": float(traj.path_score),
                        "weights": {"reconstruction": float(w_recon), "trajectory": float(w_traj)},
                        "total": float(w_recon * f02 + w_traj * float(traj.path_score)),
                    }
                else:
                    score_payload["metrics"] = None
                    score_payload["score"] = {
                        "reconstruction_f_score_002": 0.0,
                        "trajectory_path_score": float(traj.path_score),
                        "weights": {"reconstruction": float(w_recon), "trajectory": float(w_traj)},
                        "total": float(w_traj * float(traj.path_score)),
                    }

                (out_dir / "reconstruction_metrics.json").write_text(
                    json.dumps(score_payload, indent=2)
                )
                payload["reconstruction_metrics"] = str(out_dir / "reconstruction_metrics.json")
            except Exception as e:
                import traceback
                payload["reconstruction_metrics_error"] = f"{type(e).__name__}: {e}"
                try:
                    (out_dir / "reconstruction_metrics_error.txt").write_text(
                        "".join(traceback.format_exception(type(e), e, e.__traceback__))[-4000:]
                    )
                    payload["reconstruction_metrics_error_file"] = str(out_dir / "reconstruction_metrics_error.txt")
                except Exception:
                    pass

        print(json.dumps(payload, indent=2))

    finally:
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.terminate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
