#!/usr/bin/env python3
"""Train an architecture-compliant policy model (Phase 1: imitation).

This deliberately does NOT train the legacy/unified mesh-token model.
It trains a compact policy transformer on compact mesh-stat state and a
finite action grammar, per ARCHITECTURE.md.

Usage:
  /path/to/python training/train_policy.py --config config.policy_m3_quick.yaml --device auto
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

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


def _autocast_ctx(device: torch.device, mp: str):
    mp = (mp or "fp32").lower()
    if mp == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)

    # MPS autocast supports fp16/bf16 depending on torch build.
    if device.type == "mps":
        dtype = torch.bfloat16 if mp in {"bf16", "bfloat16"} else torch.float16
        return torch.autocast(device_type="mps", dtype=dtype)
    if device.type == "cuda":
        dtype = torch.bfloat16 if mp in {"bf16", "bfloat16"} else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    return torch.autocast(device_type=device.type, enabled=False)


def _save_checkpoint(out_dir: Path, step: int, model: torch.nn.Module, opt: torch.optim.Optimizer, cfg: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    # torch.compile wraps modules and prefixes parameters with `_orig_mod.`.
    # Save the underlying module weights for maximal compatibility.
    raw_model = getattr(model, "_orig_mod", model)
    payload = {
        "step": step,
        "model": raw_model.state_dict(),
        "optimizer": opt.state_dict(),
        "config": cfg,
    }
    latest = out_dir / "latest.pt"
    torch.save(payload, latest)
    if step % int(cfg["training"].get("save_every", 500)) == 0:
        torch.save(payload, out_dir / f"step_{step}.pt")

def _canonicalize_state_dict(sd: dict) -> dict:
    """Strip torch.compile's `_orig_mod.` prefix if present.

    We save checkpoints from the *unwrapped* module for compatibility, but
    older checkpoints (or other tooling) may include `_orig_mod.`-prefixed keys.
    """
    if not isinstance(sd, dict) or not sd:
        return {}
    if any(str(k).startswith("_orig_mod.") for k in sd.keys()):
        return {str(k).replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd


def main() -> int:
    from models.policy_transformer import PolicyTransformer, count_parameters
    from policy.dataset import SyntheticImitationStream, RealMeshBuildTraceStream
    from policy.goal_sampler import (
        ListGoalSampler,
        UniformGoalSampler,
        load_goal_records_from_geometry_jsonl,
        load_goal_records_from_mesh_cache_dir,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config.policy_m3_quick.yaml"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compile", action="store_true", help="Try torch.compile(model) for speed (experimental)")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    goal_sampler = None
    goal_src = data_cfg.get("goal_source")
    if isinstance(goal_src, dict):
        src_type = str(goal_src.get("type") or "").strip().lower()
        src_path = goal_src.get("path")
        if src_type in {"geometry_jsonl", "geometry"} and isinstance(src_path, str):
            recs = load_goal_records_from_geometry_jsonl(
                path=Path(src_path),
                max_records=int(goal_src.get("max_records", 50_000)),
                min_vertices=int(goal_src.get("min_vertices", 50)),
                max_vertices=int(goal_src.get("max_vertices", 20_000)),
            )
            goal_sampler = ListGoalSampler(recs)
        elif src_type in {"mesh_cache", "mesh_cache_dir"} and isinstance(src_path, str):
            recs = load_goal_records_from_mesh_cache_dir(
                cache_dir=Path(src_path),
                max_files=int(goal_src.get("max_files", 20_000)),
                min_vertices=int(goal_src.get("min_vertices", 50)),
                max_vertices=int(goal_src.get("max_vertices", 20_000)),
                min_quality_weight=(float(goal_src["min_quality_weight"]) if "min_quality_weight" in goal_src else None),
            )
            goal_sampler = ListGoalSampler(recs)
        elif src_type in {"uniform", "random"}:
            goal_sampler = UniformGoalSampler()

    device = _detect_device(args.device)

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

    if args.compile or bool(train_cfg.get("torch_compile", False)):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception as e:
            print(json.dumps({"warn": "torch.compile_failed", "error": str(e), "device": device.type}))

    param_count = count_parameters(model)
    if param_count > 50_000_000:
        raise RuntimeError(f"Policy model too large for ARCHITECTURE.md: {param_count:,} params")

    lr = float(train_cfg.get("learning_rate", 3e-4))
    wd = float(train_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    out_dir = PROJECT_ROOT / str(train_cfg.get("checkpoint_dir", "checkpoints/policy"))
    start_step = 0

    if args.resume:
        ckpt_path = out_dir / "latest.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            raw_model = getattr(model, "_orig_mod", model)
            raw_model.load_state_dict(_canonicalize_state_dict(ckpt.get("model", {})), strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = int(ckpt.get("step", 0))

    stream = SyntheticImitationStream(
        seed=int(data_cfg.get("seed", 0)),
        seq_len=int(data_cfg.get("seq_len", 64)),
        batch_size=int(data_cfg.get("batch_size", 32)),
        state_dim=int(model_cfg.get("state_dim", 10)),
        goal_sampler=goal_sampler,
        mask_numeric_goal_prob=float(data_cfg.get("mask_numeric_goal_prob", 0.0)),
        trajectory_mode=str(data_cfg.get("trajectory_mode", "forward")),
        reverse_prob=float(data_cfg.get("reverse_prob", 0.5)),
    )
    it = iter(stream)

    real_trace_it = None
    trace_mix_prob = 0.0
    trace_src = data_cfg.get("collapse_trace_source")
    if isinstance(trace_src, dict):
        trace_path = trace_src.get("path")
        if isinstance(trace_path, str) and trace_path.strip():
            real_stream = RealMeshBuildTraceStream(
                trace_root=Path(trace_path),
                seed=int(data_cfg.get("seed", 0)) + 1337,
                seq_len=int(data_cfg.get("seq_len", 64)),
                batch_size=int(data_cfg.get("batch_size", 32)),
                state_dim=int(model_cfg.get("state_dim", 10)),
                max_traces=int(trace_src.get("max_traces", 50_000)),
                mask_numeric_goal_prob=float(data_cfg.get("mask_numeric_goal_prob", 0.0)),
            )
            real_trace_it = iter(real_stream)
            trace_mix_prob = float(np.clip(float(trace_src.get("mix_prob", 0.3)), 0.0, 1.0))

    sel_rng = np.random.default_rng(int(data_cfg.get("seed", 0)) + 2026)

    type_loss_fn = torch.nn.CrossEntropyLoss()
    param_loss_fn = torch.nn.CrossEntropyLoss()

    mp = str(train_cfg.get("mixed_precision", "fp16"))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    type_loss_weight = float(train_cfg.get("type_loss_weight", 1.0))
    param_loss_weight = float(train_cfg.get("param_loss_weight", 0.5))

    noop_pred_penalty = float(train_cfg.get("noop_pred_penalty", 0.0))
    noop_action_index = int(train_cfg.get("noop_action_index", 11))

    model.train()
    t0 = time.time()
    last_log_time = t0
    last_log_step = start_step
    for step in range(start_step + 1, start_step + args.max_steps + 1):
        use_real_trace = (real_trace_it is not None) and (float(sel_rng.random()) < trace_mix_prob)
        batch = next(real_trace_it if use_real_trace else it)
        states = batch.states.to(device=device, dtype=torch.float32)
        a_type = batch.action_type.to(device=device)
        a_param = batch.action_param.to(device=device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device, mp):
            type_logits, param_logits = model(states)
            # Flatten (B*T)
            bsz, seq_len = a_type.shape
            type_loss = type_loss_fn(type_logits.reshape(bsz * seq_len, -1), a_type.reshape(-1))
            param_loss = param_loss_fn(param_logits.reshape(bsz * seq_len, -1), a_param.reshape(-1))

            loss = (type_loss_weight * type_loss) + (param_loss_weight * param_loss)

            # Anti-collapse regularizer: penalize assigning probability mass to
            # NOOP on tokens where the target action is not NOOP.
            if noop_pred_penalty > 0.0 and 0 <= noop_action_index < type_logits.shape[-1]:
                with torch.no_grad():
                    non_noop_mask = (a_type != noop_action_index)
                if bool(non_noop_mask.any()):
                    probs = torch.softmax(type_logits, dim=-1)
                    p_noop = probs[..., noop_action_index]
                    penalty = p_noop[non_noop_mask].mean()
                    loss = loss + (noop_pred_penalty * penalty)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % int(train_cfg.get("log_every", 20)) == 0:
            now = time.time()
            dt_total = now - t0
            dt_window = now - last_log_time
            steps_window = step - last_log_step
            sps_avg = (step - start_step) / max(1e-6, dt_total)
            sps_window = steps_window / max(1e-6, dt_window)
            msg = {
                "step": step,
                "loss": float(loss.item()),
                "type_loss": float(type_loss.item()),
                "param_loss": float(param_loss.item()),
                "device": device.type,
                "params": int(param_count),
                "steps_per_sec_avg": float(sps_avg),
                "steps_per_sec_window": float(sps_window),
                "trace_mix_prob": float(trace_mix_prob),
                "batch_source": ("real_trace" if use_real_trace else "synthetic"),
            }
            print(json.dumps(msg))

            last_log_time = now
            last_log_step = step

        if step % int(train_cfg.get("save_every", 200)) == 0:
            _save_checkpoint(out_dir, step, model, optimizer, cfg)

    _save_checkpoint(out_dir, start_step + args.max_steps, model, optimizer, cfg)
    print("OK: policy training completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
