#!/usr/bin/env python3
"""Predict procedural-noise parameter buckets from prompt intent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.noise_intent_predictor import NoiseIntentPredictor
from processing.procedural_displacement import params_from_buckets
from policy.state import hash_text_features


def _detect_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    p = argparse.ArgumentParser(description="Predict bucketed noise params from text")
    p.add_argument("--ckpt", type=Path, default=Path("checkpoints/noise_intent/latest.pt"))
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = p.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu")
    cfg = payload.get("config", {})

    model = NoiseIntentPredictor(
        in_dim=int(cfg.get("feat_dim", 64)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        num_layers=int(cfg.get("layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        bins=int(cfg.get("bins", 32)),
    )
    model.load_state_dict(payload["model"], strict=True)

    device = _detect_device(args.device)
    model = model.to(device)
    model.eval()

    feat = hash_text_features(args.prompt, dim=int(cfg.get("feat_dim", 64)))
    x = torch.from_numpy(feat).to(device=device, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        lg_scale, lg_detail, lg_rough, lg_strength = model(x)

    b_scale = int(lg_scale[0].argmax().item())
    b_detail = int(lg_detail[0].argmax().item())
    b_rough = int(lg_rough[0].argmax().item())
    b_strength = int(lg_strength[0].argmax().item())
    params = params_from_buckets(
        seed=int(args.seed),
        scale_bucket=b_scale,
        detail_bucket=b_detail,
        roughness_bucket=b_rough,
        distortion_bucket=0,
        strength_bucket=b_strength,
        midlevel_bucket=16,
    )

    print(
        json.dumps(
            {
                "prompt": args.prompt,
                "buckets": {
                    "scale": b_scale,
                    "detail": b_detail,
                    "roughness": b_rough,
                    "strength": b_strength,
                },
                "params": {
                    "scale": float(params.scale),
                    "detail": float(params.detail),
                    "roughness": float(params.roughness),
                    "distortion": float(params.distortion),
                    "strength": float(params.strength),
                    "midlevel": float(params.midlevel),
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
