#!/usr/bin/env python3
"""Train a compact intent->noise bucket predictor.

Phase-1 goal: learn prompt semantics to bucketed procedural params without
training any image generator.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.noise_intent_predictor import NoiseIntentPredictor, count_parameters
from policy.state import hash_text_features


PROTOTYPES: dict[str, dict] = {
    "jagged_rock": {
        "templates": [
            "jagged rocky height",
            "sharp rocky displacement",
            "craggy mountain terrain",
            "steep rough cliff surface",
        ],
        "target": {"scale": 9, "detail": 24, "roughness": 25, "strength": 23},
    },
    "soft_dunes": {
        "templates": [
            "soft dunes",
            "smooth sandy hills",
            "gentle wind dunes",
            "broad soft terrain",
        ],
        "target": {"scale": 23, "detail": 8, "roughness": 8, "strength": 10},
    },
    "stylized_faceted": {
        "templates": [
            "stylized faceted terrain",
            "low poly faceted ground",
            "flat shaded terrain",
            "stylized low poly mountain",
        ],
        "target": {"scale": 16, "detail": 7, "roughness": 6, "strength": 11},
    },
    "rolling_hills": {
        "templates": [
            "rolling grassy hills",
            "hilly landscape",
            "rounded green terrain",
            "smooth mountain foothills",
        ],
        "target": {"scale": 20, "detail": 11, "roughness": 12, "strength": 14},
    },
    "volcanic": {
        "templates": [
            "volcanic rough terrain",
            "lava rock mountain",
            "charred jagged ground",
            "harsh black stone displacement",
        ],
        "target": {"scale": 12, "detail": 22, "roughness": 27, "strength": 25},
    },
}


def _detect_device(device_str: str) -> torch.device:
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _augment_prompt(base: str, rng: random.Random) -> str:
    prefixes = ["", "procedural", "terrain", "height map", "noise"]
    suffixes = ["", "for displacement", "for mountain", "for landscape", "style"]
    tokens = [t for t in [rng.choice(prefixes), base, rng.choice(suffixes)] if t]
    return " ".join(tokens)


def _jitter_bucket(value: int, rng: random.Random, jitter: int = 2, bins: int = 32) -> int:
    v = int(value) + int(rng.randint(-abs(jitter), abs(jitter)))
    return int(max(0, min(int(bins) - 1, v)))


def _build_dataset(*, n_per_proto: int, seed: int, feat_dim: int, bins: int) -> tuple[np.ndarray, dict[str, np.ndarray], list[str]]:
    rng = random.Random(int(seed))
    feat_rows: list[np.ndarray] = []
    labels = {"scale": [], "detail": [], "roughness": [], "strength": []}
    class_names: list[str] = []

    for proto_name, proto in PROTOTYPES.items():
        target = proto["target"]
        templates = list(proto["templates"])
        for _ in range(int(n_per_proto)):
            base = str(rng.choice(templates))
            prompt = _augment_prompt(base, rng)
            feat = hash_text_features(prompt, dim=int(feat_dim)).astype(np.float32, copy=False)
            feat_rows.append(feat)
            labels["scale"].append(_jitter_bucket(int(target["scale"]), rng, jitter=3, bins=bins))
            labels["detail"].append(_jitter_bucket(int(target["detail"]), rng, jitter=3, bins=bins))
            labels["roughness"].append(_jitter_bucket(int(target["roughness"]), rng, jitter=3, bins=bins))
            labels["strength"].append(_jitter_bucket(int(target["strength"]), rng, jitter=3, bins=bins))
            class_names.append(proto_name)

    x = np.stack(feat_rows, axis=0)
    y = {k: np.array(v, dtype=np.int64) for k, v in labels.items()}
    return x, y, class_names


def main() -> int:
    p = argparse.ArgumentParser(description="Train tiny intent->noise parameter predictor")
    p.add_argument("--out", type=Path, default=Path("checkpoints/noise_intent/latest.pt"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feat-dim", type=int, default=64)
    p.add_argument("--bins", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--samples-per-prototype", type=int, default=1200)
    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = _detect_device(args.device)
    x, y, _ = _build_dataset(
        n_per_proto=int(args.samples_per_prototype),
        seed=int(args.seed),
        feat_dim=int(args.feat_dim),
        bins=int(args.bins),
    )
    n = int(x.shape[0])
    perm = np.random.permutation(n)
    split = int(0.9 * n)
    idx_train, idx_val = perm[:split], perm[split:]

    x_train = torch.from_numpy(x[idx_train]).to(device=device, dtype=torch.float32)
    x_val = torch.from_numpy(x[idx_val]).to(device=device, dtype=torch.float32)
    y_train = {k: torch.from_numpy(v[idx_train]).to(device=device) for k, v in y.items()}
    y_val = {k: torch.from_numpy(v[idx_val]).to(device=device) for k, v in y.items()}

    model = NoiseIntentPredictor(
        in_dim=int(args.feat_dim),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.layers),
        dropout=float(args.dropout),
        bins=int(args.bins),
    ).to(device)
    if count_parameters(model) > 5_000_000:
        raise RuntimeError("NoiseIntentPredictor too large for this phase")

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    batch_size = max(1, int(args.batch_size))
    best_val = float("inf")
    best_state = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        total_loss = 0.0
        for start in range(0, x_train.shape[0], batch_size):
            sel = order[start : start + batch_size]
            xb = x_train[sel]
            logits = model(xb)
            loss = (
                F.cross_entropy(logits[0], y_train["scale"][sel])
                + F.cross_entropy(logits[1], y_train["detail"][sel])
                + F.cross_entropy(logits[2], y_train["roughness"][sel])
                + F.cross_entropy(logits[3], y_train["strength"][sel])
            ) / 4.0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * float(xb.shape[0])

        model.eval()
        with torch.no_grad():
            v_logits = model(x_val)
            v_loss = (
                F.cross_entropy(v_logits[0], y_val["scale"])
                + F.cross_entropy(v_logits[1], y_val["detail"])
                + F.cross_entropy(v_logits[2], y_val["roughness"])
                + F.cross_entropy(v_logits[3], y_val["strength"])
            ) / 4.0
            acc = {
                "scale": float((v_logits[0].argmax(dim=-1) == y_val["scale"]).float().mean().item()),
                "detail": float((v_logits[1].argmax(dim=-1) == y_val["detail"]).float().mean().item()),
                "roughness": float((v_logits[2].argmax(dim=-1) == y_val["roughness"]).float().mean().item()),
                "strength": float((v_logits[3].argmax(dim=-1) == y_val["strength"]).float().mean().item()),
            }

        train_loss = float(total_loss / max(1, x_train.shape[0]))
        val_loss = float(v_loss.item())
        print(
            json.dumps(
                {
                    "epoch": int(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc_scale": acc["scale"],
                    "val_acc_detail": acc["detail"],
                    "val_acc_roughness": acc["roughness"],
                    "val_acc_strength": acc["strength"],
                }
            )
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": best_state,
        "config": {
            "feat_dim": int(args.feat_dim),
            "bins": int(args.bins),
            "hidden_dim": int(args.hidden_dim),
            "layers": int(args.layers),
            "dropout": float(args.dropout),
            "prototypes": sorted(PROTOTYPES.keys()),
        },
        "best_val_loss": float(best_val),
        "params": int(count_parameters(model)),
        "seed": int(args.seed),
    }
    torch.save(payload, out)
    print(json.dumps({"saved": str(out), "best_val_loss": float(best_val), "device": device.type}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
