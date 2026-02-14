"""Training script for the modifier prediction model.

Trains a ModifierModel to predict modifier stacks from text + mesh statistics.

Usage:
    python -m training.train_modifiers \
        --config config.yaml \
        --data data/datasets/modifiers_train.jsonl \
        --val_data data/datasets/modifiers_val.jsonl \
        --output checkpoints/modifiers/
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import yaml

logger = logging.getLogger(__name__)


class ModifierDataset(Dataset):
    """Dataset of (text, mesh_stats, modifier_stack) triples."""

    def __init__(self, jsonl_path: str, max_text_len: int = 128):
        self.max_text_len = max_text_len
        self.samples = []

        from models.modifiers.model import (
            MODIFIER_TYPE_TO_ID, MAX_MODIFIERS, PARAMS_PER_MODIFIER,
        )
        self.type_to_id = MODIFIER_TYPE_TO_ID
        self.max_mods = MAX_MODIFIERS
        self.params_per_mod = PARAMS_PER_MODIFIER

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} modifier samples from {jsonl_path}")

    def text_to_ids(self, text: str) -> torch.Tensor:
        ids = [ord(c) % 32000 for c in text[:self.max_text_len]]
        ids = ids + [0] * (self.max_text_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def text_to_mask(self, text: str) -> torch.Tensor:
        length = min(len(text), self.max_text_len)
        mask = [1.0] * length + [0.0] * (self.max_text_len - length)
        return torch.tensor(mask, dtype=torch.float)

    def encode_mesh_stats(self, stats: dict) -> torch.Tensor:
        """Encode mesh statistics into a fixed-size vector."""
        features = [
            math.log1p(stats.get("vertex_count", 0)),
            math.log1p(stats.get("face_count", 0)),
            math.log1p(stats.get("edge_count", 0)),
            stats.get("bbox_width", 1.0),
            stats.get("bbox_depth", 1.0),
            stats.get("bbox_height", 1.0),
            math.log1p(stats.get("surface_area", 0)),
            stats.get("avg_edge_length", 0.1),
            float(stats.get("has_ngons", False)),
            float(stats.get("has_quads_only", False)),
            float(stats.get("has_tris_only", False)),
            float(stats.get("is_manifold", True)),
        ]
        return torch.tensor(features, dtype=torch.float)

    def encode_modifier_stack(self, modifiers: list) -> tuple:
        """Encode modifier stack into targets.

        Returns:
            count: int
            type_ids: list of int
            params: list of float lists
        """
        count = min(len(modifiers), self.max_mods)
        type_ids = []
        params = []

        for i in range(self.max_mods):
            if i < len(modifiers):
                mod = modifiers[i]
                mod_type = mod.get("type", "NONE")
                tid = self.type_to_id.get(mod_type, 0)
                type_ids.append(tid)
                # Encode params as raw floats (the model learns to predict these)
                param_vals = self._extract_params(mod)
                params.append(param_vals)
            else:
                type_ids.append(0)  # NONE
                params.append([0.0] * self.params_per_mod)

        return count, type_ids, params

    def _extract_params(self, mod: dict) -> list:
        """Extract parameter values from a modifier dict as a flat float list."""
        p = [0.0] * self.params_per_mod
        mod_type = mod.get("type", "")

        if mod_type == "SUBSURF":
            p[0] = float(mod.get("levels", 1)) / 4.0
            p[1] = float(mod.get("render_levels", 2)) / 6.0
            p[2] = 1.0 if mod.get("use_creases", False) else 0.0
        elif mod_type == "MIRROR":
            p[0] = 1.0 if mod.get("use_axis_x", True) else 0.0
            p[1] = 1.0 if mod.get("use_axis_y", False) else 0.0
            p[2] = 1.0 if mod.get("use_axis_z", False) else 0.0
            p[3] = 1.0 if mod.get("use_clip", True) else 0.0
            p[4] = mod.get("merge_threshold", 0.001) * 100
        elif mod_type == "BEVEL":
            p[0] = mod.get("width", 0.02) * 10
            p[1] = float(mod.get("segments", 1)) / 10.0
            p[2] = 1.0 if mod.get("limit_method", "NONE") == "ANGLE" else 0.0
            p[3] = mod.get("angle_limit", 0.524) / math.pi
        elif mod_type == "SOLIDIFY":
            p[0] = mod.get("thickness", 0.01) * 10
            p[1] = mod.get("offset", -1.0)
            p[2] = 1.0 if mod.get("use_even_offset", True) else 0.0
        elif mod_type == "ARRAY":
            p[0] = float(mod.get("count", 2)) / 20.0
            p[1] = 1.0 if mod.get("use_relative_offset", True) else 0.0
            offset = mod.get("relative_offset_displace", [1, 0, 0])
            if isinstance(offset, list) and len(offset) >= 3:
                p[2:5] = [float(x) for x in offset[:3]]
        elif mod_type == "SMOOTH":
            p[0] = mod.get("factor", 0.5)
            p[1] = float(mod.get("iterations", 1)) / 20.0

        return p

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.get("text", "")
        mesh_stats = sample.get("mesh_stats", {})
        modifiers = sample.get("modifier_stack", [])

        text_ids = self.text_to_ids(text)
        text_mask = self.text_to_mask(text)
        stats_vec = self.encode_mesh_stats(mesh_stats)
        count, type_ids, params = self.encode_modifier_stack(modifiers)

        return {
            "text_ids": text_ids,
            "text_mask": text_mask,
            "mesh_stats": stats_vec,
            "target_count": torch.tensor(count - 1, dtype=torch.long),  # 0-indexed
            "target_types": torch.tensor(type_ids, dtype=torch.long),
            "target_params": torch.tensor(params, dtype=torch.float),
        }


def compute_loss(outputs: dict, batch: dict, device: torch.device) -> torch.Tensor:
    """Compute combined loss for modifier prediction.

    Loss = count_loss + type_loss + param_loss
    """
    from models.modifiers.model import MAX_MODIFIERS, NUM_MODIFIER_TYPES

    # Count loss
    count_loss = F.cross_entropy(
        outputs["count_logits"],
        batch["target_count"].to(device),
    )

    # Type loss (per slot)
    type_loss = 0
    for i in range(MAX_MODIFIERS):
        type_loss += F.cross_entropy(
            outputs["type_logits"][i],
            batch["target_types"][:, i].to(device),
        )
    type_loss /= MAX_MODIFIERS

    # Parameter loss (MSE, only for non-NONE slots)
    param_loss = 0
    n_valid = 0
    target_types = batch["target_types"].to(device)
    target_params = batch["target_params"].to(device)

    for i in range(MAX_MODIFIERS):
        # Mask: only compute param loss for non-NONE slots
        mask = (target_types[:, i] > 0).float()
        if mask.sum() == 0:
            continue

        # Get predicted params for the target type
        pred_all_params = outputs["param_values"][i]  # (B, NUM_TYPES, PARAMS)
        target_type_ids = target_types[:, i]  # (B,)

        # Gather predicted params for the correct type
        pred_params = pred_all_params[
            torch.arange(pred_all_params.shape[0], device=device),
            target_type_ids,
        ]  # (B, PARAMS)

        target_p = target_params[:, i, :]  # (B, PARAMS)
        slot_loss = F.mse_loss(pred_params, target_p, reduction="none")
        slot_loss = (slot_loss.mean(dim=-1) * mask).sum() / mask.sum()
        param_loss += slot_loss
        n_valid += 1

    if n_valid > 0:
        param_loss /= n_valid

    total_loss = count_loss + type_loss + 0.5 * param_loss
    return total_loss


def train(config: dict, data_path: str, val_path: str, output_dir: str):
    """Main training loop."""
    from models.modifiers.model import ModifierModel

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    lr = train_cfg.get("learning_rate", 1e-3)
    max_steps = train_cfg.get("max_steps", 20000)
    warmup_steps = train_cfg.get("warmup_steps", 200)
    eval_every = train_cfg.get("eval_every", 300)
    save_every = train_cfg.get("save_every", 1000)

    dataset = ModifierDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)

    val_loader = None
    if val_path and Path(val_path).exists():
        val_dataset = ModifierDataset(val_path)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=1, pin_memory=True)

    model = ModifierModel(config).to(device)
    param_count = model.count_parameters()
    logger.info(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(output_dir, exist_ok=True)

    model.train()
    step = 0
    best_val_loss = float("inf")
    data_iter = iter(dataloader)

    logger.info(f"Starting training for {max_steps} steps")

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        mesh_stats = batch["mesh_stats"].to(device)

        outputs = model(text_ids, text_mask, mesh_stats)
        loss = compute_loss(outputs, batch, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        step += 1

        if step % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Step {step}/{max_steps} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

        if val_loader and step % eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            logger.info(f"  Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "loss": val_loss,
                }, os.path.join(output_dir, "best.pt"))
                logger.info(f"  New best model saved!")

            model.train()

        if step % save_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "loss": loss.item(),
            }, os.path.join(output_dir, f"step_{step}.pt"))

    torch.save({
        "model_state_dict": model.state_dict(),
        "step": step,
        "loss": loss.item(),
    }, os.path.join(output_dir, "final.pt"))
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)
            mesh_stats = batch["mesh_stats"].to(device)

            outputs = model(text_ids, text_mask, mesh_stats)
            loss = compute_loss(outputs, batch, device)
            total_loss += loss.item()
            n += 1
            if n >= 30:
                break

    return total_loss / max(1, n)


def main():
    parser = argparse.ArgumentParser(description="Train modifier prediction model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", required=True, help="Training data JSONL")
    parser.add_argument("--val_data", default="", help="Validation data JSONL")
    parser.add_argument("--output", default="checkpoints/modifiers/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args.data, args.val_data, args.output)


if __name__ == "__main__":
    main()
