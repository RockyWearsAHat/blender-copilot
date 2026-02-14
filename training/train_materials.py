"""Training script for the material generation model.

Trains a MaterialModel to generate shader node graphs from text descriptions.

Usage:
    python -m training.train_materials \
        --config config.yaml \
        --data data/datasets/materials_train.jsonl \
        --val_data data/datasets/materials_val.jsonl \
        --output checkpoints/materials/
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import yaml

logger = logging.getLogger(__name__)


class MaterialDataset(Dataset):
    """Dataset of (text, material_tokens) pairs."""

    def __init__(self, jsonl_path: str, max_text_len: int = 128,
                 max_material_len: int = 512, vocab_size: int = 4096):
        self.max_text_len = max_text_len
        self.max_material_len = max_material_len
        self.vocab_size = vocab_size
        self.samples = []

        from models.materials.model import MaterialEncoder
        self.encoder = MaterialEncoder(vocab_size=vocab_size)

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} material samples from {jsonl_path}")

    def text_to_ids(self, text: str) -> torch.Tensor:
        """Character-level text tokenization."""
        ids = [ord(c) % 32000 for c in text[:self.max_text_len]]
        ids = ids + [0] * (self.max_text_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def text_to_mask(self, text: str) -> torch.Tensor:
        length = min(len(text), self.max_text_len)
        mask = [1.0] * length + [0.0] * (self.max_text_len - length)
        return torch.tensor(mask, dtype=torch.float)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.get("text", "material")
        material_data = sample.get("node_tree", {})

        # Tokenize material
        tokens = self.encoder.encode_material(material_data)

        # Pad/truncate
        if len(tokens) > self.max_material_len:
            tokens = tokens[:self.max_material_len]
        else:
            tokens = tokens + [0] * (self.max_material_len - len(tokens))

        # Input tokens = tokens[:-1], target = tokens[1:]
        input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(tokens[1:], dtype=torch.long)

        text_ids = self.text_to_ids(text)
        text_mask = self.text_to_mask(text)

        return {
            "text_ids": text_ids,
            "text_mask": text_mask,
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
        }


def train(config: dict, data_path: str, val_path: str, output_dir: str):
    """Main training loop."""
    from models.materials.model import MaterialModel

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Training config
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 16)
    lr = train_cfg.get("learning_rate", 3e-4)
    max_steps = train_cfg.get("max_steps", 50000)
    warmup_steps = train_cfg.get("warmup_steps", 500)
    eval_every = train_cfg.get("eval_every", 500)
    save_every = train_cfg.get("save_every", 2000)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)

    # Dataset
    dataset = MaterialDataset(data_path, vocab_size=config.get("tokenization", {}).get("vocab_size", 4096))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)

    val_dataset = None
    val_loader = None
    if val_path and Path(val_path).exists():
        val_dataset = MaterialDataset(val_path, vocab_size=config.get("tokenization", {}).get("vocab_size", 4096))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=1, pin_memory=True)

    # Model
    model = MaterialModel(config).to(device)
    param_count = model.count_parameters()
    logger.info(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # LR schedule
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Output directory
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    model.train()
    step = 0
    best_val_loss = float("inf")
    data_iter = iter(dataloader)

    logger.info(f"Starting training for {max_steps} steps")

    while step < max_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        input_tokens = batch["input_tokens"].to(device)
        target_tokens = batch["target_tokens"].to(device)

        # Forward
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(text_ids, text_mask, input_tokens)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_tokens.view(-1),
                    ignore_index=0,
                )
                loss = loss / grad_accum
        else:
            logits = model(text_ids, text_mask, input_tokens)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_tokens.view(-1),
                ignore_index=0,
            )
            loss = loss / grad_accum

        # Backward
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step
        if (step + 1) % grad_accum == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        step += 1

        # Log
        if step % 50 == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Step {step}/{max_steps} | Loss: {loss.item() * grad_accum:.4f} | LR: {current_lr:.2e}")

        # Evaluate
        if val_loader and step % eval_every == 0:
            val_loss = evaluate(model, val_loader, device, use_amp)
            logger.info(f"  Validation loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, step, val_loss,
                              os.path.join(output_dir, "best.pt"))
                logger.info(f"  New best model saved!")

            model.train()

        # Save checkpoint
        if step % save_every == 0:
            save_checkpoint(model, optimizer, step, loss.item(),
                          os.path.join(output_dir, f"step_{step}.pt"))

    # Final save
    save_checkpoint(model, optimizer, step, loss.item(),
                  os.path.join(output_dir, "final.pt"))
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


def evaluate(model, val_loader, device, use_amp):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)
            input_tokens = batch["input_tokens"].to(device)
            target_tokens = batch["target_tokens"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(text_ids, text_mask, input_tokens)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target_tokens.view(-1),
                        ignore_index=0,
                    )
            else:
                logits = model(text_ids, text_mask, input_tokens)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_tokens.view(-1),
                    ignore_index=0,
                )

            total_loss += loss.item()
            n_batches += 1

            if n_batches >= 50:
                break

    return total_loss / max(1, n_batches)


def save_checkpoint(model, optimizer, step, loss, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train material generation model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", required=True, help="Training data JSONL")
    parser.add_argument("--val_data", default="", help="Validation data JSONL")
    parser.add_argument("--output", default="checkpoints/materials/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train(config, args.data, args.val_data, args.output)


if __name__ == "__main__":
    main()
