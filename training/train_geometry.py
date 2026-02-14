"""Training script for the geometry (text-to-mesh) model.

Usage:
    python -m training.train_geometry \
        --dataset data/datasets/geometry \
        --output models/geometry/checkpoints \
        --config config.yaml
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


class MeshDataset(Dataset):
    """Dataset of (text, mesh_tokens) pairs from JSONL files."""

    def __init__(self, data_path: str, max_text_length: int = 256,
                 max_mesh_tokens: int = 18432,
                 text_tokenizer=None):
        self.examples = []
        self.max_text_length = max_text_length
        self.max_mesh_tokens = max_mesh_tokens
        self.text_tokenizer = text_tokenizer

        with open(data_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                self.examples.append(ex)

        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        text = ex["text"]

        if self.text_tokenizer is not None:
            # Proper word-level tokenizer
            text_ids, text_mask = self.text_tokenizer.encode_padded(
                text, max_length=self.max_text_length)
        else:
            # Fallback: character-level (legacy compat)
            text_ids = [ord(c) % 32000 for c in text[:self.max_text_length]]
            text_ids = text_ids + [0] * (self.max_text_length - len(text_ids))
            text_mask = [1] * min(len(text), self.max_text_length)
            text_mask = text_mask + [0] * (self.max_text_length - len(text_mask))

        # Mesh tokens (already tokenized in dataset)
        mesh_tokens = ex["tokens"][:self.max_mesh_tokens]
        # Pad
        mesh_len = len(mesh_tokens)
        mesh_tokens = mesh_tokens + [0] * (self.max_mesh_tokens - mesh_len)
        mesh_mask = [1] * mesh_len + [0] * (self.max_mesh_tokens - mesh_len)

        return {
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.float),
            "mesh_tokens": torch.tensor(mesh_tokens, dtype=torch.long),
            "mesh_mask": torch.tensor(mesh_mask, dtype=torch.float),
        }


def train(args):
    from models.geometry.model import GeometryModel

    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_config = config.get("training", {})
    geo_config = config.get("models", {}).get("geometry", {})

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Text tokenizer — build from training data or load saved
    from processing.text_tokenizer import TextTokenizer
    dataset_dir = Path(args.dataset)
    tokenizer_path = dataset_dir / "text_tokenizer.json"

    if tokenizer_path.exists():
        text_tokenizer = TextTokenizer.load(tokenizer_path)
        logger.info(f"Loaded text tokenizer: {text_tokenizer}")
    else:
        train_jsonl = dataset_dir / "train.jsonl"
        text_tokenizer = TextTokenizer.from_dataset(train_jsonl)
        text_tokenizer.save(tokenizer_path)
        logger.info(f"Built and saved text tokenizer: {text_tokenizer}")

    # Model
    model = GeometryModel(config).to(device)
    param_count = model.count_parameters()
    logger.info(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Data
    train_dataset = MeshDataset(
        dataset_dir / "train.jsonl",
        max_mesh_tokens=geo_config.get("max_sequence_length", 18432),
        text_tokenizer=text_tokenizer,
    )
    val_dataset = MeshDataset(
        dataset_dir / "val.jsonl",
        max_mesh_tokens=geo_config.get("max_sequence_length", 18432),
        text_tokenizer=text_tokenizer,
    )

    batch_size = train_config.get("batch_size", 8)
    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=pin_mem)

    # Optimizer
    lr = float(train_config.get("learning_rate", 1e-4))
    weight_decay = float(train_config.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)

    # Scheduler
    max_steps = train_config.get("max_steps", 100000)
    warmup_steps = train_config.get("warmup_steps", 1000)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Mixed precision
    use_amp = train_config.get("mixed_precision", "fp16") != "fp32"
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD token

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    grad_accum = train_config.get("gradient_accumulation_steps", 4)
    eval_every = train_config.get("eval_every", 1000)
    save_every = train_config.get("save_every", 5000)

    global_step = 0
    best_val_loss = float("inf")

    logger.info(f"Starting training: {max_steps} steps, batch={batch_size}, "
                f"grad_accum={grad_accum}, lr={lr}")

    # Wandb logging (optional)
    use_wandb = False
    try:
        import wandb
        wandb.init(project="blender-mesh-gen", config=config)
        use_wandb = True
    except Exception:
        logger.info("Wandb not available, logging to stdout only")

    model.train()
    optimizer.zero_grad()

    while global_step < max_steps:
        for batch in train_loader:
            if global_step >= max_steps:
                break

            text_ids = batch["text_ids"].to(device)
            text_mask = batch["text_mask"].to(device)
            mesh_tokens = batch["mesh_tokens"].to(device)

            # Input: all tokens except last
            # Target: all tokens except first (shifted right)
            input_tokens = mesh_tokens[:, :-1]
            target_tokens = mesh_tokens[:, 1:]

            with autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(text_ids, text_mask, input_tokens)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_tokens.reshape(-1),
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (global_step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            actual_loss = loss.item() * grad_accum

            if global_step % 10 == 0:
                lr_current = scheduler.get_last_lr()[0]
                logger.info(f"Step {global_step}/{max_steps} | "
                            f"Loss: {actual_loss:.4f} | LR: {lr_current:.2e}")
                if use_wandb:
                    wandb.log({"train/loss": actual_loss,
                               "train/lr": lr_current,
                               "step": global_step})

            # Evaluation
            if global_step > 0 and global_step % eval_every == 0:
                val_loss = evaluate(model, val_loader, criterion, device, use_amp)
                logger.info(f"  Val loss: {val_loss:.4f}")
                if use_wandb:
                    wandb.log({"val/loss": val_loss, "step": global_step})
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, scheduler, global_step,
                                    val_loss, output_dir / "best.pt")

            # Save checkpoint
            if global_step > 0 and global_step % save_every == 0:
                save_checkpoint(model, optimizer, scheduler, global_step,
                                actual_loss, output_dir / f"step_{global_step}.pt")

            global_step += 1

    # Final save
    save_checkpoint(model, optimizer, scheduler, global_step,
                    actual_loss, output_dir / "final.pt")
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0
    count = 0

    for batch in val_loader:
        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        mesh_tokens = batch["mesh_tokens"].to(device)

        input_tokens = mesh_tokens[:, :-1]
        target_tokens = mesh_tokens[:, 1:]

        with autocast(device_type="cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(text_ids, text_mask, input_tokens)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
            )

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


def save_checkpoint(model, optimizer, scheduler, step, loss, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "loss": loss,
    }, path)
    logger.info(f"  Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train geometry model")
    parser.add_argument("--dataset", required=True,
                        help="Path to geometry dataset directory")
    parser.add_argument("--output", required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    train(args)


if __name__ == "__main__":
    main()
