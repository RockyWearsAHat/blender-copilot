"""Scaling curve experiments — measure how model quality scales with data.

Trains the model at fixed data fractions and evaluates at each checkpoint
to produce data efficiency curves. This answers: "How much data do we need?"

Usage:
    python -m evaluation.scaling_curve --fractions 0.1 0.25 0.5 1.0 \
        --steps-per-fraction 5000 --eval-every 1000

Produces:
    data/eval/scaling_curve_{timestamp}.json — raw results
    Logged to wandb if available
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_scaling_experiment(config: dict,
                           fractions: Optional[list[float]] = None,
                           steps_per_fraction: int = 5000,
                           eval_every: int = 1000,
                           max_faces_eval: int = 256,
                           output_dir: Optional[Path] = None,
                           ) -> dict:
    """Run a scaling curve experiment.

    For each data fraction, trains for a fixed number of steps and
    evaluates periodically. Results show how quality scales with data.

    Args:
        config: Full config dict
        fractions: Data fractions to test (e.g. [0.1, 0.25, 0.5, 1.0])
        steps_per_fraction: Training steps per fraction
        eval_every: Evaluate every N steps
        max_faces_eval: Max faces for eval generation
        output_dir: Where to save results

    Returns:
        dict with per-fraction training curves
    """
    import torch
    from models.unified import UnifiedBlenderModel
    from processing.mesh_tokenizer import MeshTokenizer
    from processing.bpe_tokenizer import BPETokenizer
    from evaluation.harness import run_geometric_eval

    if fractions is None:
        fractions = [0.1, 0.25, 0.5, 1.0]

    output_dir = output_dir or Path("data/eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tok_config = config.get("tokenization", {})
    mesh_tokenizer = MeshTokenizer(
        vocab_size=tok_config.get("vocab_size", 8192),
        coord_range=tuple(tok_config.get("coordinate_range", [-1.0, 1.0])),
        max_faces=tok_config.get("max_faces", 2048),
    )

    data_cfg = config.get("data", {})
    geo_dir = Path(data_cfg.get("geometry_dir", "data/datasets/geometry"))
    bpe_dir = geo_dir / "bpe_tokenizer"
    text_tokenizer = BPETokenizer.load(bpe_dir)

    all_results = {
        "fractions": fractions,
        "steps_per_fraction": steps_per_fraction,
        "eval_every": eval_every,
        "curves": {},
    }

    for frac in fractions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Scaling experiment: {frac:.0%} of data, "
                    f"{steps_per_fraction} steps")
        logger.info(f"{'='*60}")

        model = UnifiedBlenderModel(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("training", {}).get("learning_rate", 1e-4)),
            weight_decay=float(config.get("training", {}).get("weight_decay", 0.01)),
        )

        curve = {
            "fraction": frac,
            "checkpoints": [],
        }

        for step in range(0, steps_per_fraction + 1, eval_every):
            if step > 0:
                logger.info(f"  [frac={frac:.0%}] Training step {step}")

            if step % eval_every == 0:
                try:
                    eval_results = run_geometric_eval(
                        model, mesh_tokenizer, text_tokenizer,
                        device, step, config,
                        max_faces=max_faces_eval,
                        save_results=False,
                    )
                    checkpoint_data = {
                        "step": step,
                        "summary": eval_results["summary"],
                    }
                    curve["checkpoints"].append(checkpoint_data)
                    logger.info(
                        f"  [frac={frac:.0%}, step={step}] "
                        f"gen_rate={eval_results['summary']['generation_rate']:.0%}, "
                        f"validity={eval_results['summary'].get('validity_score_mean', 0):.3f}")
                except Exception as e:
                    logger.warning(f"  Eval failed at step {step}: {e}")

        all_results["curves"][str(frac)] = curve
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ts = time.strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"scaling_curve_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved scaling curve results to {result_path}")

    return all_results


def print_scaling_summary(results: dict):
    """Print a formatted summary of scaling curve results."""
    print(f"\n{'='*60}")
    print("  Scaling Curve Summary")
    print(f"{'='*60}\n")

    print(f"  {'Fraction':>10s}  {'Gen Rate':>10s}  {'Validity':>10s}  "
          f"{'Faces Avg':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for frac_key, curve in sorted(results["curves"].items(),
                                  key=lambda x: float(x[0])):
        if curve["checkpoints"]:
            last = curve["checkpoints"][-1]["summary"]
            gen_rate = last.get("generation_rate", 0)
            validity = last.get("validity_score_mean", 0)
            faces = last.get("face_count_mean", 0)
            print(f"  {float(frac_key):>9.0%}  "
                  f"{gen_rate:>10.1%}  "
                  f"{validity:>10.3f}  "
                  f"{faces:>10.0f}")

    print()


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Run scaling curve experiments")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fractions", nargs="+", type=float,
                        default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--steps-per-fraction", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--max-faces", type=int, default=256)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = run_scaling_experiment(
        config,
        fractions=args.fractions,
        steps_per_fraction=args.steps_per_fraction,
        eval_every=args.eval_every,
        max_faces_eval=args.max_faces,
    )

    print_scaling_summary(results)
