"""Evaluation harness — runs geometric evaluation during training.

Bridges the model's generate_geometry() with the evaluation metrics.
Handles tokenization, decoding, and result logging.

Usage from training:
    from evaluation.harness import run_geometric_eval
    results = run_geometric_eval(model, mesh_tokenizer, text_tokenizer,
                                 device, step, config)

Usage standalone:
    python -m evaluation.harness --checkpoint checkpoints/unified/latest.pt
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _encode_prompt(text: str, text_tokenizer, max_len: int,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a text prompt using the BPE or word-level tokenizer.

    Returns:
        (text_ids, text_mask) both shape (1, max_len) on device
    """
    from processing.bpe_tokenizer import BPETokenizer

    if isinstance(text_tokenizer, BPETokenizer):
        token_ids = text_tokenizer.encode(text)
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        pad_len = max_len - len(token_ids)
        mask = [1] * len(token_ids) + [0] * pad_len
        token_ids = token_ids + [0] * pad_len
    else:
        token_ids = text_tokenizer.encode(text)
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        pad_len = max_len - len(token_ids)
        mask = [1] * len(token_ids) + [0] * pad_len
        token_ids = token_ids + [0] * pad_len

    text_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    text_mask = torch.tensor([mask], dtype=torch.long, device=device)
    return text_ids, text_mask


def generate_mesh_from_model(model, mesh_tokenizer, text_tokenizer,
                             prompt: str, device: torch.device,
                             max_faces: int = 512,
                             temperature: float = 0.7,
                             top_k: int = 50,
                             top_p: float = 0.9,
                             max_text_len: int = 256,
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Generate a mesh from a text prompt using the model.

    Returns:
        (vertices, faces) — vertices is (V, 3), faces is (F, 3)
    """
    model.eval()

    text_ids, text_mask = _encode_prompt(
        prompt, text_tokenizer, max_text_len, device)

    max_tokens = 2 + max_faces * mesh_tokenizer.tokens_per_face

    with torch.no_grad():
        generated_tokens = model.generate_geometry(
            text_ids, text_mask,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    token_list = generated_tokens[0].cpu().tolist()

    vertices, faces = mesh_tokenizer.decode_tokens(token_list)

    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)

    return vertices, faces


def run_geometric_eval(model, mesh_tokenizer, text_tokenizer,
                       device: torch.device,
                       global_step: int,
                       config: dict,
                       max_faces: int = 256,
                       temperature: float = 0.7,
                       save_results: bool = True,
                       test_cases: Optional[list[dict]] = None,
                       ) -> dict:
    """Run full geometric evaluation and return results.

    This is the main entry point called from the training loop.
    Generates meshes for all test suite prompts, evaluates them,
    and logs/saves results.

    Args:
        model: UnifiedBlenderModel
        mesh_tokenizer: MeshTokenizer instance
        text_tokenizer: BPETokenizer or word-level tokenizer
        device: torch device
        global_step: current training step (for logging)
        config: full config dict
        max_faces: max faces per generation (lower = faster)
        temperature: sampling temperature
        save_results: whether to save results to disk
        test_cases: optional custom test cases

    Returns:
        dict with per-case results and summary
    """
    from evaluation.test_suite import run_test_suite, load_test_suite

    max_text_len = config.get("unified", {}).get("text_max_length", 256)
    cases = test_cases or load_test_suite()

    logger.info(f"Running geometric eval on {len(cases)} test cases "
                f"(max_faces={max_faces})...")
    start_time = time.time()

    def generate_fn(prompt, mf):
        return generate_mesh_from_model(
            model, mesh_tokenizer, text_tokenizer,
            prompt, device,
            max_faces=mf,
            temperature=temperature,
            max_text_len=max_text_len,
        )

    results = run_test_suite(generate_fn, cases, max_faces=max_faces)
    elapsed = time.time() - start_time

    try:
        from evaluation.domain_kpis import summarize_professional_kpis
        results["professional_kpis"] = summarize_professional_kpis(results)
    except Exception:
        pass

    results["step"] = global_step
    results["elapsed_seconds"] = elapsed
    results["max_faces"] = max_faces
    results["temperature"] = temperature

    summary = results["summary"]
    logger.info(
        f"  Geometric eval (step {global_step}): "
        f"gen_rate={summary['generation_rate']:.0%}, "
        f"expectations={summary['expectations_rate']:.0%}, "
        f"validity={summary.get('validity_score_mean', 0):.3f}, "
        f"faces_avg={summary.get('face_count_mean', 0):.0f}, "
        f"time={elapsed:.1f}s"
    )

    if save_results:
        _save_eval_results(results, global_step)

    return results


def _save_eval_results(results: dict, global_step: int):
    """Save evaluation results to data/eval/."""
    eval_dir = Path("data/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = f"geometric_eval_step{global_step}_{ts}.json"

    serializable = _make_serializable(results)

    with open(eval_dir / filename, "w") as f:
        json.dump(serializable, f, indent=2)

    with open(eval_dir / "results.jsonl", "a") as f:
        summary = {
            "step": global_step,
            "timestamp": ts,
            "type": "geometric_eval",
            **results["summary"],
        }
        f.write(json.dumps(_make_serializable(summary)) + "\n")

    logger.info(f"  Saved eval results to {eval_dir / filename}")


def _make_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def get_wandb_log_dict(results: dict, prefix: str = "eval") -> dict:
    """Extract wandb-loggable metrics from eval results."""
    summary = results["summary"]
    log_dict = {
        f"{prefix}/generation_rate": summary["generation_rate"],
        f"{prefix}/expectations_rate": summary["expectations_rate"],
        f"{prefix}/total_cases": summary["total_cases"],
    }

    if "validity_score_mean" in summary:
        log_dict[f"{prefix}/validity_score_mean"] = summary[
            "validity_score_mean"]
    if "validity_score_min" in summary:
        log_dict[f"{prefix}/validity_score_min"] = summary[
            "validity_score_min"]
    if "face_count_mean" in summary:
        log_dict[f"{prefix}/face_count_mean"] = summary["face_count_mean"]

    for cat, cat_data in summary.get("by_category", {}).items():
        if cat_data["total"] > 0:
            log_dict[f"{prefix}/cat_{cat}_gen_rate"] = (
                cat_data["generated"] / cat_data["total"])

    for domain, domain_data in summary.get("by_domain", {}).items():
        if domain_data["total"] > 0:
            log_dict[f"{prefix}/domain_{domain}_gen_rate"] = (
                domain_data["generated"] / domain_data["total"]
            )
            log_dict[f"{prefix}/domain_{domain}_success_rate"] = (
                domain_data["valid"] / domain_data["total"]
            )

    pro = results.get("professional_kpis") or {}
    if pro:
        log_dict[f"{prefix}/promotion_ready"] = float(bool(pro.get("promotion_ready", False)))
        log_dict[f"{prefix}/domains_passed"] = float(pro.get("domains_passed", 0))

    return log_dict


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(
        description="Run geometric evaluation on a checkpoint")
    parser.add_argument("--checkpoint",
                        default="checkpoints/unified/latest.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config file")
    parser.add_argument("--max-faces", type=int, default=512,
                        help="Max faces per generation")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--device", default=None,
                        help="Device (auto-detect if not specified)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)

    from models.unified import UnifiedBlenderModel
    model = UnifiedBlenderModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    from processing.mesh_tokenizer import MeshTokenizer
    tok_config = config.get("tokenization", {})
    mesh_tokenizer = MeshTokenizer(
        vocab_size=tok_config.get("vocab_size", 8192),
        coord_range=tuple(tok_config.get("coordinate_range", [-1.0, 1.0])),
        max_faces=tok_config.get("max_faces", 2048),
    )

    from processing.bpe_tokenizer import BPETokenizer
    data_cfg = config.get("data", {})
    geo_dir = Path(data_cfg.get("geometry_dir",
                                "data/datasets/geometry"))
    bpe_dir = geo_dir / "bpe_tokenizer"
    text_tokenizer = BPETokenizer.load(bpe_dir)

    step = checkpoint.get("global_step", 0)
    results = run_geometric_eval(
        model, mesh_tokenizer, text_tokenizer,
        device, step, config,
        max_faces=args.max_faces,
        temperature=args.temperature,
    )

    print(f"\n{'='*60}")
    print(f"GEOMETRIC EVALUATION RESULTS (step {step})")
    print(f"{'='*60}")
    s = results["summary"]
    print(f"  Generation rate:  {s['generation_rate']:.0%}")
    print(f"  Expectations met: {s['expectations_rate']:.0%}")
    if "validity_score_mean" in s:
        print(f"  Validity score:   {s['validity_score_mean']:.3f}")
    if "face_count_mean" in s:
        print(f"  Avg face count:   {s['face_count_mean']:.0f}")
    print(f"  Time elapsed:     {results['elapsed_seconds']:.1f}s")
    print()

    for cat, data in s.get("by_category", {}).items():
        gen_rate = data["generated"] / max(data["total"], 1)
        print(f"  [{cat}] {data['generated']}/{data['total']} "
              f"generated ({gen_rate:.0%})")
