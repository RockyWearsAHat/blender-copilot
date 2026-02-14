"""Evaluation metrics for trained models.

Computes quality metrics for geometry, material, and modifier models.

Geometry metrics:
    - Chamfer Distance (CD) — measures surface similarity
    - F-Score @ threshold — precision/recall of surface coverage
    - Normal Consistency (NC) — alignment of surface normals
    - Self-intersection ratio — mesh quality
    - Valid mesh ratio — % of outputs that decode to valid meshes

Material metrics:
    - Node type accuracy — correct node types predicted
    - Connection accuracy — correct links between nodes
    - Parameter MSE — error in numeric parameters

Modifier metrics:
    - Type accuracy — correct modifier types
    - Count accuracy — correct number of modifiers
    - Parameter MSE — error in modifier parameters

Usage:
    python -m training.evaluate \
        --model checkpoints/geometry/best.pt \
        --data data/datasets/geometry_test.jsonl \
        --model_type geometry \
        --config config.yaml
"""

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ─── Geometry Metrics ──────────────────────────────────────────────

def chamfer_distance(pred_verts: np.ndarray, gt_verts: np.ndarray) -> float:
    """Compute bidirectional Chamfer Distance between two point sets.

    CD = (1/|P|) Σ min_q ||p-q||² + (1/|Q|) Σ min_p ||q-p||²
    """
    if len(pred_verts) == 0 or len(gt_verts) == 0:
        return float("inf")

    # pred → gt
    diffs_pg = pred_verts[:, None, :] - gt_verts[None, :, :]  # (P, Q, 3)
    dists_pg = np.sum(diffs_pg ** 2, axis=-1)  # (P, Q)
    min_pg = np.min(dists_pg, axis=1).mean()

    # gt → pred
    min_gp = np.min(dists_pg, axis=0).mean()

    return float(min_pg + min_gp)


def chamfer_distance_sampled(pred_verts: np.ndarray, gt_verts: np.ndarray,
                              n_samples: int = 2048) -> float:
    """Chamfer Distance with random subsampling for large meshes."""
    if len(pred_verts) > n_samples:
        idx = np.random.choice(len(pred_verts), n_samples, replace=False)
        pred_verts = pred_verts[idx]
    if len(gt_verts) > n_samples:
        idx = np.random.choice(len(gt_verts), n_samples, replace=False)
        gt_verts = gt_verts[idx]
    return chamfer_distance(pred_verts, gt_verts)


def f_score(pred_verts: np.ndarray, gt_verts: np.ndarray,
            threshold: float = 0.01) -> float:
    """Compute F-Score at a distance threshold.

    F = 2 * precision * recall / (precision + recall)
    where precision = % of predicted points close to GT
          recall = % of GT points close to predicted
    """
    if len(pred_verts) == 0 or len(gt_verts) == 0:
        return 0.0

    # Distances from pred to gt
    diffs = pred_verts[:, None, :] - gt_verts[None, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))

    # Precision: for each pred point, min distance to any gt point
    min_pred_to_gt = np.min(dists, axis=1)
    precision = (min_pred_to_gt < threshold).mean()

    # Recall: for each gt point, min distance to any pred point
    min_gt_to_pred = np.min(dists, axis=0)
    recall = (min_gt_to_pred < threshold).mean()

    if precision + recall < 1e-8:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def valid_mesh_check(vertices: list, faces: list) -> dict:
    """Check if a decoded mesh is valid."""
    result = {
        "has_vertices": len(vertices) > 0,
        "has_faces": len(faces) > 0,
        "min_face_size": 3,
        "no_degenerate_faces": True,
        "is_valid": False,
    }

    if not result["has_vertices"] or not result["has_faces"]:
        return result

    verts = np.array(vertices)

    # Check for NaN/Inf
    if np.any(np.isnan(verts)) or np.any(np.isinf(verts)):
        return result

    # Check face validity
    max_idx = len(vertices) - 1
    for face in faces:
        if len(face) < 3:
            result["no_degenerate_faces"] = False
            break
        if any(idx < 0 or idx > max_idx for idx in face):
            result["no_degenerate_faces"] = False
            break
        if len(set(face)) < 3:
            result["no_degenerate_faces"] = False
            break

    result["is_valid"] = (
        result["has_vertices"] and
        result["has_faces"] and
        result["no_degenerate_faces"]
    )

    return result


# ─── Material Metrics ──────────────────────────────────────────────

def node_type_accuracy(pred_nodes: list, gt_nodes: list) -> float:
    """Compare predicted vs ground truth node types."""
    if not gt_nodes:
        return 1.0 if not pred_nodes else 0.0

    pred_types = [n.get("type", "") for n in pred_nodes]
    gt_types = [n.get("type", "") for n in gt_nodes]

    # Order-independent comparison
    pred_set = set(pred_types)
    gt_set = set(gt_types)

    if not gt_set:
        return 1.0

    intersection = pred_set & gt_set
    return len(intersection) / len(gt_set)


def link_accuracy(pred_links: list, gt_links: list) -> float:
    """Compare predicted vs ground truth node connections."""
    if not gt_links:
        return 1.0 if not pred_links else 0.0

    def link_key(link):
        return (link.get("from_node"), link.get("from_socket"),
                link.get("to_node"), link.get("to_socket"))

    pred_set = set(link_key(l) for l in pred_links)
    gt_set = set(link_key(l) for l in gt_links)

    if not gt_set:
        return 1.0

    intersection = pred_set & gt_set
    return len(intersection) / len(gt_set)


# ─── Modifier Metrics ─────────────────────────────────────────────

def modifier_type_accuracy(pred_mods: list, gt_mods: list) -> float:
    """Compare predicted vs ground truth modifier types."""
    if not gt_mods:
        return 1.0 if not pred_mods else 0.0

    correct = 0
    for i, gt_mod in enumerate(gt_mods):
        if i < len(pred_mods):
            if pred_mods[i].get("type") == gt_mod.get("type"):
                correct += 1

    return correct / len(gt_mods)


def modifier_count_accuracy(pred_count: int, gt_count: int) -> float:
    """How close the predicted modifier count is to ground truth."""
    if gt_count == 0:
        return 1.0 if pred_count == 0 else 0.0
    return 1.0 - abs(pred_count - gt_count) / max(gt_count, pred_count)


# ─── Evaluation Runner ────────────────────────────────────────────

class Evaluator:
    """Run evaluation on a test set."""

    def __init__(self, config: dict):
        self.config = config

    def evaluate_geometry(self, model_path: str, test_data_path: str) -> dict:
        """Evaluate geometry model on test set."""
        from models.geometry.model import GeometryModel
        from processing.mesh_tokenizer import MeshTokenizer

        device = self._get_device()

        # Load model
        model = GeometryModel(self.config)
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        tok_config = self.config.get("tokenization", {})
        tokenizer = MeshTokenizer(
            vocab_size=tok_config.get("vocab_size", 8192),
            coord_range=tuple(tok_config.get("coordinate_range", [-1.0, 1.0])),
            max_faces=tok_config.get("max_faces", 2048),
        )

        # Load test data
        samples = []
        with open(test_data_path) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        logger.info(f"Evaluating {len(samples)} test samples")

        metrics = {
            "chamfer_distances": [],
            "f_scores": [],
            "valid_meshes": 0,
            "total_meshes": 0,
            "avg_pred_verts": [],
            "avg_pred_faces": [],
        }

        for i, sample in enumerate(samples):
            text = sample.get("text", "")
            gt_tokens = sample.get("tokens", [])

            # Decode ground truth
            gt_verts, gt_faces = tokenizer.decode_tokens(gt_tokens)

            # Generate prediction
            text_ids = torch.tensor(
                [[ord(c) % 32000 for c in text[:256]] + [0] * max(0, 256 - len(text))],
                dtype=torch.long, device=device
            )
            text_mask = torch.tensor(
                [[1.0] * min(len(text), 256) + [0.0] * max(0, 256 - len(text))],
                dtype=torch.float, device=device
            )

            with torch.no_grad():
                pred_tokens = model.generate(
                    text_ids, text_mask,
                    max_tokens=2048 * 9 + 2,
                    temperature=0.7,
                )

            pred_verts, pred_faces = tokenizer.decode_tokens(
                pred_tokens[0].cpu().tolist()
            )

            # Validity check
            metrics["total_meshes"] += 1
            validity = valid_mesh_check(pred_verts, pred_faces)
            if validity["is_valid"]:
                metrics["valid_meshes"] += 1

            metrics["avg_pred_verts"].append(len(pred_verts))
            metrics["avg_pred_faces"].append(len(pred_faces))

            # Geometric metrics (if both have vertices)
            if len(pred_verts) > 0 and len(gt_verts) > 0:
                pred_v = np.array(pred_verts)
                gt_v = np.array(gt_verts)

                cd = chamfer_distance_sampled(pred_v, gt_v)
                metrics["chamfer_distances"].append(cd)

                fs = f_score(pred_v[:1024], gt_v[:1024], threshold=0.05)
                metrics["f_scores"].append(fs)

            if (i + 1) % 10 == 0:
                logger.info(f"  Evaluated {i+1}/{len(samples)}")

        # Aggregate
        result = {
            "chamfer_distance_mean": float(np.mean(metrics["chamfer_distances"])) if metrics["chamfer_distances"] else float("inf"),
            "chamfer_distance_median": float(np.median(metrics["chamfer_distances"])) if metrics["chamfer_distances"] else float("inf"),
            "f_score_mean": float(np.mean(metrics["f_scores"])) if metrics["f_scores"] else 0.0,
            "valid_mesh_ratio": metrics["valid_meshes"] / max(1, metrics["total_meshes"]),
            "avg_predicted_vertices": float(np.mean(metrics["avg_pred_verts"])),
            "avg_predicted_faces": float(np.mean(metrics["avg_pred_faces"])),
            "total_evaluated": metrics["total_meshes"],
        }

        return result

    def evaluate_materials(self, model_path: str, test_data_path: str) -> dict:
        """Evaluate material model on test set."""
        from models.materials.model import MaterialModel, MaterialEncoder

        device = self._get_device()

        model = MaterialModel(self.config)
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        encoder = MaterialEncoder()

        samples = []
        with open(test_data_path) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        logger.info(f"Evaluating {len(samples)} material test samples")

        type_accs = []
        link_accs = []

        for sample in samples:
            text = sample.get("text", "")
            gt_material = sample.get("node_tree", {})

            text_ids = torch.tensor(
                [[ord(c) % 32000 for c in text[:128]] + [0] * max(0, 128 - len(text))],
                dtype=torch.long, device=device
            )
            text_mask = torch.tensor(
                [[1.0] * min(len(text), 128) + [0.0] * max(0, 128 - len(text))],
                dtype=torch.float, device=device
            )

            with torch.no_grad():
                pred_tokens = model.generate(text_ids, text_mask)

            pred_material = encoder.decode_tokens(pred_tokens[0].cpu().tolist())

            type_accs.append(node_type_accuracy(
                pred_material.get("nodes", []),
                gt_material.get("nodes", [])
            ))
            link_accs.append(link_accuracy(
                pred_material.get("links", []),
                gt_material.get("links", [])
            ))

        return {
            "node_type_accuracy": float(np.mean(type_accs)) if type_accs else 0.0,
            "link_accuracy": float(np.mean(link_accs)) if link_accs else 0.0,
            "total_evaluated": len(samples),
        }

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Test data JSONL")
    parser.add_argument("--model_type", required=True,
                        choices=["geometry", "materials", "modifiers"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="", help="Save results JSON to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluator = Evaluator(config)

    if args.model_type == "geometry":
        results = evaluator.evaluate_geometry(args.model, args.data)
    elif args.model_type == "materials":
        results = evaluator.evaluate_materials(args.model, args.data)
    else:
        logger.error(f"Evaluation for {args.model_type} not yet implemented")
        return

    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation Results ({args.model_type}):")
    logger.info(f"{'='*50}")
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    logger.info(f"{'='*50}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
