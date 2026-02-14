"""Dataset Builder — assemble processed data into training-ready datasets.

Takes extracted JSON files (from blend_extractor) and builds:
1. Geometry dataset — (text_label, tokenized_mesh) pairs
2. Material dataset — (text_label, node_tree_json) pairs
3. Modifier dataset — (text_label + mesh_stats, modifier_stack) pairs

Usage:
    python -m processing.dataset_builder \
        --input data/processed \
        --output data/datasets \
        --config config.yaml
"""

import argparse
import json
import logging
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_text_label(obj_data: dict, scene_data: dict) -> str:
    """Generate a text description for a mesh object.

    Combines available metadata: object name, material names,
    modifier types, dimensions, and source metadata.
    Uses Objaverse metadata (name, description, tags) when available.
    """
    import re
    parts = []

    # Prefer Objaverse metadata if available
    meta = scene_data.get("metadata", {})
    meta_name = meta.get("name", "")
    meta_desc = meta.get("description", "")
    meta_tags = meta.get("tags", [])

    if meta_name and meta_name != "Object":
        # Objaverse model name (usually more descriptive)
        parts.append(meta_name)
    else:
        # Fall back to object name
        name = obj_data.get("name", "Unknown")
        clean_name = name.replace(".", " ").replace("_", " ")
        clean_name = re.sub(r"\s*\d+$", "", clean_name).strip()
        if clean_name and clean_name != "Object":
            parts.append(clean_name)

    # Add Objaverse description if short enough
    if meta_desc and len(meta_desc) < 200:
        parts.append(meta_desc)

    # Add tags
    if meta_tags:
        parts.append("tags: " + ", ".join(meta_tags[:5]))

    # Dimensions for scale context
    dims = obj_data.get("dimensions", [0, 0, 0])
    if any(d > 0 for d in dims):
        parts.append(f"({dims[0]:.1f}m × {dims[1]:.1f}m × {dims[2]:.1f}m)")

    # Material names
    mat_names = [m.get("name", "") for m in obj_data.get("materials", [])]
    mat_names = [n for n in mat_names if n and n != "Material"]
    if mat_names:
        parts.append("materials: " + ", ".join(mat_names[:3]))

    # Modifier types
    mod_types = [m.get("type", "") for m in obj_data.get("modifiers", [])]
    if mod_types:
        parts.append("modifiers: " + ", ".join(mod_types))

    # Face/vertex count for complexity context
    mesh = obj_data.get("mesh", {})
    if mesh:
        parts.append(f"{mesh.get('num_faces', 0)} faces")

    return " | ".join(parts)


def mesh_hash(vertices: list, faces: list) -> str:
    """Create a hash for deduplication of similar meshes."""
    # Use quantized vertex data for fuzzy dedup
    data = json.dumps({
        "v": [[round(c, 2) for c in v] for v in vertices[:100]],
        "nf": len(faces),
    }, sort_keys=True)
    return hashlib.md5(data.encode()).hexdigest()


def build_geometry_dataset(processed_dir: Path, output_dir: Path,
                           config: dict):
    """Build the geometry training dataset.

    Each example: (text_description, tokenized_mesh_sequence)
    """
    from .mesh_tokenizer import MeshTokenizer

    tok_config = config.get("tokenization", {})
    tokenizer = MeshTokenizer(
        vocab_size=tok_config.get("vocab_size", 8192),
        coord_range=tuple(tok_config.get("coordinate_range", [-1.0, 1.0])),
        max_faces=tok_config.get("max_faces", 2048),
    )

    quality_config = config.get("processing", {}).get("quality_filter", {})
    min_faces = quality_config.get("min_faces", 12)
    dedup = quality_config.get("deduplicate", True)
    dedup_threshold = quality_config.get("dedup_similarity_threshold", 0.95)

    seen_hashes = set()
    examples = []
    skipped = defaultdict(int)

    logger.info("Building geometry dataset...")

    for json_file in sorted(processed_dir.rglob("*.json")):
        if json_file.name.endswith(".meta.json"):
            continue

        try:
            with open(json_file) as f:
                scene = json.load(f)
        except Exception as e:
            skipped["json_error"] += 1
            continue

        for obj in scene.get("objects", []):
            mesh = obj.get("mesh")
            if not mesh:
                skipped["no_mesh"] += 1
                continue

            vertices = mesh.get("vertices", [])
            faces = mesh.get("faces", [])

            if len(faces) < min_faces:
                skipped["too_few_faces"] += 1
                continue

            # Deduplication
            if dedup:
                h = mesh_hash(vertices, faces)
                if h in seen_hashes:
                    skipped["duplicate"] += 1
                    continue
                seen_hashes.add(h)

            # Tokenize
            try:
                tokens = tokenizer.encode_mesh(vertices, faces)
            except Exception as e:
                skipped["tokenize_error"] += 1
                continue

            # Generate text label
            label = generate_text_label(obj, scene)

            examples.append({
                "text": label,
                "tokens": tokens,
                "num_faces": len(faces),
                "num_vertices": len(vertices),
                "source": scene.get("source_file", ""),
            })

    # Save dataset
    geo_dir = output_dir / "geometry"
    geo_dir.mkdir(parents=True, exist_ok=True)

    # Split into train/val/test (90/5/5)
    np.random.seed(42)
    indices = np.random.permutation(len(examples))
    n_train = int(len(examples) * 0.90)
    n_val = int(len(examples) * 0.05)

    splits = {
        "train": [examples[i] for i in indices[:n_train]],
        "val": [examples[i] for i in indices[n_train:n_train + n_val]],
        "test": [examples[i] for i in indices[n_train + n_val:]],
    }

    for split_name, split_data in splits.items():
        out_path = geo_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        logger.info(f"  {split_name}: {len(split_data)} examples → {out_path}")

    # Save tokenizer
    tokenizer.save(geo_dir / "tokenizer.json")

    # Stats
    logger.info(f"Geometry dataset: {len(examples)} total examples")
    logger.info(f"Skipped: {dict(skipped)}")

    return len(examples)


def build_material_dataset(processed_dir: Path, output_dir: Path,
                            config: dict):
    """Build the material training dataset.

    Each example: (text_description, material_node_tree)
    """
    examples = []
    seen = set()

    logger.info("Building material dataset...")

    for json_file in sorted(processed_dir.rglob("*.json")):
        if json_file.name.endswith(".meta.json"):
            continue

        try:
            with open(json_file) as f:
                scene = json.load(f)
        except Exception:
            continue

        for obj in scene.get("objects", []):
            for mat in obj.get("materials", []):
                if mat.get("type") != "node_tree":
                    continue

                nodes = mat.get("nodes", [])
                if len(nodes) < 2:  # At least output + one shader
                    continue

                # Deduplicate by node structure
                mat_key = json.dumps(
                    {"nodes": [n["type"] for n in nodes],
                     "links": len(mat.get("links", []))},
                    sort_keys=True)
                if mat_key in seen:
                    continue
                seen.add(mat_key)

                # Text label from material name + context
                label_parts = [mat.get("name", "Material")]
                # Add shader types
                shader_types = [n["type"] for n in nodes
                                if "BSDF" in n.get("type", "")]
                if shader_types:
                    label_parts.append("shaders: " + ", ".join(shader_types))

                examples.append({
                    "text": " | ".join(label_parts),
                    "node_tree": {
                        "nodes": nodes,
                        "links": mat.get("links", []),
                    },
                    "source": scene.get("source_file", ""),
                })

    # Save
    mat_dir = output_dir / "materials"
    mat_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    indices = np.random.permutation(len(examples))
    n_train = int(len(examples) * 0.90)
    n_val = int(len(examples) * 0.05)

    splits = {
        "train": [examples[i] for i in indices[:n_train]],
        "val": [examples[i] for i in indices[n_train:n_train + n_val]],
        "test": [examples[i] for i in indices[n_train + n_val:]],
    }

    for split_name, split_data in splits.items():
        out_path = mat_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        logger.info(f"  {split_name}: {len(split_data)} examples → {out_path}")

    logger.info(f"Material dataset: {len(examples)} total examples")
    return len(examples)


def build_modifier_dataset(processed_dir: Path, output_dir: Path,
                            config: dict):
    """Build the modifier stack training dataset.

    Each example: (object_description + mesh_stats, modifier_stack)
    """
    examples = []

    logger.info("Building modifier dataset...")

    for json_file in sorted(processed_dir.rglob("*.json")):
        if json_file.name.endswith(".meta.json"):
            continue

        try:
            with open(json_file) as f:
                scene = json.load(f)
        except Exception:
            continue

        for obj in scene.get("objects", []):
            modifiers = obj.get("modifiers", [])
            if not modifiers:
                continue

            mesh = obj.get("mesh", {})
            label = generate_text_label(obj, scene)

            examples.append({
                "text": label,
                "mesh_stats": {
                    "num_vertices": mesh.get("num_vertices", 0),
                    "num_faces": mesh.get("num_faces", 0),
                    "dimensions": obj.get("dimensions", [0, 0, 0]),
                },
                "modifiers": modifiers,
                "source": scene.get("source_file", ""),
            })

    # Save
    mod_dir = output_dir / "modifiers"
    mod_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    indices = np.random.permutation(len(examples))
    n_train = int(len(examples) * 0.90)
    n_val = int(len(examples) * 0.05)

    splits = {
        "train": [examples[i] for i in indices[:n_train]],
        "val": [examples[i] for i in indices[n_train:n_train + n_val]],
        "test": [examples[i] for i in indices[n_train + n_val:]],
    }

    for split_name, split_data in splits.items():
        out_path = mod_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        logger.info(f"  {split_name}: {len(split_data)} examples → {out_path}")

    logger.info(f"Modifier dataset: {len(examples)} total examples")
    return len(examples)


def main():
    parser = argparse.ArgumentParser(description="Build training datasets")
    parser.add_argument("--input", required=True,
                        help="Directory with processed JSON files")
    parser.add_argument("--output", required=True,
                        help="Output directory for datasets")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--datasets", nargs="*",
                        default=["geometry", "materials", "modifiers"],
                        help="Which datasets to build")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    config = load_config(args.config)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if "geometry" in args.datasets:
        build_geometry_dataset(input_dir, output_dir, config)
    if "materials" in args.datasets:
        build_material_dataset(input_dir, output_dir, config)
    if "modifiers" in args.datasets:
        build_modifier_dataset(input_dir, output_dir, config)


if __name__ == "__main__":
    main()
