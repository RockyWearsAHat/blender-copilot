"""Local inference server for trained Blender models.

Runs as a lightweight FastAPI server on localhost.
The Blender addon sends requests here instead of to OpenAI.

Endpoints:
    POST /generate/mesh     — text → mesh data (vertices, faces, materials)
    POST /generate/material — text → material node tree
    POST /generate/modifiers — text + mesh stats → modifier stack
    GET  /health            — server health check

Usage:
    python -m inference.server --model models/geometry/checkpoints/best.pt --port 8420
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, config: dict, device: str = "auto"):
    """Load a trained geometry model from checkpoint."""
    from models.geometry.model import GeometryModel
    from processing.mesh_tokenizer import MeshTokenizer
    from processing.text_tokenizer import TextTokenizer

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info(f"Loading model on {device}...")
    dev = torch.device(device)

    # Load model
    model = GeometryModel(config)
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(dev)
    model.eval()

    # Load mesh tokenizer
    tok_config = config.get("tokenization", {})
    tokenizer = MeshTokenizer(
        vocab_size=tok_config.get("vocab_size", 8192),
        coord_range=tuple(tok_config.get("coordinate_range", [-1.0, 1.0])),
        max_faces=tok_config.get("max_faces", 2048),
    )

    # Load text tokenizer
    text_tokenizer = None
    # Look for text_tokenizer.json next to the checkpoint, or in datasets
    cp_dir = Path(checkpoint_path).parent
    for search_path in [
        cp_dir / "text_tokenizer.json",
        cp_dir.parent / "text_tokenizer.json",
        Path("data/datasets/geometry/text_tokenizer.json"),
    ]:
        if search_path.exists():
            text_tokenizer = TextTokenizer.load(search_path)
            logger.info(f"Loaded text tokenizer from {search_path}: {text_tokenizer}")
            break

    if text_tokenizer is None:
        logger.warning("No text_tokenizer.json found — falling back to character-level")

    param_count = model.count_parameters()
    logger.info(f"Model loaded: {param_count:,} parameters ({param_count / 1e6:.1f}M)")

    return model, tokenizer, dev, text_tokenizer


def text_to_tokens(text: str, max_length: int = 256,
                   text_tokenizer=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert text to model input tokens.

    Uses word-level TextTokenizer if available, falls back to character-level.
    """
    if text_tokenizer is not None:
        ids, mask = text_tokenizer.encode_padded(text, max_length=max_length)
    else:
        # Legacy fallback
        ids = [ord(c) % 32000 for c in text[:max_length]]
        mask = [1] * len(ids)
        ids = ids + [0] * (max_length - len(ids))
        mask = mask + [0] * (max_length - len(mask))

    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([mask], dtype=torch.float),
    )


def generate_mesh(model, tokenizer, text: str, device: torch.device,
                  temperature: float = 0.8, top_k: int = 50,
                  max_faces: int = 2048,
                  text_tokenizer=None) -> dict:
    """Generate a mesh from a text prompt.

    Returns structured mesh data ready for injection into Blender.
    """
    start = time.time()

    text_ids, text_mask = text_to_tokens(text, text_tokenizer=text_tokenizer)
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)

    # Generate tokens
    with torch.no_grad():
        tokens = model.generate(
            text_ids, text_mask,
            max_tokens=max_faces * 9 + 2,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode tokens to mesh
    token_list = tokens[0].cpu().tolist()
    vertices, faces = tokenizer.decode_tokens(token_list)

    elapsed = time.time() - start

    # Post-process: deduplicate vertices
    vertices, faces = _merge_duplicate_vertices(vertices, faces)

    return {
        "objects": [{
            "name": _clean_name(text),
            "mesh": {
                "vertices": vertices,
                "faces": faces,
                "num_vertices": len(vertices),
                "num_faces": len(faces),
            },
            "materials": [],
            "modifiers": [],
            "transforms": {
                "location": [0, 0, 0],
                "rotation_euler": [0, 0, 0],
                "scale": [1, 1, 1],
            },
            "shade_smooth": True,
        }],
        "generation_time": round(elapsed, 2),
        "token_count": len(token_list),
    }


def _merge_duplicate_vertices(vertices, faces, threshold=0.001):
    """Merge duplicate vertices (common after face-based decoding)."""
    import numpy as np

    if not vertices:
        return vertices, faces

    verts = np.array(vertices)
    unique_map = {}
    unique_verts = []
    index_remap = {}

    for i, v in enumerate(verts):
        key = tuple(np.round(v / threshold).astype(int))
        if key not in unique_map:
            unique_map[key] = len(unique_verts)
            unique_verts.append(v.tolist())
        index_remap[i] = unique_map[key]

    new_faces = []
    for face in faces:
        new_face = [index_remap.get(vi, vi) for vi in face]
        # Skip degenerate faces
        if len(set(new_face)) >= 3:
            new_faces.append(new_face)

    return unique_verts, new_faces


def _clean_name(text: str) -> str:
    """Generate a clean object name from prompt text."""
    # Take first few words
    words = text.strip().split()[:4]
    name = "_".join(words)
    # Remove special chars
    name = "".join(c for c in name if c.isalnum() or c == "_")
    return name[:30] or "Generated"


def create_app(model, tokenizer, device, config, text_tokenizer=None):
    """Create FastAPI application."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Blender Model Server", version="0.1.0")

    class MeshRequest(BaseModel):
        prompt: str
        temperature: float = 0.8
        top_k: int = 50
        max_faces: int = 2048

    class HealthResponse(BaseModel):
        status: str
        device: str
        model_params: int

    @app.get("/health")
    def health():
        return HealthResponse(
            status="ok",
            device=str(device),
            model_params=model.count_parameters(),
        )

    @app.post("/generate/mesh")
    def gen_mesh(req: MeshRequest):
        result = generate_mesh(
            model, tokenizer, req.prompt, device,
            temperature=req.temperature,
            top_k=req.top_k,
            max_faces=req.max_faces,
            text_tokenizer=text_tokenizer,
        )
        return result

    @app.post("/generate/material")
    def gen_material(req: MeshRequest):
        # TODO: Material model integration
        return {"error": "Material model not yet trained"}

    @app.post("/generate/modifiers")
    def gen_modifiers(req: MeshRequest):
        # TODO: Modifier model integration
        return {"error": "Modifier model not yet trained"}

    return app


def main():
    parser = argparse.ArgumentParser(description="Blender model inference server")
    parser.add_argument("--model", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model, tokenizer, device, text_tokenizer = load_model(args.model, config, args.device)
    app = create_app(model, tokenizer, device, config, text_tokenizer=text_tokenizer)

    import uvicorn
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
