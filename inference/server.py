"""Local inference server for trained Blender models.

Runs as a lightweight FastAPI server on localhost.
The Blender addon sends requests here instead of to OpenAI.

Supports both legacy GeometryModel and new UnifiedBlenderModel checkpoints.
Hot-reloads weights automatically when checkpoints change on disk.

Endpoints:
    POST /generate/mesh       — text → mesh data (vertices, faces)
    POST /generate/material   — text → material node tree
    POST /generate/modifiers  — text + mesh stats → modifier stack
    GET  /health              — server health + model info
    POST /reload              — force reload latest checkpoint

Usage:
    python -m inference.server --model checkpoints/unified/latest.pt --port 8420
"""

import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path

import torch
import yaml
from processing.prompt_semantics import enrich_prompt_text

logger = logging.getLogger(__name__)

# ── Global model state (hot-reloadable) ─────────────────────────────

_MODEL_LOCK = threading.Lock()


class ModelState:
    """Holds current model + tokenizers. Thread-safe swap on reload."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.text_tokenizer = None
        self.device = None
        self.config = None
        self.model_type = "unknown"
        self.checkpoint_path = None
        self.checkpoint_mtime = 0
        self.param_count = 0
        self.step = 0

    def load(self, checkpoint_path: str, config: dict,
             device_str: str = "auto"):
        """Load or reload model from checkpoint."""
        if device_str == "auto":
            if torch.cuda.is_available():
                device_str = "cuda"
            elif (hasattr(torch.backends, "mps")
                  and torch.backends.mps.is_available()):
                device_str = "mps"
            else:
                device_str = "cpu"

        dev = torch.device(device_str)
        logger.info(f"Loading checkpoint from {checkpoint_path} on {dev}")

        # Always deserialize to CPU first — MPS deserialization is very slow
        ckpt = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)

        if "model_state_dict" not in ckpt:
            top_keys = sorted(str(k) for k in ckpt.keys())
            if "model" in ckpt:
                raise ValueError(
                    "Checkpoint format mismatch: this appears to be a policy checkpoint "
                    "(contains 'model'), but the local inference server expects a unified "
                    "mesh checkpoint (requires 'model_state_dict'). "
                    "Use something like checkpoints/unified/latest.pt with run.py serve. "
                    f"Checkpoint: {checkpoint_path}. Keys: {top_keys}"
                )
            raise ValueError(
                "Unsupported checkpoint format for inference server: missing 'model_state_dict'. "
                "Use a unified mesh checkpoint such as checkpoints/unified/latest.pt. "
                f"Checkpoint: {checkpoint_path}. Keys: {top_keys}"
            )

        if "config" in ckpt:
            config = ckpt["config"]

        model_type = ckpt.get("model_type", "unified")

        from models.unified import UnifiedBlenderModel
        model = UnifiedBlenderModel(config)

        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model = model.to(dev)
        model.eval()

        # MPS stability: run inference in fp16 to reduce memory pressure.
        # This is especially important for long autoregressive geometry
        # sequences (KV caches scale with sequence length).
        try:
            if dev.type == "mps":
                model = model.to(dtype=torch.float16)
        except Exception:
            pass

        pc = model.count_parameters()
        # MPS is consistently faster for autoregressive generation on M3
        # (1000+ sequential token steps) — do NOT downgrade to CPU.

        from processing.mesh_tokenizer import MeshTokenizer

        tok_config = config.get("tokenization", {})
        tokenizer = MeshTokenizer(
            vocab_size=tok_config.get("vocab_size", 8192),
            coord_range=tuple(
                tok_config.get("coordinate_range", [-1.0, 1.0])),
            max_faces=tok_config.get("max_faces", 2048),
        )

        # ── Text tokenizer: prefer BPE, fall back to legacy ──
        text_tokenizer = None
        cp_dir = Path(checkpoint_path).parent

        # Search paths for BPE tokenizer directory
        bpe_search = [
            cp_dir / "bpe_tokenizer",
            cp_dir.parent / "bpe_tokenizer",
            Path("data/datasets/geometry/bpe_tokenizer"),
        ]
        for bp in bpe_search:
            if bp.is_dir() and (bp / "tokenizer.model").exists():
                from processing.bpe_tokenizer import BPETokenizer
                text_tokenizer = BPETokenizer.load(bp)
                logger.info(f"BPE tokenizer: {bp} ({text_tokenizer.vocab_size} subwords)")
                break

        # Fall back to legacy word-level tokenizer
        if text_tokenizer is None:
            from processing.text_tokenizer import TextTokenizer
            for sp in [
                cp_dir / "text_tokenizer.json",
                cp_dir.parent / "text_tokenizer.json",
                Path("data/datasets/geometry/text_tokenizer.json"),
            ]:
                if sp.exists():
                    text_tokenizer = TextTokenizer.load(sp)
                    logger.info(f"Legacy text tokenizer: {sp}")
                    break

        # If tokenizer vocab doesn't match model embedding, rebuild model
        # with correct vocab size. This happens when the tokenizer was
        # rebuilt (expanded) after the checkpoint was saved.
        if text_tokenizer is not None:
            from processing.bpe_tokenizer import BPETokenizer as _BPE
            # BPE includes specials in vocab_size; legacy needs +4
            actual_vocab = (text_tokenizer.vocab_size
                            if isinstance(text_tokenizer, _BPE)
                            else text_tokenizer.vocab_size + 4)
            model_vocab = model.text_encoder.embed.num_embeddings
            if actual_vocab != model_vocab:
                logger.warning(
                    f"Vocab mismatch: model has {model_vocab}, "
                    f"tokenizer needs {actual_vocab}. Rebuilding model.")
                config.setdefault("unified", config)
                config["unified"]["text_vocab_size"] = actual_vocab
                model = UnifiedBlenderModel(config)
                model.load_state_dict(
                    ckpt["model_state_dict"], strict=False)
                model = model.to(dev)
                model.eval()
                pc = model.count_parameters()
                # MPS is faster for autoregressive generation — no CPU fallback.

        mtime = os.path.getmtime(checkpoint_path)

        with _MODEL_LOCK:
            self.model = model
            self.tokenizer = tokenizer
            self.text_tokenizer = text_tokenizer
            self.device = dev
            self.config = config
            self.model_type = model_type
            self.checkpoint_path = checkpoint_path
            self.checkpoint_mtime = mtime
            self.param_count = pc
            self.step = ckpt.get("step", 0)

        logger.info(
            f"Loaded {model_type} model: {pc:,} params "
            f"({pc/1e6:.1f}M), step {self.step}")

    def check_and_reload(self):
        """Reload if checkpoint file has been updated on disk."""
        if not self.checkpoint_path:
            return False
        try:
            current_mtime = os.path.getmtime(self.checkpoint_path)
            if current_mtime > self.checkpoint_mtime:
                logger.info(
                    "Checkpoint updated on disk — hot-reloading...")
                self.load(
                    self.checkpoint_path, self.config or {},
                    str(self.device) if self.device else "auto",
                )
                return True
        except Exception as e:
            logger.debug(f"Hot-reload check failed: {e}")
        return False


_STATE = ModelState()


# ── Hot-reload watcher thread ───────────────────────────────────────

def _watch_checkpoint(state: ModelState, interval: float = 30.0):
    """Background thread that checks for checkpoint updates."""
    while True:
        time.sleep(interval)
        try:
            state.check_and_reload()
        except Exception as e:
            logger.debug(f"Watcher error: {e}")


# ── Text encoding ───────────────────────────────────────────────────

def text_to_tokens(text: str, max_length: int = 256,
                   text_tokenizer=None):
    """Convert text to model input tokens."""
    text = enrich_prompt_text(text, max_hints=8, stochastic=False)
    if text_tokenizer is not None:
        ids, mask = text_tokenizer.encode_padded(
            text, max_length=max_length)
    else:
        ids = [ord(c) % 32000 for c in text[:max_length]]
        mask = [1] * len(ids)
        ids += [0] * (max_length - len(ids))
        mask += [0] * (max_length - len(mask))

    return (
        torch.tensor([ids], dtype=torch.long),
        torch.tensor([mask], dtype=torch.float),
    )


def _prompt_allows_boxlike(text: str) -> bool:
    t = (text or "").lower()
    keywords = ("cube", "box", "block", "dice", "voxel", "rectangular prism")
    return any(k in t for k in keywords)


def _is_trivial_boxlike(vertices, faces) -> bool:
    if not vertices or not faces:
        return True
    # Typical collapsed outputs are tiny primitive-like meshes (e.g., 5-6 faces).
    return len(vertices) <= 10 and len(faces) <= 8


def _mesh_is_degenerate(vertices, faces) -> bool:
    """Heuristic check for clearly unusable geometry.

    We want to reject things like a single triangle/quad, or meshes that are
    essentially 2D/1D due to a collapsed bounding box.
    """
    try:
        vcount = len(vertices)
        fcount = len(faces)
    except Exception:
        return True

    if vcount < 4 or fcount < 4:
        return True

    try:
        xs = [float(v[0]) for v in vertices]
        ys = [float(v[1]) for v in vertices]
        zs = [float(v[2]) for v in vertices]
    except Exception:
        return True

    xspan = max(xs) - min(xs)
    yspan = max(ys) - min(ys)
    zspan = max(zs) - min(zs)
    spans = [abs(xspan), abs(yspan), abs(zspan)]
    max_span = max(spans)
    if max_span <= 1e-9:
        return True
    min_span = min(spans)
    if (min_span / max_span) < 1e-3:
        return True

    return False


def _sample_mesh_tokens(
    model,
    model_type: str,
    text_ids,
    text_mask,
    *,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    cfg_scale: float,
):
    with torch.no_grad():
        if model_type == "unified":
            return model.generate_geometry(
                text_ids, text_mask,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cfg_scale=cfg_scale,
            )
        return model.generate(
            text_ids, text_mask,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=cfg_scale,
        )


# ── Generation functions ────────────────────────────────────────────

def generate_mesh(state: ModelState, text: str,
                  temperature: float = 0.6, top_k: int = 0,
                  top_p: float = 0.9,
                  max_faces: int = 4096,
                  cfg_scale: float = 3.5) -> dict:
    """Generate a mesh from text prompt."""
    start = time.time()

    with _MODEL_LOCK:
        model = state.model
        tokenizer = state.tokenizer
        device = state.device
        text_tok = state.text_tokenizer
        model_type = state.model_type

    if model_type == "unified":
        geo_dec = model.geometry_decoder
        max_seq = getattr(geo_dec, "max_seq_length", 16202)
    else:
        max_seq = getattr(
            model.mesh_decoder, "max_seq_length", 18432)

    effective_max = min(max_faces, (max_seq - 2) // 9)
    allow_boxlike = _prompt_allows_boxlike(text)
    if allow_boxlike:
        prompt_max_faces = effective_max
    else:
        # Non-box prompts are more stable with a moderate token budget.
        prompt_max_faces = min(effective_max, 512)
        if "low poly" in (text or "").lower():
            prompt_max_faces = min(prompt_max_faces, 256)
    max_tokens = prompt_max_faces * 9 + 2

    text_max = getattr(
        model.text_encoder, "max_length",
        getattr(model.text_encoder, "embed",
                None) and 256 or 64,
    )
    if hasattr(model.text_encoder, "pos_embed"):
        text_max = model.text_encoder.pos_embed.num_embeddings

    text_ids, text_mask = text_to_tokens(
        text, max_length=text_max, text_tokenizer=text_tok)
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)

    logger.info(
        f"Generating mesh: prompt={text!r}, max_tokens={max_tokens}")

    best = None
    max_attempts = 3
    min_faces_nonbox = 12

    for attempt in range(max_attempts):
        temp_i = float(temperature)
        top_k_i = int(top_k)
        top_p_i = float(top_p)

        if attempt > 0:
            temp_i = min(1.2, max(0.35, temp_i) + 0.15 * attempt)
            top_k_i = max(top_k_i, 32)
            top_p_i = min(0.97, max(top_p_i, 0.92))

        tokens = _sample_mesh_tokens(
            model,
            model_type,
            text_ids,
            text_mask,
            max_tokens=max_tokens,
            temperature=temp_i,
            top_k=top_k_i,
            top_p=top_p_i,
            cfg_scale=cfg_scale,
        )
        token_list = tokens[0].cpu().tolist()
        if not token_list:
            continue

        logger.info(
            f"Attempt {attempt + 1}/{max_attempts}: "
            f"tokens={len(token_list)}, range=[{min(token_list)}, {max(token_list)}], "
            f"unique={len(set(token_list))}, temp={temp_i:.2f}, top_k={top_k_i}, top_p={top_p_i:.2f}")

        vertices, faces = tokenizer.decode_tokens(token_list)
        logger.info(f"Decoded: {len(vertices)} vertices, {len(faces)} faces (before merge)")
        vertices, faces = _merge_duplicate_vertices(vertices, faces)
        faces = _recalculate_normals_consistent(vertices, faces)
        logger.info(f"After merge: {len(vertices)} vertices, {len(faces)} faces")

        complexity = int(len(faces) * 1000 + len(vertices))
        is_boxlike = _is_trivial_boxlike(vertices, faces)
        candidate = {
            "token_list": token_list,
            "vertices": vertices,
            "faces": faces,
            "complexity": complexity,
            "is_boxlike": is_boxlike,
        }

        if best is None or candidate["complexity"] > best["complexity"]:
            best = candidate

        # Early exit only when geometry is non-degenerate.
        # Boxlike prompts are allowed to be low-complexity, but should still be
        # a valid 3D mesh (not a single triangle or collapsed plane).
        is_degenerate = _mesh_is_degenerate(vertices, faces)
        if (not is_degenerate) and (allow_boxlike or (not is_boxlike and len(faces) >= min_faces_nonbox)):
            best = candidate
            break

    token_list = (best or {}).get("token_list", [])
    vertices = (best or {}).get("vertices", [])
    faces = (best or {}).get("faces", [])

    elapsed = time.time() - start

    if not faces or not vertices:
        # Log token distribution for debugging
        from collections import Counter
        token_counts = Counter(token_list)
        logger.warning(
            f"No geometry produced. Token distribution: {token_counts.most_common(10)}")
        return {
            "error": f"Model produced no geometry ({len(token_list)} tokens decoded to 0 faces). "
                     f"Unique tokens: {len(set(token_list))}. "
                     f"The model may need more training. Try rephrasing the prompt or increasing temperature.",
            "generation_time": round(elapsed, 2),
            "token_count": len(token_list),
            "unique_tokens": len(set(token_list)),
        }

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
        "tokens": token_list,
    }


def generate_material(state: ModelState, text: str,
                      temperature: float = 0.7, top_k: int = 30,
                      max_tokens: int = 512) -> dict:
    """Generate a material from text prompt (unified model only)."""
    with _MODEL_LOCK:
        model = state.model
        device = state.device
        text_tok = state.text_tokenizer
        model_type = state.model_type

    if model_type != "unified":
        return {"error": "Material generation requires unified model"}

    if not getattr(model, "enable_materials", False):
        return {"error": "Material decoder disabled (enable_materials=false in config)"}

    text_max = 256
    if hasattr(model.text_encoder, "pos_embed"):
        text_max = model.text_encoder.pos_embed.num_embeddings

    text_ids, text_mask = text_to_tokens(
        text, max_length=text_max, text_tokenizer=text_tok)
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)

    start = time.time()
    tokens = model.generate_materials(
        text_ids, text_mask,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    elapsed = time.time() - start

    token_list = tokens[0].cpu().tolist()

    try:
        from models.encoders import MaterialEncoder
        encoder = MaterialEncoder()
        node_tree = encoder.decode_tokens(token_list)
    except Exception:
        node_tree = {"tokens": token_list}

    return {
        "node_tree": node_tree,
        "generation_time": round(elapsed, 2),
        "token_count": len(token_list),
    }


def generate_modifiers(state: ModelState, text: str,
                       mesh_stats: dict = None) -> dict:
    """Predict modifiers from text + mesh stats (unified model only)."""
    with _MODEL_LOCK:
        model = state.model
        device = state.device
        text_tok = state.text_tokenizer
        model_type = state.model_type

    if model_type != "unified":
        return {"error": "Modifier prediction requires unified model"}

    if not getattr(model, "enable_modifiers", False):
        return {"error": "Modifier head disabled (enable_modifiers=false in config)"}

    import math

    text_max = 256
    if hasattr(model.text_encoder, "pos_embed"):
        text_max = model.text_encoder.pos_embed.num_embeddings

    text_ids, text_mask = text_to_tokens(
        text, max_length=text_max, text_tokenizer=text_tok)
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)

    if mesh_stats is None:
        mesh_stats = {
            "vertex_count": 500, "face_count": 400,
            "edge_count": 900, "bbox_width": 1.0,
            "bbox_depth": 1.0, "bbox_height": 1.0,
            "surface_area": 6.0, "avg_edge_length": 0.1,
            "has_ngons": False, "has_quads_only": False,
            "has_tris_only": False, "is_manifold": True,
        }

    stats_vec = torch.tensor([
        math.log1p(mesh_stats.get("vertex_count", 0)),
        math.log1p(mesh_stats.get("face_count", 0)),
        math.log1p(mesh_stats.get("edge_count", 0)),
        mesh_stats.get("bbox_width", 1.0),
        mesh_stats.get("bbox_depth", 1.0),
        mesh_stats.get("bbox_height", 1.0),
        math.log1p(mesh_stats.get("surface_area", 0)),
        mesh_stats.get("avg_edge_length", 0.1),
        float(mesh_stats.get("has_ngons", False)),
        float(mesh_stats.get("has_quads_only", False)),
        float(mesh_stats.get("has_tris_only", False)),
        float(mesh_stats.get("is_manifold", True)),
    ], dtype=torch.float).unsqueeze(0).to(device)

    start = time.time()
    modifiers = model.predict_modifiers(
        text_ids, text_mask, stats_vec)
    elapsed = time.time() - start

    return {
        "modifiers": modifiers,
        "generation_time": round(elapsed, 2),
    }


def generate_mesh_from_image(state: ModelState, image_b64: str,
                             prompt: str = "",
                             temperature: float = 0.6,
                             top_k: int = 0,
                             top_p: float = 0.9,
                             max_faces: int = 2048,
                             cfg_scale: float = 2.0) -> dict:
    """Generate a mesh conditioned on an image (+ optional text prompt).

    Image is base64-encoded PNG/JPEG.  The model uses its ImageEncoder's
    spatial features (4×4 patch grid, 16 tokens) as cross-attention
    conditioning for the GeometryDecoder, replacing or augmenting the
    text conditioning.

    Requires enable_image_to_mesh=true in config and a checkpoint trained
    with that flag.  Falls back to text-only generation when the model does
    not support image conditioning.
    """
    import base64
    import io
    import numpy as np

    start = time.time()

    with _MODEL_LOCK:
        model = state.model
        tokenizer = state.tokenizer
        device = state.device
        text_tok = state.text_tokenizer
        model_type = state.model_type

    if model_type != "unified":
        return {"error": "Image-to-mesh requires the unified model"}

    if not getattr(model, "enable_image_to_mesh", False):
        # Graceful fallback: text-only generation
        logger.warning(
            "enable_image_to_mesh=false — falling back to text-only generation")
        if not prompt:
            return {
                "error": (
                    "enable_image_to_mesh is disabled in config. "
                    "Set enable_image_to_mesh: true and retrain, or provide a prompt."
                )
            }
        return generate_mesh(
            state, prompt,
            temperature=temperature, top_k=top_k, top_p=top_p,
            max_faces=max_faces, cfg_scale=cfg_scale,
        )

    # ── Decode and preprocess image ──────────────────────────────────
    try:
        # Strip data-URL prefix if present (e.g. coming from Blender addon)
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        raw = base64.b64decode(image_b64)

        try:
            from PIL import Image as PILImage
            img_pil = PILImage.open(io.BytesIO(raw)).convert("RGB")
            img_pil = img_pil.resize((64, 64), PILImage.LANCZOS)
            img_np = np.array(img_pil, dtype=np.float32) / 255.0
        except ImportError:
            # Fallback: use PNG header parsing via stdlib
            # Very basic: only works for simple uncompressed-like PNGs
            return {
                "error": (
                    "Pillow not installed. Run: pip install Pillow. "
                    "Required for image preprocessing."
                )
            }

        # (H, W, 3) → (1, 3, 64, 64)
        img_tensor = (
            torch.tensor(img_np, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        # Normalize to ImageNet mean/std (our CNN was trained on Blender renders
        # which have roughly similar statistics)
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

    except Exception as e:
        return {"error": f"Image preprocessing failed: {e}"}

    # ── Optionally encode text ────────────────────────────────────────
    text_ids, text_mask = None, None
    if prompt.strip():
        text_max = 256
        if hasattr(model.text_encoder, "pos_embed"):
            text_max = model.text_encoder.pos_embed.num_embeddings
        text_ids, text_mask = text_to_tokens(
            prompt, max_length=text_max, text_tokenizer=text_tok)
        text_ids = text_ids.to(device)
        text_mask = text_mask.to(device)

    # ── Generate ─────────────────────────────────────────────────────
    geo_dec = model.geometry_decoder
    max_seq = getattr(geo_dec, "max_seq_length", 16202)
    effective_max = min(max_faces, (max_seq - 2) // 9)
    max_tokens = effective_max * 9 + 2

    prompt_label = repr(prompt) if prompt else "'<image only>'"
    logger.info(
        f"Image-to-mesh: prompt={prompt_label}, "
        f"max_tokens={max_tokens}")

    with torch.no_grad():
        tokens = model.generate_from_image(
            img_tensor,
            text_ids=text_ids,
            text_mask=text_mask,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_scale=cfg_scale,
        )

    token_list = tokens[0].cpu().tolist()
    logger.info(
        f"Generated {len(token_list)} tokens (image-conditioned), "
        f"unique={len(set(token_list))}")

    vertices, faces = tokenizer.decode_tokens(token_list)
    vertices, faces = _merge_duplicate_vertices(vertices, faces)
    faces = _recalculate_normals_consistent(vertices, faces)
    elapsed = time.time() - start

    if not faces or not vertices:
        return {
            "error": (
                f"Model produced no geometry from image ({len(token_list)} tokens). "
                "The image-to-mesh head may need more training. "
                "Try adding a text prompt alongside the image."
            ),
            "generation_time": round(elapsed, 2),
            "token_count": len(token_list),
        }

    name = _clean_name(prompt) if prompt else "FromImage"
    return {
        "objects": [{
            "name": name,
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
        "source": "image_conditioned",
        "prompt": prompt,
    }


def _merge_duplicate_vertices(vertices, faces, threshold=0.0002):
    """Merge duplicate vertices (common after face-based decoding).
    
    If merging causes all faces to become degenerate, returns the
    original unmerged mesh so the user at least gets *something*.
    """
    import numpy as np

    if not vertices or not faces:
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
    seen_faces = set()
    for face in faces:
        new_face = [index_remap.get(vi, vi) for vi in face]
        if len(set(new_face)) >= 3:
            face_key = tuple(sorted(new_face))
            if face_key not in seen_faces:
                seen_faces.add(face_key)
                new_faces.append(new_face)

    # If merge killed all faces, return original unmerged mesh
    if not new_faces and faces:
        logger.warning(
            f"Vertex merge eliminated all {len(faces)} faces "
            f"({len(vertices)} verts -> {len(unique_verts)} unique). "
            f"Returning unmerged mesh.")
        # Center the original vertices
        verts_np = np.array(vertices)
        centroid = verts_np.mean(axis=0)
        verts_np -= centroid
        return verts_np.tolist(), faces

    # Center vertices at origin
    if unique_verts:
        verts_np = np.array(unique_verts)
        centroid = verts_np.mean(axis=0)
        verts_np -= centroid
        unique_verts = verts_np.tolist()

    return unique_verts, new_faces


def _recalculate_normals_consistent(vertices, faces):
    """Ensure winding order is consistent using centroid-based outward normals."""
    import numpy as np

    if not vertices or not faces:
        return faces

    verts = np.asarray(vertices, dtype=np.float32)
    center = verts.mean(axis=0)
    fixed = []

    for face in faces:
        if len(face) < 3:
            continue
        try:
            i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
            if i0 >= len(verts) or i1 >= len(verts) or i2 >= len(verts):
                continue
            v0, v1, v2 = verts[i0], verts[i1], verts[i2]
            normal = np.cross(v1 - v0, v2 - v0)
            if np.linalg.norm(normal) < 1e-12:
                continue
            face_center = np.mean(verts[np.asarray(face, dtype=np.int64)], axis=0)
            outward = face_center - center
            if float(np.dot(normal, outward)) < 0.0:
                fixed.append(list(reversed(face)))
            else:
                fixed.append(face)
        except Exception:
            continue

    return fixed if fixed else faces


def _clean_name(text: str) -> str:
    """Generate a clean object name from prompt text."""
    words = text.strip().split()[:4]
    name = "_".join(words)
    name = "".join(c for c in name if c.isalnum() or c == "_")
    return name[:30] or "Generated"


# ── FastAPI app ──────────────────────────────────────────────────────

def create_app(state: ModelState):
    """Create FastAPI application with hot-reload support."""
    from fastapi import FastAPI
    from pydantic import BaseModel
    from typing import Optional

    app = FastAPI(
        title="Blender Model Server", version="0.2.0",
        description="Serves trained Blender models with hot-reload",
    )

    class MeshRequest(BaseModel):
        prompt: str
        reference_image: Optional[str] = None
        temperature: float = 0.6
        top_k: int = 0  # 0 = disabled, use nucleus sampling only
        top_p: float = 0.9
        # Safety default: huge max_faces can allocate very large KV caches
        # during autoregressive decoding and can hard-crash on MPS.
        max_faces: int = 512
        cfg_scale: float = 2.0  # Lower CFG until model is well-trained

    class MaterialRequest(BaseModel):
        prompt: str
        temperature: float = 0.7
        top_k: int = 30
        max_tokens: int = 512

    class ModifierRequest(BaseModel):
        prompt: str
        mesh_stats: Optional[dict] = None

    class FeedbackPairwise(BaseModel):
        prompt: str
        chosen_tokens: list
        rejected_tokens: list
        metadata: Optional[dict] = None

    class FeedbackAcceptReject(BaseModel):
        prompt: str
        tokens: list
        metadata: Optional[dict] = None

    class FeedbackRating(BaseModel):
        prompt: str
        tokens: list
        rating: float
        metadata: Optional[dict] = None

    class ComparisonRequest(BaseModel):
        prompt: str
        n_candidates: int = 4

    # ── RLHF state (lazy init) ──
    _rlhf_state = {"trainer": None}

    def _get_rlhf_trainer():
        """Lazy-init RLHF trainer on first feedback call."""
        if _rlhf_state["trainer"] is None:
            try:
                from training.rlhf import RLHFTrainer
                from processing.mesh_tokenizer import MeshTokenizer
                tok_config = (state.config or {}).get("tokenization", {})
                mesh_tok = MeshTokenizer(
                    vocab_size=tok_config.get("vocab_size", 8192),
                    coord_range=tuple(tok_config.get("coordinate_range", [-1.0, 1.0])),
                    max_faces=tok_config.get("max_faces", 2048),
                )
                with _MODEL_LOCK:
                    _rlhf_state["trainer"] = RLHFTrainer(
                        state.model, state.config or {},
                        text_tokenizer=state.text_tokenizer,
                        mesh_tokenizer=mesh_tok,
                        device=state.device,
                    )
                logger.info("RLHF trainer initialized")
            except Exception as e:
                logger.warning(f"Failed to init RLHF trainer: {e}")
                return None
        return _rlhf_state["trainer"]

    @app.get("/health")
    def health():
        with _MODEL_LOCK:
            rlhf = _rlhf_state.get("trainer")
            rlhf_status = rlhf.get_status() if rlhf else None
            return {
                "status": "ok",
                "device": str(state.device),
                "model_type": state.model_type,
                "model_params": state.param_count,
                "model_params_m": round(
                    state.param_count / 1e6, 1),
                "step": state.step,
                "checkpoint": state.checkpoint_path,
                "rlhf": rlhf_status,
            }

    @app.post("/generate/mesh")
    def gen_mesh(req: MeshRequest):
        if req.reference_image:
            return generate_mesh_from_image(
                state,
                image_b64=req.reference_image,
                prompt=req.prompt,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                max_faces=req.max_faces,
                cfg_scale=req.cfg_scale,
            )
        return generate_mesh(
            state, req.prompt,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            max_faces=req.max_faces,
            cfg_scale=req.cfg_scale,
        )

    @app.post("/generate/material")
    def gen_material(req: MaterialRequest):
        return generate_material(
            state, req.prompt,
            temperature=req.temperature,
            top_k=req.top_k,
            max_tokens=req.max_tokens,
        )

    @app.post("/generate/modifiers")
    def gen_modifiers(req: ModifierRequest):
        return generate_modifiers(
            state, req.prompt,
            mesh_stats=req.mesh_stats,
        )

    class ImageMeshRequest(BaseModel):
        image: str               # base64-encoded PNG/JPEG
        prompt: str = ""         # optional text hint
        temperature: float = 0.6
        top_k: int = 0
        top_p: float = 0.9
        max_faces: int = 2048
        cfg_scale: float = 2.0

    @app.post("/generate/mesh-from-image")
    def gen_mesh_from_image(req: ImageMeshRequest):
        """Generate a 3D mesh conditioned on an image.

        The model uses its CNN ImageEncoder to extract 16 spatial patch
        features (4×4 grid from a 64×64 render), which are prepended to the
        text cross-attention context in the GeometryDecoder.  Provide an
        optional text prompt to steer semantics.

        Requires the server to be running a checkpoint trained with
        enable_image_to_mesh=true.  Falls back to text-only when unavailable.
        """
        return generate_mesh_from_image(
            state,
            image_b64=req.image,
            prompt=req.prompt,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            max_faces=req.max_faces,
            cfg_scale=req.cfg_scale,
        )

    @app.post("/reload")
    def reload_model():
        """Force hot-reload from current checkpoint path."""
        reloaded = state.check_and_reload()
        if reloaded:
            return {"status": "reloaded", "step": state.step}
        try:
            state.load(
                state.checkpoint_path, state.config or {},
                str(state.device) if state.device else "auto",
            )
            return {"status": "force_reloaded", "step": state.step}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── RLHF Feedback Endpoints ──────────────────────────────────

    @app.post("/feedback/pairwise")
    def feedback_pairwise(req: FeedbackPairwise):
        """Submit pairwise preference: chosen is better than rejected."""
        trainer = _get_rlhf_trainer()
        if trainer is None:
            return {"status": "error", "error": "RLHF trainer not available"}
        trainer.add_pairwise_feedback(
            req.prompt, req.chosen_tokens, req.rejected_tokens,
            metadata=req.metadata)
        result = trainer.maybe_update()
        return {
            "status": "ok",
            "feedback_stats": trainer.feedback_buffer.get_stats(),
            "update_result": result,
        }

    @app.post("/feedback/accept")
    def feedback_accept(req: FeedbackAcceptReject):
        """Submit positive feedback: user accepted this output."""
        trainer = _get_rlhf_trainer()
        if trainer is None:
            return {"status": "error", "error": "RLHF trainer not available"}
        trainer.add_accept_feedback(
            req.prompt, req.tokens, metadata=req.metadata)
        result = trainer.maybe_update()
        return {
            "status": "ok",
            "feedback_stats": trainer.feedback_buffer.get_stats(),
            "update_result": result,
        }

    @app.post("/feedback/reject")
    def feedback_reject(req: FeedbackAcceptReject):
        """Submit negative feedback: user rejected this output."""
        trainer = _get_rlhf_trainer()
        if trainer is None:
            return {"status": "error", "error": "RLHF trainer not available"}
        trainer.add_reject_feedback(
            req.prompt, req.tokens, metadata=req.metadata)
        result = trainer.maybe_update()
        return {
            "status": "ok",
            "feedback_stats": trainer.feedback_buffer.get_stats(),
            "update_result": result,
        }

    @app.post("/feedback/rating")
    def feedback_rating(req: FeedbackRating):
        """Submit scalar rating feedback (1-5 scale)."""
        trainer = _get_rlhf_trainer()
        if trainer is None:
            return {"status": "error", "error": "RLHF trainer not available"}
        trainer.add_rating_feedback(
            req.prompt, req.tokens, req.rating, metadata=req.metadata)
        result = trainer.maybe_update()
        return {
            "status": "ok",
            "feedback_stats": trainer.feedback_buffer.get_stats(),
            "update_result": result,
        }

    @app.post("/feedback/compare")
    def feedback_compare(req: ComparisonRequest):
        """Generate multiple outputs and return the most informative pair for comparison."""
        trainer = _get_rlhf_trainer()
        if trainer is None:
            return {"status": "error", "error": "RLHF trainer not available"}
        return trainer.select_comparison_pair(
            req.prompt, n_candidates=req.n_candidates)

    @app.get("/feedback/status")
    def feedback_status():
        """Get current RLHF training status and feedback statistics."""
        trainer = _get_rlhf_trainer()
        if trainer is None:
            return {"status": "not_initialized"}
        return trainer.get_status()

    @app.post("/feedback/evaluate")
    def feedback_evaluate():
        """Run evaluation dashboard on fixed scenarios."""
        trainer = _get_rlhf_trainer()
        if trainer is None:
            return {"status": "error", "error": "RLHF trainer not available"}
        return trainer.evaluate()

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Blender model inference server (hot-reload)")
    parser.add_argument("--model", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", default="config_unified.yaml")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--watch-interval", type=float, default=30.0,
                        help="Seconds between hot-reload checks")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    _STATE.load(args.model, config, args.device)

    watcher = threading.Thread(
        target=_watch_checkpoint,
        args=(_STATE, args.watch_interval),
        daemon=True,
    )
    watcher.start()
    logger.info(
        f"Hot-reload watcher: checking every {args.watch_interval}s")

    app = create_app(_STATE)

    import uvicorn
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
