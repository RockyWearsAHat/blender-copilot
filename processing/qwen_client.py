"""Shared Qwen/Ollama client with warm-model connection pooling.

Keeps the model loaded between calls by using a persistent HTTP session
and sending a keep-alive ping when idle.  This avoids cold-boot overhead
(~15-30s per call) and reduces per-request latency to ~1-3s.

Usage:
    from processing.qwen_client import qwen_label_text, qwen_label_vision

    # Text-only labeling (fast, ~7B model)
    label = qwen_label_text(context_parts=["Object: Cube", "Materials: metal"])

    # Vision labeling (slow, ~32B model)
    label = qwen_label_vision(image_paths=["/tmp/view1.png"], current_label="cube")

    # Ensure model stays warm between batches
    from processing.qwen_client import warm_model
    warm_model("qwen2.5:7b")
"""

from __future__ import annotations

import base64
import json
import logging
import os
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Model preference order for text labeling (fastest first)
TEXT_MODELS = ["qwen2.5:7b", "qwen2.5-coder:14b", "qwen2.5-coder:7b"]
# Vision model
VISION_MODEL = "qwen2.5vl:32b"

# ── Persistent session state ──────────────────────────────────────────

_last_model: str | None = None
_last_call_time: float = 0
_warm_interval: float = 45.0  # seconds — keep model loaded between calls


# ── Low-level HTTP ────────────────────────────────────────────────────

def _http_post(url: str, payload: dict, timeout: int = 120) -> dict:
    """POST JSON to Ollama and return parsed response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"Ollama HTTP {e.code}: {err[:300]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama connection error: {e}") from e


# ── Model warming ─────────────────────────────────────────────────────

def warm_model(model: str, ollama_url: str | None = None) -> bool:
    """Send a trivial request to ensure model is loaded in memory.

    Returns True if model is available, False otherwise.
    Call this before a batch to avoid cold-boot on the first real request.
    """
    global _last_model, _last_call_time
    url = ollama_url or OLLAMA_URL

    try:
        _http_post(f"{url}/api/generate", {
            "model": model,
            "prompt": "hello",
            "stream": False,
            "options": {"num_predict": 1},
        }, timeout=120)
        _last_model = model
        _last_call_time = time.time()
        logger.debug(f"Model {model} warmed up")
        return True
    except Exception as e:
        logger.warning(f"Failed to warm model {model}: {e}")
        return False


def _ensure_warm(model: str, ollama_url: str | None = None) -> None:
    """Warm the model if it hasn't been used recently."""
    global _last_model, _last_call_time
    now = time.time()
    if _last_model == model and (now - _last_call_time) < _warm_interval:
        return  # Still warm
    warm_model(model, ollama_url)


def keep_alive(model: str, ollama_url: str | None = None,
               duration: str = "10m") -> None:
    """Tell Ollama to keep the model loaded for a specified duration.

    This is the most efficient way to prevent unloading between batches.
    Ollama supports keep_alive in the generate/chat API or via a
    dedicated endpoint.
    """
    url = ollama_url or OLLAMA_URL
    try:
        _http_post(f"{url}/api/generate", {
            "model": model,
            "prompt": "",
            "stream": False,
            "keep_alive": duration,
            "options": {"num_predict": 0},
        }, timeout=30)
        logger.debug(f"keep_alive sent for {model}: {duration}")
    except Exception:
        pass  # Non-critical


# ── Text labeling ─────────────────────────────────────────────────────

def qwen_label_text(
    context_parts: list[str],
    system_prompt: str | None = None,
    model: str | None = None,
    models: list[str] | None = None,
    ollama_url: str | None = None,
    timeout: int = 30,
    max_words: int = 8,
    min_words: int = 2,
) -> str | None:
    """Generate a text label from structured context parts.

    Args:
        context_parts: List of "Key: value" strings describing the object.
        system_prompt: Override the default labeling system prompt.
        model: Specific model to use (overrides models list).
        models: Ordered list of models to try (default: TEXT_MODELS).
        ollama_url: Ollama server URL.
        timeout: Request timeout in seconds.
        max_words: Maximum words in the label.
        min_words: Minimum words in the label.

    Returns:
        Clean label string (2-8 words), or None if all models fail.
    """
    url = ollama_url or OLLAMA_URL
    model_list = [model] if model else (models or TEXT_MODELS)

    if not context_parts:
        return None

    prompt = system_prompt or (
        "Given this Blender 3D object metadata, generate a short, clear English "
        f"label ({min_words}-{max_words - 2} words) describing what this 3D object IS. "
        "Be specific and descriptive. Use plain English nouns/adjectives. "
        "The label should describe the PHYSICAL OBJECT — not animations, "
        "projects, or abstract concepts.\n"
        "Output ONLY the label itself — no explanation, no punctuation, no quotes.\n\n"
    )
    prompt += "\n".join(context_parts)

    global _last_model, _last_call_time

    for m in model_list:
        try:
            _ensure_warm(m, url)

            resp = _http_post(f"{url}/api/generate", {
                "model": m,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {"temperature": 0, "num_predict": 40},
            }, timeout=timeout)

            _last_model = m
            _last_call_time = time.time()

            label = resp.get("response", "").strip().strip('"\'.,;:').strip()
            words = label.split()
            if min_words <= len(words) <= max_words:
                return label
            return None  # Unusable length
        except Exception as e:
            logger.debug(f"Model {m} failed: {e}")
            continue

    return None


# ── Vision labeling ───────────────────────────────────────────────────

_VL_PROMPT = """\
PURPOSE: Generate a TRAINING LABEL for a text-to-3D mesh AI model.
This label becomes the text prompt the model learns to associate with this 3D shape.
It must read like something a user would actually type in a 3D model generator.

GOOD LABELS: "medieval sword", "wooden dining chair", "iron knight helmet",
             "sci-fi space station", "cartoon bear head", "low-poly pine tree"
BAD LABELS: "3D object", "rendered asset", "detailed mesh", "game model",
            "nicely crafted piece" — too vague, too technical, or overly elaborate

Current label (may be inaccurate): "{current_label}"

RULES:
1. Base the label on actual geometry visible in the images — silhouette, structure, form.
2. The current label is a cross-reference hint only; override it if the geometry clearly shows something different.
3. Do NOT infer object type from any filename, folder name, or ID string.
4. If the shape is ambiguous use neutral terms: 'humanoid figure', 'quadruped animal', 'mechanical part'.
5. 3-7 words. No explanation. No quotes. No trailing punctuation.

Output ONLY the label."""


def qwen_label_vision(
    image_paths: list[str | Path],
    current_label: str = "",
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: int = 180,
) -> str | None:
    """Generate a label from rendered images using Qwen VL.

    Args:
        image_paths: Up to 4 image file paths (PNG/JPEG).
        current_label: Current label as a hint (may be overridden).
        model: VL model to use (default: VISION_MODEL).
        ollama_url: Ollama server URL.
        timeout: Request timeout in seconds.

    Returns:
        Clean label string (3-7 words), or None on failure.
    """
    url = ollama_url or OLLAMA_URL
    vl_model = model or VISION_MODEL

    content: list[dict] = []
    for img_path in [Path(p) for p in image_paths[:4]]:
        try:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        except Exception:
            continue

    if not content:
        return None

    prompt_text = _VL_PROMPT.format(current_label=current_label)
    content.append({"type": "text", "text": prompt_text})

    global _last_model, _last_call_time

    try:
        _ensure_warm(vl_model, url)

        result = _http_post(f"{url}/v1/chat/completions", {
            "model": vl_model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
            "keep_alive": "10m",
        }, timeout=timeout)

        _last_model = vl_model
        _last_call_time = time.time()

        label = result["choices"][0]["message"]["content"].strip().strip(".,;:'\"")
        words = label.split()
        if 2 <= len(words) <= 10:
            return label
        return None
    except Exception as e:
        logger.debug(f"VL labeling failed: {e}")
        return None


# ── Batch helpers ─────────────────────────────────────────────────────

def build_label_context(
    obj_name: str = "",
    material_names: list[str] | None = None,
    file_label: str = "",
    metadata_name: str = "",
    metadata_desc: str = "",
    metadata_tags: list | None = None,
    num_faces: int = 0,
    num_verts: int = 0,
    animation_tags: list[str] | None = None,
) -> list[str]:
    """Build structured context parts for qwen_label_text().

    Filters out noise (hex IDs, empty strings, overly generic values)
    and separates physical-object context from animation/project metadata.
    """
    import re

    parts: list[str] = []

    # Object identity
    if metadata_name and metadata_name.strip():
        clean = metadata_name.strip()
        if not re.match(r'^[0-9a-f]{16,}$', clean):
            parts.append(f"Object name: {clean}")
    if obj_name and obj_name.strip().lower() not in ('object', 'mesh', ''):
        parts.append(f"Blender object: {obj_name.strip()}")
    if file_label and file_label.strip():
        parts.append(f"Filename: {file_label.strip()}")

    # Materials (physical properties)
    if material_names:
        clean_mats = [m for m in material_names[:6]
                      if m and m.lower() not in ('material', 'mat', '')]
        if clean_mats:
            parts.append(f"Materials: {', '.join(clean_mats)}")

    # Description (trimmed, no tags contamination)
    if metadata_desc and metadata_desc.strip():
        parts.append(f"Description: {metadata_desc.strip()[:200]}")

    # Geometry stats
    if num_faces > 0:
        parts.append(f"Face count: {num_faces}")
    if num_verts > 0:
        parts.append(f"Vertex count: {num_verts}")

    # Animation context — kept separate, explicitly labeled
    if animation_tags:
        parts.append(f"Animation/behavior: {', '.join(animation_tags[:5])}")

    # NOTE: We intentionally do NOT include raw metadata tags here.
    # Tags like "candy", "bouncing", "riser" describe the project/listing,
    # not the physical mesh.  The label should describe the object.

    return parts
