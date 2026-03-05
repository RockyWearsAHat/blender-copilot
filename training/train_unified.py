"""Unified training — fully autonomous, runs forever, pulls all data sources.

Start it and walk away. It will:
  - Generate infinite synthetic 3D shapes on-the-fly (never repeats)
  - Pull real meshes from Objaverse, BlendSwap, GitHub, Blender demos
  - Generate synthetic materials and modifiers when no real data exists
  - Render images for visual grounding (CLIP-style contrastive)
  - Auto-resume from the latest checkpoint
  - Save continuously so you can stop/restart anytime
  - Background-pull new data from all sources while training

Usage:
    python run.py train                    # just start — does everything
    python run.py train --resume latest    # explicit resume (default)

Architecture:
    InfiniteShapeStream  ->  geometry + contrastive batches (never exhausts)
    RealMeshStream       ->  geometry batches from real 3D data
    MaterialStream       ->  material batches (real or synthetic)
    ModifierStream       ->  modifier batches (real or synthetic)
    ContrastiveStream    ->  image-text pairs for visual grounding

    All streams are infinite. When a source exhausts, it reshuffles or
    regenerates. Training runs until Ctrl+C.
"""

import json
import logging
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import threading
import queue as _queue_mod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler

import yaml
from processing.prompt_semantics import enrich_prompt_text

logger = logging.getLogger(__name__)

# Graceful shutdown
_STOP_TRAINING = False


def _signal_handler(sig, frame):
    global _STOP_TRAINING
    if _STOP_TRAINING:
        logger.info("Force quit.")
        sys.exit(1)
    logger.info(
        "\nGraceful shutdown requested — finishing current step and saving..."
    )
    _STOP_TRAINING = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# =====================================================================
# DeepSeek GRPO-Inspired Training Utilities
# =====================================================================

def grpo_quality_weights(losses: torch.Tensor,
                         temperature: float = 1.0) -> torch.Tensor:
    """Compute group-relative quality weights for a batch of losses.

    Inspired by DeepSeek's GRPO: instead of using a separate reward model,
    we use the inverse loss as a quality signal and normalize within the
    group (batch). Samples with lower loss (higher quality / easier) get
    upweighted; samples with higher loss (lower quality / harder) get
    downweighted. This acts as a form of curriculum learning + data
    quality filtering through gradient weighting.

    GRPO formula adapted: A_i = (r_i - mean(r)) / std(r)
    where r_i = -loss_i (negative loss as reward).

    Args:
        losses: Per-sample losses (batch_size,)
        temperature: Controls sharpness of weighting (default 1.0)

    Returns:
        weights: Normalized weights (batch_size,) that sum to batch_size
    """
    if losses.numel() <= 1:
        return torch.ones_like(losses)

    # Quality signal: lower loss = higher reward.
    # (We intentionally do NOT hard-mine high-loss outliers early.)
    rewards = -losses.detach()

    # Group-relative normalization (GRPO core formula)
    mean_r = rewards.mean()
    std_r = rewards.std()
    if std_r < 1e-8:
        return torch.ones_like(losses)

    advantages = (rewards - mean_r) / (std_r * temperature)

    # Convert advantages to positive weights via softmax-like transform
    # Clamp to prevent extreme weights
    advantages = advantages.clamp(-3.0, 3.0)
    weights = torch.exp(advantages)

    # Normalize so weights sum to batch_size (preserves gradient scale)
    weights = weights * (losses.numel() / weights.sum())

    return weights


def curriculum_max_faces(global_step: int,
                         warmup_steps: int = 2000,
                         min_faces: int = 32,
                         max_faces: int = 512) -> int:
    """Curriculum learning: gradually increase mesh complexity.

    Inspired by DeepSeek's staged training where data complexity
    increases over training. Start with simple shapes (few faces),
    ramp up to full complexity.

    Args:
        global_step: Current training step
        warmup_steps: Steps over which to ramp from min to max
        min_faces: Starting face count
        max_faces: Final face count

    Returns:
        Current max face count for this step
    """
    if global_step >= warmup_steps:
        return max_faces
    progress = global_step / warmup_steps
    # Smooth ramp using cosine schedule
    scale = 0.5 * (1.0 - math.cos(math.pi * progress))
    return int(min_faces + (max_faces - min_faces) * scale)


def deepseek_lr_schedule(step: int, warmup_steps: int = 500,
                         total_steps: int = 100000) -> float:
    """DeepSeek-style multi-step learning rate schedule.

    DeepSeek uses: peak → 31.6% at 80% → 10% at 90%.
    This is more aggressive than pure cosine but prevents
    catastrophic oscillation from cosine restarts.

    Args:
        step: Current step
        warmup_steps: Linear warmup steps
        total_steps: Approximate total training steps for decay

    Returns:
        LR multiplier (0.0 to 1.0)
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)

    if progress < 0.8:
        # Cosine decay from 1.0 to 0.316 over first 80% of training
        t = progress / 0.8
        return 0.316 + 0.684 * 0.5 * (1 + math.cos(math.pi * t))
    elif progress < 0.9:
        # Linear decay from 0.316 to 0.1 over 80-90%
        t = (progress - 0.8) / 0.1
        return 0.316 - 0.216 * t
    else:
        # Constant 0.1 for final 10%
        return 0.1


# =====================================================================
# Focal Cross-Entropy Loss
# =====================================================================

def focal_cross_entropy(
    logits: "torch.Tensor",
    targets: "torch.Tensor",
    gamma: float = 2.0,
    ignore_index: int = 0,
    label_smoothing: float = 0.1,
) -> "torch.Tensor":
    """Cross-entropy loss with focal modulation (Lin et al., 2017).

    Standard CE is dominated by easy/frequent tokens (like coordinate
    bins at ±1.0). Focal loss multiplies each token's CE by
    ``(1 - p_t)^gamma`` where ``p_t`` is the model's predicted
    probability for the correct token. This down-weights "easy"
    (high-confidence) predictions and up-weights rare/hard ones.

    Args:
        logits:  (N, V)  raw logits from the model
        targets: (N,)    ground-truth token indices
        gamma:   focal exponent (0 = plain CE, 2 = strong focal)
        ignore_index: target index to ignore (PAD)
        label_smoothing: standard label smoothing

    Returns:
        Scalar loss (mean over non-ignored tokens).
    """
    import torch
    import torch.nn.functional as F

    # Standard per-token cross-entropy (no reduction)
    ce = F.cross_entropy(
        logits, targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="none",
    )

    # Compute p_t — the model's probability on the correct class
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        # Gather the probability assigned to the correct token
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t).pow(gamma)
        # Zero-out weight for ignored positions
        mask = (targets != ignore_index).float()
        focal_weight = focal_weight * mask

    loss = (ce * focal_weight).sum() / mask.sum().clamp(min=1)
    return loss


# =====================================================================
# Mesh Augmentation Utilities
# =====================================================================

def _random_rotation_matrix(rng: np.random.RandomState | None = None) -> np.ndarray:
    """Generate a random 3D rotation matrix using QR decomposition."""
    if rng is None:
        rng = np.random.RandomState()
    # Random rotation via QR of random normal matrix
    H = rng.randn(3, 3)
    Q, R = np.linalg.qr(H)
    # Ensure proper rotation (det=+1)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def augment_vertices(
    verts: list | np.ndarray,
    *,
    rotate: bool = True,
    jitter_std: float = 0.003,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Apply random augmentation to mesh vertices.

    Augmentation is applied BEFORE normalize_mesh, so the normalization
    will still fill [-1, 1] properly. This breaks axis-alignment that
    causes coordinate tokens to cluster at extreme bins.

    Args:
        verts: (V, 3) vertex positions
        rotate: Apply random 3D rotation
        jitter_std: Gaussian noise std added after rotation
        rng: Numpy RandomState for reproducibility

    Returns:
        (V, 3) augmented vertices
    """
    arr = np.array(verts, dtype=np.float64)
    if len(arr) == 0:
        return arr
    if rng is None:
        rng = np.random.RandomState()

    if rotate:
        R = _random_rotation_matrix(rng)
        arr = arr @ R.T

    if jitter_std > 0:
        arr += rng.randn(*arr.shape) * jitter_std

    return arr


# =====================================================================
# Infinite Synthetic Shape Stream
# =====================================================================

class InfiniteShapeStream(IterableDataset):
    """Generates random synthetic 3D shapes forever. Never repeats.

    Each call produces a fresh random shape with random parameters,
    random rotation, random label style. The parameter space is
    continuous so no two shapes are ever identical.
    """

    def __init__(self, mesh_tokenizer, text_tokenizer=None,
                 max_text_length: int = 256, max_mesh_tokens: int = 16202,
                 image_size: int = 64, render_prob: float = 0.3,
                 include_scenes: bool = True):
        self.mesh_tokenizer = mesh_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.max_mesh_tokens = max_mesh_tokens
        self.image_size = image_size
        self.render_prob = render_prob
        self.include_scenes = include_scenes

        from processing.generate_synthetic import (
            SHAPE_SPECS, COMPOSITE_SPECS, generate_label,
            normalize_mesh, apply_rotation,
        )
        self._shape_specs = SHAPE_SPECS
        self._composite_specs = COMPOSITE_SPECS
        # Include BOTH primitives AND composites for diverse training data.
        # Composites (tables, chairs, houses, etc.) teach the model to
        # differentiate shapes based on text prompts — essential for
        # breaking mode collapse where the model ignores text conditioning.
        self._all_specs = {**SHAPE_SPECS, **COMPOSITE_SPECS}
        self._all_keys = list(self._all_specs.keys())
        self._generate_label = generate_label
        self._normalize = normalize_mesh
        self._apply_rotation = apply_rotation

        self._renderer = None
        if render_prob > 0:
            try:
                from processing.render_shapes import render_and_encode
                self._renderer = render_and_encode
            except ImportError:
                pass

    def __iter__(self):
        while True:
            try:
                if self.include_scenes and random.random() < 0.15:
                    result = self._generate_scene()
                else:
                    result = self._generate_single()
                if result is not None:
                    yield result
            except Exception:
                continue

    def _generate_single(self):
        shape_key = random.choice(self._all_keys)
        spec = self._all_specs[shape_key]

        params = spec["params"]()
        verts, faces = spec["generator"](params)

        if len(verts) < 3 or len(faces) < 1:
            return None

        if random.random() < 0.6:
            angle = random.uniform(0, 360)
            axis = random.choice(["x", "y", "z"])
            verts = self._apply_rotation(verts, angle, axis)

        verts = self._normalize(verts, target_range=(-1.0, 1.0))

        max_faces = (self.max_mesh_tokens - 2) // 9
        if len(faces) > max_faces or len(faces) < 2:
            return None

        tokens = self.mesh_tokenizer.encode_mesh(verts, faces)
        if len(tokens) > self.max_mesh_tokens:
            return None
        if (tokens[0] != self.mesh_tokenizer.BOS
                or tokens[-1] != self.mesh_tokenizer.EOS):
            return None

        label = self._generate_label(shape_key, params)
        text_ids, text_mask = self._encode_text(label)

        result = {
            "task": "geometry",
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.float),
            "mesh_tokens": torch.tensor(tokens, dtype=torch.long),
        }

        if self._renderer and random.random() < self.render_prob:
            try:
                img_data = self._renderer(
                    verts, faces, size=self.image_size)
                img = np.array(img_data, dtype=np.uint8).reshape(
                    self.image_size, self.image_size, 3)
                img_t = torch.tensor(
                    img, dtype=torch.float32).permute(2, 0, 1) / 255.0
                result["image"] = img_t
            except Exception:
                pass

        return result

    def _generate_scene(self):
        from processing.generate_synthetic import _merge, _offset_verts

        n_objects = random.randint(2, 4)
        all_verts, all_faces = [], []
        names = []

        for i in range(n_objects):
            shape_key = random.choice(self._all_keys)
            spec = self._all_specs[shape_key]
            params = spec["params"]()
            verts, faces = spec["generator"](params)

            if len(verts) < 3 or len(faces) < 1:
                continue

            dx = random.uniform(-0.4, 0.4) * (i - n_objects / 2)
            dy = random.uniform(-0.4, 0.4) * (i - n_objects / 2)
            verts = _offset_verts(verts, dx, dy, 0)
            all_verts, all_faces = _merge(
                all_verts, all_faces, verts, faces)
            names.append(random.choice(spec["names"]))

        all_verts = self._normalize(all_verts, target_range=(-1.0, 1.0))

        max_faces = (self.max_mesh_tokens - 2) // 9
        if len(all_faces) > max_faces or len(all_faces) < 4:
            return None

        tokens = self.mesh_tokenizer.encode_mesh(all_verts, all_faces)
        if len(tokens) > self.max_mesh_tokens:
            return None
        if (tokens[0] != self.mesh_tokenizer.BOS
                or tokens[-1] != self.mesh_tokenizer.EOS):
            return None

        label = self._scene_label(names)
        text_ids, text_mask = self._encode_text(label)

        return {
            "task": "geometry",
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.float),
            "mesh_tokens": torch.tensor(tokens, dtype=torch.long),
        }

    def _scene_label(self, names):
        templates = [
            "a scene with {obj_list}",
            "{obj_list} together",
            "a 3D scene containing {obj_list}",
            "create {obj_list} in a scene",
            "{obj_list} arranged together",
            "a composition of {obj_list}",
            "model {obj_list} as a group",
        ]
        if len(names) == 2:
            obj_list = f"a {names[0]} and a {names[1]}"
        else:
            parts = ", ".join(f"a {n}" for n in names[:-1])
            obj_list = f"{parts}, and a {names[-1]}"
        return random.choice(templates).format(obj_list=obj_list)

    def _encode_text(self, text):
        text = enrich_prompt_text(
            text,
            max_hints=8,
            stochastic=True,
            keep_prob=0.65,
            rng=random,
        )
        if self.text_tokenizer is not None:
            return self.text_tokenizer.encode_padded(
                text, self.max_text_length)
        ids = [ord(c) % 32000 for c in text[:self.max_text_length]]
        mask = [1] * len(ids)
        ids += [0] * (self.max_text_length - len(ids))
        mask += [0] * (self.max_text_length - len(mask))
        return ids, mask


# =====================================================================
# Pre-built Training Cache Stream (from scripts/build_training_cache.py)
# =====================================================================

class PrebuiltCacheStream(IterableDataset):
    """Streams training items from a pre-built training cache.

    Reads batch_XXXXX.pt files produced by scripts/build_training_cache.py.
    Items are already tokenized, filtered, and quality-scored — no inline
    processing needed. This is dramatically faster than RealMeshStream
    since it skips source-file scanning, JSON parsing, trimesh processing,
    and cache building entirely.

    Each .pt file contains a list of dicts with keys:
      text_ids, text_mask, mesh_tokens, quality_weight (required)
      label, data_source, scene_context (optional metadata)
    """

    def __init__(self, cache_dir: str, max_mesh_tokens: int = 16202,
                 prefetch_size: int = 256, prefetch_threads: int = 2):
        self.cache_dir = Path(cache_dir)
        self.max_mesh_tokens = max_mesh_tokens
        self._stop_event = threading.Event()

        # Discover batch files
        self._cache_paths = sorted(
            str(p) for p in self.cache_dir.glob("batch_*.pt")
            if p.stat().st_size > 200
        )
        random.shuffle(self._cache_paths)

        # Count total items (quick scan of first file)
        n_items_sample = 0
        if self._cache_paths:
            try:
                sample = torch.load(self._cache_paths[0], weights_only=False)
                n_items_sample = len(sample)
            except Exception:
                pass

        logger.info(
            f"PrebuiltCacheStream: {len(self._cache_paths)} batch files "
            f"(~{n_items_sample} items/batch) from {cache_dir}"
        )

        # Prefetch queue
        self._prefetch_queue = _queue_mod.Queue(maxsize=prefetch_size)
        self._prefetch_threads_list = []
        self._prefetch_threads_count = prefetch_threads
        self._file_lock = threading.Lock()
        self._cache_idx = 0
        self._started = False

    def _start_prefetch(self):
        if self._started:
            return
        self._started = True
        for i in range(self._prefetch_threads_count):
            t = threading.Thread(
                target=self._prefetch_worker, daemon=True,
                name=f"PrebuiltCachePrefetch-{i}")
            t.start()
            self._prefetch_threads_list.append(t)

    def _next_cache_file(self):
        with self._file_lock:
            if not self._cache_paths:
                return None
            if self._cache_idx >= len(self._cache_paths):
                self._cache_idx = 0
                random.shuffle(self._cache_paths)
            path = self._cache_paths[self._cache_idx]
            self._cache_idx += 1
            return path

    def _prefetch_worker(self):
        """Background thread: loads batch .pt files into the queue."""
        while not self._stop_event.is_set():
            path = self._next_cache_file()
            if path is None:
                time.sleep(2)
                continue
            try:
                cached = torch.load(path, weights_only=False)
                if not cached:
                    continue
                for item in cached:
                    mesh_len = len(item["mesh_tokens"])
                    if mesh_len > self.max_mesh_tokens:
                        continue
                    result = {
                        "task": "geometry",
                        "text_ids": item["text_ids"],
                        "text_mask": item["text_mask"],
                        "mesh_tokens": item["mesh_tokens"],
                        "quality_weight": item["quality_weight"],
                    }
                    # Pass through optional metadata
                    for key in ("label_confidence", "scene_complexity_score",
                                "composition", "workflow_supervision",
                                "quality_tier", "label_quality_score",
                                "mesh_quality_score", "image"):
                        if key in item:
                            val = item[key]
                            if key == "image":
                                if val.dtype == torch.uint8:
                                    val = val.float() / 255.0
                                if val.dim() == 3 and val.shape[-1] == 3:
                                    val = val.permute(2, 0, 1)
                            result[key] = val
                    while not self._stop_event.is_set():
                        try:
                            self._prefetch_queue.put(result, timeout=1)
                            break
                        except _queue_mod.Full:
                            continue
            except Exception as exc:
                logger.debug(f"PrebuiltCacheStream: error loading {path}: {exc}")

    def __iter__(self):
        self._start_prefetch()
        while True:
            try:
                yield self._prefetch_queue.get(timeout=10)
            except _queue_mod.Empty:
                continue

    def stop(self):
        self._stop_event.set()
        for t in self._prefetch_threads_list:
            t.join(timeout=5)


# =====================================================================
# Real Mesh Stream (Objaverse, BlendSwap, GitHub, Blender demos)
# =====================================================================

class RealMeshStream(IterableDataset):
    """Streams real 3D meshes from processed data directories.

    Architecture:
      - ALL files are included regardless of size (no skipping)
      - Each file is processed once and cached as a compact .pt file
      - Background threads pre-process uncached files during init
      - At training time, only .pt cache files are read (instant)
      - Quality scoring runs once per object during cache build
      - Periodically checks for NEW files added by background scrapers

    Cache strategy:
      - Source JSON (e.g., 3GB) → parsed, tokenized, scored → saved as
        compact .pt (e.g., 50KB) containing only training-ready tensors
      - First training run: slow while cache builds in background
      - All subsequent runs: instant loading from cache
    """

    def __init__(self, data_dirs, mesh_tokenizer,
                 text_tokenizer=None, max_text_length: int = 256,
                 max_mesh_tokens: int = 16202, rescan_interval: int = 500,
                 prefetch_size: int = 256,
                 prefetch_threads: int = 2):
        self.data_dirs = [Path(d) for d in data_dirs if Path(d).exists()]
        self.mesh_tokenizer = mesh_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.max_mesh_tokens = max_mesh_tokens
        self.rescan_interval = rescan_interval
        self._file_paths = []        # paths to source JSONs (for scanning)
        self._cache_paths = []       # paths to .pt cache files (for training)
        self._seen_files = set()

        # Disk cache directory
        self._cache_dir = (Path(self.data_dirs[0]).parent / ".mesh_cache"
                           if self.data_dirs else Path(".mesh_cache"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Scan all source files (no size filter)
        self._scan_files()

        # Build cache SYNCHRONOUSLY before training starts.
        # Running this in a background thread caused OOM: the cache
        # builder loads multi-GB JSON files + runs trimesh decimation
        # + renders images, all competing with training for the 36GB
        # unified memory on Apple Silicon.  The model handles 4608
        # tokens (511 faces) fine when it has full memory access.
        self._cache_build_thread = None
        self._stop_event = threading.Event()
        uncached = [p for p in self._file_paths
                    if not self._cache_key(p).exists()]
        cached_count = len(self._file_paths) - len(uncached)
        logger.info(f"RealMeshStream: {len(self._file_paths)} source files, "
                    f"{cached_count} already cached, "
                    f"{len(uncached)} need processing")
        # Minimum cached items required to start training immediately.
        # Below this threshold we build synchronously first so the model
        # has a useful dataset from the very first step.
        _MIN_CACHED_TO_TRAIN = 20

        if uncached:
            if torch.cuda.is_available():
                # On cloud GPUs, SKIP uncached files entirely.
                # The 70 uncached files include multi-GB JSON blobs
                # that take 30+ minutes each for json.load() + trimesh
                # decimation on CPU. Not worth blocking GPU training.
                # Train with the 1973 already-cached files instead.
                logger.info(
                    f"CUDA: SKIPPING {len(uncached)} uncached files "
                    f"(use 'python -c \"...\"' offline to build cache). "
                    f"Training with {cached_count} cached files.")
            elif cached_count >= _MIN_CACHED_TO_TRAIN:
                # Enough cached files to train on right now.
                # Build remaining files in a background thread with a
                # small per-file sleep so it doesn't compete with the
                # training loop for unified memory on Apple Silicon.
                logger.info(
                    f"MPS: {cached_count} files already cached — starting "
                    f"training immediately. {len(uncached)} files will be "
                    f"processed in the background (throttled).")
                self._cache_build_thread = threading.Thread(
                    target=self._build_cache_worker_background,
                    args=(uncached,), daemon=True)
                self._cache_build_thread.start()
            else:
                # Not enough data yet — build synchronously up to the
                # threshold, then hand off to a background thread.
                need = _MIN_CACHED_TO_TRAIN - cached_count
                bootstrap = uncached[:need]
                rest = uncached[need:]
                logger.info(
                    f"Building initial {len(bootstrap)} cache files "
                    f"synchronously, then switching to background...")
                self._build_cache_worker(bootstrap)
                logger.info(
                    f"Initial cache ready — training can start. "
                    f"{len(rest)} files queued for background processing.")
                if rest:
                    # Refresh so the remaining uncached list is accurate
                    rest = [p for p in rest
                            if not self._cache_key(p).exists()]
                if rest:
                    self._cache_build_thread = threading.Thread(
                        target=self._build_cache_worker_background,
                        args=(rest,), daemon=True)
                    self._cache_build_thread.start()

        # Build initial list of available cache files
        self._refresh_cache_paths()

        # Prefetch buffer: background threads load cached .pt files
        # into a queue so the training loop never blocks on disk I/O.
        # These only start AFTER cache build is done, so they never
        # compete with the memory-heavy cache builder.
        self._prefetch_queue = _queue_mod.Queue(maxsize=prefetch_size)
        self._prefetch_threads_list = []
        self._prefetch_threads_count = prefetch_threads
        self._file_lock = threading.Lock()
        self._cache_idx = 0

        # DO NOT start prefetch threads here! DataLoader with
        # num_workers>0 will fork() this process. Threads don't
        # survive fork, and inherited locked mutexes cause deadlock.
        # Prefetching starts lazily in __iter__() which runs inside
        # each DataLoader worker process AFTER the fork.
        logger.info(f"RealMeshStream: {self._prefetch_threads_count} prefetch "
                     f"threads ready, {len(self._cache_paths)} cache files")

    def _scan_files(self):
        """Scan all data directories for source JSON files.

        Skips .meta.json files (metadata only, no mesh data).
        """
        new_paths = []
        for d in self.data_dirs:
            if not d.exists():
                continue
            for f in d.rglob("*.json"):
                if f.name.endswith(".meta.json"):
                    continue
                key = str(f)
                if key not in self._seen_files:
                    self._seen_files.add(key)
                    new_paths.append(key)
        if new_paths:
            self._file_paths.extend(new_paths)
            logger.info(f"RealMeshStream: +{len(new_paths)} files "
                        f"(total {len(self._file_paths)})")

    def _refresh_cache_paths(self):
        """Rebuild the list of available .pt cache files.

        Filters out empty/marker files (< 200 bytes) so prefetch
        threads never waste time loading them.
        """
        paths = sorted(self._cache_dir.glob("*.pt"))
        self._cache_paths = [
            str(p) for p in paths if p.stat().st_size > 200
        ]
        random.shuffle(self._cache_paths)

    def _cache_key(self, path):
        """Generate a cache filename from the source path."""
        import hashlib
        h = hashlib.md5(path.encode()).hexdigest()[:16]
        return self._cache_dir / f"{h}.pt"

    # ── Cache building (background) ──────────────────────────────────

    def _build_cache_worker_background(self, uncached_paths):
        """Throttled wrapper around _build_cache_worker for background use.

        Sleeps briefly between files so the cache builder doesn't compete
        with the active training loop for unified memory on Apple Silicon.
        The training loop and cache builder both load large data structures;
        1.5s between files gives the GC a chance to reclaim memory from
        the previous file before the next one is loaded.
        """
        import time as _time
        from processing.generate_synthetic import normalize_mesh
        try:
            from processing.quality_filter import MeshQualityScorer
            scorer = MeshQualityScorer()
        except Exception:
            scorer = None
        renderer = None
        try:
            from processing.render_shapes import render_mesh_to_image
            renderer = render_mesh_to_image
        except Exception:
            pass

        processed = 0
        total = len(uncached_paths)
        items_total = 0
        decimated_total = 0
        max_f = (self.max_mesh_tokens - 2) // 9

        for path in uncached_paths:
            if self._stop_event.is_set():
                break
            cache_path = self._cache_key(path)
            if cache_path.exists():
                continue
            try:
                cache_items, n_decimated = self._process_source_file(
                    path, normalize_mesh, scorer, max_f, renderer)
                items_total += len(cache_items) if cache_items else 0
                decimated_total += n_decimated
                if cache_items:
                    torch.save(cache_items, cache_path)
            except Exception as e:
                logger.debug(f"BG cache build failed for "
                             f"{Path(path).name}: {e}")
                continue
            finally:
                import gc
                gc.collect()
            processed += 1
            if processed % 20 == 0:
                self._refresh_cache_paths()
                fsize_mb = Path(path).stat().st_size / 1048576
                logger.info(f"[BG cache] {processed}/{total} files "
                            f"({len(self._cache_paths)} cached, "
                            f"{items_total} items, "
                            f"last: {Path(path).name} {fsize_mb:.0f}MB)")
            # Throttle: yield CPU + memory to the training loop between files
            _time.sleep(1.5)

        self._refresh_cache_paths()
        logger.info(f"[BG cache] complete: {processed}/{total} files, "
                    f"{len(self._cache_paths)} total cache files, "
                    f"{items_total} items")

    def _build_cache_worker(self, uncached_paths):
        """Process source JSONs into .pt cache files (runs synchronously).

        Runs BEFORE training starts so it gets full memory access.
        Each file is loaded, objects are decimated (if needed),
        tokenized, quality-scored, rendered to images, and saved as
        compact .pt files.  Explicit gc.collect() after each file
        prevents memory accumulation from multi-GB JSON parsing.
        """
        from processing.generate_synthetic import normalize_mesh
        try:
            from processing.quality_filter import MeshQualityScorer
            scorer = MeshQualityScorer()
        except Exception:
            scorer = None

        # Renderer for caching images of real meshes
        renderer = None
        try:
            from processing.render_shapes import render_mesh_to_image
            renderer = render_mesh_to_image
        except Exception:
            logger.warning("render_shapes not available — cache will lack images")

        processed = 0
        total = len(uncached_paths)
        items_total = 0
        decimated_total = 0
        max_f = (self.max_mesh_tokens - 2) // 9

        for path in uncached_paths:
            if self._stop_event.is_set():
                break

            cache_path = self._cache_key(path)
            if cache_path.exists():
                continue

            try:
                cache_items, n_decimated = self._process_source_file(
                    path, normalize_mesh, scorer, max_f, renderer)
                items_total += len(cache_items) if cache_items else 0
                decimated_total += n_decimated
                if cache_items:
                    torch.save(cache_items, cache_path)
                # Don't save empty markers — wastes disk and prefetch time
            except Exception as e:
                logger.debug(f"Cache build failed for {Path(path).name}: {e}")
                continue
            finally:
                # Free memory from large JSON parsing + trimesh objects
                import gc
                gc.collect()

            processed += 1
            if processed % 20 == 0:
                self._refresh_cache_paths()
                fsize_mb = Path(path).stat().st_size / 1048576
                logger.info(f"Cache build: {processed}/{total} files "
                            f"({len(self._cache_paths)} cached, "
                            f"{items_total} items, {decimated_total} decimated, "
                            f"last: {Path(path).name} {fsize_mb:.0f}MB)")

        # Final refresh
        self._refresh_cache_paths()
        logger.info(f"Cache build complete: {processed}/{total} files processed, "
                    f"{len(self._cache_paths)} cache files with data, "
                    f"{items_total} total items ({decimated_total} decimated)")

    def _process_source_file(self, path, normalize_mesh, scorer, max_f,
                              renderer=None):
        """Process a single source JSON file into cache items.

        Returns:
            (items, n_decimated) — cache items and count of meshes that
            were decimated to fit the token limit.
        """
        fsize = Path(path).stat().st_size
        use_streaming = fsize > 100 * 1024 * 1024  # >100MB

        if use_streaming:
            return self._process_large_file_streaming(
                path, normalize_mesh, scorer, max_f, renderer)

        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            objects = data.get("objects", [data])
            metadata = data.get("metadata", {})
        else:
            objects = [data]
            metadata = {}

        result = self._tokenize_objects(
            objects, metadata, normalize_mesh, scorer, max_f, renderer)
        del data, objects
        return result

    def _process_large_file_streaming(self, path, normalize_mesh,
                                       scorer, max_f, renderer=None):
        """Process a large JSON file by loading objects one section at a time.

        For multi-GB files with thousands of objects, we load the full
        file but process objects in chunks to limit peak memory.
        """
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, MemoryError):
            logger.debug(f"Failed to load large file: {Path(path).name}")
            return [], 0

        if isinstance(data, dict):
            objects = data.get("objects", [data])
            metadata = data.get("metadata", {})
        else:
            objects = [data]
            metadata = {}

        # Process in chunks to limit memory pressure
        all_items = []
        total_decimated = 0
        chunk_size = 50
        for i in range(0, len(objects), chunk_size):
            if self._stop_event.is_set():
                break
            chunk = objects[i:i + chunk_size]
            items, n_dec = self._tokenize_objects(
                chunk, metadata, normalize_mesh, scorer, max_f, renderer)
            all_items.extend(items)
            total_decimated += n_dec

        # Free the large data structure immediately
        del data, objects
        import gc
        gc.collect()

        return all_items, total_decimated

    @staticmethod
    def _decimate_mesh(verts, faces, target_faces):
        """Decimate a mesh to target_faces using trimesh + fast_simplification.

        Three-tier strategy:
          1. fast_simplification via trimesh (best quality)
          2. Retry with mesh cleanup (process=True) if tier-1 fails
          3. Naive random face sampling (always works, lower quality)

        The C backend in fast_simplification can crash on degenerate
        geometry or under memory pressure, so tiers 2-3 ensure we
        never drop a mesh entirely.
        """
        try:
            import trimesh
            v_arr = np.array(verts, dtype=np.float64)
            f_arr = np.array(faces, dtype=np.int64)
            if f_arr.ndim != 2 or f_arr.shape[1] != 3:
                return verts, faces, False

            # ── Tier 1: fast decimation (process=False for speed) ──
            mesh = trimesh.Trimesh(vertices=v_arr, faces=f_arr,
                                   process=False)
            decimated = None
            for aggression in [None, 7, 10]:
                try:
                    d = mesh.simplify_quadric_decimation(
                        face_count=target_faces,
                        **({"aggression": aggression} if aggression else {}))
                    if len(d.faces) <= int(target_faces * 1.1):
                        decimated = d
                        break
                except Exception:
                    continue

            # ── Tier 2: retry with mesh cleanup ──
            if decimated is None:
                try:
                    mesh_clean = trimesh.Trimesh(
                        vertices=v_arr, faces=f_arr, process=True)
                    mesh_clean.merge_vertices()
                    mesh_clean.remove_degenerate_faces()
                    mesh_clean.remove_duplicate_faces()
                    if len(mesh_clean.faces) <= target_faces:
                        decimated = mesh_clean
                    else:
                        d = mesh_clean.simplify_quadric_decimation(
                            face_count=target_faces, aggression=7)
                        if len(d.faces) <= int(target_faces * 1.1):
                            decimated = d
                except Exception:
                    pass

            # ── Tier 3: naive random face sampling (always works) ──
            if decimated is None:
                rng = np.random.default_rng(len(faces))
                idx = rng.choice(len(f_arr), size=target_faces,
                                 replace=False)
                sampled_faces = f_arr[idx]
                used_verts = np.unique(sampled_faces)
                remap = np.full(v_arr.shape[0], -1, dtype=np.int64)
                remap[used_verts] = np.arange(len(used_verts))
                new_faces = remap[sampled_faces]
                new_verts = v_arr[used_verts]
                decimated = trimesh.Trimesh(
                    vertices=new_verts, faces=new_faces, process=False)

            if len(decimated.faces) < 2:
                return verts, faces, False
            return (
                decimated.vertices.tolist(),
                decimated.faces.tolist(),
                True,
            )
        except Exception as e:
            logger.warning(f"Decimation total failure ({len(faces)} faces): "
                           f"{type(e).__name__}: {e}")
            return verts, faces, False

    def _tokenize_objects(self, objects, metadata, normalize_mesh,
                          scorer, max_f, renderer=None):
        """Tokenize a list of mesh objects into cache-ready items.

        Key quality decisions:
          - Meshes with >max_f faces are DECIMATED, not skipped.
            Complex models are the most valuable training data.
          - Each mesh is rendered to a 64x64 image and cached.
            This is the only way the model learns to associate
            visual appearance with 3D geometry.
          - Quality scoring runs on the original (pre-decimation)
            mesh so the score reflects true mesh quality.

        Returns:
            (items, n_decimated) tuple
        """
        items = []
        n_decimated = 0
        for obj in objects:
            mesh = obj.get("mesh", {})
            verts = mesh.get("vertices", [])
            faces = mesh.get("faces", [])

            if not verts or not faces:
                continue
            if len(faces) < 2:
                continue

            # Quality scoring on ORIGINAL mesh (before decimation)
            quality_info = obj.get("quality_score_info")
            if quality_info is None and scorer is not None:
                try:
                    quality_info = scorer.score(mesh)
                except Exception:
                    quality_info = {"quality_score": 0.5}
            quality = (quality_info or {}).get("quality_score", 0.5)
            # quality_weight is finalized after tokenization once we know
            # the actual fill rate (face tokens used / max possible).
            quality_base = max(0.2, 0.3 + quality * 1.2)

            # Decimate if over the face limit instead of skipping
            was_decimated = False
            original_faces = len(faces)

            # Hard cap: skip objects too large to produce useful training examples.
            # QEM decimation of 50k+ → 128 faces destroys all topology and is very
            # slow (minutes per mesh). Random-subsample is fast but meaningless at
            # this ratio. Skip entirely — real-world scene meshes with 50k+ faces
            # are almost never useful at max_faces=128.
            MAX_DECIMATE_FACES = 200_000
            if original_faces > MAX_DECIMATE_FACES:
                logger.info(
                    f"SKIP (too large): {obj.get('name','?')} "
                    f"{original_faces} faces > {MAX_DECIMATE_FACES} cap"
                )
                continue

            if len(faces) > max_f:
                # Belt-and-suspenders: also skip here in case the 50k check above
                # doesn't fire (e.g. old .pyc in __pycache__).
                if len(faces) > 200_000:
                    logger.info(
                        f"SKIP (too large) [guard2]: {obj.get('name','?')} "
                        f"{len(faces)} faces"
                    )
                    continue
                logger.info(f"DECIMATING: {obj.get('name','?')} {len(faces)} faces -> {max_f}")
                verts, faces, was_decimated = self._decimate_mesh(
                    verts, faces, max_f)
                if not was_decimated:
                    logger.warning(f"DECIMATION FAILED: {obj.get('name','?')} {original_faces} faces")
                    continue  # decimation failed, skip
                logger.info(f"DECIMATED OK: {obj.get('name','?')} {original_faces} -> {len(faces)} faces")
                n_decimated += 1

            # ── Mesh augmentation ─────────────────────────────────────
            # Apply random rotation + small jitter BEFORE normalization.
            # This breaks axis-alignment that causes coordinate tokens
            # to cluster at extreme bins (token 4 and token vocab-1),
            # spreading the token distribution more uniformly across
            # all coordinate bins.  Essential for preventing mode collapse.
            aug_rng = np.random.RandomState(hash(str(verts[:3])) & 0xFFFFFFFF)
            verts_aug = augment_vertices(
                verts, rotate=True, jitter_std=0.002, rng=aug_rng)

            try:
                verts_norm = normalize_mesh(
                    verts_aug.tolist() if isinstance(verts_aug, np.ndarray) else verts_aug,
                    target_range=(-1.0, 1.0))
                tokens = self.mesh_tokenizer.encode_mesh(
                    verts_norm, faces)
            except Exception:
                continue

            if len(tokens) > self.max_mesh_tokens:
                continue
            if (tokens[0] != self.mesh_tokenizer.BOS
                    or tokens[-1] != self.mesh_tokenizer.EOS):
                continue

            # Fill-rate: kept as monitoring metric and partial weight signal.
            # Meshes that fill a reasonable fraction of the face budget are more
            # informative; 0.3 floor means sparse objects are not discarded.
            face_tokens = len(tokens) - 2  # exclude BOS+EOS
            max_face_tokens = max(1, max_f * 9)
            fill_rate = min(1.0, face_tokens / max_face_tokens)
            fill_weight = max(0.3, fill_rate ** 0.5)
            # sample_weight is finalised below after geometric distribution scoring

            # Text label — use smart labeling from object properties
            from processing.labeler_smart import generate_smart_label, compute_bbox_aspect
            
            obj_name = obj.get("name", "")
            mat_names = [m.get("name", "") for m in obj.get("materials", [])]
            mod_types = [m.get("type", "") for m in obj.get("modifiers", [])]
            bbox_aspect = compute_bbox_aspect(verts_norm)
            
            text = generate_smart_label(
                obj_name=obj_name,
                material_names=mat_names,
                modifier_types=mod_types,
                num_faces=len(faces),
                num_verts=len(verts),
                bbox_aspect=bbox_aspect,
                file_label=obj.get("text_label", ""),
                metadata_name=metadata.get("name", ""),
                metadata_desc=str(metadata.get("description", ""))[:200],
                metadata_tags=metadata.get("tags", []),
                metadata_categories=metadata.get("categories", ""),
            )
            
            if was_decimated and original_faces > 1000:
                detail = "high-detail" if original_faces > 10000 else "detailed"
                text = f"{text} ({detail})"

            # ── Geometric distribution scoring ─────────────────────────────
            # Score the mesh based on how closely its shape distributions
            # (bounding-box proportions, height profile, radial spread, XZ
            # angular distribution) match the EMA prototype for its class.
            # Until enough samples accumulate for a class the score defaults
            # to 0.5 (neutral), then it rises as the prototype stabilises.
            #
            # We also augment the text prompt with compact distribution tokens
            # so the model learns to associate shape words with geometry:
            #   "car [wide:0.9 tall:0.3 fill:shell hpeak:base sym:0.8]"
            from processing.mesh_geometry_score import (
                compute_signature, shape_descriptor_tokens, get_global_registry
            )
            _geo_sig = compute_signature(verts_norm)
            _geo_registry = get_global_registry()
            _shape_score = _geo_registry.score(text, _geo_sig)
            _geo_registry.update(text, _geo_sig)

            _desc = shape_descriptor_tokens(_geo_sig)
            if _desc:
                text = text + _desc

            # Blend: 40 % fill-rate (non-empty guarantee)
            #      + 60 % shape-score (geometric typicality)
            # Until prototypes are established shape_score == 0.5, giving
            # ~0.7 × fill_weight — very close to the old fill-only behaviour.
            sample_weight = quality_base * (0.4 * fill_weight + 0.6 * _shape_score)
            # ─────────────────────────────────────────────────────────────

            text_ids, text_mask = self._encode_text(text)

            semantic_parts = obj.get("semantic_parts", [])
            if not isinstance(semantic_parts, list):
                semantic_parts = []
            composition = obj.get("composition")
            if not isinstance(composition, dict):
                composition = self._infer_composition(
                    label=text,
                    metadata=metadata,
                    semantic_parts=semantic_parts,
                    object_count=len(objects),
                    total_faces=original_faces,
                )
            workflow_supervision = obj.get("workflow_supervision")
            if not isinstance(workflow_supervision, dict):
                workflow_supervision = self._infer_workflow_supervision(
                    label=text,
                    composition=composition,
                    semantic_parts=semantic_parts,
                )

            label_confidence = float(obj.get("label_confidence", 1.0))
            label_confidence = max(0.0, min(1.0, label_confidence))
            scene_complexity_score = float(
                composition.get("scene_complexity_score", 0.3)
            )
            scene_complexity_score = max(0.0, min(1.0, scene_complexity_score))

            item = {
                "text_ids": torch.tensor(text_ids, dtype=torch.long),
                "text_mask": torch.tensor(text_mask, dtype=torch.float),
                "mesh_tokens": torch.tensor(tokens, dtype=torch.long),
                "quality_weight": torch.tensor(sample_weight,
                                               dtype=torch.float),
                "fill_rate": torch.tensor(fill_rate, dtype=torch.float),
                "label_confidence": torch.tensor(label_confidence, dtype=torch.float),
                "scene_complexity_score": torch.tensor(scene_complexity_score, dtype=torch.float),
                "composition": composition,
                "workflow_supervision": workflow_supervision,
            }

            # Render image and cache it — this is how the model
            # learns to associate visual appearance with geometry
            if renderer is not None:
                try:
                    img = renderer(verts_norm, faces, size=64)
                    img_t = torch.tensor(
                        np.array(img, dtype=np.uint8),
                        dtype=torch.uint8,
                    )
                    item["image"] = img_t  # (64, 64, 3) uint8
                except Exception:
                    pass  # image rendering failed, still cache mesh data

            items.append(item)

        return items, n_decimated

    def _infer_composition(self, label, metadata, semantic_parts,
                           object_count, total_faces):
        text = f"{label} {' '.join(semantic_parts)} {metadata.get('name', '')}".lower()
        if any(k in text for k in ("character", "creature", "animal", "humanoid", "armature")):
            domain = "character"
        elif any(k in text for k in ("car", "vehicle", "robot", "mech", "engine", "ship", "plane")):
            domain = "vehicle"
        elif any(k in text for k in ("house", "building", "room", "interior", "architecture", "bridge")):
            domain = "environment"
        elif any(k in text for k in ("chair", "table", "desk", "shelf", "sofa", "lamp")):
            domain = "prop_set"
        else:
            domain = "object"

        if object_count >= 8 or total_faces >= 120000:
            tier, score = "hero", 1.0
        elif object_count >= 4 or total_faces >= 50000:
            tier, score = "complex", 0.8
        elif object_count >= 2 or total_faces >= 12000:
            tier, score = "medium", 0.55
        else:
            tier, score = "simple", 0.3

        tags = []
        if object_count > 1:
            tags.append("multi_object")
        if semantic_parts:
            tags.append("semantic_parts")
        if not tags:
            tags.append("single_object")

        return {
            "scene_domain": domain,
            "composition_label": label,
            "composition_tags": tags,
            "semantic_part_count": len(semantic_parts),
            "object_count": int(max(1, object_count)),
            "total_face_count": int(max(0, total_faces)),
            "complexity_tier": tier,
            "scene_complexity_score": float(score),
        }

    def _infer_workflow_supervision(self, label, composition, semantic_parts):
        domain = composition.get("scene_domain", "object")
        object_count = int(composition.get("object_count", 1))
        actions = ["decompose_prompt", "generate_base_mesh", "inspect_scene"]
        if object_count > 1:
            actions.append("arrange_scene")
        if domain in ("vehicle", "environment", "character"):
            actions.append("apply_modifiers")
        actions.extend(["assign_materials", "capture_viewport", "declare_complete"])

        targets = ["modeling", "shading", "qa"]
        if domain == "character":
            targets.append("rigging")
        if object_count > 1:
            targets.append("scene_assembly")

        return {
            "initial_state_summary": "empty_scene",
            "target_instruction": label,
            "workflow_targets": targets,
            "action_sequence": actions,
            "final_state_checks": [
                "non_empty_scene",
                "reasonable_scale",
                "materials_assigned",
                "completion_declared",
            ],
            "semantic_parts": list(semantic_parts)[:12],
        }

    # ── Prefetch (training time — reads only from cache) ─────────────

    def _start_prefetching(self):
        """Start background prefetch threads if not running."""
        alive = [t for t in self._prefetch_threads_list if t.is_alive()]
        if len(alive) >= self._prefetch_threads_count:
            return
        self._stop_event.clear()
        for i in range(self._prefetch_threads_count - len(alive)):
            t = threading.Thread(
                target=self._prefetch_worker, daemon=True,
                name=f"RealMeshPrefetch-{i}")
            t.start()
            self._prefetch_threads_list.append(t)
        logger.info(f"RealMeshStream: {self._prefetch_threads_count} prefetch "
                     f"threads, {len(self._cache_paths)} cache files ready")

    def _next_cache_file(self):
        """Thread-safe: get next cache .pt file path."""
        with self._file_lock:
            if not self._cache_paths:
                self._refresh_cache_paths()
                if not self._cache_paths:
                    return None

            if self._cache_idx >= len(self._cache_paths):
                self._cache_idx = 0
                random.shuffle(self._cache_paths)
                # Periodically pick up new cache files + new source files
                self._refresh_cache_paths()
                self._scan_files()

            path = self._cache_paths[self._cache_idx]
            self._cache_idx += 1
            return path

    def _prefetch_worker(self):
        """Background thread: loads .pt cache files into the queue."""
        while not self._stop_event.is_set():
            path = self._next_cache_file()
            if path is None:
                time.sleep(2)
                continue

            try:
                cached = torch.load(path, weights_only=False)
                if not cached:
                    continue
                for item in cached:
                    mesh_len = len(item["mesh_tokens"])
                    if mesh_len > self.max_mesh_tokens:
                        continue
                    result = {
                        "task": "geometry",
                        "text_ids": item["text_ids"],
                        "text_mask": item["text_mask"],
                        "mesh_tokens": item["mesh_tokens"],
                        "quality_weight": item["quality_weight"],
                    }
                    if "label_confidence" in item:
                        result["label_confidence"] = item["label_confidence"]
                    if "scene_complexity_score" in item:
                        result["scene_complexity_score"] = item["scene_complexity_score"]
                    if "composition" in item:
                        result["composition"] = item["composition"]
                    if "workflow_supervision" in item:
                        result["workflow_supervision"] = item["workflow_supervision"]
                    # Quality tier metadata from restructured cache
                    if "quality_tier" in item:
                        result["quality_tier"] = item["quality_tier"]
                    if "label_quality_score" in item:
                        result["label_quality_score"] = item["label_quality_score"]
                    if "mesh_quality_score" in item:
                        result["mesh_quality_score"] = item["mesh_quality_score"]
                    # Pass cached image through if present
                    # Convert (H,W,3) uint8 → (3,H,W) float32 for model
                    if "image" in item:
                        img = item["image"]
                        if img.dtype == torch.uint8:
                            img = img.float() / 255.0
                        if img.dim() == 3 and img.shape[-1] == 3:
                            img = img.permute(2, 0, 1)
                        result["image"] = img
                    while not self._stop_event.is_set():
                        try:
                            self._prefetch_queue.put(result, timeout=1)
                            break
                        except _queue_mod.Full:
                            continue
            except Exception:
                continue

    def __iter__(self):
        self._start_prefetching()
        while True:
            try:
                item = self._prefetch_queue.get(timeout=2)
                yield item
            except _queue_mod.Empty:
                if not self._cache_paths and not self._file_paths:
                    return
                continue

    def _encode_text(self, text):
        text = enrich_prompt_text(
            text,
            max_hints=8,
            stochastic=True,
            keep_prob=0.7,
            rng=random,
        )
        if self.text_tokenizer is not None:
            return self.text_tokenizer.encode_padded(
                text, self.max_text_length)
        ids = [ord(c) % 32000 for c in text[:self.max_text_length]]
        mask = [1] * len(ids)
        ids += [0] * (self.max_text_length - len(ids))
        mask += [0] * (self.max_text_length - len(mask))
        return ids, mask


# =====================================================================
# Material Stream
# =====================================================================

class MaterialStream(IterableDataset):
    """Streams material data. If no real data exists, generates synthetic."""

    def __init__(self, data_path: str, text_tokenizer=None,
                 max_text_length: int = 256, max_material_len: int = 512,
                 vocab_size: int = 4096):
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.max_material_len = max_material_len
        self.vocab_size = vocab_size
        self.samples = []
        self._material_encoder = None  # Cached encoder instance

        if Path(data_path).exists():
            with open(data_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))
            logger.info(
                f"MaterialStream: {len(self.samples)} real samples"
            )

        if not self.samples:
            self.samples = self._generate_synthetic_materials()
            logger.info(
                f"MaterialStream: generated {len(self.samples)} "
                f"synthetic materials"
            )

    def _generate_synthetic_materials(self):
        materials = []
        colors = {
            "red": [0.8, 0.1, 0.1], "blue": [0.1, 0.2, 0.8],
            "green": [0.1, 0.7, 0.2], "white": [0.9, 0.9, 0.9],
            "black": [0.05, 0.05, 0.05], "gray": [0.5, 0.5, 0.5],
            "brown": [0.5, 0.3, 0.1], "yellow": [0.9, 0.8, 0.1],
            "orange": [0.9, 0.4, 0.1], "purple": [0.5, 0.1, 0.7],
            "pink": [0.9, 0.4, 0.6], "gold": [0.8, 0.7, 0.2],
            "silver": [0.7, 0.7, 0.75], "copper": [0.7, 0.4, 0.2],
            "teal": [0.1, 0.6, 0.6], "navy": [0.1, 0.1, 0.4],
            "cream": [0.95, 0.9, 0.8], "olive": [0.4, 0.5, 0.1],
            "maroon": [0.5, 0.1, 0.1], "beige": [0.9, 0.85, 0.7],
            "turquoise": [0.2, 0.7, 0.7], "coral": [0.9, 0.4, 0.3],
            "indigo": [0.2, 0.1, 0.5], "lime": [0.5, 0.9, 0.1],
        }
        surfaces = {
            "matte": {"roughness": 0.8, "metallic": 0.0},
            "glossy": {"roughness": 0.2, "metallic": 0.0},
            "metallic": {"roughness": 0.3, "metallic": 1.0},
            "rough": {"roughness": 0.95, "metallic": 0.0},
            "polished": {"roughness": 0.05, "metallic": 0.5},
            "satin": {"roughness": 0.4, "metallic": 0.1},
            "brushed metal": {"roughness": 0.5, "metallic": 0.9},
            "mirror": {"roughness": 0.01, "metallic": 1.0},
            "rubber": {"roughness": 0.9, "metallic": 0.0},
            "ceramic": {"roughness": 0.3, "metallic": 0.0},
            "plastic": {"roughness": 0.4, "metallic": 0.0},
            "wood": {"roughness": 0.7, "metallic": 0.0},
            "glass": {"roughness": 0.05, "metallic": 0.0,
                       "transmission": 0.9},
            "stone": {"roughness": 0.8, "metallic": 0.0},
            "concrete": {"roughness": 0.9, "metallic": 0.0},
            "leather": {"roughness": 0.6, "metallic": 0.0},
            "fabric": {"roughness": 0.85, "metallic": 0.0},
            "velvet": {"roughness": 0.95, "metallic": 0.0},
            "clay": {"roughness": 0.85, "metallic": 0.0},
            "ice": {"roughness": 0.1, "metallic": 0.0,
                     "transmission": 0.5},
            "chrome": {"roughness": 0.05, "metallic": 1.0},
            "rusted metal": {"roughness": 0.9, "metallic": 0.6},
            "frosted glass": {"roughness": 0.4, "metallic": 0.0,
                               "transmission": 0.7},
            "silk": {"roughness": 0.3, "metallic": 0.05},
            "sandstone": {"roughness": 0.85, "metallic": 0.0},
        }

        for color_name, rgb in colors.items():
            for surf_name, props in surfaces.items():
                node_tree = {
                    "nodes": [
                        {"type": "BSDF_PRINCIPLED",
                         "base_color": rgb + [1.0],
                         "roughness": props["roughness"],
                         "metallic": props["metallic"],
                         "transmission": props.get("transmission", 0.0)},
                        {"type": "OUTPUT_MATERIAL"},
                    ],
                    "links": [{"from": 0, "to": 1}],
                }
                text = random.choice([
                    f"{color_name} {surf_name} material",
                    f"a {surf_name} {color_name} surface",
                    f"{color_name} {surf_name}",
                    f"create a {color_name} {surf_name} material",
                    f"{surf_name} finish in {color_name}",
                ])
                materials.append({
                    "text": text, "node_tree": node_tree,
                })

        return materials

    def __iter__(self):
        while True:
            random.shuffle(self.samples)
            for sample in self.samples:
                try:
                    yield self._process(sample)
                except Exception:
                    continue

    def _process(self, sample):
        text = sample.get("text", "material")
        text_ids, text_mask = self._encode_text(text)

        if self._material_encoder is None:
            from models.encoders import MaterialEncoder
            self._material_encoder = MaterialEncoder(
                vocab_size=self.vocab_size)
        tokens = self._material_encoder.encode_material(
            sample.get("node_tree", {}))

        if len(tokens) > self.max_material_len:
            tokens = tokens[:self.max_material_len]
        else:
            tokens += [0] * (self.max_material_len - len(tokens))

        return {
            "task": "materials",
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.float),
            "input_tokens": torch.tensor(tokens[:-1], dtype=torch.long),
            "target_tokens": torch.tensor(tokens[1:], dtype=torch.long),
        }

    def _encode_text(self, text):
        if self.text_tokenizer is not None:
            return self.text_tokenizer.encode_padded(
                text, self.max_text_length)
        ids = [ord(c) % 32000 for c in text[:self.max_text_length]]
        mask = [1] * len(ids)
        ids += [0] * (self.max_text_length - len(ids))
        mask += [0] * (self.max_text_length - len(mask))
        return ids, mask


# =====================================================================
# Modifier Stream
# =====================================================================

class ModifierStream(IterableDataset):
    """Streams modifier data. If no real data, generates synthetic."""

    def __init__(self, data_path: str, text_tokenizer=None,
                 max_text_length: int = 256):
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.samples = []

        from models.encoders import (
            MODIFIER_TYPE_TO_ID, MAX_MODIFIERS, PARAMS_PER_MODIFIER,
        )
        self.type_to_id = MODIFIER_TYPE_TO_ID
        self.max_mods = MAX_MODIFIERS
        self.params_per_mod = PARAMS_PER_MODIFIER

        if Path(data_path).exists():
            with open(data_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))
            logger.info(
                f"ModifierStream: {len(self.samples)} real samples"
            )

        if not self.samples:
            self.samples = self._generate_synthetic_modifiers()
            logger.info(
                f"ModifierStream: generated {len(self.samples)} "
                f"synthetic modifiers"
            )

    def _generate_synthetic_modifiers(self):
        samples = []
        shapes = [
            "cube", "sphere", "cylinder", "cone", "torus",
            "table", "chair", "car body", "house", "vase",
            "bottle", "sword", "tree", "rock", "barrel",
            "lamp", "bench", "sofa", "bed", "bookshelf",
            "arch", "tower", "bridge", "fence", "door",
        ]

        modifier_templates = [
            {"type": "SUBSURF", "levels": 1, "render_levels": 2},
            {"type": "SUBSURF", "levels": 2, "render_levels": 3},
            {"type": "SUBSURF", "levels": 3, "render_levels": 4},
            {"type": "MIRROR", "use_axis_x": True,
             "use_axis_y": False, "use_axis_z": False},
            {"type": "MIRROR", "use_axis_x": True,
             "use_axis_y": True, "use_axis_z": False},
            {"type": "BEVEL", "width": 0.02, "segments": 2},
            {"type": "BEVEL", "width": 0.05, "segments": 3},
            {"type": "BEVEL", "width": 0.01, "segments": 1},
            {"type": "SOLIDIFY", "thickness": 0.02, "offset": -1.0},
            {"type": "SOLIDIFY", "thickness": 0.05, "offset": -1.0},
        ]

        mod_descriptions = {
            "SUBSURF": ["smooth", "subdivided", "rounded", "soft"],
            "MIRROR": ["mirrored", "symmetric", "symmetrical"],
            "BEVEL": ["beveled", "chamfered", "with rounded edges"],
            "SOLIDIFY": ["hollow", "with thickness", "shell"],
        }

        for _ in range(3000):
            shape = random.choice(shapes)
            n_mods = random.randint(1, 3)
            mods = random.sample(
                modifier_templates,
                min(n_mods, len(modifier_templates)),
            )

            descs = []
            for m in mods:
                d = random.choice(
                    mod_descriptions.get(m["type"], ["modified"]))
                descs.append(d)

            desc_str = ", ".join(descs)
            text = random.choice([
                f"a {desc_str} {shape}",
                f"{shape} with {desc_str} applied",
                f"create a {desc_str} {shape}",
                f"{desc_str} {shape} model",
            ])

            mesh_stats = {
                "vertex_count": random.randint(8, 5000),
                "face_count": random.randint(6, 3000),
                "edge_count": random.randint(12, 8000),
                "bbox_width": random.uniform(0.5, 3.0),
                "bbox_depth": random.uniform(0.5, 3.0),
                "bbox_height": random.uniform(0.3, 3.0),
                "surface_area": random.uniform(1.0, 50.0),
                "avg_edge_length": random.uniform(0.01, 0.5),
                "has_ngons": random.random() < 0.1,
                "has_quads_only": random.random() < 0.3,
                "has_tris_only": random.random() < 0.4,
                "is_manifold": random.random() < 0.8,
            }

            samples.append({
                "text": text,
                "mesh_stats": mesh_stats,
                "modifier_stack": mods,
            })

        return samples

    def __iter__(self):
        while True:
            random.shuffle(self.samples)
            for sample in self.samples:
                try:
                    yield self._process(sample)
                except Exception:
                    continue

    def _process(self, sample):
        text = sample.get("text", "")
        text_ids, text_mask = self._encode_text(text)
        stats_vec = self._encode_mesh_stats(
            sample.get("mesh_stats", {}))
        count, type_ids, params = self._encode_modifier_stack(
            sample.get("modifier_stack", []))

        return {
            "task": "modifiers",
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.float),
            "mesh_stats": stats_vec,
            "target_count": torch.tensor(
                max(0, count - 1), dtype=torch.long),
            "target_types": torch.tensor(type_ids, dtype=torch.long),
            "target_params": torch.tensor(params, dtype=torch.float),
        }

    def _encode_text(self, text):
        if self.text_tokenizer is not None:
            return self.text_tokenizer.encode_padded(
                text, self.max_text_length)
        ids = [ord(c) % 32000 for c in text[:self.max_text_length]]
        mask = [1] * len(ids)
        ids += [0] * (self.max_text_length - len(ids))
        mask += [0] * (self.max_text_length - len(mask))
        return ids, mask

    def _encode_mesh_stats(self, stats):
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

    def _encode_modifier_stack(self, modifiers):
        count = min(len(modifiers), self.max_mods)
        type_ids = []
        params = []
        for i in range(self.max_mods):
            if i < len(modifiers):
                mod = modifiers[i]
                tid = self.type_to_id.get(
                    mod.get("type", "NONE"), 0)
                type_ids.append(tid)
                params.append(self._extract_params(mod))
            else:
                type_ids.append(0)
                params.append([0.0] * self.params_per_mod)
        return count, type_ids, params

    def _extract_params(self, mod):
        p = [0.0] * self.params_per_mod
        t = mod.get("type", "")
        if t == "SUBSURF":
            p[0] = float(mod.get("levels", 1)) / 4.0
            p[1] = float(mod.get("render_levels", 2)) / 6.0
        elif t == "MIRROR":
            p[0] = 1.0 if mod.get("use_axis_x", True) else 0.0
            p[1] = 1.0 if mod.get("use_axis_y", False) else 0.0
            p[2] = 1.0 if mod.get("use_axis_z", False) else 0.0
        elif t == "BEVEL":
            p[0] = mod.get("width", 0.02) * 10
            p[1] = float(mod.get("segments", 1)) / 10.0
        elif t == "SOLIDIFY":
            p[0] = mod.get("thickness", 0.01) * 10
            p[1] = mod.get("offset", -1.0)
        return p


# =====================================================================
# Real Contrastive Stream (images from cached real meshes)
# =====================================================================

class RealContrastiveStream(IterableDataset):
    """Image-text pairs from cached real mesh data.

    This is critical for model quality: without seeing rendered images
    of REAL Blender geometry (not just synthetic cubes/spheres), the
    model has no way to learn visual grounding for actual 3D content.

    Reads from the same .pt cache files built by RealMeshStream.
    Items with 'image' keys are yielded as contrastive training pairs.
    """

    def __init__(self, cache_dir, text_tokenizer=None,
                 max_text_length: int = 256, image_size: int = 64,
                 prefetch_size: int = 128):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.image_size = image_size
        self._stop_event = threading.Event()

        # Prefetch queue
        self._prefetch_queue = _queue_mod.Queue(maxsize=prefetch_size)
        self._cache_paths = []
        self._refresh_cache_paths()

        n_with_images = 0
        if self._cache_paths:
            # Quick scan: count how many cache files have images
            sample_paths = random.sample(
                self._cache_paths, min(20, len(self._cache_paths)))
            for p in sample_paths:
                try:
                    items = torch.load(p, weights_only=False)
                    if items and any("image" in it for it in items):
                        n_with_images += 1
                except Exception:
                    pass

        logger.info(f"RealContrastiveStream: {len(self._cache_paths)} cache files, "
                    f"~{n_with_images}/{min(20, len(self._cache_paths))} sampled have images")

        # Start prefetch thread
        t = threading.Thread(target=self._prefetch_worker, daemon=True,
                             name="RealContrastivePrefetch")
        t.start()

    def _refresh_cache_paths(self):
        if self.cache_dir and self.cache_dir.exists():
            paths = sorted(self.cache_dir.glob("*.pt"))
            self._cache_paths = [
                str(p) for p in paths if p.stat().st_size > 200
            ]
            random.shuffle(self._cache_paths)

    def _prefetch_worker(self):
        idx = 0
        while not self._stop_event.is_set():
            if not self._cache_paths:
                self._refresh_cache_paths()
                if not self._cache_paths:
                    time.sleep(5)
                    continue

            if idx >= len(self._cache_paths):
                idx = 0
                random.shuffle(self._cache_paths)
                self._refresh_cache_paths()

            path = self._cache_paths[idx]
            idx += 1

            try:
                items = torch.load(path, weights_only=False)
                if not items:
                    continue
                for item in items:
                    if "image" not in item:
                        continue
                    try:
                        result = self._make_contrastive_item(item)
                        while not self._stop_event.is_set():
                            try:
                                self._prefetch_queue.put(result, timeout=1)
                                break
                            except _queue_mod.Full:
                                continue
                    except Exception:
                        continue
            except Exception:
                continue

    def _make_contrastive_item(self, item):
        """Convert a cached item with image into a contrastive training sample."""
        # Image: stored as (64, 64, 3) uint8 tensor
        img_raw = item["image"]
        if img_raw.dim() == 3 and img_raw.shape[-1] == 3:
            # (H, W, 3) -> (3, H, W) float
            img_t = img_raw.float().permute(2, 0, 1) / 255.0
        elif img_raw.dim() == 3 and img_raw.shape[0] == 3:
            img_t = img_raw.float() / 255.0
        else:
            img_t = img_raw.float().reshape(
                3, self.image_size, self.image_size) / 255.0

        # Text: use cached text or the text_ids from the item
        text_ids = item["text_ids"]
        text_mask = item["text_mask"]

        return {
            "task": "contrastive",
            "text_ids": text_ids,
            "text_mask": text_mask,
            "image": img_t,
        }

    def __iter__(self):
        while True:
            try:
                item = self._prefetch_queue.get(timeout=2)
                yield item
            except _queue_mod.Empty:
                continue


# =====================================================================
# Merged Contrastive Stream (real + synthetic)
# =====================================================================

class _MergedContrastiveStream(IterableDataset):
    """Merges synthetic and real contrastive streams.

    Preferentially samples from real data (actual Blender meshes) because
    that's what teaches the model to recognize real geometry. Falls back
    to synthetic when real data isn't available yet (cache still building).
    """

    def __init__(self, synthetic_stream, real_stream, real_weight=0.6):
        self.synthetic = synthetic_stream
        self.real = real_stream
        self.real_weight = real_weight

    def __iter__(self):
        synth_iter = iter(self.synthetic)
        real_iter = iter(self.real)
        while True:
            # Prefer real data
            if random.random() < self.real_weight:
                try:
                    item = next(real_iter)
                    yield item
                    continue
                except StopIteration:
                    real_iter = iter(self.real)
                except Exception:
                    pass
            # Fallback to synthetic
            try:
                yield next(synth_iter)
            except StopIteration:
                synth_iter = iter(self.synthetic)
            except Exception:
                continue


# =====================================================================
# Contrastive (image-text grounding) Stream
# =====================================================================

class ContrastiveStream(IterableDataset):
    """Infinite image-text pairs for visual grounding.

    Architecture (prefetched):
      - Background render threads generate image-text pairs
      - Rendered pairs are put into a prefetch queue
      - Training thread pulls from queue instantly (no blocking)
      - Matplotlib rendering (slow, ~0.3s/image) happens off-thread

    Sources:
      1. Rendered synthetic shapes (pre-rendered in background)
      2. Stored image-text pairs from JSONL files
      3. Real pre-rendered PNGs from data/renders/ (manifest JSONs)
    """

    def __init__(self, geometry_jsonl: str | None = None,
                 text_tokenizer=None,
                 max_text_length: int = 256, image_size: int = 64,
                 prefetch_size: int = 128, render_threads: int = 3,
                 renders_dir: str | None = None):
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.stored = []
        self._png_pairs: list[tuple[str, str]] = []  # (png_path, label)
        self._stop_event = threading.Event()

        if geometry_jsonl and Path(geometry_jsonl).exists():
            with open(geometry_jsonl) as f:
                for line in f:
                    ex = json.loads(line.strip())
                    if "image" in ex:
                        self.stored.append(ex)
            if self.stored:
                logger.info(
                    f"ContrastiveStream: {len(self.stored)} stored pairs"
                )

        # Load real Blender render PNGs from data/renders/ via manifest files.
        # Each *_manifest.json has {"label": str, "renders": [{"filepath": ...}]}
        # We add one (png_path, label) entry per view per mesh.
        _renders_dir = Path(renders_dir) if renders_dir else None
        if _renders_dir is None:
            _candidate = Path(__file__).resolve().parent.parent / "data" / "renders"
            if _candidate.exists():
                _renders_dir = _candidate
        if _renders_dir is not None and _renders_dir.exists():
            for manifest_path in sorted(_renders_dir.glob("**/*_manifest.json")):
                try:
                    with open(manifest_path) as mf:
                        manifest = json.load(mf)
                    label = manifest.get("label", "").strip()
                    if not label:
                        continue
                    for render_entry in manifest.get("renders", []):
                        png = render_entry.get("filepath", "")
                        if png and Path(png).exists():
                            self._png_pairs.append((png, label))
                except Exception:
                    pass
            if self._png_pairs:
                logger.info(
                    f"ContrastiveStream: {len(self._png_pairs)} real render PNGs "
                    f"from {_renders_dir}"
                )

        self._renderer = None
        try:
            from processing.render_shapes import render_and_encode
            self._renderer = render_and_encode
        except ImportError:
            pass

        # Prefetch queue filled by background render threads
        self._prefetch_queue = _queue_mod.Queue(maxsize=prefetch_size)
        self._render_threads = []
        for i in range(render_threads):
            t = threading.Thread(
                target=self._render_worker, daemon=True,
                name=f"ContrastiveRender-{i}")
            t.start()
            self._render_threads.append(t)
        logger.info(f"ContrastiveStream: {render_threads} render threads, "
                     f"prefetch queue size {prefetch_size}")

    def _render_worker(self):
        """Background thread: renders image-text pairs into the queue.

        Sampling priority (when all sources available):
          40% — real PNG renders from data/renders/ (highest quality)
          30% — stored JSONL pairs
          30% — freshly rendered synthetic shapes
        """
        while not self._stop_event.is_set():
            try:
                r = random.random()
                has_png = bool(self._png_pairs)
                has_stored = bool(self.stored)
                has_renderer = bool(self._renderer)

                if has_png and r < 0.4:
                    item = self._process_png(random.choice(self._png_pairs))
                elif has_stored and (r < 0.7 or not has_renderer):
                    item = self._process_stored(random.choice(self.stored))
                elif has_renderer:
                    item = self._generate_fresh()
                elif has_stored:
                    item = self._process_stored(random.choice(self.stored))
                elif has_png:
                    item = self._process_png(random.choice(self._png_pairs))
                else:
                    time.sleep(0.5)
                    continue

                while not self._stop_event.is_set():
                    try:
                        self._prefetch_queue.put(item, timeout=1)
                        break
                    except _queue_mod.Full:
                        continue
            except Exception:
                continue

    def __iter__(self):
        while True:
            try:
                item = self._prefetch_queue.get(timeout=2)
                yield item
            except _queue_mod.Empty:
                continue

    def _process_png(self, png_pair: tuple[str, str]):
        """Load a real rendered PNG from disk and return a contrastive batch item.

        Args:
            png_pair: (png_path, label) from self._png_pairs

        Returns:
            dict with task, text_ids, text_mask, image keys
        """
        from PIL import Image as _PILImage
        png_path, label = png_pair
        img_pil = _PILImage.open(png_path).convert("RGB")
        img_pil = img_pil.resize(
            (self.image_size, self.image_size), _PILImage.LANCZOS)
        img_t = torch.tensor(
            np.array(img_pil, dtype=np.float32),
            dtype=torch.float32,
        ).permute(2, 0, 1) / 255.0  # (3, H, W) in [0, 1]
        text_ids, text_mask = self._encode_text(label)
        return {
            "task": "contrastive",
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.float),
            "image": img_t,
        }

    def _process_stored(self, ex):
        text_ids, text_mask = self._encode_text(ex["text"])
        img = np.array(ex["image"], dtype=np.uint8).reshape(
            self.image_size, self.image_size, 3)
        img_t = torch.tensor(
            img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return {
            "task": "contrastive",
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.float),
            "image": img_t,
        }

    def _generate_fresh(self):
        from processing.generate_synthetic import (
            SHAPE_SPECS, COMPOSITE_SPECS, generate_label,
            normalize_mesh, apply_rotation,
        )

        all_specs = {**SHAPE_SPECS, **COMPOSITE_SPECS}
        key = random.choice(list(all_specs.keys()))
        spec = all_specs[key]
        params = spec["params"]()
        verts, faces = spec["generator"](params)

        if random.random() < 0.5:
            verts = apply_rotation(
                verts, random.uniform(0, 360),
                random.choice(["x", "y", "z"]),
            )
        verts = normalize_mesh(verts, target_range=(-1.0, 1.0))

        label = generate_label(key, params)
        text_ids, text_mask = self._encode_text(label)

        assert self._renderer is not None, "Renderer not available"
        img_data = self._renderer(verts, faces, size=self.image_size)
        img = np.array(img_data, dtype=np.uint8).reshape(
            self.image_size, self.image_size, 3)
        img_t = torch.tensor(
            img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return {
            "task": "contrastive",
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.float),
            "image": img_t,
        }

    def _encode_text(self, text):
        if self.text_tokenizer is not None:
            return self.text_tokenizer.encode_padded(
                text, self.max_text_length)
        ids = [ord(c) % 32000 for c in text[:self.max_text_length]]
        mask = [1] * len(ids)
        ids += [0] * (self.max_text_length - len(ids))
        mask += [0] * (self.max_text_length - len(mask))
        return ids, mask


# =====================================================================
# Background Data Puller
# =====================================================================

class BackgroundDataPuller:
    """Extraction-first data pipeline that runs alongside training.

    Design principle: PROCESS FIRST, DOWNLOAD SECOND.

    The loop checks how many raw files are waiting to be extracted.
    If the pending queue is above LOW_WATER_MARK, it focuses entirely
    on extraction (Blender headless / mesh_extractor).  Only when the
    queue drops below LOW_WATER_MARK does it download from the next
    source in rotation to refill the hopper.

    Download sources (rotated one-per-cycle when queue is low):
      - Objaverse-XL: Sketchfab, GitHub, Thingiverse, Smithsonian
      - BlendSwap (CC-0/CC-BY)
      - SmutBase / Open3DLab character models
      - GitHub .blend repos
      - Blender official demos

    Lightweight tasks (metadata enrichment, vocab expansion) run
    periodically every N cycles.
    """

    # Objaverse-XL sources to cycle through
    XL_SOURCES = ["sketchfab", "github", "smithsonian", "thingiverse"]

    # When pending files drop below this, download more
    LOW_WATER_MARK = 50
    # Max files to extract per mini-cycle (keeps loop responsive)
    EXTRACT_BATCH = 30

    # All download sources — rotated one at a time
    DOWNLOAD_SOURCES = [
        "objaverse_xl",
        "blendswap",
        "smutbase",
        "open3dlab",
        "github",
        "blender_official",
    ]

    def __init__(self, config: dict, project_root: str):
        self.config = config
        self.root = Path(project_root)
        self._thread = None
        self._stop = threading.Event()
        self._xl_source_idx = 0  # Rotate through XL sources
        self._dl_source_idx = 0  # Rotate through download sources
        self._cycle_count = 0    # Total cycles completed

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True)
        self._thread.start()
        logger.info("Background data puller started (no caps, all sources)")

    def stop(self):
        self._stop.set()

    def _count_pending(self):
        """Count raw 3D files not yet extracted to processed JSON."""
        model_exts = {".glb", ".gltf", ".obj", ".stl",
                      ".ply", ".blend", ".off", ".3ds"}

        # Build set of already-processed stems
        proc_base = self.root / "data" / "processed"
        processed_stems = set()
        if proc_base.exists():
            for d in proc_base.iterdir():
                if d.is_dir():
                    for f in d.rglob("*.json"):
                        processed_stems.add(f.stem)

        # Also exclude stems of files that failed extraction this session
        if hasattr(self, '_failed_extractions'):
            for fp in self._failed_extractions:
                processed_stems.add(Path(fp).stem)

        # Count raw files whose stem is NOT in processed
        raw_dirs = [
            self.root / "data" / "raw" / "objaverse",
            self.root / "data" / "raw" / "blender_official" / "models",
            self.root / "data" / "raw" / "blendswap",
            self.root / "data" / "raw" / "smutbase" / "files",
            self.root / "data" / "raw" / "open3dlab" / "files",
            self.root / "data" / "raw" / "github",
        ]
        pending = 0
        for raw_dir in raw_dirs:
            if not raw_dir.exists():
                continue
            for f in raw_dir.rglob("*"):
                if f.suffix.lower() in model_exts and f.stem not in processed_stems:
                    pending += 1
        return pending

    def _download_next_source(self):
        """Download from the next source in rotation (one source per call)."""
        source = self.DOWNLOAD_SOURCES[
            self._dl_source_idx % len(self.DOWNLOAD_SOURCES)
        ]
        self._dl_source_idx += 1

        dispatch = {
            "objaverse_xl": self._pull_objaverse_xl,
            "blendswap": self._pull_blendswap,
            "smutbase": self._pull_smutbase,
            "open3dlab": self._pull_open3dlab,
            "github": self._pull_github,
            "blender_official": self._pull_blender_official,
        }

        fn = dispatch.get(source)
        if fn is None:
            return

        try:
            logger.info(f"Background: downloading from {source}...")
            fn()
        except Exception as e:
            logger.warning(f"{source} download failed: {e}")

    def _pull_blender_official(self):
        """Re-run the blender_official scraper for any new/missing files."""
        try:
            from scrapers.blender_official import download_blender_official
            from scrapers.utils import ensure_dir
        except ImportError as e:
            logger.warning(f"Cannot import blender_official scraper: {e}")
            return

        out_dir = ensure_dir(self.root / "data" / "raw" / "blender_official")
        download_blender_official(out_dir, max_size_mb=500, crawl=False,
                                  curated_only=True)

    def _run(self):
        """Extraction-first loop: process pending files, download when low."""
        while not self._stop.is_set():
          try:
            self._cycle_count += 1

            # ── Step 1: Count pending files ──
            pending = self._count_pending()
            logger.info(
                f"Background cycle {self._cycle_count}: "
                f"{pending} files pending extraction"
            )

            if self._stop.is_set():
                break

            # ── Step 2: ALWAYS extract first ──
            # If lots pending, do a big batch. If few, do a small batch.
            if pending > 0:
                batch = self.EXTRACT_BATCH if pending <= self.LOW_WATER_MARK else self.EXTRACT_BATCH * 2
                try:
                    self._extract_pending(max_files=batch)
                except Exception as e:
                    logger.warning(f"Extraction failed: {e}")

            if self._stop.is_set():
                break

            # ── Step 3: Download more if queue is low ──
            if pending < self.LOW_WATER_MARK:
                self._download_next_source()

                if self._stop.is_set():
                    break

                # Extract whatever we just downloaded
                try:
                    self._extract_pending(max_files=self.EXTRACT_BATCH)
                except Exception as e:
                    logger.warning(f"Post-download extraction failed: {e}")
            else:
                logger.info(
                    f"Background: {pending} pending > {self.LOW_WATER_MARK} "
                    f"low-water mark, skipping downloads to focus on extraction"
                )

            if self._stop.is_set():
                break

            # ── Step 4: Lightweight housekeeping (every 5 cycles) ──
            if self._cycle_count % 5 == 0:
                for name, fn in [
                    ("metadata enrichment", self._enrich_objaverse_metadata),
                    ("YouTube transcripts", self._pull_youtube_transcripts),
                    ("Wikimedia captions", self._pull_wikimedia),
                    ("terminology", self._pull_terminology),
                    ("vocab expansion", self._expand_vocabulary),
                ]:
                    if self._stop.is_set():
                        break
                    try:
                        fn()
                    except Exception as e:
                        logger.warning(f"{name} failed: {e}")

            # ── Sleep ──
            # Short sleep if lots pending (keep processing), longer if idle
            sleep_time = 10 if pending > self.LOW_WATER_MARK else 60
            logger.info(
                f"Background cycle {self._cycle_count} done, "
                f"sleeping {sleep_time}s..."
            )
            self._stop.wait(sleep_time)

          except KeyboardInterrupt:
            logger.info("Background data puller: interrupted, will retry next cycle")
            if self._stop.is_set():
                break
            continue

    def _pull_objaverse_xl(self):
        """Pull a batch from the current Objaverse-XL source, then rotate."""
        try:
            from scrapers.objaverse_scraper import download_objaverse_batch
        except ImportError as e:
            logger.warning(f"Cannot import objaverse scraper: {e}")
            return

        source = self.XL_SOURCES[self._xl_source_idx % len(self.XL_SOURCES)]
        self._xl_source_idx += 1

        raw_dir = self.root / "data" / "raw" / "objaverse"
        raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Background: pulling Objaverse-XL batch "
            f"(source={source})..."
        )
        try:
            # macOS often emits libmalloc stack-logging warnings when many
            # short-lived Python processes are spawned. Default to 1 process
            # on Darwin for stability; allow override via env var.
            import platform as _platform
            default_procs = 1 if _platform.system() == "Darwin" else 4
            processes = int(os.environ.get("OBJAVERSE_PROCESSES", str(default_procs)))
            count = download_objaverse_batch(
                output_dir=str(raw_dir),
                batch_size=500,
                source=source,
                processes=processes,
            )
            if count > 0:
                logger.info(
                    f"Background: downloaded {count} new "
                    f"{source} objects"
                )
        except (KeyboardInterrupt, SystemExit):
            logger.info(f"Objaverse-XL {source} download interrupted, continuing training")
        except Exception as e:
            logger.warning(f"Objaverse-XL {source} download: {e}")

    # ── Blender headless extraction ────────────────────────────
    BLENDER_EXE = "/Applications/Blender.app/Contents/MacOS/Blender"

    def _extract_blend_file(self, blend_path: Path, output_dir: Path) -> bool:
        """Extract a .blend file via Blender headless → JSON.

        Shells out to Blender in background mode running
        processing/blend_extractor.py which produces a JSON with
        the same schema as mesh_extractor (objects → mesh → vertices/faces).
        Returns True on success.
        """
        if not Path(self.BLENDER_EXE).exists():
            return False

        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"{blend_path.stem}.json"
        if out_file.exists():
            return True

        extractor_script = self.root / "processing" / "blend_extractor.py"
        if not extractor_script.exists():
            logger.warning("blend_extractor.py not found")
            return False

        cmd = [
            str(self.BLENDER_EXE),
            "--background",
            "--factory-startup",
            "--python", str(extractor_script),
            "--",
            "--input", str(blend_path),
            "--output", str(output_dir),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if out_file.exists():
                logger.info(f"Blender extracted: {blend_path.name}")
                return True
            else:
                if result.returncode != 0:
                    logger.debug(
                        f"Blender extraction failed for {blend_path.name}: "
                        f"{result.stderr[-300:] if result.stderr else 'no stderr'}"
                    )
                return False
        except subprocess.TimeoutExpired:
            logger.warning(f"Blender extraction timed out: {blend_path.name}")
            return False
        except Exception as e:
            logger.warning(f"Blender extraction error: {e}")
            return False

    def _enrich_objaverse_metadata(self):
        """Backfill Objaverse Sketchfab metadata with v1 annotations.

        Objaverse-XL metadata only has sha256/license/uid.  Objaverse v1
        has rich text: name, description, tags, categories.  This merges
        them so extraction produces useful text_label for training.
        """
        try:
            from scrapers.objaverse_scraper import enrich_existing_metadata
        except ImportError:
            return

        md_dir = (self.root / "data" / "raw" / "objaverse"
                  / "sketchfab" / "metadata")
        if not md_dir.exists():
            return

        enriched = enrich_existing_metadata(md_dir)
        if enriched > 0:
            logger.info(
                f"Background: enriched {enriched} Objaverse metadata files"
            )

    def _extract_pending(self, max_files=None):
        """Extract raw 3D files to processed JSON, with hash dedup.

        Args:
            max_files: Maximum number of files to process before returning.
                       None = process all pending (original behavior).
        """
        # Track files that failed extraction so we don't retry every cycle
        if not hasattr(self, '_failed_extractions'):
            self._failed_extractions: set[str] = set()
            self._failed_error_types: dict[str, int] = {}

        # (raw_dir, processed_subdir) pairs — route each source
        # to its own processed directory so RealMeshStream finds them.
        raw_dir_map = [
            # Objaverse-XL sources
            (self.root / "data" / "raw" / "objaverse" / "sketchfab" / "models", "objaverse"),
            (self.root / "data" / "raw" / "objaverse" / "github" / "models", "objaverse"),
            (self.root / "data" / "raw" / "objaverse" / "smithsonian" / "models", "objaverse"),
            (self.root / "data" / "raw" / "objaverse" / "thingiverse" / "models", "objaverse"),
            (self.root / "data" / "raw" / "objaverse", "objaverse"),
            # Other scrapers — each gets its own processed dir
            (self.root / "data" / "raw" / "blender_official" / "models", "blender_official"),
            (self.root / "data" / "raw" / "blendswap", "blendswap"),
            (self.root / "data" / "raw" / "smutbase" / "files", "smutbase"),
            (self.root / "data" / "raw" / "open3dlab" / "files", "open3dlab"),
            (self.root / "data" / "raw" / "youtube" / "blend_files", "objaverse"),
            (self.root / "data" / "raw" / "github", "github"),
        ]

        # Build set of already-processed stems across ALL processed dirs
        proc_base = self.root / "data" / "processed"
        proc_base.mkdir(parents=True, exist_ok=True)
        processed_names = set()
        for d in proc_base.iterdir():
            if d.is_dir():
                for f in d.rglob("*.json"):
                    processed_names.add(f.stem)

        # Load global hash registry for dedup
        try:
            from scrapers.utils import GlobalHashRegistry
            hash_reg = GlobalHashRegistry(
                self.root / "data" / "raw"
            )
        except ImportError:
            hash_reg = None

        extracted = 0
        # Processable extensions
        model_exts = (
            "*.glb", "*.gltf", "*.obj", "*.stl",
            "*.ply", "*.blend", "*.off", "*.3ds",
        )

        for raw_dir, proc_subdir in raw_dir_map:
            if not raw_dir.exists():
                continue

            proc_dir = proc_base / proc_subdir
            proc_dir.mkdir(parents=True, exist_ok=True)

            for ext in model_exts:
                files = list(raw_dir.rglob(ext))
                for model_file in files:
                    if model_file.name.startswith("._"):
                        continue
                    if model_file.stem in processed_names:
                        continue
                    if self._stop.is_set() or (max_files is not None and extracted >= max_files):
                        if hash_reg:
                            hash_reg.save()
                        if extracted > 0:
                            logger.info(f"Background: extracted {extracted} new models (batch limit)")
                        return extracted

                    # Hash dedup: skip if we've seen this exact file
                    file_key = str(model_file)
                    if hash_reg and not hash_reg.is_new(model_file):
                        processed_names.add(model_file.stem)
                        self._failed_extractions.add(file_key)
                        continue

                    # Skip files that already failed extraction
                    if file_key in self._failed_extractions:
                        continue

                    try:
                        if model_file.suffix.lower() == ".blend":
                            ok = self._extract_blend_file(
                                model_file, proc_dir
                            )
                            if ok:
                                processed_names.add(model_file.stem)
                                extracted += 1
                                if hash_reg:
                                    hash_reg.add_file(model_file)
                            else:
                                # Track failed .blend extraction
                                self._failed_extractions.add(file_key)
                                processed_names.add(model_file.stem)
                        else:
                            from processing.mesh_extractor import (
                                extract_from_file,
                            )
                            meta_dir = raw_dir.parent / "metadata"
                            if not meta_dir.exists():
                                meta_dir = raw_dir

                            result = extract_from_file(
                                str(model_file),
                                metadata_dir=str(meta_dir),
                            )
                            if not result:
                                # No mesh data extracted (too large, no
                                # geometry, unsupported format, etc.)
                                # Mark as failed so we stop retrying.
                                self._failed_extractions.add(file_key)
                                processed_names.add(model_file.stem)
                                if hash_reg:
                                    hash_reg.add_file(model_file)
                                continue

                            # Quality filter check before saving
                            from processing.quality_filter import QualityFilter
                            qf = QualityFilter(self.config)
                            objects = result.get("objects", [])
                            good_objects = []
                            for obj in objects:
                                mesh = obj.get("mesh")
                                if not mesh:
                                    continue
                                passed, reason = qf.check_mesh(mesh)
                                if passed:
                                    good_objects.append(obj)
                            if not good_objects:
                                processed_names.add(model_file.stem)
                                self._failed_extractions.add(file_key)
                                if hash_reg:
                                    hash_reg.add_file(model_file)
                                continue
                            result["objects"] = good_objects

                            out = proc_dir / f"{model_file.stem}.json"
                            with open(out, "w") as f:
                                json.dump(result, f)
                            processed_names.add(model_file.stem)
                            extracted += 1
                            if hash_reg:
                                hash_reg.add_file(model_file)
                    except Exception as e:
                        self._failed_extractions.add(file_key)
                        err_type = type(e).__name__
                        self._failed_error_types[err_type] = (
                            self._failed_error_types.get(err_type, 0) + 1
                        )
                        # Log first occurrence of each error type at WARNING,
                        # then DEBUG for subsequent ones
                        if self._failed_error_types[err_type] == 1:
                            logger.warning(
                                f"Extraction failed for {model_file.name} "
                                f"({err_type}): {e}"
                            )
                        elif self._failed_error_types[err_type] <= 5:
                            logger.debug(
                                f"Extraction failed for {model_file.name} "
                                f"({err_type}): {e}"
                            )
                        continue

        if hash_reg:
            hash_reg.save()

        if extracted > 0:
            logger.info(
                f"Background: extracted {extracted} new models"
            )
        if self._failed_error_types:
            summary = ", ".join(
                f"{k}={v}" for k, v in self._failed_error_types.items()
            )
            logger.info(
                f"Background: {len(self._failed_extractions)} files "
                f"failed extraction ({summary})"
            )
        return extracted

    def _pull_youtube_transcripts(self):
        """Pull YouTube transcripts and blend files (no cap)."""
        try:
            from scrapers.youtube_scraper import (
                search_youtube_videos,
                process_video,
            )
        except ImportError as e:
            logger.warning(f"Cannot import YouTube scraper: {e}")
            return

        yt_dir = self.root / "data" / "raw" / "youtube"
        yt_dir.mkdir(parents=True, exist_ok=True)

        # Load progress to skip already-processed videos
        progress_file = yt_dir / ".progress"
        progress = set()
        if progress_file.exists():
            progress = set(progress_file.read_text().splitlines())

        queries = [
            "blender modeling tutorial free blend file",
            "blender tutorial project file download",
            "blender hard surface modeling tutorial",
        ]

        try:
            total = 0
            for query in queries:
                if self._stop.is_set():
                    break
                videos = search_youtube_videos(query, max_results=50)
                for video in videos:
                    if self._stop.is_set():
                        break
                    vid = video.get("id", "")
                    if vid in progress:
                        continue
                    result = process_video(video, yt_dir)
                    if result.get("transcript_saved") or result.get("blend_files"):
                        total += 1
                    progress.add(vid)
                    with open(progress_file, "a") as f:
                        f.write(vid + "\n")
            if total > 0:
                logger.info(f"Background: processed {total} YouTube videos")
        except Exception as e:
            logger.warning(f"YouTube scrape: {e}")

    def _pull_smutbase(self):
        """Pull a batch of character models from SmutBase (no cap)."""
        try:
            from scrapers.smutbase_scraper import scrape_batch
        except ImportError as e:
            logger.warning(f"Cannot import SmutBase scraper: {e}")
            return

        raw_dir = self.root / "data" / "raw" / "smutbase"
        raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Background: pulling SmutBase models...")
        try:
            count = scrape_batch(
                site_key="smutbase",
                output_dir=str(raw_dir),
                batch_size=50,
            )
            if count > 0:
                logger.info(
                    f"Background: downloaded {count} SmutBase models"
                )
            else:
                logger.info(
                    "SmutBase: 0 downloads (site likely requires login)"
                )
        except Exception as e:
            logger.warning(f"SmutBase scrape: {e}")

    def _pull_open3dlab(self):
        """Pull a batch of models from Open3DLab (no cap)."""
        try:
            from scrapers.smutbase_scraper import scrape_batch
        except ImportError as e:
            logger.warning(f"Cannot import Open3DLab scraper: {e}")
            return

        raw_dir = self.root / "data" / "raw" / "open3dlab"
        raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Background: pulling Open3DLab models...")
        try:
            count = scrape_batch(
                site_key="open3dlab",
                output_dir=str(raw_dir),
                batch_size=50,
            )
            if count > 0:
                logger.info(
                    f"Background: downloaded {count} Open3DLab models"
                )
            else:
                logger.info(
                    "Open3DLab: 0 downloads (site likely requires login)"
                )
        except Exception as e:
            logger.warning(f"Open3DLab scrape: {e}")

    def _pull_blendswap(self):
        """Pull popular models from BlendSwap (CC-0/CC-BY licensed).

        Downloads most-popular models first using /blends/{page}/mostDownloads.
        BlendSwap has a monthly download quota (~20 downloads/month on free),
        so popularity-first maximizes training data quality per download.

        Requires BLENDSWAP_EMAIL and BLENDSWAP_PASSWORD in .env.
        """
        try:
            from scrapers.blendswap_scraper import (
                get_blend_detail, create_session, BASE_URL,
            )
            from scrapers.utils import (
                ensure_dir, load_progress, save_progress, download_file,
                save_metadata,
            )
            from scrapers.quality_filter import passes_quality_filter, MAX_FILE_SIZE_MB
        except ImportError as e:
            logger.warning(f"Cannot import BlendSwap scraper: {e}")
            return

        raw_dir = self.root / "data" / "raw" / "blendswap"
        raw_dir.mkdir(parents=True, exist_ok=True)

        progress_file = raw_dir / ".progress"
        progress = load_progress(progress_file)

        session = create_session()

        # ── Pre-flight: check download quota ──
        # BlendSwap has a monthly download limit. If exhausted, the download
        # page returns HTML with title "No Downloads Remaining".
        # Use a known popular blend ID for the check (ID 1 is 404).
        try:
            test_url = f"{BASE_URL}/blend/18735/download"
            test_resp = session.get(test_url, timeout=15)
            if "no downloads remaining" in test_resp.text.lower():
                logger.warning(
                    "BlendSwap: download quota exhausted — "
                    "skipping this cycle (resets monthly)"
                )
                return
        except Exception:
            pass

        logger.info("Background: pulling BlendSwap (most-popular first)...")

        import re as _re
        from bs4 import BeautifulSoup as _BS

        downloaded = 0
        skipped = 0
        consecutive_fails = 0
        max_consecutive_fails = 5
        target = 20
        session_dead = False

        # ── Browse /blends/{page}/mostDownloads (all categories, sorted) ──
        for page in range(1, 200):
            if downloaded >= target or self._stop.is_set() or session_dead:
                break

            url = f"{BASE_URL}/blends/{page}/mostDownloads"
            try:
                resp = session.get(url, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                logger.warning(f"BlendSwap page {page} fetch failed: {e}")
                break

            soup = _BS(resp.text, "html.parser")

            # Parse blend IDs and licenses from the listing page
            seen_on_page = set()
            listings = []
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                m = _re.search(r"/blend/(\d+)", href)
                if not m:
                    continue
                bid = m.group(1)
                if bid in seen_on_page:
                    continue
                seen_on_page.add(bid)

                # Find license text near this listing
                lic = ""
                parent = link.parent
                while parent and parent.name not in ("body", None):
                    txt = parent.get_text()
                    for tag in ["CC-0", "CC-BY-NC", "CC-BY-SA", "CC-BY", "GAL"]:
                        if tag in txt:
                            lic = tag
                            break
                    if lic:
                        break
                    parent = parent.parent

                blend_url = href if href.startswith("http") else BASE_URL + href
                listings.append({"id": bid, "url": blend_url, "license": lic})

            if not listings:
                break

            for listing in listings:
                if downloaded >= target or self._stop.is_set() or session_dead:
                    break

                bid = listing["id"]
                if bid in progress:
                    continue

                # License filter — skip restrictive licenses
                lic = listing.get("license", "")
                if not any(tag in lic for tag in ["CC-0", "CC-BY", "GAL"]):
                    save_progress(progress_file, bid)
                    progress.add(bid)
                    continue

                # Get detail page for download URL + metadata
                detail = get_blend_detail(listing["url"], session)
                if not detail or not detail.get("download_url"):
                    save_progress(progress_file, bid)
                    progress.add(bid)
                    continue

                # ── Quality gate ──
                dl_count = (detail.get("stats") or {}).get("downloads")
                # Use first tag as title (often the model name on BlendSwap)
                tags = detail.get("tags") or []
                title = tags[0] if tags else detail.get("description", "")[:80] or f"blend_{bid}"
                passed, reason = passes_quality_filter(
                    title=title,
                    description=detail.get("description", ""),
                    tags=detail.get("tags", []),
                    downloads=dl_count,
                    likes=(detail.get("stats") or {}).get("likes"),
                    category="popular",
                )
                if not passed:
                    logger.info(
                        f"BlendSwap skip '{title[:40]}': {reason}")
                    save_progress(progress_file, bid)
                    progress.add(bid)
                    skipped += 1
                    continue

                # Download into a flat "popular" directory
                pop_dir = ensure_dir(raw_dir / "popular")
                out_path = pop_dir / f"{bid}.blend"
                success = download_file(
                    detail["download_url"], out_path,
                    max_size_mb=MAX_FILE_SIZE_MB,
                    session=session,
                )
                if success:
                    save_metadata(str(pop_dir), bid, {
                        **listing, **detail, "source": "blendswap",
                    })
                    downloaded += 1
                    consecutive_fails = 0
                    logger.info(
                        f"BlendSwap #{downloaded}: blend {bid} "
                        f"({dl_count or '?'} downloads)")
                else:
                    consecutive_fails += 1
                    if consecutive_fails >= max_consecutive_fails:
                        logger.warning(
                            f"BlendSwap: {consecutive_fails} "
                            f"consecutive download failures — "
                            f"download quota likely exhausted, stopping")
                        session_dead = True

                save_progress(progress_file, bid)
                progress.add(bid)
                time.sleep(2.5)

        if downloaded > 0 or skipped > 0:
            logger.info(
                f"Background: downloaded {downloaded} BlendSwap models "
                f"(skipped {skipped} by quality filter)")
        elif session_dead:
            pass
        else:
            logger.info("BlendSwap: 0 new downloads this cycle")

    def _pull_github(self):
        """Pull .blend files from GitHub repositories."""
        try:
            from scrapers.github_scraper import (
                search_blend_files, get_repo_info, get_repo_blend_files,
                download_blend_from_repo, ALLOWED_LICENSES,
            )
            from scrapers.utils import (
                ensure_dir, load_progress, save_progress, save_metadata,
            )
        except ImportError as e:
            logger.warning(f"Cannot import GitHub scraper: {e}")
            return

        raw_dir = self.root / "data" / "raw" / "github"
        raw_dir.mkdir(parents=True, exist_ok=True)

        progress_file = raw_dir / ".progress"
        progress = load_progress(progress_file)

        token = os.environ.get("GITHUB_TOKEN", "").strip()

        import requests as _requests
        session = _requests.Session()
        if token:
            session.headers["Authorization"] = f"token {token}"
        session.headers["Accept"] = "application/vnd.github.v3+json"
        session.headers["User-Agent"] = "BlenderModelTraining/0.1"

        logger.info("Background: pulling GitHub .blend files...")

        queries = [
            "extension:blend blender model",
            "extension:blend 3d",
            "extension:blend character",
            "extension:blend vehicle",
        ]

        downloaded = 0
        target = 20

        import random as _random
        query = _random.choice(queries)
        page = _random.randint(1, 10)

        try:
            items = search_blend_files(session, query, page=page)
            if not items:
                logger.info("GitHub: no results this cycle")
                return

            repos_seen = set()
            for item in items:
                if downloaded >= target or self._stop.is_set():
                    break

                repo_full = item.get("repository", {}).get("full_name", "")
                if not repo_full or repo_full in repos_seen:
                    continue
                repos_seen.add(repo_full)

                if repo_full in progress:
                    continue

                owner, repo = repo_full.split("/", 1)
                repo_info = get_repo_info(session, owner, repo)
                if not repo_info:
                    save_progress(progress_file, repo_full)
                    progress.add(repo_full)
                    continue

                # Skip huge repos (size is in KB) — tree API chokes on them
                repo_size_kb = repo_info.get("size", 0)
                if repo_size_kb > 500_000:  # >500 MB
                    logger.debug(f"GitHub: skipping {repo_full} ({repo_size_kb/1000:.0f}MB)")
                    save_progress(progress_file, repo_full)
                    progress.add(repo_full)
                    continue

                lic = (repo_info.get("license") or {}).get("spdx_id", "").lower()
                if lic not in ALLOWED_LICENSES and lic != "noassertion":
                    save_progress(progress_file, repo_full)
                    progress.add(repo_full)
                    continue

                branch = repo_info.get("default_branch", "main")
                blend_files = get_repo_blend_files(session, owner, repo, branch)
                if not blend_files:
                    save_progress(progress_file, repo_full)
                    progress.add(repo_full)
                    continue

                repo_dir = ensure_dir(raw_dir / owner / repo)
                for bf in blend_files[:5]:
                    if downloaded >= target:
                        break
                    fpath = bf["path"]
                    safe = fpath.replace("/", "__")
                    out = repo_dir / safe
                    if bf.get("size", 0) > 200 * 1024 * 1024:
                        continue
                    success = download_blend_from_repo(
                        session, owner, repo, fpath, branch, out, 200,
                    )
                    if success:
                        save_metadata(str(repo_dir), safe, {
                            "source": "github", "repo": repo_full,
                            "file_path": fpath, "license": lic,
                            "description": repo_info.get("description", ""),
                        })
                        downloaded += 1

                save_progress(progress_file, repo_full)
                progress.add(repo_full)
                time.sleep(2)

        except Exception as e:
            logger.warning(f"GitHub scrape: {e}")

        if downloaded > 0:
            logger.info(f"Background: downloaded {downloaded} GitHub .blend files")
        else:
            logger.info("GitHub: 0 new downloads this cycle")

    def _pull_wikimedia(self):
        """Pull image captions from Wikimedia Commons for vocab expansion."""
        try:
            from scrapers.wikimedia_scraper import scrape_batch
        except ImportError as e:
            logger.warning(f"Cannot import Wikimedia scraper: {e}")
            return

        raw_dir = self.root / "data" / "raw" / "wikimedia"
        raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Background: pulling Wikimedia Commons captions...")
        try:
            count = scrape_batch(output_dir=str(raw_dir), batch_size=100)
            if count > 0:
                logger.info(
                    f"Background: scraped {count} Wikimedia captions"
                )
        except Exception as e:
            logger.warning(f"Wikimedia scrape: {e}")

    def _pull_terminology(self):
        """Pull 3D/CG terminology from Wikipedia & Polycount."""
        try:
            from scrapers.terminology_scraper import scrape_batch
        except ImportError as e:
            logger.warning(f"Cannot import terminology scraper: {e}")
            return

        raw_dir = self.root / "data" / "raw" / "terminology"
        raw_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Background: pulling 3D terminology...")
        try:
            count = scrape_batch(output_dir=str(raw_dir))
            if count > 0:
                logger.info(
                    f"Background: scraped {count} terminology entries"
                )
        except Exception as e:
            logger.warning(f"Terminology scrape: {e}")

    def _expand_vocabulary(self):
        """Expand text tokenizer vocabulary from all scraped text.

        With BPE tokenizer, this is a no-op — BPE handles any word
        by splitting into known subword pieces. Only needed for the
        legacy word-level tokenizer.
        """
        # Check if we're using BPE — if so, skip
        bpe_dir = (
            self.root / "data" / "datasets" / "geometry" / "bpe_tokenizer"
        )
        if bpe_dir.exists() and (bpe_dir / "tokenizer.model").exists():
            logger.info("BPE tokenizer in use — vocabulary expansion skipped")
            return

        from processing.text_tokenizer import TextTokenizer, _tokenize_text

        tok_path = (
            self.root / "data" / "datasets" / "geometry"
            / "text_tokenizer.json"
        )
        if not tok_path.exists():
            return

        tokenizer = TextTokenizer.load(tok_path)
        old_size = tokenizer.vocab_size

        new_words: set[str] = set()

        # 1. Wikimedia captions and terminology
        text_files = [
            self.root / "data" / "raw" / "wikimedia" / "captions.jsonl",
            self.root / "data" / "raw" / "terminology" / "terms.jsonl",
        ]
        for text_file in text_files:
            if not text_file.exists():
                continue
            try:
                with open(text_file) as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        for field in ("title", "description",
                                      "term", "definition",
                                      "categories"):
                            text = entry.get(field, "")
                            if text:
                                tokens = _tokenize_text(text)
                                new_words.update(tokens)
            except Exception:
                continue

        # 2. Real model text labels from processed JSONs
        proc_base = self.root / "data" / "processed"
        if proc_base.exists():
            for subdir in proc_base.iterdir():
                if not subdir.is_dir():
                    continue
                for jf in subdir.glob("*.json"):
                    try:
                        with open(jf) as f:
                            data = json.load(f)
                        for obj in data.get("objects", [data]):
                            for field in ("text_label", "name",
                                          "resolved_name"):
                                text = obj.get(field, "")
                                if text:
                                    new_words.update(
                                        _tokenize_text(text))
                        meta = data.get("metadata", {})
                        for field in ("name", "description"):
                            text = meta.get(field, "")
                            if text:
                                new_words.update(
                                    _tokenize_text(text))
                        for tag in meta.get("tags", []):
                            t = tag if isinstance(tag, str) else ""
                            if t:
                                new_words.update(_tokenize_text(t))
                    except Exception:
                        continue

        # 3. Objaverse metadata files (name, description, tags)
        meta_dirs = [
            self.root / "data" / "raw" / "objaverse" / "sketchfab" / "metadata",
        ]
        for md in meta_dirs:
            if not md.exists():
                continue
            for mf in md.glob("*.meta.json"):
                try:
                    with open(mf) as f:
                        meta = json.load(f)
                    for field in ("name", "description"):
                        text = meta.get(field, "")
                        if text:
                            new_words.update(_tokenize_text(text))
                    for tag in meta.get("tags", []):
                        t = tag if isinstance(tag, str) else ""
                        if t:
                            new_words.update(_tokenize_text(t))
                except Exception:
                    continue

        # Add new words not already in vocab
        added = 0
        max_vocab = 8000
        idx = tokenizer.vocab_size
        for word in sorted(new_words):
            if idx >= max_vocab:
                break
            if word not in tokenizer.vocab:
                tokenizer.vocab[word] = idx
                tokenizer.id_to_token[idx] = word
                idx += 1
                added += 1

        if added > 0:
            tokenizer.vocab_size = len(tokenizer.vocab)
            tokenizer.save(tok_path)
            logger.info(
                f"Vocabulary expanded: {old_size} -> "
                f"{tokenizer.vocab_size} words (+{added} new)"
            )


# =====================================================================
# Collation
# =====================================================================

def collate_geometry(batch):
    """Dynamic padding for geometry batches."""
    text_ids = [b["text_ids"] for b in batch]
    text_masks = [b["text_mask"] for b in batch]
    text_lens = [int(m.sum().item()) for m in text_masks]
    max_text = max(max(text_lens), 1)

    text_ids = torch.stack([t[:max_text] for t in text_ids])
    text_masks = torch.stack([m[:max_text] for m in text_masks])

    mesh_lists = [b["mesh_tokens"] for b in batch]
    max_mesh = max(len(m) for m in mesh_lists)
    # Round up to face-aligned boundary (multiple of 9)
    max_mesh = ((max_mesh + 8) // 9) * 9
    padded = torch.zeros(len(batch), max_mesh, dtype=torch.long)
    for i, m in enumerate(mesh_lists):
        padded[i, :len(m)] = m

    result = {
        "task": "geometry",
        "text_ids": text_ids,
        "text_mask": text_masks,
        "mesh_tokens": padded,
    }

    # Quality weights for sample weighting (from RealMeshStream)
    if "quality_weight" in batch[0]:
        result["quality_weight"] = torch.stack(
            [b.get("quality_weight", torch.tensor(1.0)) for b in batch]
        )

    if "fill_rate" in batch[0]:
        result["fill_rate"] = torch.stack(
            [b.get("fill_rate", torch.tensor(0.5)) for b in batch]
        )

    if "label_confidence" in batch[0]:
        result["label_confidence"] = torch.stack(
            [b.get("label_confidence", torch.tensor(1.0)) for b in batch]
        )

    if "scene_complexity_score" in batch[0]:
        result["scene_complexity_score"] = torch.stack(
            [b.get("scene_complexity_score", torch.tensor(0.3)) for b in batch]
        )

    if "composition" in batch[0]:
        result["composition"] = [b.get("composition", {}) for b in batch]

    if "workflow_supervision" in batch[0]:
        result["workflow_supervision"] = [b.get("workflow_supervision", {}) for b in batch]

    if "image" in batch[0]:
        images = []
        for b in batch:
            if "image" in b:
                images.append(b["image"])
            else:
                images.append(torch.zeros(3, 64, 64))
        result["image"] = torch.stack(images)

    return result


def collate_generic(batch):
    """Stack all tensors in a batch dict."""
    result = {"task": batch[0]["task"]}
    for key in batch[0]:
        if key == "task":
            continue
        result[key] = torch.stack([b[key] for b in batch])
    return result


# =====================================================================
# Multi-task infinite sampler
# =====================================================================

class InfiniteMultiTaskSampler:
    """Samples from multiple infinite streams, weighted."""

    def __init__(self, loaders, weights=None):
        self.loaders = loaders
        self.weights = weights or {}
        self.iterators = {}

        self.tasks = []
        for name in loaders:
            w = int(self.weights.get(name, 1.0) * 10)
            if w <= 0:
                continue
            self.tasks.extend([name] * max(w, 1))
        from collections import Counter
        logger.info(f"Sampler task distribution: {dict(Counter(self.tasks))}")

    def __iter__(self):
        self.iterators = {
            k: iter(v) for k, v in self.loaders.items()
        }
        step = 0
        _real_attempts = 0

        while True:
            task = self.tasks[step % len(self.tasks)]
            step += 1

            if task not in self.iterators:
                continue

            if task == "real_geometry":
                _real_attempts += 1
                if _real_attempts <= 3 or _real_attempts % 100 == 0:
                    logger.info(f"Sampler: attempting real_geometry fetch #{_real_attempts}")

            try:
                batch = next(self.iterators[task])
            except StopIteration:
                self.iterators[task] = iter(self.loaders[task])
                try:
                    batch = next(self.iterators[task])
                except StopIteration:
                    if task == "real_geometry":
                        logger.warning(f"real_geometry StopIteration (attempt #{_real_attempts})")
                    continue

            yield task, batch


# =====================================================================
# Loss computation
# =====================================================================

def compute_modifier_loss(outputs, batch, device):
    from models.unified import MAX_MODIFIERS

    count_loss = F.cross_entropy(
        outputs["count_logits"],
        batch["target_count"].to(device),
    )

    type_loss = 0
    for i in range(MAX_MODIFIERS):
        type_loss += F.cross_entropy(
            outputs["type_logits"][i],
            batch["target_types"][:, i].to(device),
        )
    type_loss /= MAX_MODIFIERS

    param_loss = 0
    n_valid = 0
    target_types = batch["target_types"].to(device)
    target_params = batch["target_params"].to(device)

    for i in range(MAX_MODIFIERS):
        mask = (target_types[:, i] > 0).float()
        if mask.sum() == 0:
            continue
        pred_all = outputs["param_values"][i]
        tgt_ids = target_types[:, i]
        pred_params = pred_all[
            torch.arange(pred_all.shape[0], device=device),
            tgt_ids,
        ]
        tgt_p = target_params[:, i, :]
        slot_loss = F.mse_loss(
            pred_params, tgt_p, reduction="none")
        slot_loss = (
            (slot_loss.mean(dim=-1) * mask).sum() / mask.sum()
        )
        param_loss += slot_loss
        n_valid += 1

    if n_valid > 0:
        param_loss /= n_valid

    return count_loss + type_loss + 0.5 * param_loss


# =====================================================================
# Checkpoint management
# =====================================================================

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the most recent checkpoint."""
    cp_dir = Path(checkpoint_dir)
    if not cp_dir.exists():
        return None

    latest = cp_dir / "latest.pt"
    if latest.exists():
        return str(latest)

    best = cp_dir / "best.pt"
    if best.exists():
        return str(best)

    step_files = sorted(
        cp_dir.glob("step_*.pt"),
        key=lambda f: int(f.stem.split("_")[1]),
        reverse=True,
    )
    if step_files:
        return str(step_files[0])

    return None


def _get_free_disk_gb(path) -> float:
    """Return free disk space in GB for the filesystem containing `path`."""
    try:
        usage = shutil.disk_usage(Path(path).parent)
        return usage.free / (1024 ** 3)
    except Exception:
        return float('inf')


def cleanup_old_checkpoints(checkpoint_dir, keep=5):
    """Delete old step_*.pt files, keeping only the `keep` most recent."""
    cp_dir = Path(checkpoint_dir)
    step_files = sorted(
        cp_dir.glob("step_*.pt"),
        key=lambda f: int(f.stem.split("_")[1]),
        reverse=True,
    )
    to_delete = step_files[keep:]
    for f in to_delete:
        try:
            f.unlink()
            logger.info(f"  Pruned old checkpoint: {f.name}")
        except OSError as e:
            logger.warning(f"  Failed to prune {f.name}: {e}")
    if to_delete:
        logger.info(f"  Cleaned up {len(to_delete)} old checkpoints, "
                     f"kept {min(keep, len(step_files))} most recent")


def save_checkpoint(model, optimizer, scheduler, step, loss, path,
                    config=None, best_val_loss=None,
                    grad_accum: Optional[int] = None,
                    min_free_gb=3.0):
    """Atomic save with tmp file. Skips save if disk space < min_free_gb."""
    free_gb = _get_free_disk_gb(path)
    if free_gb < min_free_gb:
        logger.warning(
            f"  SKIPPING save ({path}) - only {free_gb:.1f} GB free "
            f"(need {min_free_gb:.0f} GB). Run cleanup or free disk space."
        )
        return False

    model_to_save = model
    if hasattr(model, "_orig_mod"):
        model_to_save = model._orig_mod

    data = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "loss": loss,
        "model_type": "unified",
        "timestamp": time.time(),
    }
    if grad_accum is not None:
        ga = int(max(1, grad_accum))
        data["grad_accum"] = ga
        data["optimizer_step"] = int(step) // ga
    if config is not None:
        data["config"] = config
    if best_val_loss is not None:
        data["best_val_loss"] = best_val_loss

    tmp_path = str(path) + ".tmp"
    try:
        torch.save(data, tmp_path)
        os.replace(tmp_path, str(path))
        logger.info(f"  Saved: {path}")
        return True
    except (RuntimeError, OSError) as e:
        logger.error(f"  FAILED to save {path}: {e}")
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return False


# =====================================================================
# Main training loop
# =====================================================================


def _oom_retry_per_sample(
    model, batch, task_name, B, geo_criterion,
    loss_weights, grad_accum, use_scaler, scaler,
    amp_enabled, amp_dtype, device, global_step,
):
    """Retry a geometry batch sample-by-sample after OOM.

    When a batched forward/backward OOMs, this function processes each
    sample individually with trimmed padding.  Detailed meshes with many
    faces are *valuable* training data — skipping them wastes the most
    informative examples.

    Returns the average loss across recovered samples, or None if all
    samples failed.
    """
    from torch.amp import autocast  # noqa: F811
    import gc as _gc

    seq_len = batch.get("mesh_tokens", torch.empty(0, 0)).shape[-1]
    logger.info(
        f"Step {global_step} OOM on batch of {B} "
        f"({task_name}, seq_len={seq_len}), retrying per-sample...")

    # Thorough memory cleanup before retrying
    _gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    success = 0
    total_loss = 0.0
    consecutive_fails = 0
    lw = loss_weights.get(task_name, loss_weights.get("geometry", 1.0))

    for si in range(B):
        # Fail-fast: if 3+ consecutive samples fail, CUDA is likely
        # in a corrupted state after OOM — stop wasting time.
        if consecutive_fails >= 3:
            logger.warning(
                f"  {consecutive_fails} consecutive failures — "
                f"CUDA likely corrupted after OOM, aborting retry "
                f"(recovered {success}/{si} so far)")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Extract single sample and trim padding zeros
        mt = batch["mesh_tokens"][si:si + 1]
        nonzero = (mt[0] != 0).nonzero(as_tuple=True)[0]
        if len(nonzero) == 0:
            continue
        alen = int(nonzero[-1].item()) + 1
        if alen <= 1:
            continue
        mt = mt[:, :alen]

        ti = batch["text_ids"][si:si + 1]
        tm = batch["text_mask"][si:si + 1]
        tlen = max(1, int(tm.sum().item()))
        ti = ti[:, :tlen]
        tm = tm[:, :tlen]

        try:
            with autocast(
                device_type=device.type, dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                inp = mt[:, :-1]
                tgt = mt[:, 1:]
                logits = model.forward_geometry(ti, tm, inp)
                loss = geo_criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                )
                if ("quality_weight" in batch
                        and task_name == "real_geometry"):
                    qw = batch["quality_weight"][si].item()
                    loss = loss * qw
                # Each sample contributes 1/B of original batch gradient
                wl = loss * lw / (grad_accum * B)

            if use_scaler:
                scaler.scale(wl).backward()
            else:
                wl.backward()

            success += 1
            consecutive_fails = 0
            total_loss += loss.item()
            del logits, loss, wl, inp, tgt

        except Exception as e2:
            e2_msg = str(e2).lower()
            if ("out of memory" in e2_msg
                    or "cublas" in e2_msg
                    or "alloc_failed" in e2_msg):
                consecutive_fails += 1
                logger.warning(
                    f"  Sample {si} OOM even alone "
                    f"(seq_len={alen}), skipping")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()
            else:
                logger.warning(f"  Sample {si} error: {e2}")

    if success > 0:
        avg = total_loss / success
        logger.info(
            f"  Recovered {success}/{B} samples (avg_loss={avg:.4f})")
        return avg
    return None


def train(config: dict, args):
    """Autonomous training. Runs until Ctrl+C.

    - Infinite synthetic data on-the-fly
    - Real mesh data from all processed sources
    - Auto-resumes from latest checkpoint
    - Saves continuously (latest.pt always current)
    - Background thread pulls new data
    """
    global _STOP_TRAINING
    _STOP_TRAINING = False

    # ── Quality gate: validate training data before starting ──
    try:
        from scripts.validate_training_quality import validate_training_data
        gate_passed, gate_report = validate_training_data(quick=True)
        if gate_passed:
            logger.info("Training data quality gate: PASSED")
        else:
            issues = gate_report.get("issues", [])
            logger.warning("Training data quality gate: ISSUES DETECTED")
            for issue in issues:
                logger.warning(f"  - {issue}")
            logger.warning(
                "Consider running: python scripts/enrich_training_data.py --apply"
            )
            # Don't block training — just warn so users can decide
    except ImportError:
        pass  # Validator not available, skip
    except Exception as e:
        logger.debug(f"Quality gate check failed: {e}")

    from models.unified import UnifiedBlenderModel

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (hasattr(torch.backends, "mps")
          and torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Config
    train_cfg = config.get("training", {})
    unified_cfg = config.get("unified", {})
    data_cfg = config.get("data", {})

    # On macOS/MPS, the background puller can spawn many short-lived Python
    # processes (multi-process downloads) and Blender headless extractions.
    # That often causes libmalloc warnings and can exacerbate swap thrashing.
    # Keep training stable by defaulting it off unless explicitly enabled.
    if device.type == "mps" and "background_data_pull" not in train_cfg:
        train_cfg["background_data_pull"] = False
        logger.info(
            "MPS default: background_data_pull=False (set training.background_data_pull=true to enable)"
        )

    batch_size = train_cfg.get("batch_size", 4)
    lr = float(train_cfg.get("learning_rate", 3e-4))
    warmup_steps = train_cfg.get("warmup_steps", 500)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 4)

    # ── Cloud GPU overrides via environment variables ──
    # Set by cloud/train_cloud.sh based on detected GPU memory.
    if os.environ.get("CLOUD_BATCH_SIZE"):
        batch_size = int(os.environ["CLOUD_BATCH_SIZE"])
        logger.info(f"Cloud override: batch_size={batch_size}")
    if os.environ.get("CLOUD_GRAD_ACCUM"):
        grad_accum = int(os.environ["CLOUD_GRAD_ACCUM"])
        logger.info(f"Cloud override: grad_accum={grad_accum}")
    if os.environ.get("CLOUD_SAVE_EVERY"):
        train_cfg["save_every"] = int(os.environ["CLOUD_SAVE_EVERY"])
    if os.environ.get("CLOUD_EVAL_EVERY"):
        train_cfg["eval_every"] = int(os.environ["CLOUD_EVAL_EVERY"])
    if os.environ.get("GEO_EVAL_EVERY"):
        train_cfg["geo_eval_every"] = int(os.environ["GEO_EVAL_EVERY"])
        logger.info(f"Env override: geo_eval_every={train_cfg['geo_eval_every']}")
    if os.environ.get("DISABLE_BACKGROUND_DATA_PULL"):
        train_cfg["background_data_pull"] = False
        logger.info("Env override: background_data_pull=False")

    # MPS needs batch=1 for long sequences (no flash attention).
    # Compensate with higher gradient accumulation to keep effective
    # batch size the same:  batch=1 × grad_accum=8 = effective 8.
    if device.type == "mps" and batch_size > 1:
        old_effective = batch_size * grad_accum
        batch_size = 1
        grad_accum = old_effective
        logger.info(f"MPS: batch_size=1, grad_accum={grad_accum} "
                    f"(effective batch={old_effective})")
    eval_every = train_cfg.get("eval_every", 500)
    save_every = train_cfg.get("save_every", 1000)
    max_text_len = unified_cfg.get("text_max_length", 256)
    image_size = unified_cfg.get("image_size", 64)
    naninf_guard_threshold = int(train_cfg.get("naninf_guard_threshold", 64))
    naninf_guard_mode = str(
        train_cfg.get("naninf_guard_mode", "advance")
    ).strip().lower()
    if naninf_guard_mode not in ("advance", "abort"):
        logger.warning(
            f"Unknown naninf_guard_mode={naninf_guard_mode!r}; "
            "falling back to 'advance'"
        )
        naninf_guard_mode = "advance"

    task_weights = train_cfg.get("task_weights", {
        "geometry": 6.0, "materials": 0.0,
        "modifiers": 0.0, "contrastive": 0.5,
        "image_geometry": 0.0,  # enabled below when model supports it
    })
    loss_weights = train_cfg.get("loss_weights", {
        "geometry": 1.0, "materials": 1.0,
        "modifiers": 0.5, "contrastive": 0.15,
        "image_geometry": 1.2,  # slightly upweight: rarer supervised signal
    })
    # Materials=0.0000 loss and modifiers=0.12 are fully converged.
    # Every step spent on them is a step NOT spent improving geometry
    # which is the core task and still learning (loss 0.12-0.28).
    # Disable them to put 100% of GPU compute toward mesh quality.
    logger.info("Task weights: geometry=6.0, materials=DISABLED (converged), "
                "modifiers=DISABLED (converged)")

    # Output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = str(Path(__file__).parent.parent)

    # Tokenizers
    from processing.mesh_tokenizer import MeshTokenizer

    tok_config = config.get("tokenization", {})
    mesh_tokenizer = MeshTokenizer(
        vocab_size=tok_config.get("vocab_size", 8192),
        coord_range=tuple(
            tok_config.get("coordinate_range", [-1.0, 1.0])),
        max_faces=tok_config.get("max_faces", 2048),
    )

    geo_dir = Path(
        data_cfg.get("geometry_dir", "data/datasets/geometry"))

    # ── Text tokenizer: prefer BPE, fall back to legacy word-level ──
    text_tokenizer = None
    bpe_dir = geo_dir / "bpe_tokenizer"

    if bpe_dir.exists() and (bpe_dir / "tokenizer.model").exists():
        from processing.bpe_tokenizer import BPETokenizer
        text_tokenizer = BPETokenizer.load(bpe_dir)
        logger.info(
            f"Loaded BPE tokenizer: {text_tokenizer.vocab_size} subwords"
        )
    else:
        logger.info("Training BPE tokenizer from synthetic + real labels...")
        from processing.bpe_tokenizer import BPETokenizer
        from processing.generate_synthetic import (
            SHAPE_SPECS, COMPOSITE_SPECS, generate_label,
        )

        # Gather text from synthetic labels
        all_specs = {**SHAPE_SPECS, **COMPOSITE_SPECS}
        texts = []
        for _ in range(20000):
            key = random.choice(list(all_specs.keys()))
            spec = all_specs[key]
            params = spec["params"]()
            texts.append(generate_label(key, params))

        # Also gather text from real mesh labels (Objaverse etc.)
        real_dirs = [Path(d) for d in data_cfg.get("real_mesh_dirs", [
            "data/processed/objaverse",
            "data/processed/blendswap",
            "data/processed/blender_official",
        ]) if Path(d).exists()]
        for d in real_dirs:
            for jf in d.glob("*.json"):
                if jf.name.endswith(".meta.json"):
                    continue
                try:
                    with open(jf) as f:
                        data = json.load(f)
                    objs = data.get("objects", [data]) if isinstance(data, dict) else [data]
                    for obj in objs:
                        label = (obj.get("text_label")
                                 or obj.get("name")
                                 or obj.get("text") or "")
                        if label and label.lower() not in ("3d object", "object", "mesh"):
                            texts.append(label)
                except Exception:
                    pass

        logger.info(f"Training BPE on {len(texts)} text samples...")
        geo_dir.mkdir(parents=True, exist_ok=True)
        model_prefix = str(geo_dir / "bpe_model")
        text_tokenizer = BPETokenizer.train(
            texts, vocab_size=8000, model_prefix=model_prefix)
        logger.info(
            f"Built BPE tokenizer: {text_tokenizer.vocab_size} subwords"
        )

    if text_tokenizer is not None:
        # BPE tokenizer already includes special tokens in vocab_size.
        # Legacy word-level tokenizer needed +4 for PAD/BOS/EOS/UNK.
        from processing.bpe_tokenizer import BPETokenizer
        if isinstance(text_tokenizer, BPETokenizer):
            unified_cfg["text_vocab_size"] = text_tokenizer.vocab_size
        else:
            unified_cfg["text_vocab_size"] = text_tokenizer.vocab_size + 4
    config["unified"] = unified_cfg

    # Model
    model: UnifiedBlenderModel = UnifiedBlenderModel(config).to(device)  # type: ignore[assignment]
    param_count = model.count_parameters()
    logger.info(f"Model: {param_count:,} params ({param_count/1e6:.1f}M)")

    # Enable gradient checkpointing for geometry decoder.
    # Trades ~2x compute for ~6x less activation memory.
    # On MPS: essential (no flash attention, O(n^2) attention matrix).
    # On CUDA with large batch: enables much bigger batches by freeing
    # intermediate activations and recomputing them during backward pass.
    if device.type == "mps" or (device.type == "cuda" and batch_size >= 32):
        model.geometry_decoder.use_gradient_checkpointing = True
        logger.info(
            f"Gradient checkpointing enabled "
            f"(batch_size={batch_size}, saves ~6x activation memory)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    # DeepSeek-style multi-step LR schedule
    # (peak → 31.6% at 80% → 10% at 90% of estimated total steps)
    # IMPORTANT: scheduler.step() is called once per *optimizer* step
    # (i.e., once per `grad_accum` micro-steps). Convert configured
    # micro-step counts into optimizer-step units so the schedule decays
    # as intended.
    estimated_total_steps_micro = int(train_cfg.get("max_steps", 200000))
    estimated_total_steps = max(
        1, int(math.ceil(estimated_total_steps_micro / float(grad_accum)))
    )
    warmup_steps_micro = int(warmup_steps)
    warmup_steps_sched = max(
        1, int(math.ceil(warmup_steps_micro / float(grad_accum)))
    )
    logger.info(
        "LR schedule steps (micro→opt): "
        f"max_steps={estimated_total_steps_micro}→{estimated_total_steps}, "
        f"warmup_steps={warmup_steps_micro}→{warmup_steps_sched}, "
        f"grad_accum={grad_accum}"
    )

    def lr_schedule(step):
        # LambdaLR applies lr_lambda(0) after the first optimizer step.
        # Shift by +1 so the first applied warmup LR is non-zero.
        return deepseek_lr_schedule(
            step + 1, warmup_steps_sched, estimated_total_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_schedule)

    # Mixed precision override.
    # Historical name: CLOUD_MIXED_PRECISION (set by train_cloud.sh), but this
    # also applies to local runs. Prefer the clearer aliases below.
    # H100/A100 Tensor Cores are optimized for bf16, not fp16.
    # bf16 doesn't need GradScaler (8-bit exponent = no overflow risk).
    mixed_prec = (
        os.environ.get("TRAIN_MIXED_PRECISION")
        or os.environ.get("MIXED_PRECISION")
        or os.environ.get("CLOUD_MIXED_PRECISION")
        or train_cfg.get("mixed_precision", "fp16")
    )
    use_amp = mixed_prec != "fp32"
    use_bf16 = mixed_prec == "bf16" and device.type == "cuda"
    use_scaler = use_amp and device.type == "cuda" and not use_bf16
    scaler = GradScaler(enabled=use_scaler)
    if use_bf16:
        logger.info("Using bf16 mixed precision (native H100/A100, no GradScaler)")

    geo_criterion = nn.CrossEntropyLoss(
        ignore_index=0, label_smoothing=0.1)
    # Focal loss modulator: down-weights easy (high-probability) tokens,
    # up-weights rare/hard tokens. Prevents mode collapse where the model
    # learns to only predict the 2-3 most frequent coordinate bins.
    focal_gamma = float(train_cfg.get("focal_gamma", 2.0))
    use_focal = focal_gamma > 0
    if use_focal:
        logger.info(f"Focal loss enabled (gamma={focal_gamma})")
    mat_criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Auto-resume
    start_step = 0
    best_val_loss = float("inf")

    resume_path = getattr(args, "resume", None)
    if resume_path == "latest" or resume_path is None:
        resume_path = find_latest_checkpoint(str(output_dir))

    if resume_path and Path(resume_path).exists():
        logger.info(f"Resuming from {resume_path}")
        ckpt = torch.load(
            resume_path, map_location=device, weights_only=False)

        # 'step' is the micro-step counter (increments every batch).
        # Scheduler/optimizer step once per grad_accum micro-steps.
        start_step = int(ckpt.get("step", 0))
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

        saved_grad_accum = int(ckpt.get("grad_accum", grad_accum) or grad_accum)
        saved_opt_step = ckpt.get("optimizer_step")
        if saved_opt_step is None:
            # Fallback: derive optimizer steps from micro-steps.
            saved_opt_step = start_step // max(1, saved_grad_accum)
        opt_step = int(saved_opt_step)

        # Handle shape mismatches (e.g. vocab size changed)
        saved_state = ckpt["model_state_dict"]
        current_state = model.state_dict()
        compatible_state = {}
        skipped = []
        for k, v in saved_state.items():
            if k in current_state and v.shape == current_state[k].shape:
                compatible_state[k] = v
            else:
                skipped.append(k)
        if skipped:
            logger.warning(
                f"Skipped {len(skipped)} mismatched checkpoint keys "
                f"(vocab/arch change): {skipped[:5]}"
            )
        model.load_state_dict(compatible_state, strict=False)

        if skipped:
            logger.warning(
                "Skipping optimizer/scheduler restore due to arch changes"
            )
            # Fast-forward scheduler so LR matches resumed step.
            # LambdaLR tracks state in last_epoch; setting it directly
            # is O(1) vs looping start_step times.
            if opt_step > 0:
                scheduler.last_epoch = opt_step
                lr_now = lr_schedule(opt_step) * lr
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
                logger.info(
                    f"Scheduler fast-forwarded to opt_step {opt_step} "
                    f"(LR: {lr_now:.2e})")
        else:
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(
                        ckpt["optimizer_state_dict"])
                except Exception:
                    logger.warning("Could not restore optimizer state")
            if "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(
                        ckpt["scheduler_state_dict"])
                except Exception:
                    pass

        # Sanity-check: scheduler.last_epoch should be close to start_step.
        # If a previous buggy run saved a checkpoint with last_epoch stuck
        # near 0, the restored LR will be near-zero (warmup region).
        # Detect and force-fix.
        sched_epoch = int(getattr(scheduler, 'last_epoch', 0) or 0)
        if opt_step > 100 and abs(sched_epoch - opt_step) > opt_step * 0.5:
            logger.warning(
                f"Scheduler last_epoch ({sched_epoch}) is far from "
                f"opt_step ({opt_step}) — corrupted checkpoint state. "
                f"Force fast-forwarding scheduler.")
            scheduler.last_epoch = opt_step
            lr_now = lr_schedule(opt_step) * lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now
            logger.info(
                f"Scheduler corrected to opt_step {opt_step} "
                f"(LR: {lr_now:.2e})")
        else:
            lr_now = scheduler.get_last_lr()[0]
            logger.info(f"Scheduler OK: last_epoch={sched_epoch}, LR={lr_now:.2e}")

        logger.info(f"Resumed at step {start_step}")
    else:
        logger.info("Starting fresh training")

        # Prime warmup LR so the first optimizer update doesn't happen at full LR.
        # (LambdaLR otherwise applies lr_lambda(0) only after the first update.)
        lr_now = lr * lr_schedule(0)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        logger.info(f"Warmup primed: initial LR {lr_now:.2e}")

    # torch.compile: Skip for now.  The model is small (86M) and uses
    # gradient checkpointing + variable sequence lengths, which causes
    # torch.compile to spend 10+ minutes recompiling CUDA graphs.
    # The bf16 + larger batch size provides the main speedup instead.
    # if device.type == \"cuda\":
    #     model = torch.compile(model, dynamic=True)
    logger.info("torch.compile SKIPPED (86M model, gradient checkpointing "
                "makes compile overhead > benefit)")

    # Data streams — RoPE has no architectural limit, but MPS has a
    # practical ceiling: 8 heads × seq² must fit in a 32-bit tensor.
    # At 1800 faces (seq=16202), elements = 8 × 16201² = 2.1B < 2^31.
    max_mesh_tok = unified_cfg.get("geometry", {}).get(
        "max_seq_length", 16202)

    # MPS memory guard: O(n²) attention on 16202 tokens needs ~8GB for
    # the attention matrix alone plus activations.  On a MacBook with
    # unified memory this routinely pushes RAM >90% and triggers swap
    # thrashing that freezes the system for minutes.
    # Cap to 900 faces (8102 tokens) → 4× less attention memory.
    if device.type == "mps":
        mps_max_tok = 8102   # 900 faces × 9 + 2
        if max_mesh_tok > mps_max_tok:
            logger.info(
                f"MPS memory guard: capping max_seq_length "
                f"{max_mesh_tok} → {mps_max_tok} (900 faces)")
            max_mesh_tok = mps_max_tok

    # CUDA memory guard: allow env var override for cloud GPUs.
    # With Flash Attention (SDPA) on H100/A100, attention is O(n) memory,
    # so full 16202-token sequences are tractable.  The OOM handler in the
    # training loop catches any edge cases and clears the CUDA cache.
    cloud_max_seq = int(os.environ.get("CLOUD_MAX_SEQ_LEN", "0"))
    if cloud_max_seq > 0 and max_mesh_tok > cloud_max_seq:
        logger.info(
            f"CLOUD_MAX_SEQ_LEN guard: capping max_seq_length "
            f"{max_mesh_tok} -> {cloud_max_seq}")
        max_mesh_tok = cloud_max_seq

    # num_workers: use 4 on CUDA (saturate GPU), 0 on MPS (fork + Metal is unstable)
    dl_workers = 0 if device.type == "mps" else train_cfg.get("num_workers", 4)

    # Adaptive batch sizes per task:
    # - Synthetic geometry: short sequences (100-500 tokens) → full batch
    # - Real geometry: long sequences (1000-16000 tokens) → smaller batch
    #   to avoid OOM on padded tensors (collator pads to max in batch)
    # - Materials/modifiers: short sequences → full batch
    real_geom_batch = max(1, batch_size // 4)  # 1/4 of main batch (real seqs can be 72K tokens)
    if os.environ.get("CLOUD_REAL_GEOM_BATCH"):
        real_geom_batch = int(os.environ["CLOUD_REAL_GEOM_BATCH"])
    logger.info(f"Batch sizes: synth={batch_size}, real_geom={real_geom_batch}, "
                f"materials={batch_size}, modifiers={batch_size}")

    # Cap synthetic max tokens based on batch size to prevent OOM.
    # collate_geometry now drops outliers that would cause excessive padding,
    # so we can be more generous with the cap.
    # H100 80GB with bf16: batch=128 × 4000 tokens = 512K tokens → ~20GB
    synth_max_tok = min(max_mesh_tok, max(2000, 500_000 // batch_size))
    if synth_max_tok < max_mesh_tok:
        logger.info(
            f"Synthetic seq cap: {max_mesh_tok} → {synth_max_tok} "
            f"(batch_size={batch_size}, prevents padding OOM)")

    loaders = {}

    # 1. Infinite synthetic shapes
    # render_prob: fraction of synthetic examples that get a rendered image.
    # When image_to_mesh is enabled, increase to 0.6 so the image_geometry
    # task receives enough image-conditioned batches.  At 0.3 (default),
    # with batch_size=8 only ~2-3 examples per batch have images, which
    # is too sparse for stable contrastive + image-geometry training.
    _img_render_prob = 0.6 if model.enable_image_to_mesh else 0.3
    synth_stream = InfiniteShapeStream(
        mesh_tokenizer, text_tokenizer,
        max_text_length=max_text_len,
        max_mesh_tokens=synth_max_tok,
        image_size=image_size,
        render_prob=_img_render_prob,
        include_scenes=True,
    )
    loaders["geometry"] = DataLoader(
        synth_stream, batch_size=batch_size,
        num_workers=dl_workers, collate_fn=collate_geometry,
        prefetch_factor=2 if dl_workers > 0 else None,
    )

    # 2. Real mesh data — prefer pre-built training cache if available
    _training_cache_dir = Path(project_root) / "data" / "training_cache" / "default"
    _has_training_cache = (
        _training_cache_dir.exists()
        and any(_training_cache_dir.glob("batch_*.pt"))
    )

    if _has_training_cache:
        # Use pre-built training cache (from scripts/build_training_cache.py)
        # This is dramatically faster: items are already tokenized, filtered,
        # and quality-scored. No source-file scanning or inline processing.
        logger.info(
            f"Using pre-built training cache: {_training_cache_dir} "
            f"({len(list(_training_cache_dir.glob('batch_*.pt')))} batches)"
        )
        prebuilt_stream = PrebuiltCacheStream(
            str(_training_cache_dir),
            max_mesh_tokens=max_mesh_tok,
        )
        loaders["real_geometry"] = DataLoader(
            prebuilt_stream, batch_size=real_geom_batch,
            num_workers=0, collate_fn=collate_geometry,
        )
        task_weights["real_geometry"] = 4.0
        loss_weights["real_geometry"] = 1.5
    else:
        # Fallback: scan source JSONs and build cache inline (legacy path)
        real_data_dirs = [
            str(Path(project_root) / "data" / "processed" / d)
            for d in [
                "objaverse", "blendswap", "blender_official", "github",
                "smutbase", "open3dlab", "youtube",
            ]
        ] + [
            # Objaverse-XL per-source metadata dirs (for metadata enrichment)
            str(Path(project_root) / "data" / "raw" / "objaverse" / s / "metadata")
            for s in ["sketchfab", "github", "smithsonian", "thingiverse"]
        ] + [
            str(Path(project_root) / "data" / "filtered"),
            str(Path(project_root) / "data" / "labeled"),
        ]
        existing_dirs = [
            d for d in real_data_dirs
            if Path(d).exists() and any(Path(d).rglob("*.json"))
        ]
        if existing_dirs:
            real_stream = RealMeshStream(
                existing_dirs, mesh_tokenizer, text_tokenizer,
                max_text_length=max_text_len,
                max_mesh_tokens=max_mesh_tok,
            )
            loaders["real_geometry"] = DataLoader(
                real_stream, batch_size=real_geom_batch,
                num_workers=0, collate_fn=collate_geometry,
            )
            task_weights["real_geometry"] = 4.0
            loss_weights["real_geometry"] = 1.5

    # 3. Materials (only when model has material decoder enabled)
    if model.enable_materials:
        mat_path = data_cfg.get(
            "materials_train",
            str(geo_dir / "materials_train.jsonl"))
        mat_stream = MaterialStream(
            mat_path, text_tokenizer,
            max_text_length=max_text_len,
            vocab_size=unified_cfg.get(
                "materials", {}).get("vocab_size", 4096),
        )
        loaders["materials"] = DataLoader(
            mat_stream, batch_size=batch_size,
            num_workers=dl_workers, collate_fn=collate_generic,
            prefetch_factor=2 if dl_workers > 0 else None,
        )
    else:
        logger.info("Materials task DISABLED (enable_materials=false)")
        task_weights.pop("materials", None)

    # 4. Modifiers (only when model has modifier head enabled)
    if model.enable_modifiers:
        mod_path = data_cfg.get(
            "modifiers_train",
            str(geo_dir / "modifiers_train.jsonl"))
        mod_stream = ModifierStream(
            mod_path, text_tokenizer,
            max_text_length=max_text_len,
        )
        loaders["modifiers"] = DataLoader(
            mod_stream, batch_size=batch_size,
            num_workers=dl_workers, collate_fn=collate_generic,
            prefetch_factor=2 if dl_workers > 0 else None,
        )
    else:
        logger.info("Modifiers task DISABLED (enable_modifiers=false)")
        task_weights.pop("modifiers", None)

    # 5. Contrastive (only when model has contrastive head enabled)
    # NOTE: Even when enabled, disabled at batch_size<=2 because
    # CLIP-style contrastive loss = log(2) = 0.693 (random chance).
    if model.enable_contrastive and task_weights.get("contrastive", 0) > 0:
        geo_train = geo_dir / "train.jsonl"
        contrastive_stream = ContrastiveStream(
            str(geo_train) if geo_train.exists() else None,
            text_tokenizer,
            max_text_length=max_text_len,
            image_size=image_size,
        )

        real_cache_dir = None
        # Prefer pre-built training cache, fall back to old .mesh_cache
        for d in [
            str(_training_cache_dir),
            str(Path(project_root) / "data" / "processed" / ".mesh_cache"),
        ]:
            if Path(d).exists():
                real_cache_dir = d
                break
        real_contrastive = None
        if real_cache_dir:
            real_contrastive = RealContrastiveStream(
                real_cache_dir, text_tokenizer,
                max_text_length=max_text_len,
                image_size=image_size,
            )

        if real_contrastive is not None:
            merged_contrastive = _MergedContrastiveStream(
                contrastive_stream, real_contrastive, real_weight=0.6)
            loaders["contrastive"] = DataLoader(
                merged_contrastive, batch_size=batch_size,
                num_workers=dl_workers, collate_fn=collate_generic,
                prefetch_factor=2 if dl_workers > 0 else None,
            )
        else:
            loaders["contrastive"] = DataLoader(
                contrastive_stream, batch_size=batch_size,
                num_workers=dl_workers, collate_fn=collate_generic,
                prefetch_factor=2 if dl_workers > 0 else None,
            )
    else:
        if not model.enable_contrastive:
            logger.info("Contrastive task DISABLED (enable_contrastive=false)")
        else:
            logger.info("Contrastive task DISABLED (weight=0, useless at batch_size=%d)", batch_size)
        task_weights.pop("contrastive", None)

    # 6. Image-conditioned geometry (image + optional text → mesh tokens)
    #    Uses the same InfiniteShapeStream + RealMeshStream data that already
    #    renders images, but passes them through forward_image_conditioned()
    #    instead of forward_geometry().  This trains the ImageEncoder's
    #    spatial features to condition the GeometryDecoder.
    if model.enable_image_to_mesh and batch_size >= 4:
        task_weights["image_geometry"] = 1.5
        # Re-use the same contrastive stream (it already has rendered images)
        # by wrapping the synth_stream as an image-geometry source.
        # The training step (below) will detect task=="image_geometry" and
        # call forward_image_conditioned() when "image" is in the batch.
        loaders["image_geometry"] = DataLoader(
            synth_stream, batch_size=max(1, batch_size // 2),
            num_workers=dl_workers, collate_fn=collate_geometry,
            prefetch_factor=2 if dl_workers > 0 else None,
        )
        logger.info(
            "Image-geometry task ENABLED (image→mesh training, weight=1.5, "
            "batch_size=%d)", max(1, batch_size // 2))
    else:
        task_weights.pop("image_geometry", None)
        if not model.enable_image_to_mesh:
            logger.info("Image-geometry task DISABLED (enable_image_to_mesh=false)")
        else:
            logger.info(
                "Image-geometry task DISABLED (batch_size=%d < 4, "
                "need at least 4 for image conditioning)", batch_size)

    # Validation
    val_loader = None
    geo_val = geo_dir / "val.jsonl"
    if geo_val.exists():
        class _ValDS(Dataset):
            def __init__(self, path, tt, mtl, mmt, mesh_vocab_size: int):
                self.examples = []
                self.mesh_vocab_size = int(mesh_vocab_size)
                self.max_token_id = -1
                with open(path) as f:
                    for ln in f:
                        ex = json.loads(ln.strip())
                        toks = ex.get("tokens") or []
                        if toks:
                            try:
                                self.max_token_id = max(
                                    self.max_token_id, int(max(toks))
                                )
                            except Exception:
                                pass
                        self.examples.append(ex)
                self.tt, self.mtl, self.mmt = tt, mtl, mmt

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, idx):
                ex = self.examples[idx]
                if self.tt:
                    ids, mask = self.tt.encode_padded(
                        ex["text"], self.mtl)
                else:
                    t = ex["text"][:self.mtl]
                    ids = [ord(c) % 32000 for c in t]
                    mask = [1] * len(ids)
                    ids += [0] * (self.mtl - len(ids))
                    mask += [0] * (self.mtl - len(mask))
                toks = ex["tokens"][:self.mmt]
                return {
                    "task": "geometry",
                    "text_ids": torch.tensor(
                        ids, dtype=torch.long),
                    "text_mask": torch.tensor(
                        mask, dtype=torch.float),
                    "mesh_tokens": torch.tensor(
                        toks, dtype=torch.long),
                }

        val_ds = _ValDS(
            str(geo_val), text_tokenizer,
            max_text_len, max_mesh_tok,
            mesh_tokenizer.vocab_size,
        )

        if val_ds.max_token_id >= mesh_tokenizer.vocab_size:
            logger.warning(
                "Validation set appears incompatible with current mesh vocab: "
                "%s has max_token_id=%d but vocab_size=%d. "
                "Disabling val loss evaluation to avoid crashes/misleading metrics. "
                "Regenerate val.jsonl with the current tokenizer.",
                str(geo_val), val_ds.max_token_id, mesh_tokenizer.vocab_size,
            )
            val_loader = None
        else:
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=0, collate_fn=collate_geometry,
            )

    # Background data puller
    puller = BackgroundDataPuller(config, project_root)
    if train_cfg.get("background_data_pull", True):
        puller.start()
    else:
        logger.info("Background data puller disabled")

    # Wandb (optional)
    wandb: Any = None
    use_wandb = False
    try:
        import wandb  # type: ignore[no-redef]
        wandb.init(
            project="blender-copilot-unified",
            config=config, resume="allow",
        )
        use_wandb = True
    except Exception:
        pass

    # RLHF trainer (human feedback integration)
    rlhf_trainer = None
    rlhf_cfg = config.get("rlhf", {})
    if rlhf_cfg.get("enabled", False):
        try:
            from training.rlhf import RLHFTrainer
            rlhf_trainer = RLHFTrainer(
                model, config,
                text_tokenizer=text_tokenizer,
                mesh_tokenizer=mesh_tokenizer,
                device=device)
            logger.info("RLHF trainer enabled - accepting human feedback")
        except Exception as e:
            logger.warning(f"RLHF init failed (non-fatal): {e}")
            rlhf_trainer = None

    # Info
    active_tasks = list(loaders.keys())
    logger.info("=" * 60)
    logger.info("AUTONOMOUS TRAINING - runs until Ctrl+C")
    logger.info("=" * 60)
    logger.info(f"  Device:     {device}")
    logger.info(f"  Model:      {param_count/1e6:.1f}M params")
    logger.info(f"  Tasks:      {', '.join(active_tasks)}")
    logger.info(f"  Batch:      {batch_size}")
    logger.info(f"  LR:         {lr}")
    logger.info(f"  Grad accum: {grad_accum}")
    logger.info(f"  Save every: {save_every} steps")
    logger.info(
        f"  NaN guard:  mode={naninf_guard_mode}, "
        f"threshold={naninf_guard_threshold}"
    )
    logger.info(f"  Output:     {output_dir}")
    if start_step > 0:
        logger.info(f"  Resumed at: step {start_step}")
    logger.info("=" * 60)

    # Training loop
    model.train()
    optimizer.zero_grad()
    step_timer = time.time()
    heartbeat_timer = time.time()
    global_step = start_step
    task_losses = {t: 0.0 for t in active_tasks}
    task_counts = {t: 0 for t in active_tasks}
    naninf_hits_step = -1
    naninf_hits_count = 0
    last_loss_value = 0.0

    sampler = InfiniteMultiTaskSampler(loaders, task_weights)
    logger.info("Starting training loop iteration...")

    for task_name, batch in sampler:
        if global_step == start_step or (global_step - start_step) % 50 == 0:
            logger.info(f"  step {global_step}: task={task_name}, "
                        f"batch_keys={list(batch.keys())}")
        if _STOP_TRAINING:
            break

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # Curriculum learning: gradually increase mesh complexity.
        # Early training uses small meshes (fast, stable gradients),
        # then ramps up to full complexity over warmup_steps.
        if task_name in ("geometry", "real_geometry"):
            seq_len = batch.get("mesh_tokens", torch.empty(0, 0)).shape[-1]

            # Curriculum cap: ramp from 32→max_faces over 5000 steps
            curriculum_faces = curriculum_max_faces(
                global_step, warmup_steps=5000,
                min_faces=32,
                max_faces=(max_mesh_tok - 2) // 9)
            curriculum_tok_cap = curriculum_faces * 9 + 2
            # Pad-aligned cap (face-aligned to next multiple of 9)
            padded_curriculum_cap = ((curriculum_tok_cap + 8) // 9) * 9

            # Skip oversized sequences BEFORE they hit the GPU.
            # collate_geometry rounds padding to next multiple of 9
            # (face-aligned), so allow that rounding slack.
            padded_cap = ((max_mesh_tok + 8) // 9) * 9
            effective_cap = min(padded_cap, padded_curriculum_cap)
            if seq_len > effective_cap:
                if global_step % 1000 == 0:
                    logger.info(
                        f"  Skipping {task_name} batch: "
                        f"seq_len={seq_len} > cap={effective_cap}")
                continue

        # CUDA cache flush for real_geometry: the memory pool from the
        # previous (larger) synthetic batch contains allocations with
        # different tensor shapes that can't be reused.  Flushing lets
        # the allocator serve the long-sequence real_geometry batch
        # without fragmentation-induced OOM.  Cost: ~1ms per call.
        if device.type == "cuda" and task_name == "real_geometry":
            torch.cuda.empty_cache()

        # MPS RAM pressure check — skip batch if system RAM is critically high
        if device.type == "mps" and global_step % 5 == 0:
            try:
                import psutil
                ram_pct = psutil.virtual_memory().percent
                if ram_pct > 88:
                    torch.mps.empty_cache()
                    if ram_pct > 96:
                        import gc
                        gc.collect()
                        torch.mps.empty_cache()
                        logger.warning(
                            f"  RAM {ram_pct:.0f}%: skipping batch and "
                            f"clearing caches")
                        continue
            except ImportError:
                pass

        amp_enabled = use_amp and device.type in ("cuda", "mps")
        amp_dtype = (
            torch.bfloat16 if use_bf16
            else torch.float16 if device.type == "cuda"
            else torch.bfloat16
        )
        force_advance_step = False
        force_abort_training = False

        try:
            with autocast(
                device_type=device.type, dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                actual_task = task_name
                if task_name in ("geometry", "real_geometry"):
                    actual_task = "geometry"
                    mesh_tokens = batch["mesh_tokens"]

                    input_tok = mesh_tokens[:, :-1]
                    target_tok = mesh_tokens[:, 1:]

                    # Classifier-free guidance training: randomly drop
                    # text conditioning 10% of the time (replace with
                    # zeros).  This teaches the model an unconditional
                    # distribution which is essential for CFG at inference.
                    text_ids = batch["text_ids"]
                    text_mask = batch["text_mask"]
                    if model.training and random.random() < 0.1:
                        text_ids = torch.zeros_like(text_ids)
                        text_mask = torch.zeros_like(text_mask)

                    logits = model.forward_geometry(
                        text_ids,
                        text_mask,
                        input_tok,
                    )

                    # GRPO-inspired per-sample quality weighting
                    # Compute per-sample loss, then weight by
                    # group-relative advantage (DeepSeek GRPO)
                    B = logits.size(0)
                    if B > 1:
                        token_loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            target_tok.reshape(-1),
                            ignore_index=0,
                            reduction="none",
                        ).reshape(B, -1)
                        # Mask out PAD/ignore_index positions so shorter
                        # sequences aren't treated as "higher quality".
                        loss_mask = (target_tok != 0).float()
                        per_sample_loss = (
                            (token_loss * loss_mask).sum(dim=1)
                            / loss_mask.sum(dim=1).clamp(min=1)
                        )
                        grpo_w = grpo_quality_weights(
                            per_sample_loss, temperature=1.0)
                        loss = (per_sample_loss * grpo_w).mean()
                    else:
                        if use_focal:
                            loss = focal_cross_entropy(
                                logits.reshape(-1, logits.size(-1)),
                                target_tok.reshape(-1),
                                gamma=focal_gamma,
                            )
                        else:
                            loss = geo_criterion(
                                logits.reshape(-1, logits.size(-1)),
                                target_tok.reshape(-1),
                            )

                    # Position-weighted loss: tokens later in the sequence get
                    # higher weight. This penalises early EOS and rewards the
                    # model for generating full-budget meshes.
                    # Weight linearly from 1.0 (first token) to 1.8 (last).
                    if B == 1:
                        seq_len = target_tok.size(1)
                        if seq_len > 1:
                            pos_w = torch.linspace(
                                1.0, 1.8, seq_len,
                                device=target_tok.device,
                                dtype=torch.float32,
                            )
                            # recompute loss with position+focal weights
                            ptok_loss = F.cross_entropy(
                                logits.reshape(-1, logits.size(-1)),
                                target_tok.reshape(-1),
                                ignore_index=0,
                                label_smoothing=0.1,
                                reduction="none",
                            ).reshape(1, seq_len)
                            pmask = (target_tok != 0).float()
                            if use_focal:
                                with torch.no_grad():
                                    fl_probs = F.softmax(
                                        logits.reshape(-1, logits.size(-1)), dim=-1)
                                    fl_pt = fl_probs.gather(
                                        1, target_tok.reshape(-1).unsqueeze(1)
                                    ).squeeze(1).reshape(1, seq_len)
                                    fl_w = (1.0 - fl_pt).pow(focal_gamma)
                                loss = ((ptok_loss * pos_w.unsqueeze(0) * fl_w * pmask).sum()
                                        / pmask.sum().clamp(min=1))
                            else:
                                loss = ((ptok_loss * pos_w.unsqueeze(0) * pmask).sum()
                                        / pmask.sum().clamp(min=1))

                    # Apply quality-based weighting for real data
                    if ("quality_weight" in batch
                            and task_name == "real_geometry"):
                        qw = batch["quality_weight"].mean().item()
                        lc = 1.0
                        if "label_confidence" in batch:
                            lc = max(0.4, float(batch["label_confidence"].mean().item()))
                        complexity_boost = 1.0
                        if "scene_complexity_score" in batch:
                            complexity_boost = 1.0 + 0.15 * float(batch["scene_complexity_score"].mean().item())
                        loss = loss * qw * lc * complexity_boost

                    if "image" in batch:
                        try:
                            clip_loss = model.forward_contrastive(
                                batch["text_ids"],
                                batch["text_mask"],
                                batch["image"],
                            )
                            loss = loss + (
                                loss_weights.get("contrastive", 0.3)
                                * clip_loss
                            )
                        except Exception:
                            pass

                elif task_name == "materials":
                    logits = model.forward_materials(
                        batch["text_ids"],
                        batch["text_mask"],
                        batch["input_tokens"],
                    )
                    loss = mat_criterion(
                        logits.reshape(-1, logits.size(-1)),
                        batch["target_tokens"].reshape(-1),
                    )

                elif task_name == "modifiers":
                    outputs = model.forward_modifiers(
                        batch["text_ids"],
                        batch["text_mask"],
                        batch["mesh_stats"],
                    )
                    loss = compute_modifier_loss(
                        outputs, batch, device)

                elif task_name == "contrastive":
                    loss = model.forward_contrastive(
                        batch["text_ids"],
                        batch["text_mask"],
                        batch["image"],
                    )

                elif task_name == "image_geometry":
                    # Image-conditioned geometry: image (+ text) → mesh tokens
                    # Skip batches that have no rendered image
                    if "image" not in batch:
                        continue
                    input_tok = batch["mesh_tokens"][:, :-1].to(device)
                    target_tok = batch["mesh_tokens"][:, 1:].to(device)
                    img = batch["image"].to(device)
                    t_ids = batch["text_ids"].to(device)
                    t_mask = batch["text_mask"].to(device)
                    # 10% chance of dropping text hint (image-only conditioning)
                    if random.random() < 0.1:
                        t_ids = None
                        t_mask = None
                    try:
                        logits = model.forward_image_conditioned(
                            img, input_tok,
                            text_ids=t_ids,
                            text_mask=t_mask,
                        )
                        loss = geo_criterion(
                            logits.reshape(-1, logits.size(-1)),
                            target_tok.reshape(-1),
                        )
                    except RuntimeError as e:
                        if "enable_image_to_mesh" in str(e):
                            continue
                        raise

                else:
                    continue

                lw = loss_weights.get(
                    task_name,
                    loss_weights.get(actual_task, 1.0),
                )
                weighted_loss = loss * lw / grad_accum

                # Check for NaN/Inf loss BEFORE backward pass
                # to prevent corrupt gradients from polluting
                # the optimizer's Adam momentum/variance state.
                if torch.isnan(loss) or torch.isinf(loss):
                    if naninf_hits_step != global_step:
                        naninf_hits_step = global_step
                        naninf_hits_count = 0
                    naninf_hits_count += 1

                    logger.warning(
                        f"Step {global_step}: NaN/Inf loss on "
                        f"{task_name}, zeroing grads and skipping")
                    optimizer.zero_grad()
                    del weighted_loss

                    if (naninf_guard_threshold > 0
                            and naninf_hits_count >= naninf_guard_threshold):
                        if naninf_guard_mode == "abort":
                            logger.error(
                                f"Step {global_step}: NaN/Inf guard hit "
                                f"({naninf_hits_count} skips). "
                                "Stopping training for safe resume."
                            )
                            force_abort_training = True
                        else:
                            logger.error(
                                f"Step {global_step}: NaN/Inf guard hit "
                                f"({naninf_hits_count} skips). "
                                "Forcing step advance."
                            )
                            force_advance_step = True
                        naninf_hits_count = 0

                    if not force_advance_step and not force_abort_training:
                        continue

            if force_abort_training:
                _STOP_TRAINING = True
                break

            if force_advance_step:
                optimizer.zero_grad()

        except Exception as e:
            err_msg = str(e).lower()
            is_oom = ("out of memory" in err_msg
                      or "cublas" in err_msg
                      or "alloc_failed" in err_msg
                      or "cufft" in err_msg)
            if is_oom and task_name in ("geometry", "real_geometry"):
                # --- OOM RETRY: process samples individually ---
                # Detailed meshes are valuable training data.  Instead
                # of skipping the whole batch, clear memory and retry
                # each sample one-at-a-time with trimmed padding.
                try:
                    del loss, weighted_loss, logits
                except NameError:
                    pass
                import gc as _gc
                _gc.collect()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                elif device.type == "mps":
                    torch.mps.empty_cache()
                optimizer.zero_grad()

                B = batch.get("mesh_tokens", torch.empty(0)).shape[0]
                seq_len = batch.get(
                    "mesh_tokens", torch.empty(0, 0)).shape[-1]
                if B <= 1:
                    logger.warning(
                        f"Step {global_step} OOM on single sample "
                        f"({task_name}, seq_len={seq_len}), skipping")
                    continue

                _oom_retry_result = _oom_retry_per_sample(
                    model, batch, task_name, B, geo_criterion,
                    loss_weights, grad_accum, use_scaler, scaler,
                    amp_enabled, amp_dtype, device, global_step,
                )
                if _oom_retry_result is not None:
                    task_losses[task_name] = (
                        task_losses.get(task_name, 0)
                        + _oom_retry_result)
                    task_counts[task_name] = (
                        task_counts.get(task_name, 0) + 1)
                else:
                    # All retries failed (CUDA likely corrupted after
                    # OOM) — skip gradient step to avoid scaler crash
                    optimizer.zero_grad()
                    continue

            elif is_oom:
                logger.warning(
                    f"Step {global_step} OOM during forward "
                    f"({task_name}), skipping batch")
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            else:
                logger.warning(
                    f"Step {global_step} error ({task_name}): {e}")
                optimizer.zero_grad()
                continue
        else:
            # Forward pass succeeded — now do backward
            try:
                if use_scaler:
                    scaler.scale(weighted_loss).backward()
                else:
                    weighted_loss.backward()
            except RuntimeError as e:
                err_msg = str(e).lower()
                if ("out of memory" in err_msg
                        or "cublas" in err_msg
                        or "alloc_failed" in err_msg
                        or "cufft" in err_msg):
                    try:
                        del weighted_loss
                    except NameError:
                        pass
                    import gc as _gc
                    _gc.collect()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    elif device.type == "mps":
                        torch.mps.empty_cache()
                    optimizer.zero_grad()

                    if task_name in ("geometry", "real_geometry"):
                        B = batch.get("mesh_tokens",
                                      torch.empty(0)).shape[0]
                        if B > 1:
                            _oom_retry_result = _oom_retry_per_sample(
                                model, batch, task_name, B,
                                geo_criterion, loss_weights,
                                grad_accum, use_scaler, scaler,
                                amp_enabled, amp_dtype, device,
                                global_step,
                            )
                            if _oom_retry_result is not None:
                                task_losses[task_name] = (
                                    task_losses.get(task_name, 0)
                                    + _oom_retry_result)
                                task_counts[task_name] = (
                                    task_counts.get(task_name, 0) + 1)
                            else:
                                # All retries failed — skip grad step
                                optimizer.zero_grad()
                                continue
                        else:
                            logger.warning(
                                f"Step {global_step} OOM backward "
                                f"single sample ({task_name}), skip")
                            continue
                    else:
                        logger.warning(
                            f"Step {global_step} OOM backward "
                            f"({task_name}), skipping")
                        continue
                else:
                    raise

            # Normal loss tracking (forward+backward both succeeded)
            task_losses[task_name] = (
                task_losses.get(task_name, 0) + loss.item()
            )
            task_counts[task_name] = (
                task_counts.get(task_name, 0) + 1
            )
            last_loss_value = float(loss.item())

        # Periodic memory cleanup for MPS / CUDA
        if global_step % 100 == 0:
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        if (global_step + 1) % grad_accum == 0:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Logging — every 10 steps (was 50, too slow for MPS feedback)
        should_log = (
            (global_step < 10) or (global_step % 10 == 0)
        )
        if should_log:
            now = time.time()
            elapsed = now - step_timer
            interval = max(
                1, 10 if global_step >= 10 else 1)
            sps = interval / max(elapsed, 0.001)
            step_timer = now
            heartbeat_timer = now

            parts = []
            for t in active_tasks:
                if task_counts.get(t, 0) > 0:
                    avg = task_losses[t] / task_counts[t]
                    short = t[:4] if len(t) > 6 else t
                    parts.append(f"{short}={avg:.4f}")
            loss_str = " | ".join(parts)

            lr_cur = scheduler.get_last_lr()[0]

            # Memory info for MPS/CUDA
            mem_str = ""
            if device.type == "mps":
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    mem_str = f" | RAM: {mem.percent:.0f}%"
                except ImportError:
                    pass
            elif device.type == "cuda":
                alloc = torch.cuda.memory_allocated() / 1e9
                mem_str = f" | GPU: {alloc:.1f}GB"

            logger.info(
                f"Step {global_step} | {loss_str} | "
                f"LR: {lr_cur:.2e} | {sps:.1f} it/s{mem_str}"
            )

            if use_wandb:
                log_dict = {
                    "step": global_step,
                    "train/lr": lr_cur,
                }
                for t in active_tasks:
                    if task_counts.get(t, 0) > 0:
                        log_dict[f"train/{t}_loss"] = (
                            task_losses[t] / task_counts[t]
                        )
                wandb.log(log_dict)

            task_losses = {t: 0.0 for t in active_tasks}
            task_counts = {t: 0 for t in active_tasks}

        # Heartbeat — log every 60s even if steps are slow
        # so user knows training is alive
        elif time.time() - heartbeat_timer > 60:
            heartbeat_timer = time.time()
            elapsed_total = heartbeat_timer - step_timer
            mem_str = ""
            if device.type == "mps":
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    mem_str = f" | RAM: {mem.percent:.0f}%"
                except ImportError:
                    pass
            logger.info(
                f"  [heartbeat] step {global_step} in progress "
                f"({elapsed_total:.0f}s since last log){mem_str}"
            )

        # Eval
        if (val_loader and global_step > 0
                and global_step % eval_every == 0):
            val_loss = evaluate(
                model, val_loader, geo_criterion,
                device, use_amp,
            )
            logger.info(
                f"  Val loss: {val_loss:.4f} "
                f"(best: {best_val_loss:.4f})"
            )
            if use_wandb:
                wandb.log({
                    "val/geometry_loss": val_loss,
                    "step": global_step,
                })
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler,
                    global_step, val_loss,
                    output_dir / "best.pt",
                    config, best_val_loss,
                    grad_accum=grad_accum,
                )

        # Geometric evaluation — run at 5× eval interval
        # Generates actual meshes and measures validity, face counts, etc.
        geo_eval_every = train_cfg.get("geo_eval_every", eval_every * 5)
        if (geo_eval_every > 0
                and global_step > 0
                and global_step % geo_eval_every == 0):
            try:
                from evaluation.harness import (
                    run_geometric_eval, get_wandb_log_dict,
                )
                geo_results = run_geometric_eval(
                    model, mesh_tokenizer, text_tokenizer,
                    device, global_step, config,
                    max_faces=256,
                    temperature=0.7,
                )
                if use_wandb:
                    wandb.log({
                        **get_wandb_log_dict(geo_results),
                        "step": global_step,
                    })
                model.train()
            except Exception as e:
                logger.warning(f"Geometric eval failed: {e}")
                model.train()

        # RLHF — check for human feedback and run DPO updates
        if (rlhf_trainer is not None
                and global_step > 0
                and global_step % 100 == 0):
            try:
                rlhf_result = rlhf_trainer.maybe_update()
                if rlhf_result:
                    logger.info(
                        f"  RLHF update: "
                        f"dpo_loss={rlhf_result.get('dpo_loss', 0):.4f}, "
                        f"reward_acc={rlhf_result.get('reward_accuracy', 0):.2%}, "
                        f"feedback={rlhf_result.get('total_feedback', 0)}")
                    if use_wandb:
                        wandb.log({
                            "rlhf/dpo_loss": rlhf_result.get("dpo_loss", 0),
                            "rlhf/reward_accuracy": rlhf_result.get("reward_accuracy", 0),
                            "rlhf/total_feedback": rlhf_result.get("total_feedback", 0),
                            "step": global_step,
                        })
                    model.train()
            except Exception as e:
                if global_step % 5000 == 0:
                    logger.warning(f"RLHF update error: {e}")

        # Save
        if global_step > 0 and global_step % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                global_step, last_loss_value,
                output_dir / "latest.pt",
                config, best_val_loss,
                grad_accum=grad_accum,
            )

        if (global_step > 0
                and global_step % (save_every * 5) == 0):
            save_checkpoint(
                model, optimizer, scheduler,
                global_step, last_loss_value,
                output_dir / f"step_{global_step}.pt",
                config, best_val_loss,
                grad_accum=grad_accum,
            )
            cleanup_old_checkpoints(output_dir, keep=5)

        global_step += 1

    # Graceful shutdown
    puller.stop()
    logger.info(f"Saving final checkpoint at step {global_step}...")
    save_checkpoint(
        model, optimizer, scheduler, global_step,
        0.0, output_dir / "latest.pt", config, best_val_loss,
        grad_accum=grad_accum,
    )
    save_checkpoint(
        model, optimizer, scheduler, global_step,
        0.0, output_dir / "final.pt", config, best_val_loss,
        grad_accum=grad_accum,
    )
    logger.info(
        f"Training stopped at step {global_step}. "
        f"Resume anytime with: python run.py train"
    )


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, use_amp):
    """Evaluate geometry on validation set."""
    model.eval()
    total_loss = 0
    count = 0

    for batch in val_loader:
        text_ids = batch["text_ids"].to(device)
        text_mask = batch["text_mask"].to(device)
        mesh_tokens = batch["mesh_tokens"].to(device)

        input_tok = mesh_tokens[:, :-1]
        target_tok = mesh_tokens[:, 1:]

        amp_enabled = use_amp and device.type in ("cuda", "mps")
        _eval_mp = (
            os.environ.get("TRAIN_MIXED_PRECISION")
            or os.environ.get("MIXED_PRECISION")
            or os.environ.get("CLOUD_MIXED_PRECISION")
            or ""
        )
        _eval_bf16 = device.type == "cuda" and _eval_mp == "bf16"
        amp_dtype = (
            torch.bfloat16 if _eval_bf16
            else torch.float16 if device.type == "cuda"
            else torch.bfloat16
        )
        with autocast(
            device_type=device.type, dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model.forward_geometry(
                text_ids, text_mask, input_tok)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tok.reshape(-1),
            )

        total_loss += loss.item()
        count += 1
        if count >= 50:
            break

    return total_loss / max(count, 1)


# =====================================================================
# Legacy compat: generate_unified_data
# =====================================================================

def generate_unified_data(config, output_dir,
                          num_examples=50000,
                          render_images=True):
    """Generate a static dataset (for validation data).

    The new training loop generates on-the-fly and does not need
    this, but it is useful for creating val/test splits.
    """
    from processing.generate_synthetic import (
        SHAPE_SPECS, COMPOSITE_SPECS, generate_label,
        normalize_mesh, apply_rotation,
    )
    from processing.mesh_tokenizer import MeshTokenizer
    from processing.text_tokenizer import TextTokenizer

    renderer = None
    if render_images:
        try:
            from processing.render_shapes import render_and_encode
            renderer = render_and_encode
        except ImportError:
            pass

    tok_config = config.get("tokenization", {})
    tokenizer = MeshTokenizer(
        vocab_size=tok_config.get("vocab_size", 8192),
        coord_range=tuple(
            tok_config.get("coordinate_range", [-1.0, 1.0])),
        max_faces=tok_config.get("max_faces", 2048),
    )

    max_seq = config.get("unified", {}).get(
        "geometry", {}).get("max_seq_length", 16202)
    max_faces = (max_seq - 2) // 9

    all_specs = {**SHAPE_SPECS, **COMPOSITE_SPECS}
    all_shapes = list(all_specs.keys())
    image_size = config.get("unified", {}).get("image_size", 64)

    examples = []
    skipped = 0

    for i in range(num_examples):
        shape_key = random.choice(all_shapes)
        spec = all_specs[shape_key]

        try:
            params = spec["params"]()
            verts, faces = spec["generator"](params)
        except Exception:
            skipped += 1
            continue

        if random.random() < 0.5:
            verts = apply_rotation(
                verts, random.uniform(0, 360),
                random.choice(["x", "y", "z"]),
            )
        verts = normalize_mesh(verts, target_range=(-1.0, 1.0))

        if len(faces) > max_faces or len(faces) < 2:
            skipped += 1
            continue

        try:
            tokens = tokenizer.encode_mesh(verts, faces)
        except Exception:
            skipped += 1
            continue

        if len(tokens) > max_seq:
            skipped += 1
            continue

        label = generate_label(shape_key, params)
        example = {
            "text": label,
            "tokens": tokens,
            "num_faces": len(faces),
            "num_vertices": len(verts),
            "source": f"synthetic_{shape_key}",
        }

        if renderer and random.random() < 0.5:
            try:
                example["image"] = renderer(
                    verts, faces, size=image_size)
            except Exception:
                pass

        examples.append(example)

        if (i + 1) % 10000 == 0:
            logger.info(f"  {len(examples)}/{num_examples}")

    all_texts = [ex["text"] for ex in examples]
    from processing.bpe_tokenizer import BPETokenizer
    model_prefix = str(Path(output_dir) / "bpe_model")
    text_tokenizer = BPETokenizer.train(
        all_texts, vocab_size=8000, model_prefix=model_prefix)

    random.seed(42)
    random.shuffle(examples)
    n_train = int(len(examples) * 0.90)
    n_val = int(len(examples) * 0.05)

    splits = {
        "train": examples[:n_train],
        "val": examples[n_train:n_train + n_val],
        "test": examples[n_train + n_val:],
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in splits.items():
        out_path = out_dir / f"{name}.jsonl"
        with open(out_path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        n_img = sum(1 for ex in data if "image" in ex)
        logger.info(f"  {name}: {len(data)} ({n_img} with images)")

    # BPE tokenizer is already saved by .train() into bpe_tokenizer/
    # Also keep legacy format for backward compat
    try:
        from processing.text_tokenizer import TextTokenizer
        legacy_tok = TextTokenizer.from_texts(all_texts)
        legacy_tok.save(out_dir / "text_tokenizer.json")
    except Exception:
        pass
    return len(examples)


# =====================================================================
# Entry point
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Autonomous unified training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="checkpoints/unified")
    parser.add_argument(
        "--resume", default="latest",
        help="'latest' (default), path, or 'none'",
    )
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--num-examples", type=int, default=50000)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.generate_data:
        data_dir = config.get("data", {}).get(
            "geometry_dir", "data/datasets/geometry")
        generate_unified_data(
            config, data_dir,
            num_examples=args.num_examples,
            render_images=not args.no_images,
        )

    if args.resume == "none":
        args.resume = None

    train(config, args)


if __name__ == "__main__":
    main()
