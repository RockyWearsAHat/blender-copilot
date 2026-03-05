"""Geometric distribution scoring for mesh training quality.

Instead of exact geometry matching, we score meshes by how well their
statistical geometric distributions match their class prototype.

Five distributions capture shape "form":
  1. bbox_ratios       — normalized W:H:D proportions (is it wide? tall? cubic?)
  2. height_profile    — histogram of vertex Y positions (flat base, peaked dome?)
  3. radial_profile    — histogram of distances from centroid (hollow? solid? shell?)
  4. angular_profile   — histogram of XZ angles (symmetric? directional?)
  5. vertical_sections — radial spread per height slice (waist/shoulder/base)

These are aggregated per object class via exponential moving average to build
"prototype" distributions. A new mesh is scored by Bhattacharyya similarity to
the prototype — well-formed, representative meshes get high scores; outliers
and primitive blobs get low scores.

The class prototypes improve during training: as more meshes are seen, the
prototype converges to a stable statistical description of "what a car looks
like geometrically" — not memorised coordinates, but learned distributions.
"""

from __future__ import annotations

import math
import threading
from typing import Optional
import numpy as np


N_BINS = 16       # histogram bins per distribution
N_VSECTIONS = 8   # vertical sections for radial-per-height
EMA_DECAY = 0.97  # prototype update decay (higher = slower to update)
MIN_COUNT_TO_SCORE = 4  # need at least this many meshes before scoring


# ── Core signature computation ────────────────────────────────────────────────

def compute_signature(verts: list | np.ndarray) -> dict:
    """Compute a 5-distribution geometric signature for a mesh.

    Args:
        verts: list of [x, y, z] vertex positions (already normalised to [-1,1])

    Returns:
        dict with keys: bbox_ratios, height_profile, radial_profile,
                        angular_profile, section_radii
        Returns None if verts is empty or degenerate.
    """
    if not verts:
        return None

    v = np.asarray(verts, dtype=np.float32)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4:
        return None

    mn, mx = v.min(0), v.max(0)
    dims = (mx - mn).clip(1e-6)
    max_dim = dims.max()

    # 1. Bounding-box ratios (all relative to longest axis → [0,1])
    bbox_ratios = (dims / max_dim).tolist()   # [x_ratio, y_ratio, z_ratio]

    # Centroid
    centroid = v.mean(0)

    # 2. Height profile: histogram of normalised Y position
    y_norm = (v[:, 1] - mn[1]) / dims[1]
    h_hist, _ = np.histogram(y_norm, bins=N_BINS, range=(0.0, 1.0))
    h_hist = _norm_hist(h_hist)

    # 3. Radial profile: histogram of distance from centroid
    centered = v - centroid
    radii = np.linalg.norm(centered, axis=1)
    r_max = radii.max().clip(1e-6)
    r_norm = radii / r_max
    r_hist, _ = np.histogram(r_norm, bins=N_BINS, range=(0.0, 1.0))
    r_hist = _norm_hist(r_hist)

    # 4. XZ angular profile: histogram of angle around Y axis
    angles = np.arctan2(centered[:, 2], centered[:, 0])   # -π … π
    a_hist, _ = np.histogram(angles, bins=N_BINS, range=(-math.pi, math.pi))
    a_hist = _norm_hist(a_hist)

    # 5. Section radii: average XZ distance from centroid per height slice
    #    Captures "is the mesh wide at the bottom and narrow at the top?" etc.
    section_radii = []
    for i in range(N_VSECTIONS):
        lo = i / N_VSECTIONS
        hi = (i + 1) / N_VSECTIONS
        mask = (y_norm >= lo) & (y_norm < hi)
        if mask.sum() == 0:
            section_radii.append(0.0)
        else:
            xz_radii = np.linalg.norm(centered[mask][:, [0, 2]], axis=1)
            section_radii.append(float(xz_radii.mean() / r_max))

    return {
        "bbox_ratios":     bbox_ratios,
        "height_profile":  h_hist.tolist(),
        "radial_profile":  r_hist.tolist(),
        "angular_profile": a_hist.tolist(),
        "section_radii":   section_radii,
    }


def signature_similarity(sig_a: dict, sig_b: dict) -> float:
    """Compute [0, 1] similarity between two geometric signatures.

    Uses Bhattacharyya coefficient for histograms (ideal for comparing
    probability distributions) and L1 distance for bbox ratios and
    section radii.

    1.0 = identical distributions, 0.0 = maximally different.
    """
    scores = []

    # Histogram distributions — Bhattacharyya coefficient
    for key in ("height_profile", "radial_profile", "angular_profile"):
        a = sig_a.get(key)
        b = sig_b.get(key)
        if a and b:
            scores.append(_bhattacharyya(a, b))

    # Bbox ratios — 1 - normalised L1 distance
    br_a = sig_a.get("bbox_ratios")
    br_b = sig_b.get("bbox_ratios")
    if br_a and br_b:
        aa, bb = np.array(br_a[:3]), np.array(br_b[:3])
        scores.append(float(1.0 - np.abs(aa - bb).mean()))

    # Section radii — Bhattacharyya-like on small vector
    sr_a = sig_a.get("section_radii")
    sr_b = sig_b.get("section_radii")
    if sr_a and sr_b:
        scores.append(_bhattacharyya(sr_a, sr_b))

    return float(np.mean(scores)) if scores else 0.5


def shape_descriptor_tokens(sig: dict) -> str:
    """Convert a signature into compact text tokens for label augmentation.

    Example output: " [wide:0.9 tall:0.3 solid:c bell:h sym:0.7]"

    These tokens are appended to the text prompt so the model learns
    to associate geometric distributions with shape words.
    """
    if not sig:
        return ""

    parts = []

    # Bbox proportions
    br = sig.get("bbox_ratios", [1, 1, 1])
    parts.append(f"wide:{br[0]:.1f}")
    parts.append(f"tall:{br[1]:.1f}")
    parts.append(f"deep:{br[2]:.1f}")

    # Radial profile character: shell (peaks at edge), solid (peaks at center),
    # uniform, or ring (peaks at mid)
    rp = sig.get("radial_profile", [])
    if rp:
        rp_arr = np.array(rp)
        peak = int(rp_arr.argmax())
        if peak < N_BINS // 4:
            char = "solid"
        elif peak > 3 * N_BINS // 4:
            char = "shell"
        elif N_BINS // 4 <= peak <= 3 * N_BINS // 4:
            char = "mid"
        else:
            char = "ring"
        parts.append(f"fill:{char}")

    # Height profile character: flat (uniform), bell (mid-peak), base-heavy, top-heavy
    hp = sig.get("height_profile", [])
    if hp:
        hp_arr = np.array(hp)
        peak = int(hp_arr.argmax())
        if hp_arr.std() < 0.05:
            char = "flat"
        elif peak < N_BINS // 4:
            char = "base"
        elif peak > 3 * N_BINS // 4:
            char = "top"
        else:
            char = "bell"
        parts.append(f"hpeak:{char}")

    # Angular symmetry score: how uniform is the XZ distribution?
    ap = sig.get("angular_profile", [])
    if ap:
        ap_arr = np.array(ap)
        uniformity = 1.0 - float(ap_arr.std())
        parts.append(f"sym:{uniformity:.1f}")

    if not parts:
        return ""
    return " [" + " ".join(parts) + "]"


# ── Prototype registry ─────────────────────────────────────────────────────────

class ShapePrototypeRegistry:
    """Maintains per-category geometric distribution prototypes via EMA.

    As training processes meshes, each object's signature is folded into
    the running prototype for its category. New meshes are then scored
    against the accumulated prototype — meshes that look like "typical"
    members of their category get high scores.

    This lets the system learn what geometry distributions correspond to
    object classes without memorising specific meshes.
    """

    def __init__(self, decay: float = EMA_DECAY, min_count: int = MIN_COUNT_TO_SCORE):
        self.decay = decay
        self.min_count = min_count
        self._prototypes: dict[str, dict] = {}   # category → avg signature
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def update(self, label: str, sig: dict) -> None:
        """Update prototype for label's category with this signature."""
        if not sig or not label:
            return
        cat = _categorize(label)
        with self._lock:
            if cat not in self._prototypes:
                self._prototypes[cat] = {k: list(v) for k, v in sig.items()}
                self._counts[cat] = 1
            else:
                _ema_update(self._prototypes[cat], sig, self.decay)
                self._counts[cat] += 1

    def score(self, label: str, sig: dict) -> float:
        """Score how closely sig matches the class prototype.

        Returns 0.5 if prototype is not yet established (not enough data).
        """
        if not sig or not label:
            return 0.5
        cat = _categorize(label)
        with self._lock:
            count = self._counts.get(cat, 0)
            proto = self._prototypes.get(cat)
        if count < self.min_count or proto is None:
            return 0.5
        return signature_similarity(sig, proto)

    def category_count(self, label: str) -> int:
        cat = _categorize(label)
        return self._counts.get(cat, 0)

    def num_categories(self) -> int:
        return len(self._prototypes)

    def get_prototype(self, label: str) -> Optional[dict]:
        cat = _categorize(label)
        with self._lock:
            proto = self._prototypes.get(cat)
            return {k: list(v) for k, v in proto.items()} if proto else None


# ── Module-level singleton registry ──────────────────────────────────────────

_GLOBAL_REGISTRY: Optional[ShapePrototypeRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_global_registry() -> ShapePrototypeRegistry:
    """Return (or create) the module-level prototype registry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        with _REGISTRY_LOCK:
            if _GLOBAL_REGISTRY is None:
                _GLOBAL_REGISTRY = ShapePrototypeRegistry()
    return _GLOBAL_REGISTRY


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm_hist(h: np.ndarray) -> np.ndarray:
    """Normalise a count histogram to sum to 1."""
    s = h.sum()
    if s < 1e-9:
        return np.ones_like(h, dtype=np.float32) / len(h)
    return (h / s).astype(np.float32)


def _bhattacharyya(h1, h2) -> float:
    """Bhattacharyya coefficient for two histograms (both sum to 1 ideally)."""
    a = np.asarray(h1, dtype=np.float32).clip(0)
    b = np.asarray(h2, dtype=np.float32).clip(0)
    # Re-normalise in case inputs aren't perfectly normalised
    a = a / a.sum().clip(1e-9)
    b = b / b.sum().clip(1e-9)
    return float(np.sqrt(a * b).sum())


def _ema_update(proto: dict, sig: dict, decay: float) -> None:
    """In-place EMA update of proto with values from sig."""
    for key, val in sig.items():
        if key not in proto:
            proto[key] = list(val)
            continue
        old = np.asarray(proto[key], dtype=np.float32)
        new = np.asarray(val, dtype=np.float32)
        if old.shape == new.shape:
            proto[key] = (decay * old + (1.0 - decay) * new).tolist()


def _categorize(label: str) -> str:
    """Extract coarse category token from a label string."""
    if not label:
        return "unknown"
    # Labels often look like "car: sports car" or "3d character: rpg mage"
    cat = label.split(":")[0].strip().lower()
    # Collapse synonyms
    _syn = {
        "automobile": "car", "vehicle": "car", "truck": "car",
        "person": "character", "human": "character", "figure": "character",
        "house": "building", "structure": "building",
        "tree": "plant", "bush": "plant",
        "weapon": "weapon", "sword": "weapon", "gun": "weapon",
    }
    for k, v in _syn.items():
        if k in cat:
            return v
    # Truncate to first 20 chars for stability
    return cat[:20]
