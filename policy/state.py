from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Text feature hashing
# ---------------------------------------------------------------------------
# Dimensionality of the hashed text feature vector.  Kept small deliberately:
# the policy model learns what these features *mean* from training data.
# This function encodes NO knowledge about objects — it is purely a stable,
# unique fingerprint so the model can learn per-prompt behaviour from examples.
PROMPT_FEAT_DIM: int = 16


def hash_text_features(text: str, dim: int = PROMPT_FEAT_DIM) -> np.ndarray:
    """Character n-gram feature hashing → compact L2-normalised float32 vector.

    Fully deterministic (SHA-256, not Python's built-in hash()).
    Zero parameters — all semantic associations (what a "lamborghini" looks
    like, how detailed a "dragon" should be, etc.) are learned by the policy
    from training data, NOT hardcoded here.

        Steps:
            1. Lowercase + strip.
            2. Char 1-, 2-, 3-, 4-grams.
            3. Each n-gram hashed via SHA-256 → bucket in [0, dim).
            4. Signed feature hashing (±1) to reduce collisions.
            5. L2-normalise.
    """
    vec = np.zeros(dim, dtype=np.float32)
    t = (text or "").lower().strip()
    if not t:
        return vec
    for n in (1, 2, 3, 4):
        for i in range(len(t) - n + 1):
            gram = t[i : i + n].encode("utf-8")
            digest = hashlib.sha256(gram).digest()
            bucket = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            vec[bucket] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# Compact mesh state  (10 features, no raw geometry)
# ---------------------------------------------------------------------------

@dataclass
class MeshState:
    vertex_count: int
    face_count: int
    edge_count: int
    bbox_x: float
    bbox_y: float
    bbox_z: float
    avg_edge_length: float
    manifold_flag: float
    symmetry_score: float
    selected_face_count: int

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                float(self.vertex_count),
                float(self.face_count),
                float(self.edge_count),
                float(self.bbox_x),
                float(self.bbox_y),
                float(self.bbox_z),
                float(self.avg_edge_length),
                float(self.manifold_flag),
                float(self.symmetry_score),
                float(self.selected_face_count),
            ],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------

@dataclass
class Goal:
    # Explicit numeric targets — only set when the user literally typed them
    # in the prompt (e.g. "2k verts", "symmetry 0.8").  Both default to 0
    # meaning "unconstrained"; the model decides from text_features alone.
    target_vertex_count: int = 0
    target_symmetry: float = 0.0

    # Compact text fingerprint (PROMPT_FEAT_DIM floats, L2-normalised).
    # ALL semantic knowledge — what a cube looks like, how many verts a
    # lamborghini needs, whether a cube should be subdivided — is learned by
    # the policy from training data via this fingerprint.  Nothing is
    # hardcoded here.
    text_features: np.ndarray = field(
        default_factory=lambda: np.zeros(PROMPT_FEAT_DIM, dtype=np.float32)
    )


# Total goal feature dim fed to the policy.
# Layout: [vert_norm | sym_norm | text_features × PROMPT_FEAT_DIM]
GOAL_DIM: int = 2 + PROMPT_FEAT_DIM   # = 18


def normalize_goal(goal: Goal) -> np.ndarray:
    """18-feature goal vector concatenated to the mesh-stat state.

    Encodes ONLY what the user explicitly specified plus the prompt
    fingerprint.  All geometry/material/subdivision decisions are learned
    by the policy model from training data — not hardcoded here.
    """
    # Numeric targets are naturally in [0, 1] after scaling.
    scalar_feats = np.array(
        [
            float(goal.target_vertex_count) / 20000.0,
            float(goal.target_symmetry),
        ],
        dtype=np.float32,
    )
    scalar_feats = np.clip(scalar_feats, 0.0, 1.0)

    # Prompt fingerprints use signed feature hashing; preserve sign.
    text_feats = goal.text_features.astype(np.float32, copy=False)
    text_feats = np.clip(text_feats, -1.0, 1.0)

    return np.concatenate([scalar_feats, text_feats], axis=0)


def normalize_state_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize mesh-stat vector to [0, 1] with hand-tuned scales."""
    v = vec.astype(np.float32, copy=False)
    scales = np.array(
        [
            20000.0,  # vertex_count
            20000.0,  # face_count
            40000.0,  # edge_count
            5.0,      # bbox_x
            5.0,      # bbox_y
            5.0,      # bbox_z
            1.0,      # avg_edge_length
            1.0,      # manifold_flag
            1.0,      # symmetry_score
            10.0,     # selected_face_count (keep this sensitive: 0/1 should matter)
        ],
        dtype=np.float32,
    )
    v = v / scales
    v = np.clip(v, 0.0, 1.0)
    return v
