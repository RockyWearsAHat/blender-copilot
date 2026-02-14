"""Modifier prediction model.

A smaller classification/regression model that predicts which Blender modifiers
to apply to a mesh and their parameter values, given:
  - Text description of the desired result
  - Mesh statistics (vertex count, face count, bounds, topology info)

This is NOT an autoregressive sequence model — it's a structured prediction
model since modifier stacks are short (typically 1-6 modifiers) and have
well-defined schemas.

Architecture:
    [Text Embedding] + [Mesh Stats] → Fusion MLP → Modifier Heads

Output:
    - num_modifiers: int (1-6)
    - For each modifier slot:
        - modifier_type: classification (SUBSURF, MIRROR, BEVEL, SOLIDIFY, ...)
        - parameters: regression values specific to each type
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Supported modifier types
MODIFIER_TYPES = [
    "NONE",          # Empty slot
    "SUBSURF",       # Subdivision Surface
    "MIRROR",        # Mirror
    "BEVEL",         # Bevel
    "SOLIDIFY",      # Solidify
    "ARRAY",         # Array
    "BOOLEAN",       # Boolean
    "SHRINKWRAP",    # Shrinkwrap
    "SMOOTH",        # Smooth / Laplacian Smooth
    "DECIMATE",      # Decimate
    "EDGE_SPLIT",    # Edge Split
    "WEIGHTED_NORMAL",  # Weighted Normal
    "SIMPLE_DEFORM", # Simple Deform (twist, bend, taper, stretch)
    "CAST",          # Cast
    "CURVE",         # Curve
    "DISPLACE",      # Displace
    "SKIN",          # Skin
    "REMESH",        # Remesh
    "WIREFRAME",     # Wireframe
    "WELD",          # Weld
]

MODIFIER_TYPE_TO_ID = {m: i for i, m in enumerate(MODIFIER_TYPES)}
NUM_MODIFIER_TYPES = len(MODIFIER_TYPES)

# Maximum number of modifiers in a stack
MAX_MODIFIERS = 8

# Number of parameter values per modifier (covers all types, padded)
# Each modifier type uses a subset of these slots
PARAMS_PER_MODIFIER = 12


class MeshStatsEncoder(nn.Module):
    """Encode mesh statistics into a fixed-size vector.

    Input features:
        - vertex_count (log-scaled)
        - face_count (log-scaled)
        - edge_count (log-scaled)
        - bounding box dimensions (width, depth, height)
        - surface area (log-scaled)
        - avg edge length
        - has_ngons (0/1)
        - has_quads_only (0/1)
        - has_tris_only (0/1)
        - is_manifold (0/1)
    """

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.GELU(),
            nn.Linear(64, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stats: (batch, 12) mesh statistics vector

        Returns:
            (batch, hidden_size)
        """
        return self.net(stats)


class TextEncoder(nn.Module):
    """Simple text encoder for modifier prediction.

    Uses learned embeddings + mean pooling.
    """

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 256,
                 max_length: int = 128, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_length, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, text_ids: torch.Tensor,
                text_mask: torch.Tensor) -> torch.Tensor:
        """Encode text and return pooled representation.

        Args:
            text_ids: (batch, seq_len)
            text_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            (batch, hidden_size) — mean-pooled text representation
        """
        B, T = text_ids.shape
        pos = torch.arange(T, device=text_ids.device).unsqueeze(0)
        x = self.embedding(text_ids) + self.pos_embedding(pos)
        x = self.encoder(x)

        # Masked mean pooling
        mask_expanded = text_mask.unsqueeze(-1)  # (B, T, 1)
        pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return pooled


class ModifierHead(nn.Module):
    """Prediction head for a single modifier slot.

    Predicts:
        - modifier type (classification)
        - parameters (regression, conditioned on predicted type)
    """

    def __init__(self, input_size: int = 512):
        super().__init__()

        # Type classifier
        self.type_head = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.GELU(),
            nn.Linear(128, NUM_MODIFIER_TYPES),
        )

        # Parameter regressors — one per modifier type
        # Each outputs PARAMS_PER_MODIFIER values
        self.param_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, 64),
                nn.GELU(),
                nn.Linear(64, PARAMS_PER_MODIFIER),
            )
            for _ in range(NUM_MODIFIER_TYPES)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_size) fused representation

        Returns:
            type_logits: (batch, NUM_MODIFIER_TYPES)
            params: (batch, PARAMS_PER_MODIFIER) — from the predicted type's head
        """
        type_logits = self.type_head(x)

        # During training, compute all param heads and select by target type
        # During inference, use argmax type
        all_params = torch.stack(
            [head(x) for head in self.param_heads], dim=1
        )  # (batch, NUM_MODIFIER_TYPES, PARAMS_PER_MODIFIER)

        return type_logits, all_params


class ModifierModel(nn.Module):
    """Complete modifier stack prediction model.

    Given text description + mesh statistics, predicts a modifier stack
    of up to MAX_MODIFIERS modifiers, each with type and parameters.

    This is a structured prediction model, NOT an autoregressive one.
    The modifier stack is predicted all at once.
    """

    def __init__(self, config: dict):
        super().__init__()

        mod_config = config.get("model", {}).get("modifiers", {})
        self.hidden_size = mod_config.get("hidden_size", 256)
        text_vocab = config.get("model", {}).get("text_encoder", {}).get(
            "vocab_size", 32000
        )

        # Encoders
        self.text_encoder = TextEncoder(
            vocab_size=text_vocab,
            hidden_size=self.hidden_size,
            max_length=128,
            num_layers=mod_config.get("num_layers", 4),
        )
        self.mesh_encoder = MeshStatsEncoder(hidden_size=self.hidden_size)

        # Fusion
        fused_size = self.hidden_size * 2
        self.fusion = nn.Sequential(
            nn.Linear(fused_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        # Number of modifiers head (1-MAX_MODIFIERS)
        self.count_head = nn.Linear(self.hidden_size, MAX_MODIFIERS)

        # Per-slot modifier heads
        self.modifier_heads = nn.ModuleList([
            ModifierHead(input_size=self.hidden_size)
            for _ in range(MAX_MODIFIERS)
        ])

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor,
                mesh_stats: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            text_ids: (batch, text_len)
            text_mask: (batch, text_len)
            mesh_stats: (batch, 12)

        Returns:
            dict with:
                count_logits: (batch, MAX_MODIFIERS)
                type_logits: list of (batch, NUM_MODIFIER_TYPES) per slot
                param_values: list of (batch, NUM_MODIFIER_TYPES, PARAMS_PER_MODIFIER)
        """
        text_vec = self.text_encoder(text_ids, text_mask)
        mesh_vec = self.mesh_encoder(mesh_stats)

        fused = self.fusion(torch.cat([text_vec, mesh_vec], dim=-1))

        count_logits = self.count_head(fused)

        type_logits_list = []
        param_values_list = []
        for head in self.modifier_heads:
            t_logits, params = head(fused)
            type_logits_list.append(t_logits)
            param_values_list.append(params)

        return {
            "count_logits": count_logits,
            "type_logits": type_logits_list,
            "param_values": param_values_list,
        }

    @torch.no_grad()
    def predict(self, text_ids: torch.Tensor, text_mask: torch.Tensor,
                mesh_stats: torch.Tensor) -> list[dict]:
        """Predict a modifier stack.

        Returns:
            List of modifier dicts, each with 'type' and 'params'.
        """
        self.eval()
        out = self.forward(text_ids, text_mask, mesh_stats)

        # Predict count
        count = out["count_logits"].argmax(dim=-1).item() + 1
        count = min(count, MAX_MODIFIERS)

        modifiers = []
        for i in range(count):
            type_id = out["type_logits"][i].argmax(dim=-1).item()
            if type_id == 0:  # NONE
                continue

            mod_type = MODIFIER_TYPES[type_id]
            params = out["param_values"][i][0, type_id].cpu().tolist()

            mod_dict = {
                "type": mod_type,
                "params": _decode_modifier_params(mod_type, params),
            }
            modifiers.append(mod_dict)

        return modifiers

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _decode_modifier_params(mod_type: str, raw_params: list[float]) -> dict:
    """Convert raw regression outputs to meaningful modifier parameters.

    Each modifier type uses the parameter slots differently.
    Values are passed through sigmoid/clamp as appropriate.
    """
    def sigmoid(x):
        return 1 / (1 + (-x).__class__(1).__class__.__bases__[0].__subclasses__()[0](1))  # avoid import

    import math

    def sig(x):
        return 1 / (1 + math.exp(-max(-10, min(10, x))))

    p = raw_params

    if mod_type == "SUBSURF":
        return {
            "levels": max(1, min(4, round(sig(p[0]) * 4))),
            "render_levels": max(1, min(6, round(sig(p[1]) * 6))),
            "use_creases": sig(p[2]) > 0.5,
        }
    elif mod_type == "MIRROR":
        return {
            "use_axis": [sig(p[0]) > 0.5, sig(p[1]) > 0.5, sig(p[2]) > 0.5],
            "use_clip": sig(p[3]) > 0.5,
            "merge_threshold": sig(p[4]) * 0.01,
        }
    elif mod_type == "BEVEL":
        return {
            "width": sig(p[0]) * 0.1,
            "segments": max(1, min(10, round(sig(p[1]) * 10))),
            "limit_method": "ANGLE" if sig(p[2]) > 0.5 else "NONE",
            "angle_limit": sig(p[3]) * 3.14159,
        }
    elif mod_type == "SOLIDIFY":
        return {
            "thickness": sig(p[0]) * 0.5,
            "offset": p[1] * 0.5,  # can be negative
            "use_even_offset": sig(p[2]) > 0.5,
        }
    elif mod_type == "ARRAY":
        return {
            "count": max(1, min(20, round(sig(p[0]) * 20))),
            "use_relative_offset": sig(p[1]) > 0.5,
            "relative_offset_displace": [p[2], p[3], p[4]],
        }
    elif mod_type == "SMOOTH":
        return {
            "factor": sig(p[0]),
            "iterations": max(1, min(20, round(sig(p[1]) * 20))),
        }
    elif mod_type == "DECIMATE":
        return {
            "ratio": max(0.01, min(1.0, sig(p[0]))),
            "decimate_type": "COLLAPSE",
        }
    elif mod_type == "EDGE_SPLIT":
        return {
            "split_angle": sig(p[0]) * 3.14159,
            "use_edge_angle": True,
        }
    elif mod_type == "WEIGHTED_NORMAL":
        return {
            "weight": max(1, min(100, round(sig(p[0]) * 100))),
            "keep_sharp": sig(p[1]) > 0.5,
        }
    elif mod_type == "SIMPLE_DEFORM":
        deform_types = ["TWIST", "BEND", "TAPER", "STRETCH"]
        idx = max(0, min(3, round(sig(p[0]) * 3)))
        return {
            "deform_method": deform_types[idx],
            "angle": p[1] * 3.14159,
            "factor": p[2],
        }
    elif mod_type == "REMESH":
        return {
            "mode": "VOXEL" if sig(p[0]) > 0.5 else "SMOOTH",
            "voxel_size": max(0.01, sig(p[1]) * 0.5),
            "octree_depth": max(1, min(8, round(sig(p[2]) * 8))),
        }
    else:
        return {"raw": raw_params[:6]}
