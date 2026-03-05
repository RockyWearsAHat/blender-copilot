from __future__ import annotations

import torch
import torch.nn as nn


class NoiseIntentPredictor(nn.Module):
    """Tiny intent -> procedural-noise bucket predictor.

    Input: compact hashed text features (optionally with extra scalar context).
    Output heads (bucket logits):
      - scale
      - detail
      - roughness
      - strength
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bins: int = 32,
    ):
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if bins <= 1:
            raise ValueError("bins must be > 1")

        layers: list[nn.Module] = []
        prev = int(in_dim)
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(prev, int(hidden_dim)))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(float(dropout)))
            prev = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)

        self.scale_head = nn.Linear(prev, int(bins))
        self.detail_head = nn.Linear(prev, int(bins))
        self.roughness_head = nn.Linear(prev, int(bins))
        self.strength_head = nn.Linear(prev, int(bins))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return (
            self.scale_head(h),
            self.detail_head(h),
            self.roughness_head(h),
            self.strength_head(h),
        )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
