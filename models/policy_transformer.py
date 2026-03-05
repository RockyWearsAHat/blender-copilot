from __future__ import annotations

import torch
import torch.nn as nn


class PolicyTransformer(nn.Module):
    """Small transformer policy: (state_t sequence) -> action logits per step.

    Designed to satisfy ARCHITECTURE.md constraints:
    - compact numeric state
    - <= 6 layers
    - hidden size <= 512
    - seq_len <= 128
    """

    def __init__(
        self,
        *,
        state_dim: int = 10,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        action_type_vocab: int = 11,
        action_param_vocab: int = 32,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_size = int(hidden_size)
        self.max_seq_len = int(max_seq_len)

        self.state_proj = nn.Linear(self.state_dim, self.hidden_size)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.hidden_size)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(self.hidden_size)

        self.action_type_head = nn.Linear(self.hidden_size, action_type_vocab)
        self.action_param_head = nn.Linear(self.hidden_size, action_param_vocab)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
        states: (B, T, state_dim) float32 in [-1, 1]

        Returns:
        type_logits: (B, T, action_type_vocab)
        param_logits: (B, T, action_param_vocab)
        """
        bsz, seq_len, dim = states.shape
        if dim != self.state_dim:
            raise ValueError(f"state_dim mismatch: got {dim}, expected {self.state_dim}")
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} > max_seq_len {self.max_seq_len}")

        x = self.state_proj(states)
        pos = torch.arange(seq_len, device=states.device)
        x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        # Causal mask: prevent attending to future timesteps.
        # Shape: (T, T) where mask[i, j] = -inf for j > i.
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=states.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        x = self.encoder(x, mask=causal_mask)
        x = self.norm(x)

        type_logits = self.action_type_head(x)
        param_logits = self.action_param_head(x)
        return type_logits, param_logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
