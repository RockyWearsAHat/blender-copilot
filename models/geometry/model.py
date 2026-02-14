"""Geometry Model — autoregressive mesh generation.

Takes a text prompt and generates a tokenized mesh sequence.
Architecture inspired by MeshGPT: a transformer decoder that
produces mesh face tokens autoregressively.

The text prompt is encoded via a frozen text encoder, then
the transformer generates face tokens conditioned on that encoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """Simple text encoder using learned embeddings.

    For v1 we use a trainable embedding + transformer encoder.
    Can be swapped for a frozen CLIP/BERT encoder later.
    """

    def __init__(self, vocab_size: int = 32000, embed_dim: int = 1024,
                 max_length: int = 256, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,  # Required for MPS compatibility
        )
        self.max_length = max_length

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode text tokens to conditioning vectors.

        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: (batch, seq_len) mask

        Returns:
            (batch, seq_len, embed_dim) encoded text
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)

        if attention_mask is not None:
            # TransformerEncoder expects src_key_padding_mask as (batch, seq_len)
            # where True means "ignore this position"
            padding_mask = ~attention_mask.bool()
        else:
            padding_mask = None

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x


class MeshTransformer(nn.Module):
    """Autoregressive transformer for mesh token generation.

    Generates mesh face tokens one at a time, conditioned on:
    1. Text encoding (cross-attention)
    2. Previously generated mesh tokens (causal self-attention)
    """

    def __init__(self, mesh_vocab_size: int = 8192,
                 hidden_size: int = 1024, num_layers: int = 24,
                 num_heads: int = 16, max_seq_length: int = 18432,
                 text_embed_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.mesh_vocab_size = mesh_vocab_size

        # Mesh token embeddings
        self.mesh_embed = nn.Embedding(mesh_vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_length, hidden_size)

        # Text conditioning projection
        self.text_proj = nn.Linear(text_embed_dim, hidden_size)

        # Transformer decoder layers with cross-attention
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, mesh_vocab_size)

        # Causal mask cache
        self._causal_mask = None

    def forward(self, mesh_tokens: torch.Tensor,
                text_encoding: torch.Tensor,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass — returns logits for next token prediction.

        Args:
            mesh_tokens: (batch, seq_len) mesh token indices
            text_encoding: (batch, text_len, hidden_size) text features
            text_mask: (batch, text_len) attention mask for text

        Returns:
            (batch, seq_len, mesh_vocab_size) next-token logits
        """
        batch, seq_len = mesh_tokens.shape
        device = mesh_tokens.device

        # Embed mesh tokens + positional
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.mesh_embed(mesh_tokens) + self.pos_embed(positions)

        # Project text encoding to model dimension
        text_cond = self.text_proj(text_encoding)

        # Causal mask
        causal_mask = self._get_causal_mask(seq_len, device)

        # Convert text_mask for cross-attention
        cross_mask = None
        if text_mask is not None:
            cross_mask = ~text_mask.bool()

        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, text_cond, causal_mask=causal_mask,
                      cross_mask=cross_mask)

        x = self.norm(x)
        logits = self.output_proj(x)
        return logits

    def _get_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask.size(0) < size:
            mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
            self._causal_mask = mask.bool()
        return self._causal_mask[:size, :size]

    @torch.no_grad()
    def generate(self, text_encoding: torch.Tensor,
                 text_mask: torch.Tensor | None = None,
                 max_tokens: int = 2048 * 9 + 2,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 bos_token: int = 1,
                 eos_token: int = 2) -> torch.Tensor:
        """Autoregressively generate mesh tokens.

        Args:
            text_encoding: (1, text_len, hidden_size) — batch size 1
            max_tokens: maximum sequence length
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold

        Returns:
            (1, generated_len) mesh token sequence
        """
        device = text_encoding.device
        generated = torch.tensor([[bos_token]], device=device, dtype=torch.long)

        text_cond = self.text_proj(text_encoding)

        for step in range(max_tokens - 1):
            seq_len = generated.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            x = self.mesh_embed(generated) + self.pos_embed(positions)

            causal_mask = self._get_causal_mask(seq_len, device)
            cross_mask = ~text_mask.bool() if text_mask is not None else None

            for layer in self.layers:
                x = layer(x, text_cond, causal_mask=causal_mask,
                          cross_mask=cross_mask)

            x = self.norm(x)
            logits = self.output_proj(x[:, -1, :])  # Last position

            # Temperature scaling
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, top_k)
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token:
                break

        return generated


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attn + cross-attn."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, text_cond: torch.Tensor,
                causal_mask: torch.Tensor | None = None,
                cross_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention (causal)
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.dropout(x) + residual

        # Cross-attention (attend to text)
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(x, text_cond, text_cond,
                                key_padding_mask=cross_mask)
        x = self.dropout(x) + residual

        # Feedforward
        residual = x
        x = self.norm3(x)
        x = self.ff(x) + residual

        return x


class GeometryModel(nn.Module):
    """Complete text-to-mesh generation model.

    Combines text encoder + mesh transformer decoder.
    """

    def __init__(self, config: dict):
        super().__init__()

        geo_config = config.get("models", {}).get("geometry", {})
        tok_config = config.get("tokenization", {})

        hidden_size = geo_config.get("hidden_size", 1024)
        num_layers = geo_config.get("num_layers", 24)
        num_heads = geo_config.get("num_heads", 16)
        mesh_vocab = tok_config.get("vocab_size", 8192)
        max_seq = geo_config.get("max_sequence_length", 18432)
        dropout = geo_config.get("dropout", 0.1)

        self.text_encoder = TextEncoder(
            embed_dim=hidden_size,
            num_layers=4,
            num_heads=min(8, num_heads),
        )

        self.mesh_decoder = MeshTransformer(
            mesh_vocab_size=mesh_vocab,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_length=max_seq,
            text_embed_dim=hidden_size,
            dropout=dropout,
        )

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor,
                mesh_tokens: torch.Tensor) -> torch.Tensor:
        """Training forward pass.

        Returns logits for next-token prediction on mesh sequence.
        """
        text_encoding = self.text_encoder(text_ids, text_mask)
        logits = self.mesh_decoder(mesh_tokens, text_encoding, text_mask)
        return logits

    @torch.no_grad()
    def generate(self, text_ids: torch.Tensor,
                 text_mask: torch.Tensor | None = None,
                 **kwargs) -> torch.Tensor:
        """Generate a mesh from text."""
        text_encoding = self.text_encoder(text_ids, text_mask)
        return self.mesh_decoder.generate(
            text_encoding, text_mask, **kwargs)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
