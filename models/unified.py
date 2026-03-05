"""Unified Blender Copilot Model — text → geometry/materials/modifiers.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
    │  │ Text Enc │   │  Image Enc   │   │ Contrastive    │  │
    │  │ (words)  │──▶│ (CNN/patch)  │──▶│ Alignment      │  │
    │  └────┬─────┘   └──────┬───────┘   │ (CLIP-style)   │  │
    │       │                │           └────────────────┘  │
    │       └────────┬───────┘                                │
    │                │ fused embedding                        │
    │       ┌────────┼────────┬──────────┐                    │
    │       ▼        ▼        ▼          ▼                    │
    │  ┌─────────┐ ┌──────┐ ┌────────┐ ┌──────────────────┐  │
    │  │Geometry │ │Mater-│ │Modifier│ │ Image-Word       │  │
    │  │Decoder  │ │ials  │ │Head    │ │ Grounding Head   │  │
    │  │(autoreg)│ │Decode│ │(struct)│ │ (contrastive)    │  │
    │  └─────────┘ └──────┘ └────────┘ └──────────────────┘  │
    └─────────────────────────────────────────────────────────┘

Optional heads (materials, modifiers, contrastive/image) are gated by
config flags: enable_materials, enable_modifiers, enable_contrastive.
Set to False on memory-constrained devices (MPS/laptop) to save ~15M
params of GPU memory for longer geometry sequences.

Two-brain architecture at inference time:
  - Qwen 2.5 Coder (via ollama) = reasoning brain (tool-calling LLM)
  - This model (~70-85M params) = mesh brain (generates actual 3D geometry)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

try:
    from transformers import CLIPVisionModel
except Exception:
    CLIPVisionModel = None


# ─── Rotary Position Embeddings (RoPE) ───────────────────────────────

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) — Su et al. 2021.

    Used by LLaMA, Mistral, Qwen, and most modern transformers.
    Unlike learned positional embeddings, RoPE:
      - Has NO maximum sequence length (works at any length)
      - Encodes relative position (better for long sequences)
      - Generalizes to longer sequences than trained on
      - Adds zero parameters to the model

    Applied to Q and K tensors before attention computation.
    """

    def __init__(self, dim: int, max_cached_len: int = 32768, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        # Pre-compute inverse frequencies (not a parameter — constant)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Cache cos/sin for common lengths to avoid recomputation
        self._cached_len = 0
        self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("_sin_cached", torch.empty(0), persistent=False)
        self._build_cache(max_cached_len)

    def _build_cache(self, seq_len: int):
        """Pre-compute cos/sin tables up to seq_len."""
        if seq_len <= self._cached_len:
            return
        inv_freq: torch.Tensor = self.inv_freq  # type: ignore[assignment]
        t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)  # (seq_len, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self._cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        self._sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)
        self._cached_len = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """Apply rotary embeddings to Q and K.

        Args:
            q: (B, heads, seq_len, head_dim)
            k: (B, heads, seq_len_k, head_dim)
            offset: position offset for cached inference (KV-cache step)

        Returns:
            (q_rotated, k_rotated) with same shapes
        """
        seq_len = max(q.shape[2] + offset, k.shape[2] + offset)
        self._build_cache(seq_len)

        cos_cache: torch.Tensor = self._cos_cached  # type: ignore[assignment]
        sin_cache: torch.Tensor = self._sin_cached  # type: ignore[assignment]

        cos = cos_cache[:, :, offset:offset + q.shape[2], :q.shape[-1]].to(q.device, q.dtype)
        sin = sin_cache[:, :, offset:offset + q.shape[2], :q.shape[-1]].to(q.device, q.dtype)
        q_rot = (q * cos) + (_rotate_half(q) * sin)

        cos_k = cos_cache[:, :, :k.shape[2], :k.shape[-1]].to(k.device, k.dtype)
        sin_k = sin_cache[:, :, :k.shape[2], :k.shape[-1]].to(k.device, k.dtype)
        k_rot = (k * cos_k) + (_rotate_half(k) * sin_k)

        return q_rot, k_rot


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of x for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RopeTextEncoderLayer(nn.Module):
    """Transformer encoder layer with RoPE self-attention."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
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
        self.dropout = nn.Dropout(dropout)

    def _proj_qkv(self, x):
        d = self.self_attn.embed_dim
        w, b = self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
        q = F.linear(x, w[:d], b[:d] if b is not None else None)
        k = F.linear(x, w[d:2*d], b[d:2*d] if b is not None else None)
        v = F.linear(x, w[2*d:], b[2*d:] if b is not None else None)
        return q, k, v

    def forward(self, x, padding_mask=None, rope=None):
        residual = x
        x_n = self.norm1(x)
        q, k, v = self._proj_qkv(x_n)

        nh = self.self_attn.num_heads
        hd = self.self_attn.head_dim
        bsz, seq_len = x_n.shape[:2]

        q = q.view(bsz, seq_len, nh, hd).transpose(1, 2)
        k = k.view(bsz, seq_len, nh, hd).transpose(1, 2)
        v = v.view(bsz, seq_len, nh, hd).transpose(1, 2)

        if rope is not None:
            q, k = rope(q, k)

        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, nh, seq_len, -1)
            attn_mask = torch.where(attn_mask, float('-inf'), 0.0)

        x_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x_attn = x_attn.transpose(1, 2).contiguous().view(bsz, seq_len, nh * hd)
        x_attn = self.self_attn.out_proj(x_attn)
        x = residual + self.dropout(x_attn)

        residual = x
        x = self.norm2(x)
        x = residual + self.ff(x)
        return x


# ─── Shared Text Encoder ──────────────────────────────────────────────

class SharedTextEncoder(nn.Module):
    """Text encoder shared across all tasks.

    Produces both sequence-level outputs (for decoders via cross-attention)
    and a pooled vector (for contrastive alignment and modifier prediction).
    """

    def __init__(self, vocab_size: int = 4096, embed_dim: int = 512,
                 max_length: int = 256, num_layers: int = 4, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        head_dim = embed_dim // num_heads
        self.rope = RotaryPositionEmbedding(head_dim)
        self.layers = nn.ModuleList([
            RopeTextEncoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Projection for contrastive alignment (text → shared space)
        self.contrastive_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        """
        Returns:
            seq_out:  (B, T, D) — full sequence for cross-attention
            pooled:   (B, D)   — mean-pooled for classification/contrastive
        """
        x = self.dropout(self.embed(input_ids))

        padding_mask = None
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()

        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask, rope=self.rope)
        x = self.norm(x)

        # Masked mean pooling
        if attention_mask is not None:
            mask_exp = attention_mask.unsqueeze(-1)  # (B, T, 1)
            pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        return x, pooled


# ─── Image Encoder (lightweight for visual grounding) ─────────────────

class ImageEncoder(nn.Module):
    """CLIP-ViT image encoder (with local CNN fallback).

    Uses pretrained CLIP vision features when available. This provides
    strong visual semantics for contrastive/image→mesh tasks while keeping
    the rest of the architecture unchanged.
    """

    _CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, embed_dim: int = 512, spatial: bool = False):
        super().__init__()
        self._spatial = spatial
        self._use_clip = False

        hidden_size = 256
        if CLIPVisionModel is not None:
            try:
                self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
                self._use_clip = True
                hidden_size = int(self.clip.config.hidden_size)
                # Freeze backbone initially; downstream heads stay trainable.
                for param in self.clip.parameters():
                    param.requires_grad = False
            except Exception:
                self.clip = None
        else:
            self.clip = None

        if not self._use_clip:
            self._backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.GELU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.GELU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.GELU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.GELU(),
                nn.BatchNorm2d(256),
            )
            self.features = nn.Sequential(self._backbone, nn.AdaptiveAvgPool2d(1))

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.spatial_proj = None
        if spatial:
            self.spatial_proj = nn.Sequential(
                nn.Linear(hidden_size, embed_dim),
                nn.LayerNorm(embed_dim),
            )

    def _to_clip_pixels(self, images: torch.Tensor) -> torch.Tensor:
        """Convert possibly-ImageNet-normalized tensors to CLIP-normalized 224x224."""
        x = images
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # If values look normalized around 0 with negatives, map back to [0,1].
        if float(x.min()) < -0.1:
            mean = torch.tensor(self._IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            std = torch.tensor(self._IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
            x = x * std + mean

        x = x.clamp(0.0, 1.0)
        clip_mean = torch.tensor(self._CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        clip_std = torch.tensor(self._CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - clip_mean) / clip_std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 64, 64) normalized RGB images

        Returns:
            (B, embed_dim) image embeddings
        """
        if self._use_clip and self.clip is not None:
            px = self._to_clip_pixels(images)
            out = self.clip(pixel_values=px)
            pooled = out.pooler_output
            if pooled is None:
                pooled = out.last_hidden_state[:, 0, :]
            return self.proj(pooled)

        x = self.features(images)
        return self.proj(x)

    def encode_spatial(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into a sequence of spatial patch features.

        Args:
            images: (B, 3, 64, 64) normalized RGB images

        Returns:
            (B, 16, embed_dim) — 16 patch tokens (4×4 spatial grid)

        Raises:
            RuntimeError: if spatial projection was not enabled.
        """
        if self.spatial_proj is None:
            raise RuntimeError(
                "ImageEncoder spatial projection not enabled. "
                "Set enable_image_to_mesh=true in config.")
        if self._use_clip and self.clip is not None:
            px = self._to_clip_pixels(images)
            out = self.clip(pixel_values=px)
            tokens = out.last_hidden_state[:, 1:, :]  # drop CLS
            bsz, n_tokens, dim = tokens.shape
            side = int(math.sqrt(n_tokens))
            if side * side != n_tokens:
                side = 7
                tokens = tokens[:, :side * side, :]
            feat = tokens.permute(0, 2, 1).reshape(bsz, dim, side, side)
            feat = F.adaptive_avg_pool2d(feat, (4, 4))
            patches = feat.reshape(bsz, dim, 16).permute(0, 2, 1)
            return self.spatial_proj(patches)

        feat = self._backbone(images)        # (B, 256, 4, 4)
        bsz = feat.shape[0]
        patches = feat.reshape(bsz, 256, 16).permute(0, 2, 1)  # (B, 16, 256)
        return self.spatial_proj(patches)    # (B, 16, embed_dim)


# ─── Contrastive Alignment (CLIP-style) ──────────────────────────────

class ContrastiveHead(nn.Module):
    """CLIP-style contrastive alignment between text and image embeddings.

    Learns a shared embedding space where text descriptions of shapes
    are close to rendered images of those shapes.
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.text_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.image_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, text_embeds: torch.Tensor,
                image_embeds: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            text_embeds:  (B, D) from text encoder pooled output
            image_embeds: (B, D) from image encoder

        Returns:
            Scalar contrastive loss
        """
        # Project to shared space
        t = F.normalize(self.text_proj(text_embeds), dim=-1)
        i = F.normalize(self.image_proj(image_embeds), dim=-1)

        # Cosine similarity matrix scaled by learned temperature
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * t @ i.t()  # (B, B)

        # Symmetric cross-entropy loss
        labels = torch.arange(len(logits), device=logits.device)
        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.t(), labels)

        return (loss_t2i + loss_i2t) / 2


# ─── Geometry Decoder (autoregressive mesh tokens) ────────────────────

class GeometryDecoderLayer(nn.Module):
    """Transformer decoder layer with causal self-attn + text cross-attn."""

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

    def _proj_qkv(self, mha, x):
        d = mha.embed_dim
        w, b = mha.in_proj_weight, mha.in_proj_bias
        q = F.linear(x, w[:d], b[:d] if b is not None else None)
        k = F.linear(x, w[d:2*d], b[d:2*d] if b is not None else None)
        v = F.linear(x, w[2*d:], b[2*d:] if b is not None else None)
        return q, k, v

    def _attn(self, mha, q, k, v):
        nh, hd = mha.num_heads, mha.head_dim
        b = q.size(0)
        q = q.view(b, -1, nh, hd).transpose(1, 2)
        k = k.view(b, -1, nh, hd).transpose(1, 2)
        v = v.view(b, -1, nh, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return mha.out_proj(out.transpose(1, 2).contiguous().view(b, -1, nh * hd))

    def forward(self, x, text_cond, cross_mask=None, rope=None):
        nh, hd = self.self_attn.num_heads, self.self_attn.head_dim

        # Causal self-attention via SDPA (with RoPE on Q,K)
        residual = x
        x_n = self.norm1(x)
        q, k, v = self._proj_qkv(self.self_attn, x_n)
        b, s = x_n.shape[:2]
        q = q.view(b, s, nh, hd).transpose(1, 2)
        k = k.view(b, s, nh, hd).transpose(1, 2)
        v = v.view(b, s, nh, hd).transpose(1, 2)
        if rope is not None:
            q, k = rope(q, k)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = self.self_attn.out_proj(x.transpose(1, 2).contiguous().view(b, s, nh * hd))
        x = self.dropout(x) + residual

        # Cross-attention to text
        residual = x
        x_n = self.norm2(x)
        d = self.cross_attn.embed_dim
        w, bw = self.cross_attn.in_proj_weight, self.cross_attn.in_proj_bias
        q_c = F.linear(x_n, w[:d], bw[:d] if bw is not None else None)
        k_c = F.linear(text_cond, w[d:2*d], bw[d:2*d] if bw is not None else None)
        v_c = F.linear(text_cond, w[2*d:], bw[2*d:] if bw is not None else None)
        b_sz, s_q = x_n.shape[:2]
        s_kv = text_cond.shape[1]
        q_c = q_c.view(b_sz, s_q, nh, hd).transpose(1, 2)
        k_c = k_c.view(b_sz, s_kv, nh, hd).transpose(1, 2)
        v_c = v_c.view(b_sz, s_kv, nh, hd).transpose(1, 2)
        attn_mask_cross = None
        if cross_mask is not None:
            attn_mask_cross = cross_mask.unsqueeze(1).unsqueeze(2)
            attn_mask_cross = attn_mask_cross.expand(-1, nh, s_q, -1)
            attn_mask_cross = torch.where(attn_mask_cross, float('-inf'), 0.0)
        x = F.scaled_dot_product_attention(q_c, k_c, v_c, attn_mask=attn_mask_cross)
        x = self.cross_attn.out_proj(x.transpose(1, 2).contiguous().view(b_sz, s_q, nh * hd))
        x = self.dropout(x) + residual

        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ff(x) + residual
        return x

    def forward_cached(self, x_new, text_cond, cache=None, rope=None, step=0):
        """Single-token step with KV cache for fast inference."""
        d = self.self_attn.embed_dim
        nh, hd = self.self_attn.num_heads, self.self_attn.head_dim
        new_cache = {}

        # Self-attention with cache + RoPE
        residual = x_new
        x_n = self.norm1(x_new)
        q, k, v = self._proj_qkv(self.self_attn, x_n)
        # Apply RoPE before caching K (position is baked into the key)
        if rope is not None:
            b = q.size(0)
            q = q.view(b, 1, nh, hd).transpose(1, 2)
            k = k.view(b, 1, nh, hd).transpose(1, 2)
            q, k = rope(q, k, offset=step)
            q = q.transpose(1, 2).contiguous().view(b, 1, nh * hd)
            k = k.transpose(1, 2).contiguous().view(b, 1, nh * hd)
        if cache is not None and 'sk' in cache:
            k = torch.cat([cache['sk'], k], dim=1)
            v = torch.cat([cache['sv'], v], dim=1)
        new_cache['sk'] = k
        new_cache['sv'] = v
        x = self.dropout(self._attn(self.self_attn, q, k, v)) + residual

        # Cross-attention (cache text K/V)
        residual = x
        x_n = self.norm2(x)
        w, b = self.cross_attn.in_proj_weight, self.cross_attn.in_proj_bias
        q_c = F.linear(x_n, w[:d], b[:d] if b is not None else None)
        if cache is not None and 'ck' in cache:
            ck, cv = cache['ck'], cache['cv']
        else:
            ck = F.linear(text_cond, w[d:2*d], b[d:2*d] if b is not None else None)
            cv = F.linear(text_cond, w[2*d:], b[2*d:] if b is not None else None)
        new_cache['ck'] = ck
        new_cache['cv'] = cv
        x = self.dropout(self._attn(self.cross_attn, q_c, ck, cv)) + residual

        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ff(x) + residual
        return x, new_cache


class GeometryDecoder(nn.Module):
    """Autoregressive mesh token decoder, conditioned on shared text encoding.

    Uses RoPE (Rotary Position Embeddings) instead of learned positional
    embeddings.  This means there is NO hard maximum sequence length —
    the model can handle any number of faces at inference time.
    max_seq_length is kept as a soft limit for training (what we
    practically fit in GPU memory), not an architectural constraint.
    """

    def __init__(self, mesh_vocab_size: int = 8192, hidden_size: int = 512,
                 num_layers: int = 12, num_heads: int = 8,
                 max_seq_length: int = 18432, text_embed_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.mesh_vocab_size = mesh_vocab_size

        self.mesh_embed = nn.Embedding(mesh_vocab_size, hidden_size)
        # RoPE replaces learned pos_embed — no maximum length, no parameters
        head_dim = hidden_size // num_heads
        self.rope = RotaryPositionEmbedding(head_dim)
        self.text_proj = nn.Linear(text_embed_dim, hidden_size)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            GeometryDecoderLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, mesh_vocab_size, bias=False)

        # Weight tying
        nn.init.normal_(self.mesh_embed.weight, std=hidden_size ** -0.5)
        self.output_proj.weight = self.mesh_embed.weight

        # Gradient checkpointing — trades compute for memory.
        # Essential for long sequences on MPS (no flash attention).
        # Without this: max ~511 faces at batch=2
        # With this: ~1500+ faces at batch=1
        self.use_gradient_checkpointing = False

    def forward(self, mesh_tokens, text_encoding, text_mask=None):
        B, S = mesh_tokens.shape

        x = self.embed_dropout(self.mesh_embed(mesh_tokens))
        text_cond = self.text_proj(text_encoding)

        cross_mask = None
        if text_mask is not None:
            cross_mask = ~text_mask.bool()

        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                x = torch_checkpoint(
                    layer, x, text_cond, cross_mask, self.rope,
                    use_reentrant=False)
            else:
                x = layer(x, text_cond, cross_mask=cross_mask, rope=self.rope)

        x = self.norm(x)
        return self.output_proj(x)

    @torch.no_grad()
    def generate(self, text_encoding, text_mask=None,
                 max_tokens=2048 * 9 + 2, temperature=0.8,
                 top_k=50, top_p=0.9, bos_token=1, eos_token=2,
                 cfg_scale=0.0):
        """Autoregressive generation with KV-cache and optional CFG.

        Args:
            cfg_scale: Classifier-free guidance scale. 0 = disabled (default).
                       Values > 0 amplify text conditioning:
                       logits = uncond + cfg_scale * (cond - uncond)
                       Typical values: 2.0 - 7.0
        """
        device = text_encoding.device
        max_tokens = min(max_tokens, self.max_seq_length)
        use_cfg = cfg_scale > 0.0

        output_tokens = torch.zeros(1, max_tokens, device=device, dtype=torch.long)
        output_tokens[0, 0] = bos_token
        gen_len = 1

        text_cond = self.text_proj(text_encoding)
        layer_caches = [None] * len(self.layers)
        top_k_clamped = min(top_k, self.mesh_vocab_size) if top_k > 0 else 0

        # CFG: prepare unconditional text conditioning (zeros)
        if use_cfg:
            uncond_text = torch.zeros_like(text_cond)
            uncond_caches = [None] * len(self.layers)

        for step in range(max_tokens - 1):
            last_tok = output_tokens[:, step:step+1]
            x = self.mesh_embed(last_tok)

            # Conditioned pass
            new_caches = []
            x_cond = x
            for i, layer in enumerate(self.layers):
                x_cond, nc = layer.forward_cached(x_cond, text_cond, cache=layer_caches[i],
                                                  rope=self.rope, step=step)
                new_caches.append(nc)
            layer_caches = new_caches

            x_cond = self.norm(x_cond)
            logits_cond = self.output_proj(x_cond[:, 0, :])

            if use_cfg:
                # Unconditioned pass (same mesh tokens, zero text)
                new_uncond_caches = []
                x_uncond = x
                for i, layer in enumerate(self.layers):
                    x_uncond, nc = layer.forward_cached(x_uncond, uncond_text,
                                                        cache=uncond_caches[i],
                                                        rope=self.rope, step=step)
                    new_uncond_caches.append(nc)
                uncond_caches = new_uncond_caches

                x_uncond = self.norm(x_uncond)
                logits_uncond = self.output_proj(x_uncond[:, 0, :])

                # CFG: amplify the difference between conditioned and unconditioned
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            else:
                logits = logits_cond

            # Guard against NaN/inf from CFG amplification or fp16 overflow
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

            logits = logits / max(temperature, 1e-6)

            # Clamp to a safe range before any masking (prevents exp() overflow)
            logits = logits.clamp(-1e4, 1e4)

            if top_k_clamped > 0:
                top_k_vals, _ = torch.topk(logits, top_k_clamped)
                logits[logits < top_k_vals[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)

            # Sanitize: replace any residual NaN/inf and ensure valid distribution
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs.clamp(min=0.0)
            prob_sum = probs.sum(dim=-1, keepdim=True)
            if (prob_sum == 0).any():
                # Fallback: uniform over vocab if all probs collapsed
                probs = torch.where(prob_sum == 0,
                                    torch.ones_like(probs) / probs.shape[-1],
                                    probs / prob_sum)
            else:
                probs = probs / prob_sum

            next_token = torch.multinomial(probs, num_samples=1)
            output_tokens[0, gen_len] = next_token[0, 0]
            gen_len += 1

            if next_token.item() == eos_token:
                break

        return output_tokens[:, :gen_len]


# ─── Material Decoder (autoregressive node-graph tokens) ──────────────

class MaterialDecoder(nn.Module):
    """Autoregressive decoder for material node graph tokens.

    Smaller and simpler than geometry — material sequences are short (~100-300 tokens).
    """

    def __init__(self, vocab_size: int = 4096, hidden_size: int = 256,
                 num_layers: int = 6, num_heads: int = 4,
                 max_seq_len: int = 512, text_embed_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        self.text_proj = nn.Linear(text_embed_dim, hidden_size)
        self.drop = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)

        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

    def forward(self, tokens, text_encoding, text_mask=None):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.drop(self.token_embed(tokens) + self.pos_embed(pos))

        text_cond = self.text_proj(text_encoding)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tokens.device)
        mem_mask = None
        if text_mask is not None:
            mem_mask = text_mask == 0

        x = self.decoder(tgt=x, memory=text_cond,
                         tgt_mask=causal_mask,
                         memory_key_padding_mask=mem_mask)
        return self.head(self.ln_f(x))


# ─── Modifier Head (structured prediction) ────────────────────────────

# Import constants from existing module for compatibility
MODIFIER_TYPES = [
    "NONE", "SUBSURF", "MIRROR", "BEVEL", "SOLIDIFY", "ARRAY",
    "BOOLEAN", "SHRINKWRAP", "SMOOTH", "DECIMATE", "EDGE_SPLIT",
    "WEIGHTED_NORMAL", "SIMPLE_DEFORM", "CAST", "CURVE", "DISPLACE",
    "SKIN", "REMESH", "WIREFRAME", "WELD",
]
NUM_MODIFIER_TYPES = len(MODIFIER_TYPES)
MAX_MODIFIERS = 8
PARAMS_PER_MODIFIER = 12


class MeshStatsEncoder(nn.Module):
    """Encode mesh statistics into a vector."""

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.GELU(),
            nn.Linear(64, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, stats):
        return self.net(stats)


class ModifierHead(nn.Module):
    """Structured prediction head for modifier stack."""

    def __init__(self, input_size: int = 512):
        super().__init__()
        self.count_head = nn.Linear(input_size, MAX_MODIFIERS)

        self.slot_heads = nn.ModuleList([
            nn.ModuleDict({
                'type_head': nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.GELU(),
                    nn.Linear(128, NUM_MODIFIER_TYPES),
                ),
                'param_heads': nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_size, 64),
                        nn.GELU(),
                        nn.Linear(64, PARAMS_PER_MODIFIER),
                    )
                    for _ in range(NUM_MODIFIER_TYPES)
                ]),
            })
            for _ in range(MAX_MODIFIERS)
        ])

    def forward(self, fused):
        """
        Args:
            fused: (B, D) fused text+mesh representation

        Returns dict with count_logits, type_logits, param_values
        """
        count_logits = self.count_head(fused)

        type_logits_list = []
        param_values_list = []
        for slot in self.slot_heads:
            slot_dict: nn.ModuleDict = slot  # type: ignore[assignment]
            t_logits = slot_dict['type_head'](fused)
            param_heads: nn.ModuleList = slot_dict['param_heads']  # type: ignore[assignment]
            all_params = torch.stack(
                [h(fused) for h in param_heads], dim=1)
            type_logits_list.append(t_logits)
            param_values_list.append(all_params)

        return {
            "count_logits": count_logits,
            "type_logits": type_logits_list,
            "param_values": param_values_list,
        }


# ─── Unified Model ────────────────────────────────────────────────────

class UnifiedBlenderModel(nn.Module):
    """Multi-task model: text encoder → geometry + (optional) materials/modifiers/contrastive.

    Optional heads are gated by config flags so they're only instantiated
    when the hardware can support them:
        enable_materials:    MaterialDecoder          (~7.6M params)
        enable_modifiers:    ModifierHead + stats     (~6.3M params)
        enable_contrastive:  ImageEncoder + CLIP head (~1.0M params)

    On a laptop (MPS, 36 GB) set all three to false → saves ~15M params
    of GPU memory, leaving more room for long geometry sequences.

    Config structure expected:
        unified:
            embed_dim: 512
            text_vocab_size: 4096
            text_max_length: 256
            text_num_layers: 4
            text_num_heads: 8
            dropout: 0.1
            enable_materials: false
            enable_modifiers: false
            enable_contrastive: false

            geometry:
                num_layers: 12
                num_heads: 8
                max_seq_length: 16202
                mesh_vocab_size: 8192

            materials:   ...  (only read when enable_materials=true)
            modifiers:   ...  (only read when enable_modifiers=true)
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = config.get("unified", config)

        embed_dim = cfg.get("embed_dim", 512)
        text_vocab = cfg.get("text_vocab_size", 4096)
        text_max_len = cfg.get("text_max_length", 256)
        text_layers = cfg.get("text_num_layers", 4)
        text_heads = cfg.get("text_num_heads", 8)
        dropout = cfg.get("dropout", 0.1)

        # Feature flags — disable optional heads on constrained hardware
        self.enable_materials = cfg.get("enable_materials", False)
        self.enable_modifiers = cfg.get("enable_modifiers", False)
        self.enable_contrastive = cfg.get("enable_contrastive", False)

        # ── Always-on: shared text encoder + geometry decoder ──
        self.text_encoder = SharedTextEncoder(
            vocab_size=text_vocab,
            embed_dim=embed_dim,
            max_length=text_max_len,
            num_layers=text_layers,
            num_heads=text_heads,
            dropout=dropout,
        )

        geo_cfg = cfg.get("geometry", {})
        self.geometry_decoder = GeometryDecoder(
            mesh_vocab_size=geo_cfg.get("mesh_vocab_size", 8192),
            hidden_size=embed_dim,
            num_layers=geo_cfg.get("num_layers", 12),
            num_heads=geo_cfg.get("num_heads", 8),
            max_seq_length=geo_cfg.get("max_seq_length", 16202),
            text_embed_dim=embed_dim,
            dropout=dropout,
        )

        # ── Optional: contrastive (image encoder + CLIP head) ──
        self.enable_image_to_mesh = cfg.get("enable_image_to_mesh", False)
        if self.enable_image_to_mesh and not self.enable_contrastive:
            raise ValueError(
                "enable_image_to_mesh requires enable_contrastive=true "
                "(the ImageEncoder is shared)")

        self.image_encoder = None
        self.contrastive_head = None
        if self.enable_contrastive:
            self.image_encoder = ImageEncoder(
                embed_dim=embed_dim,
                spatial=self.enable_image_to_mesh,
            )
            self.contrastive_head = ContrastiveHead(embed_dim=embed_dim)

        # ── Optional: material decoder ──
        self.material_decoder = None
        if self.enable_materials:
            mat_cfg = cfg.get("materials", {})
            self.material_decoder = MaterialDecoder(
                vocab_size=mat_cfg.get("vocab_size", 4096),
                hidden_size=mat_cfg.get("hidden_size", 256),
                num_layers=mat_cfg.get("num_layers", 6),
                num_heads=mat_cfg.get("num_heads", 4),
                max_seq_len=mat_cfg.get("max_seq_len", 512),
                text_embed_dim=embed_dim,
                dropout=dropout,
            )

        # ── Optional: modifier head ──
        self.mesh_stats_encoder = None
        self.modifier_fusion = None
        self.modifier_head = None
        if self.enable_modifiers:
            mod_cfg = cfg.get("modifiers", {})
            mod_hidden = mod_cfg.get("hidden_size", 256)
            self.mesh_stats_encoder = MeshStatsEncoder(hidden_size=mod_hidden)
            self.modifier_fusion = nn.Sequential(
                nn.Linear(embed_dim + mod_hidden, embed_dim),
                nn.GELU(),
                nn.LayerNorm(embed_dim),
            )
            self.modifier_head = ModifierHead(input_size=embed_dim)

    # ── Forward methods per task ──

    def forward_image_conditioned(self, images, mesh_tokens,
                                  text_ids=None, text_mask=None):
        """Image-conditioned geometry training: image (+optional text) → mesh logits.

        Concatenates text conditioning (if provided) with image spatial
        features along the sequence dimension, then feeds into the
        geometry decoder's cross-attention.

        Args:
            images:      (B, 3, 64, 64) normalized RGB images
            mesh_tokens: (B, S) mesh token input sequence
            text_ids:    (B, T) optional text token ids
            text_mask:   (B, T) optional text attention mask

        Returns:
            (B, S, mesh_vocab_size) logits
        """
        if not self.enable_image_to_mesh:
            raise RuntimeError(
                "Image-to-mesh disabled (enable_image_to_mesh=false)")
        assert self.image_encoder is not None

        # Image spatial features: (B, 16, embed_dim)
        img_spatial = self.image_encoder.encode_spatial(images)

        if text_ids is not None:
            # Text sequence features: (B, T, embed_dim)
            text_seq, _ = self.text_encoder(text_ids, text_mask)
            # Concatenate: (B, T+16, embed_dim)
            cond = torch.cat([text_seq, img_spatial], dim=1)
            # Build combined cross-attention mask
            if text_mask is not None:
                img_mask = torch.ones(
                    img_spatial.shape[0], img_spatial.shape[1],
                    device=text_mask.device, dtype=text_mask.dtype)
                cond_mask = torch.cat([text_mask, img_mask], dim=1)
            else:
                cond_mask = None
        else:
            # Image-only conditioning: (B, 16, embed_dim)
            cond = img_spatial
            cond_mask = None

        return self.geometry_decoder(mesh_tokens, cond, cond_mask)

    def forward_geometry(self, text_ids, text_mask, mesh_tokens):
        """Geometry training: text → mesh token logits. Always available."""
        text_seq, _ = self.text_encoder(text_ids, text_mask)
        return self.geometry_decoder(mesh_tokens, text_seq, text_mask)

    def forward_materials(self, text_ids, text_mask, material_tokens):
        """Material training: text → material token logits."""
        if not self.enable_materials:
            raise RuntimeError("Material decoder disabled (enable_materials=false)")
        assert self.material_decoder is not None
        text_seq, _ = self.text_encoder(text_ids, text_mask)
        return self.material_decoder(material_tokens, text_seq, text_mask)

    def forward_modifiers(self, text_ids, text_mask, mesh_stats):
        """Modifier training: text + mesh stats → modifier predictions."""
        if not self.enable_modifiers:
            raise RuntimeError("Modifier head disabled (enable_modifiers=false)")
        assert self.mesh_stats_encoder is not None
        assert self.modifier_fusion is not None
        assert self.modifier_head is not None
        _, text_pooled = self.text_encoder(text_ids, text_mask)
        mesh_vec = self.mesh_stats_encoder(mesh_stats)
        fused = self.modifier_fusion(torch.cat([text_pooled, mesh_vec], dim=-1))
        return self.modifier_head(fused)

    def forward_contrastive(self, text_ids, text_mask, images):
        """Contrastive training: align text and image embeddings."""
        if not self.enable_contrastive:
            raise RuntimeError("Contrastive head disabled (enable_contrastive=false)")
        assert self.image_encoder is not None
        assert self.contrastive_head is not None
        _, text_pooled = self.text_encoder(text_ids, text_mask)
        image_embeds = self.image_encoder(images)
        text_proj = self.text_encoder.contrastive_proj(text_pooled)
        return self.contrastive_head(text_proj, image_embeds)

    # ── Inference ──

    @torch.no_grad()
    def generate_geometry(self, text_ids, text_mask=None, **kwargs):
        """Generate mesh tokens from text."""
        text_seq, _ = self.text_encoder(text_ids, text_mask)
        return self.geometry_decoder.generate(text_seq, text_mask, **kwargs)

    @torch.no_grad()
    def generate_from_image(self, images, text_ids=None, text_mask=None,
                            **kwargs):
        """Generate mesh tokens from image (+optional text).

        Uses the same GeometryDecoder as generate_geometry but conditions
        on spatial image features (and optionally concatenated text).

        Args:
            images:    (B, 3, 64, 64) normalized RGB images
            text_ids:  (B, T) optional text token ids
            text_mask: (B, T) optional text attention mask
            **kwargs:  forwarded to GeometryDecoder.generate()

        Returns:
            (1, gen_len) mesh token ids
        """
        if not self.enable_image_to_mesh:
            raise RuntimeError(
                "Image-to-mesh disabled (enable_image_to_mesh=false)")
        assert self.image_encoder is not None

        img_spatial = self.image_encoder.encode_spatial(images)

        if text_ids is not None:
            text_seq, _ = self.text_encoder(text_ids, text_mask)
            cond = torch.cat([text_seq, img_spatial], dim=1)
            if text_mask is not None:
                img_mask = torch.ones(
                    img_spatial.shape[0], img_spatial.shape[1],
                    device=text_mask.device, dtype=text_mask.dtype)
                cond_mask = torch.cat([text_mask, img_mask], dim=1)
            else:
                cond_mask = None
        else:
            cond = img_spatial
            cond_mask = None

        return self.geometry_decoder.generate(cond, cond_mask, **kwargs)

    @torch.no_grad()
    def generate_materials(self, text_ids, text_mask, max_tokens=512,
                           temperature=0.7, top_k=30):
        """Generate material tokens from text."""
        if not self.enable_materials:
            raise RuntimeError("Material decoder disabled (enable_materials=false)")
        assert self.material_decoder is not None
        self.eval()
        text_seq, _ = self.text_encoder(text_ids, text_mask)
        _text_cond = self.material_decoder.text_proj(text_seq)  # noqa: F841

        B = text_ids.shape[0]
        device = text_ids.device
        generated = torch.full((B, 1), 1, dtype=torch.long, device=device)

        for _ in range(max_tokens):
            logits = self.material_decoder(generated, text_seq, text_mask)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == 2).all():
                break
        return generated

    @torch.no_grad()
    def predict_modifiers(self, text_ids, text_mask, mesh_stats):
        """Predict modifier stack from text + mesh stats."""
        if not self.enable_modifiers:
            raise RuntimeError("Modifier head disabled (enable_modifiers=false)")
        self.eval()
        out = self.forward_modifiers(text_ids, text_mask, mesh_stats)

        count = out["count_logits"].argmax(dim=-1).item() + 1
        count = min(count, MAX_MODIFIERS)

        modifiers = []
        for i in range(count):
            type_id = out["type_logits"][i].argmax(dim=-1).item()
            if type_id == 0:
                continue
            mod_type = MODIFIER_TYPES[type_id]
            params = out["param_values"][i][0, type_id].cpu().tolist()
            modifiers.append({"type": mod_type, "params": params})
        return modifiers

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
