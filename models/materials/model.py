"""Material generation model.

A text-conditioned model that outputs Blender shader node trees.

Unlike the geometry model (which generates mesh tokens autoregressively),
the material model generates a structured node graph as a sequence of
node placement + connection operations.

Architecture:
    Text → TextEncoder → cross-attention
    Autoregressive decoder generates a sequence of node-graph tokens:
      [NODE_TYPE, PARAM_1, ..., PARAM_N, CONNECT_FROM, CONNECT_TO, ...]

Node Graph Tokenization:
    Each material is serialized as a sequence of operations:
    1. ADD_NODE <type_id>                         — add a Principled BSDF, MixRGB, etc.
    2. SET_INPUT <node_idx> <input_idx> <value>   — set a numeric input
    3. SET_COLOR <node_idx> <input_idx> <r> <g> <b> — set a color input
    4. LINK <from_node> <from_socket> <to_node> <to_socket> — connect sockets
    5. SET_LOCATION <node_idx> <x> <y>            — set node editor location
    6. END_MATERIAL                               — done

    All values are quantized to integers in [0, vocab_size).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Node type vocabulary ---

# Common Blender shader node types (ordered by frequency in training data)
NODE_TYPES = [
    "ShaderNodeBsdfPrincipled",
    "ShaderNodeMixRGB",
    "ShaderNodeTexImage",
    "ShaderNodeTexNoise",
    "ShaderNodeTexCoord",
    "ShaderNodeMapping",
    "ShaderNodeBump",
    "ShaderNodeNormalMap",
    "ShaderNodeTexMusgrave",
    "ShaderNodeValToRGB",  # ColorRamp
    "ShaderNodeMath",
    "ShaderNodeSeparateXYZ",
    "ShaderNodeCombineXYZ",
    "ShaderNodeInvert",
    "ShaderNodeMixShader",
    "ShaderNodeBsdfGlass",
    "ShaderNodeBsdfTransparent",
    "ShaderNodeEmission",
    "ShaderNodeTexVoronoi",
    "ShaderNodeTexWave",
    "ShaderNodeFresnel",
    "ShaderNodeLayerWeight",
    "ShaderNodeRGBCurve",
    "ShaderNodeHueSaturation",
    "ShaderNodeBrightContrast",
    "ShaderNodeGamma",
    "ShaderNodeTexChecker",
    "ShaderNodeTexBrick",
    "ShaderNodeTexGradient",
    "ShaderNodeTexMagic",
    "ShaderNodeOutputMaterial",
    "ShaderNodeVectorMath",
    "ShaderNodeMapRange",
    "ShaderNodeClamp",
    "ShaderNodeRGB",
    "ShaderNodeValue",
    "ShaderNodeAmbientOcclusion",
    "ShaderNodeBsdfDiffuse",
    "ShaderNodeBsdfGlossy",
    "ShaderNodeBsdfAnisotropic",
    "ShaderNodeSubsurfaceScattering",
]

NODE_TYPE_TO_ID = {n: i for i, n in enumerate(NODE_TYPES)}


# --- Special tokens ---
PAD = 0
BOS = 1
EOS = 2
ADD_NODE = 3
SET_INPUT = 4
SET_COLOR = 5
LINK = 6
END_MATERIAL = 7
SPECIAL_TOKENS = 8


class MaterialEncoder:
    """Encode/decode material node trees as token sequences."""

    def __init__(self, vocab_size: int = 4096, num_bins: int = 256):
        self.vocab_size = vocab_size
        self.num_bins = num_bins
        # Offset for regular values after special tokens + node types
        self.value_offset = SPECIAL_TOKENS + len(NODE_TYPES)

    def quantize_float(self, value: float, min_val: float = -10.0,
                       max_val: float = 10.0) -> int:
        """Quantize a float to an integer bin."""
        clamped = max(min_val, min(max_val, value))
        normalized = (clamped - min_val) / (max_val - min_val)
        bin_idx = int(normalized * (self.num_bins - 1))
        return bin_idx + self.value_offset

    def dequantize(self, token: int, min_val: float = -10.0,
                   max_val: float = 10.0) -> float:
        """Convert a quantized token back to float."""
        bin_idx = token - self.value_offset
        normalized = bin_idx / (self.num_bins - 1)
        return normalized * (max_val - min_val) + min_val

    def quantize_color(self, r: float, g: float, b: float) -> tuple[int, int, int]:
        """Quantize RGB values (0-1 range) to tokens."""
        return (
            self.quantize_float(r, 0.0, 1.0),
            self.quantize_float(g, 0.0, 1.0),
            self.quantize_float(b, 0.0, 1.0),
        )

    def encode_material(self, material_data: dict) -> list[int]:
        """Encode a material node tree dict into a token sequence.

        Args:
            material_data: Dict with 'nodes' and 'links' from blend_extractor.

        Returns:
            List of integer tokens.
        """
        tokens = [BOS]

        nodes = material_data.get("nodes", [])
        links = material_data.get("links", [])

        # Encode nodes
        node_name_to_idx = {}
        for i, node in enumerate(nodes):
            node_type = node.get("type", "")
            if node_type not in NODE_TYPE_TO_ID:
                continue

            type_id = SPECIAL_TOKENS + NODE_TYPE_TO_ID[node_type]
            tokens.append(ADD_NODE)
            tokens.append(type_id)
            node_name_to_idx[node.get("name", f"node_{i}")] = i

            # Encode inputs (scalar values)
            for inp in node.get("inputs", []):
                value = inp.get("default_value")
                if value is None:
                    continue

                input_idx = self.quantize_float(
                    inp.get("index", 0), 0, 30
                )

                if isinstance(value, (list, tuple)):
                    if len(value) >= 3:
                        # Color input
                        tokens.append(SET_COLOR)
                        tokens.append(self.quantize_float(i, 0, 50))
                        tokens.append(input_idx)
                        r, g, b = self.quantize_color(
                            value[0], value[1], value[2]
                        )
                        tokens.extend([r, g, b])
                elif isinstance(value, (int, float)):
                    tokens.append(SET_INPUT)
                    tokens.append(self.quantize_float(i, 0, 50))
                    tokens.append(input_idx)
                    tokens.append(self.quantize_float(float(value)))

        # Encode links
        for link in links:
            from_node = link.get("from_node", "")
            to_node = link.get("to_node", "")
            from_idx = node_name_to_idx.get(from_node, 0)
            to_idx = node_name_to_idx.get(to_node, 0)

            tokens.append(LINK)
            tokens.append(self.quantize_float(from_idx, 0, 50))
            tokens.append(self.quantize_float(
                link.get("from_socket_index", 0), 0, 20
            ))
            tokens.append(self.quantize_float(to_idx, 0, 50))
            tokens.append(self.quantize_float(
                link.get("to_socket_index", 0), 0, 20
            ))

        tokens.append(END_MATERIAL)
        tokens.append(EOS)

        return tokens

    def decode_tokens(self, tokens: list[int]) -> dict:
        """Decode a token sequence back to a material node tree dict.

        Returns:
            Dict with 'nodes' and 'links' keys.
        """
        nodes = []
        links = []
        i = 0

        while i < len(tokens):
            t = tokens[i]

            if t == BOS or t == PAD:
                i += 1
                continue
            elif t == EOS or t == END_MATERIAL:
                break
            elif t == ADD_NODE:
                if i + 1 < len(tokens):
                    type_id = tokens[i + 1] - SPECIAL_TOKENS
                    if 0 <= type_id < len(NODE_TYPES):
                        nodes.append({
                            "type": NODE_TYPES[type_id],
                            "inputs": {},
                        })
                    i += 2
                else:
                    i += 1
            elif t == SET_INPUT:
                if i + 3 < len(tokens):
                    node_idx = int(self.dequantize(tokens[i + 1], 0, 50))
                    input_idx = int(self.dequantize(tokens[i + 2], 0, 30))
                    value = self.dequantize(tokens[i + 3])
                    if 0 <= node_idx < len(nodes):
                        nodes[node_idx]["inputs"][input_idx] = value
                    i += 4
                else:
                    i += 1
            elif t == SET_COLOR:
                if i + 5 < len(tokens):
                    node_idx = int(self.dequantize(tokens[i + 1], 0, 50))
                    input_idx = int(self.dequantize(tokens[i + 2], 0, 30))
                    r = self.dequantize(tokens[i + 3], 0, 1)
                    g = self.dequantize(tokens[i + 4], 0, 1)
                    b = self.dequantize(tokens[i + 5], 0, 1)
                    if 0 <= node_idx < len(nodes):
                        nodes[node_idx]["inputs"][input_idx] = [r, g, b, 1.0]
                    i += 6
                else:
                    i += 1
            elif t == LINK:
                if i + 4 < len(tokens):
                    links.append({
                        "from_node": int(self.dequantize(tokens[i + 1], 0, 50)),
                        "from_socket": int(self.dequantize(tokens[i + 2], 0, 20)),
                        "to_node": int(self.dequantize(tokens[i + 3], 0, 50)),
                        "to_socket": int(self.dequantize(tokens[i + 4], 0, 20)),
                    })
                    i += 5
                else:
                    i += 1
            else:
                i += 1

        return {"nodes": nodes, "links": links}


class MaterialTransformer(nn.Module):
    """Autoregressive transformer for material node graph generation.

    Smaller than the geometry model since material node trees are
    much shorter sequences than mesh face lists.
    """

    def __init__(self, config: dict):
        super().__init__()

        mat_config = config.get("model", {}).get("materials", {})
        self.hidden_size = mat_config.get("hidden_size", 512)
        self.num_layers = mat_config.get("num_layers", 12)
        self.num_heads = mat_config.get("num_heads", 8)
        self.dropout = mat_config.get("dropout", 0.1)
        self.vocab_size = config.get("tokenization", {}).get("vocab_size", 4096)
        self.max_seq_len = mat_config.get("max_seq_len", 512)

        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.hidden_size)
        self.drop = nn.Dropout(self.dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.num_layers
        )

        # Output projection
        self.ln_f = nn.LayerNorm(self.hidden_size)
        self.head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens: torch.Tensor,
                text_memory: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: (batch, seq_len) token indices
            text_memory: (batch, text_len, hidden) encoded text
            text_mask: (batch, text_len) text attention mask

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0)

        x = self.token_embedding(tokens) + self.pos_embedding(pos)
        x = self.drop(x)

        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=tokens.device
        )

        # Memory key padding mask
        mem_mask = None
        if text_mask is not None:
            mem_mask = text_mask == 0

        x = self.decoder(
            tgt=x,
            memory=text_memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=mem_mask,
        )

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class MaterialModel(nn.Module):
    """Complete material generation model: text → node graph tokens.

    Components:
        1. TextEncoder — shared with geometry model architecture
        2. MaterialTransformer — autoregressive decoder for node graph tokens
    """

    def __init__(self, config: dict):
        super().__init__()

        mat_config = config.get("model", {}).get("materials", {})
        self.hidden_size = mat_config.get("hidden_size", 512)
        self.vocab_size = config.get("tokenization", {}).get("vocab_size", 4096)

        # Text encoder (smaller than geometry model's)
        text_config = config.get("model", {}).get("text_encoder", {})
        text_vocab = text_config.get("vocab_size", 32000)
        text_hidden = text_config.get("hidden_size", 512)
        text_layers = text_config.get("num_layers", 4)

        self.text_embedding = nn.Embedding(text_vocab, text_hidden)
        self.text_pos = nn.Embedding(
            text_config.get("max_length", 256), text_hidden
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_hidden,
            nhead=text_config.get("num_heads", 8),
            dim_feedforward=text_hidden * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=text_layers
        )

        # Project text hidden to material hidden if different
        if text_hidden != self.hidden_size:
            self.text_proj = nn.Linear(text_hidden, self.hidden_size)
        else:
            self.text_proj = nn.Identity()

        # Material decoder
        self.decoder = MaterialTransformer(config)

        # Encoder for material tokens
        self.encoder = MaterialEncoder(vocab_size=self.vocab_size)

    def encode_text(self, text_ids: torch.Tensor,
                    text_mask: torch.Tensor) -> torch.Tensor:
        """Encode text into hidden representations."""
        B, T = text_ids.shape
        pos = torch.arange(T, device=text_ids.device).unsqueeze(0)
        x = self.text_embedding(text_ids) + self.text_pos(pos)
        x = self.text_encoder(x)
        x = self.text_proj(x)
        return x

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor,
                tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass for training.

        Args:
            text_ids: (batch, text_len)
            text_mask: (batch, text_len)
            tokens: (batch, seq_len) target material token sequence

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        text_memory = self.encode_text(text_ids, text_mask)
        logits = self.decoder(tokens, text_memory, text_mask)
        return logits

    @torch.no_grad()
    def generate(self, text_ids: torch.Tensor, text_mask: torch.Tensor,
                 max_tokens: int = 512, temperature: float = 0.7,
                 top_k: int = 30) -> torch.Tensor:
        """Autoregressively generate material tokens.

        Returns:
            tokens: (batch, generated_len) token indices
        """
        self.eval()
        B = text_ids.shape[0]
        device = text_ids.device

        text_memory = self.encode_text(text_ids, text_mask)

        # Start with BOS
        generated = torch.full((B, 1), BOS, dtype=torch.long, device=device)

        for _ in range(max_tokens):
            logits = self.decoder(generated, text_memory, text_mask)
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop on EOS or END_MATERIAL
            if (next_token == EOS).all() or (next_token == END_MATERIAL).all():
                break

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
