"""Shared encoders and constants for material/modifier processing.

Extracted from legacy models/materials/model.py and models/modifiers/model.py
so the old model subdirectories can be removed. Used by:
  - training/train_unified.py (MaterialEncoder, modifier constants)
  - inference/server.py (MaterialEncoder for decode)
"""

import math
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Material Encoder — encode/decode Blender shader node trees as tokens
# ═══════════════════════════════════════════════════════════════════════════

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
    "ShaderNodeValToRGB",       # ColorRamp
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

# Special tokens for material sequences
MAT_PAD = 0
MAT_BOS = 1
MAT_EOS = 2
MAT_ADD_NODE = 3
MAT_SET_INPUT = 4
MAT_SET_COLOR = 5
MAT_LINK = 6
MAT_END_MATERIAL = 7
MAT_SPECIAL_TOKENS = 8


class MaterialEncoder:
    """Encode/decode material node trees as token sequences.

    Node Graph Tokenization:
      1. ADD_NODE <type_id>                             — add node
      2. SET_INPUT <node_idx> <input_idx> <value>       — set numeric input
      3. SET_COLOR <node_idx> <input_idx> <r> <g> <b>   — set color input
      4. LINK <from_node> <from_socket> <to_node> <to_socket> — connect
      5. END_MATERIAL                                   — done

    All values are quantized to integers in [0, vocab_size).
    """

    def __init__(self, vocab_size: int = 4096, num_bins: int = 256):
        self.vocab_size = vocab_size
        self.num_bins = num_bins
        self.value_offset = MAT_SPECIAL_TOKENS + len(NODE_TYPES)

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
        tokens = [MAT_BOS]

        nodes = material_data.get("nodes", [])
        links = material_data.get("links", [])

        # Encode nodes
        node_name_to_idx = {}
        for i, node in enumerate(nodes):
            node_type = node.get("type", "")
            if node_type not in NODE_TYPE_TO_ID:
                continue

            type_id = MAT_SPECIAL_TOKENS + NODE_TYPE_TO_ID[node_type]
            tokens.append(MAT_ADD_NODE)
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
                        tokens.append(MAT_SET_COLOR)
                        tokens.append(self.quantize_float(i, 0, 50))
                        tokens.append(input_idx)
                        r, g, b = self.quantize_color(
                            value[0], value[1], value[2]
                        )
                        tokens.extend([r, g, b])
                elif isinstance(value, (int, float)):
                    tokens.append(MAT_SET_INPUT)
                    tokens.append(self.quantize_float(i, 0, 50))
                    tokens.append(input_idx)
                    tokens.append(self.quantize_float(float(value)))

        # Encode links
        for link in links:
            from_node = link.get("from_node", "")
            to_node = link.get("to_node", "")
            from_idx = node_name_to_idx.get(from_node, 0)
            to_idx = node_name_to_idx.get(to_node, 0)

            tokens.append(MAT_LINK)
            tokens.append(self.quantize_float(from_idx, 0, 50))
            tokens.append(self.quantize_float(
                link.get("from_socket_index", 0), 0, 20
            ))
            tokens.append(self.quantize_float(to_idx, 0, 50))
            tokens.append(self.quantize_float(
                link.get("to_socket_index", 0), 0, 20
            ))

        tokens.append(MAT_END_MATERIAL)
        tokens.append(MAT_EOS)

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

            if t == MAT_BOS or t == MAT_PAD:
                i += 1
                continue
            elif t == MAT_EOS or t == MAT_END_MATERIAL:
                break
            elif t == MAT_ADD_NODE:
                if i + 1 < len(tokens):
                    type_id = tokens[i + 1] - MAT_SPECIAL_TOKENS
                    if 0 <= type_id < len(NODE_TYPES):
                        nodes.append({
                            "type": NODE_TYPES[type_id],
                            "inputs": {},
                        })
                    i += 2
                else:
                    i += 1
            elif t == MAT_SET_INPUT:
                if i + 3 < len(tokens):
                    node_idx = int(self.dequantize(tokens[i + 1], 0, 50))
                    input_idx = int(self.dequantize(tokens[i + 2], 0, 30))
                    value = self.dequantize(tokens[i + 3])
                    if 0 <= node_idx < len(nodes):
                        nodes[node_idx]["inputs"][input_idx] = value
                    i += 4
                else:
                    i += 1
            elif t == MAT_SET_COLOR:
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
            elif t == MAT_LINK:
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

    # Alias for backward compatibility (server.py was calling decode_material)
    decode_material = decode_tokens


# ═══════════════════════════════════════════════════════════════════════════
# Modifier Constants — shared vocabulary for modifier prediction
# ═══════════════════════════════════════════════════════════════════════════

MODIFIER_TYPES = [
    "NONE",             # Empty slot
    "SUBSURF",          # Subdivision Surface
    "MIRROR",           # Mirror
    "BEVEL",            # Bevel
    "SOLIDIFY",         # Solidify
    "ARRAY",            # Array
    "BOOLEAN",          # Boolean
    "SHRINKWRAP",       # Shrinkwrap
    "SMOOTH",           # Smooth / Laplacian Smooth
    "DECIMATE",         # Decimate
    "EDGE_SPLIT",       # Edge Split
    "WEIGHTED_NORMAL",  # Weighted Normal
    "SIMPLE_DEFORM",    # Simple Deform (twist, bend, taper, stretch)
    "CAST",             # Cast
    "CURVE",            # Curve
    "DISPLACE",         # Displace
    "SKIN",             # Skin
    "REMESH",           # Remesh
    "WIREFRAME",        # Wireframe
    "WELD",             # Weld
]

MODIFIER_TYPE_TO_ID = {m: i for i, m in enumerate(MODIFIER_TYPES)}
NUM_MODIFIER_TYPES = len(MODIFIER_TYPES)

MAX_MODIFIERS = 8           # Maximum modifiers in a stack
PARAMS_PER_MODIFIER = 12    # Parameter slots per modifier (covers all types)
