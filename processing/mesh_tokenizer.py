"""Mesh Tokenizer — converts 3D meshes to/from token sequences.

Inspired by MeshGPT: meshes are represented as sequences of quantized
vertex coordinates. Each triangular face becomes 9 tokens
(3 vertices × 3 coordinates), and the mesh is an ordered sequence of faces.

The tokenizer handles:
1. Coordinate quantization to a fixed vocabulary
2. Face ordering (spatial ordering for locality)
3. Encoding/decoding between mesh data and token sequences
"""

import math
import numpy as np
from pathlib import Path


class MeshTokenizer:
    """Tokenize meshes into integer sequences for transformer training.

    Coordinate space is normalized to [-1, 1] and quantized to
    `vocab_size` bins.  Each triangle face = 9 tokens (v1x v1y v1z
    v2x v2y v2z v3x v3y v3z).

    Special tokens:
        PAD = 0
        BOS = 1  (beginning of mesh)
        EOS = 2  (end of mesh)
        SEP = 3  (separator between faces, optional)

    Coordinate tokens start at index 4.
    """

    SPECIAL_TOKENS = 4  # PAD, BOS, EOS, SEP
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3

    def __init__(self, vocab_size: int = 8192,
                 coord_range: tuple[float, float] = (-1.0, 1.0),
                 max_faces: int = 2048):
        self.vocab_size = vocab_size
        self.coord_bins = vocab_size - self.SPECIAL_TOKENS
        self.coord_min = coord_range[0]
        self.coord_max = coord_range[1]
        self.coord_range = coord_range[1] - coord_range[0]
        self.max_faces = max_faces
        self.tokens_per_face = 9  # 3 verts × 3 coords (triangulated)

    def quantize_coord(self, value: float) -> int:
        """Map a float coordinate to an integer token."""
        # Clamp to range
        value = max(self.coord_min, min(self.coord_max, value))
        # Normalize to [0, 1]
        t = (value - self.coord_min) / self.coord_range
        # Map to bin index
        bin_idx = int(t * (self.coord_bins - 1))
        return bin_idx + self.SPECIAL_TOKENS

    def dequantize_token(self, token: int) -> float:
        """Map an integer token back to a float coordinate."""
        if token < self.SPECIAL_TOKENS:
            return 0.0
        bin_idx = token - self.SPECIAL_TOKENS
        t = bin_idx / (self.coord_bins - 1)
        return self.coord_min + t * self.coord_range

    def encode_mesh(self, vertices: list[list[float]],
                    faces: list[list[int]],
                    add_special: bool = True) -> list[int]:
        """Convert a triangulated mesh to a token sequence.

        Args:
            vertices: List of [x, y, z] coordinates (normalized to [-1, 1])
            faces: List of [v1, v2, v3] vertex index triples
            add_special: Whether to add BOS/EOS tokens

        Returns:
            List of integer tokens
        """
        verts = np.array(vertices)
        ordered_faces = self._order_faces(verts, faces)

        tokens = []
        if add_special:
            tokens.append(self.BOS)

        for face_indices in ordered_faces[:self.max_faces]:
            for vi in face_indices:
                if vi < len(verts):
                    for coord in verts[vi]:
                        tokens.append(self.quantize_coord(float(coord)))
                else:
                    # Invalid index — use center
                    for _ in range(3):
                        tokens.append(self.quantize_coord(0.0))

        if add_special:
            tokens.append(self.EOS)

        return tokens

    def decode_tokens(self, tokens: list[int]) -> tuple[list[list[float]], list[list[int]]]:
        """Convert a token sequence back to vertices and faces.

        Returns:
            (vertices, faces) where vertices may contain duplicates
            (one set of 3 coords per face vertex).
        """
        # Strip special tokens
        clean = [t for t in tokens if t >= self.SPECIAL_TOKENS]

        vertices = []
        faces = []

        for i in range(0, len(clean) - 8, 9):
            face_verts = []
            face_indices = []
            for j in range(3):
                x = self.dequantize_token(clean[i + j * 3])
                y = self.dequantize_token(clean[i + j * 3 + 1])
                z = self.dequantize_token(clean[i + j * 3 + 2])
                face_verts.append([x, y, z])
                face_indices.append(len(vertices) + j)
            vertices.extend(face_verts)
            faces.append(face_indices)

        return vertices, faces

    def _order_faces(self, verts: np.ndarray,
                     faces: list[list[int]]) -> list[list[int]]:
        """Order faces spatially for better sequence locality.

        Uses Z-curve (Morton code) ordering of face centers so
        nearby faces in 3D space are nearby in the sequence.
        """
        if not faces:
            return faces

        # Compute face centers
        centers = []
        for face in faces:
            valid = [fi for fi in face if fi < len(verts)]
            if valid:
                center = verts[valid].mean(axis=0)
            else:
                center = np.zeros(3)
            centers.append(center)

        centers = np.array(centers)

        # Normalize centers to [0, 1023] for Morton code
        c_min = centers.min(axis=0)
        c_max = centers.max(axis=0)
        c_range = c_max - c_min
        c_range[c_range < 1e-6] = 1.0
        normalized = ((centers - c_min) / c_range * 1023).astype(int)
        normalized = np.clip(normalized, 0, 1023)

        # Compute Morton code (Z-curve) for each face
        morton_codes = []
        for c in normalized:
            code = self._morton_encode_3d(c[0], c[1], c[2])
            morton_codes.append(code)

        # Sort faces by Morton code
        order = np.argsort(morton_codes)
        return [faces[i] for i in order]

    @staticmethod
    def _morton_encode_3d(x: int, y: int, z: int) -> int:
        """Compute 3D Morton code (interleave bits of x, y, z)."""
        def spread_bits(v):
            v = v & 0x3FF  # 10 bits
            v = (v | (v << 16)) & 0x030000FF
            v = (v | (v << 8)) & 0x0300F00F
            v = (v | (v << 4)) & 0x030C30C3
            v = (v | (v << 2)) & 0x09249249
            return v
        return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)

    def pad_sequence(self, tokens: list[int], max_length: int) -> list[int]:
        """Pad a token sequence to fixed length."""
        if len(tokens) >= max_length:
            return tokens[:max_length]
        return tokens + [self.PAD] * (max_length - len(tokens))

    def sequence_length_for_faces(self, num_faces: int) -> int:
        """Calculate token sequence length for a given number of faces."""
        return 2 + num_faces * self.tokens_per_face  # +2 for BOS/EOS

    @property
    def total_vocab_size(self) -> int:
        return self.vocab_size

    def save(self, path: str | Path):
        """Save tokenizer config."""
        import json
        config = {
            "vocab_size": self.vocab_size,
            "coord_range": [self.coord_min, self.coord_max],
            "max_faces": self.max_faces,
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "MeshTokenizer":
        """Load tokenizer from config."""
        import json
        with open(path) as f:
            config = json.load(f)
        return cls(
            vocab_size=config["vocab_size"],
            coord_range=tuple(config["coord_range"]),
            max_faces=config["max_faces"],
        )


def test_roundtrip():
    """Test encode/decode roundtrip."""
    tokenizer = MeshTokenizer(vocab_size=8192)

    # Simple cube
    vertices = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ]
    faces = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 5, 1], [0, 4, 5],  # front
        [2, 7, 3], [2, 6, 7],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 5, 6], [1, 6, 2],  # right
    ]

    tokens = tokenizer.encode_mesh(vertices, faces)
    print(f"Cube: {len(vertices)} verts, {len(faces)} faces → {len(tokens)} tokens")

    decoded_verts, decoded_faces = tokenizer.decode_tokens(tokens)
    print(f"Decoded: {len(decoded_verts)} verts, {len(decoded_faces)} faces")

    # Check roundtrip accuracy
    max_error = 0
    for i, face in enumerate(faces[:len(decoded_faces)]):
        for j, vi in enumerate(face):
            for k in range(3):
                orig = vertices[vi][k]
                decoded = decoded_verts[decoded_faces[i][j]][k]
                max_error = max(max_error, abs(orig - decoded))

    print(f"Max roundtrip error: {max_error:.6f}")
    print(f"Vocab size: {tokenizer.total_vocab_size}")
    print(f"Tokens per face: {tokenizer.tokens_per_face}")


if __name__ == "__main__":
    test_roundtrip()
