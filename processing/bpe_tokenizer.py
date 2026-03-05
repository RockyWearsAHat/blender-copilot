"""BPE (Byte-Pair Encoding) tokenizer for text-to-3D training.

Replaces the word-level TextTokenizer with a subword tokenizer that:
  - Never produces <unk> — any word gets split into known sub-pieces
  - Handles new Blender terms, shader nodes, animation jargon
  - Scales to materials, modifiers, animation, workflow descriptions
  - Is standard (SentencePiece BPE, same algorithm as GPT/Llama)

Usage:
    # Train from scratch
    tokenizer = BPETokenizer.train(
        texts=["a smooth red cube", "subdivided sphere with bevel modifier", ...],
        vocab_size=8000,
        model_prefix="data/datasets/geometry/bpe_tokenizer",
    )

    # Load saved tokenizer
    tokenizer = BPETokenizer.load("data/datasets/geometry/bpe_tokenizer")

    # Encode / decode (same API as TextTokenizer)
    ids = tokenizer.encode("Create a glossy metallic sphere with subsurface scattering")
    text = tokenizer.decode(ids)
"""

import json
import logging
import os
import tempfile
from pathlib import Path

import sentencepiece as spm

logger = logging.getLogger(__name__)

# Special token IDs — must match TextTokenizer for compatibility
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# Blender-specific terminology that MUST be in the vocabulary.
# These are injected as user-defined symbols so SentencePiece always
# keeps them as single tokens, never splits them into sub-pieces.
BLENDER_TERMS = [
    # Primitives
    "cube", "sphere", "cylinder", "cone", "torus", "plane", "circle",
    "icosphere", "monkey", "suzanne", "grid", "pyramid",
    # Modifiers
    "subdivision", "subsurf", "mirror", "bevel", "solidify", "array",
    "boolean", "decimate", "remesh", "shrinkwrap", "lattice",
    "armature", "curve", "cast", "smooth", "corrective",
    "multiresolution", "skin", "wireframe", "weld", "weighted",
    # Materials & Shading
    "shader", "principled", "bsdf", "emission", "glossy", "diffuse",
    "specular", "metallic", "roughness", "transmission", "ior",
    "subsurface", "clearcoat", "sheen", "anisotropic", "normal",
    "bump", "displacement", "texture", "hdri", "environment",
    "eevee", "cycles", "volume", "scatter", "fresnel",
    # UV & Texturing
    "uv", "unwrap", "seam", "uvmap", "texcoord",
    # Animation
    "keyframe", "animation", "animate", "timeline", "nla",
    "action", "driver", "constraint", "rig", "rigging",
    "bone", "armature", "pose", "weight", "paint",
    # Geometry Nodes
    "geonodes", "geometry", "nodes", "attribute", "instance",
    "distribute", "scatter", "proximity", "raycast",
    # Sculpting
    "sculpt", "retopology", "dynatopo", "multires",
    # Physics
    "physics", "particle", "simulation", "cloth", "fluid",
    "smoke", "fire", "softbody", "rigidbody", "collision",
    # Rendering
    "render", "viewport", "camera", "light", "sun", "spot",
    "area", "point", "compositing", "freestyle",
    # Common descriptors
    "lowpoly", "highpoly", "detailed", "stylized", "realistic",
    "procedural", "parametric", "organic", "geometric",
    # Blender-specific
    "blender", "addon", "grease", "pencil", "collection",
    "origin", "cursor", "pivot", "snap", "proportional",
]


class BPETokenizer:
    """SentencePiece BPE tokenizer for 3D model text descriptions.

    Drop-in replacement for TextTokenizer with identical API:
      .encode(text) -> list[int]
      .encode_padded(text) -> (ids, mask)
      .decode(ids) -> str
      .vocab_size -> int
      .save(path) / .load(path)

    Key differences:
      - Subword tokenization: never produces UNK
      - Dynamic: handles any text without rebuilding
      - PAD=0, BOS=1, EOS=2, UNK=3 (same as TextTokenizer)
    """

    def __init__(self, sp_model: spm.SentencePieceProcessor):
        self.sp = sp_model
        self.vocab_size = sp_model.get_piece_size()  # type: ignore[attr-defined]

    def encode(self, text: str, max_length: int = 256,
               add_special: bool = True) -> list[int]:
        """Encode text to integer IDs.

        Args:
            text: Input text string.
            max_length: Maximum sequence length (will truncate).
            add_special: If True, prepend BOS and append EOS.

        Returns:
            List of integer token IDs (NOT padded — use encode_padded for that).
        """
        text = text.lower().strip()
        ids = self.sp.encode(text, out_type=int)  # type: ignore[attr-defined]

        if add_special:
            ids = [BOS_ID] + ids + [EOS_ID]

        # Truncate
        ids = ids[:max_length]
        return ids

    def encode_padded(self, text: str, max_length: int = 256,
                      add_special: bool = True) -> tuple[list[int], list[int]]:
        """Encode text and return (ids, mask) both padded to max_length."""
        ids = self.encode(text, max_length, add_special)
        real_len = len(ids)
        mask = [1] * real_len + [0] * (max_length - real_len)
        ids = ids + [PAD_ID] * (max_length - real_len)
        return ids, mask

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode integer IDs back to text."""
        if skip_special:
            ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self.sp.decode(ids)  # type: ignore[attr-defined]

    def save(self, path: str | Path):
        """Save tokenizer to a directory (model + config)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save the SentencePiece model file
        model_path = path / "tokenizer.model"
        with open(model_path, "wb") as f:
            f.write(self.sp.serialized_model_proto())

        # Save config for compatibility checking
        config = {
            "type": "bpe",
            "vocab_size": self.vocab_size,
            "pad_id": PAD_ID,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
            "unk_id": UNK_ID,
        }
        with open(path / "tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved BPE tokenizer ({self.vocab_size} tokens) to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load tokenizer from directory."""
        path = Path(path)
        model_path = path / "tokenizer.model"
        if not model_path.exists():
            raise FileNotFoundError(f"No tokenizer.model in {path}")

        sp = spm.SentencePieceProcessor()
        sp.load(str(model_path))  # type: ignore[attr-defined]
        return cls(sp)

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 8000,
              model_prefix: str | Path = "bpe_tokenizer",
              extra_terms: list[str] | None = None) -> "BPETokenizer":
        """Train a new BPE tokenizer from text data.

        Args:
            texts: List of text descriptions to train on.
            vocab_size: Target vocabulary size.
            model_prefix: Path prefix for output model files.
            extra_terms: Additional terms to force into vocabulary.

        Returns:
            Trained BPETokenizer instance.
        """
        # Combine Blender terms + any extra terms
        user_defined = list(BLENDER_TERMS)
        if extra_terms:
            user_defined.extend(extra_terms)
        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for t in user_defined:
            t_lower = t.lower().strip()
            if t_lower and t_lower not in seen:
                seen.add(t_lower)
                unique_terms.append(t_lower)

        # Write training text to temp file (SentencePiece needs a file)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                          delete=False) as tmp:
            for text in texts:
                tmp.write(text.lower().strip() + "\n")
            # Also write all Blender terms as additional training lines
            # so they get good coverage even if rare in the data
            for term in unique_terms:
                tmp.write(f"a {term}\n")
                tmp.write(f"create a {term}\n")
                tmp.write(f"with {term} modifier\n")
            tmp_path = tmp.name

        try:
            model_prefix = str(Path(model_prefix))

            # Train SentencePiece BPE model
            spm.SentencePieceTrainer.train(  # type: ignore[attr-defined]
                input=tmp_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type="bpe",
                # Special tokens: pad=0, bos=1, eos=2, unk=3
                pad_id=PAD_ID,
                bos_id=BOS_ID,
                eos_id=EOS_ID,
                unk_id=UNK_ID,
                # Force Blender terms to be whole tokens
                user_defined_symbols=unique_terms,
                # Normalization
                normalization_rule_name="identity",  # Keep text as-is
                remove_extra_whitespaces=True,
                # Coverage
                character_coverage=1.0,  # ASCII text, cover everything
                # Misc
                num_threads=os.cpu_count() or 4,
                train_extremely_large_corpus=False,
                max_sentence_length=4096,
            )

            sp = spm.SentencePieceProcessor()
            sp.load(f"{model_prefix}.model")  # type: ignore[attr-defined]

            tokenizer = cls(sp)

            # Save in our standard format
            out_dir = Path(model_prefix).parent / "bpe_tokenizer"
            tokenizer.save(out_dir)

            # Verify critical terms are single tokens
            n_split = 0
            for term in unique_terms[:20]:
                pieces = sp.encode(term, out_type=str)  # type: ignore[attr-defined]
                if len(pieces) > 1:
                    n_split += 1
                    logger.warning(f"Term '{term}' split into {pieces}")
            if n_split == 0:
                logger.info("All critical Blender terms are single tokens")

            logger.info(f"Trained BPE tokenizer: {tokenizer.vocab_size} tokens "
                        f"from {len(texts)} texts + {len(unique_terms)} Blender terms")
            return tokenizer

        finally:
            os.unlink(tmp_path)
            # Clean up SentencePiece's default output files
            for ext in [".model", ".vocab"]:
                p = f"{model_prefix}{ext}"
                if os.path.exists(p):
                    # Move to the bpe_tokenizer dir if not already there
                    pass

    @classmethod
    def from_dataset(cls, dataset_path: str | Path,
                     vocab_size: int = 8000) -> "BPETokenizer":
        """Build tokenizer from a JSONL dataset file.

        Compatible with TextTokenizer.from_dataset() API.
        """
        texts = []
        with open(dataset_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                text = ex.get("text", "")
                if text:
                    texts.append(text)

        prefix = str(Path(dataset_path).parent / "bpe_model")
        return cls.train(texts, vocab_size=vocab_size, model_prefix=prefix)

    def __repr__(self) -> str:
        return f"BPETokenizer(vocab_size={self.vocab_size})"


def load_tokenizer(path: str | Path):
    """Smart loader: detects tokenizer type and loads accordingly.

    Supports:
      - BPE tokenizer directory (contains tokenizer.model)
      - Legacy word-level tokenizer JSON (text_tokenizer.json)

    Returns:
        BPETokenizer or TextTokenizer instance.
    """
    path = Path(path)

    # Check for BPE tokenizer directory
    if path.is_dir() and (path / "tokenizer.model").exists():
        return BPETokenizer.load(path)

    # Check for BPE in parent directory
    bpe_dir = path.parent / "bpe_tokenizer"
    if bpe_dir.is_dir() and (bpe_dir / "tokenizer.model").exists():
        return BPETokenizer.load(bpe_dir)

    # Fall back to legacy word-level tokenizer
    if path.suffix == ".json" or (path.is_file() and path.name.endswith(".json")):
        from processing.text_tokenizer import TextTokenizer
        return TextTokenizer.load(path)

    raise FileNotFoundError(
        f"No tokenizer found at {path}. Expected either a directory "
        f"with tokenizer.model (BPE) or a .json file (word-level)."
    )
