"""Word-level text tokenizer for Blender model training.

Replaces the naive `ord(c) % 32000` character-level tokenizer with a
proper word-level tokenizer that builds a vocabulary from training data.

Vocabulary is small (~2-4k tokens) because our text descriptions are
short structured strings like:
    "Lantern | tags: lamp, assets | (1.3m × 2.0m × 1.3m) | 7296 faces"

Usage:
    # Build from training data
    tokenizer = TextTokenizer.from_dataset("data/datasets/geometry/train.jsonl")
    tokenizer.save("data/datasets/geometry/text_tokenizer.json")

    # Load saved tokenizer
    tokenizer = TextTokenizer.load("data/datasets/geometry/text_tokenizer.json")

    # Encode / decode
    ids = tokenizer.encode("Create a red cube")
    text = tokenizer.decode(ids)
"""

import json
import re
from collections import Counter
from pathlib import Path


# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def _tokenize_text(text: str) -> list[str]:
    """Split text into tokens.

    Splits on:
    - Words (alphabetic sequences)
    - Numbers (including decimals like 1.3)
    - Individual punctuation / symbols
    """
    return re.findall(r"[a-zA-Z]+|[0-9]+\.?[0-9]*|[^\s]", text.lower())


class TextTokenizer:
    """Word-level tokenizer for 3D model text descriptions.

    Attributes:
        vocab: Dict mapping token string → integer ID.
        id_to_token: Dict mapping integer ID → token string.
        vocab_size: Total vocabulary size including special tokens.
    """

    def __init__(self, vocab: dict[str, int] | None = None):
        if vocab is None:
            vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)

    def encode(self, text: str, max_length: int = 256,
               add_special: bool = True) -> list[int]:
        """Encode text to integer IDs.

        Args:
            text: Input text string.
            max_length: Maximum sequence length (will pad/truncate).
            add_special: If True, prepend BOS and append EOS.

        Returns:
            List of integer token IDs, padded to max_length.
        """
        tokens = _tokenize_text(text)
        ids = [self.vocab.get(t, UNK_ID) for t in tokens]

        if add_special:
            ids = [BOS_ID] + ids + [EOS_ID]

        # Truncate
        ids = ids[:max_length]

        return ids

    def encode_padded(self, text: str, max_length: int = 256,
                      add_special: bool = True) -> tuple[list[int], list[int]]:
        """Encode text and return (ids, mask) both padded to max_length.

        Args:
            text: Input text.
            max_length: Pad/truncate to this length.
            add_special: Add BOS/EOS tokens.

        Returns:
            (ids, mask) where mask is 1 for real tokens, 0 for padding.
        """
        ids = self.encode(text, max_length, add_special)
        real_len = len(ids)
        mask = [1] * real_len + [0] * (max_length - real_len)
        ids = ids + [PAD_ID] * (max_length - real_len)
        return ids, mask

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode integer IDs back to text.

        Args:
            ids: List of token IDs.
            skip_special: If True, skip PAD/BOS/EOS tokens in output.

        Returns:
            Reconstructed text string.
        """
        skip = {PAD_ID, BOS_ID, EOS_ID} if skip_special else set()
        tokens = []
        for i in ids:
            if i in skip:
                continue
            tokens.append(self.id_to_token.get(i, UNK_TOKEN))
        return " ".join(tokens)

    def save(self, path: str | Path):
        """Save tokenizer vocabulary to JSON file."""
        data = {
            "version": 1,
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "TextTokenizer":
        """Load tokenizer from saved JSON file."""
        with open(path) as f:
            data = json.load(f)
        vocab = {k: int(v) for k, v in data["vocab"].items()}
        return cls(vocab=vocab)

    @classmethod
    def from_dataset(cls, dataset_path: str | Path,
                     min_freq: int = 1,
                     max_vocab: int = 8000) -> "TextTokenizer":
        """Build tokenizer vocabulary from a JSONL dataset.

        Args:
            dataset_path: Path to train.jsonl file.
            min_freq: Minimum token frequency to include in vocab.
            max_vocab: Maximum vocabulary size (including special tokens).

        Returns:
            TextTokenizer with vocabulary built from the data.
        """
        counter: Counter[str] = Counter()

        with open(dataset_path) as f:
            for line in f:
                ex = json.loads(line.strip())
                text = ex.get("text", "")
                tokens = _tokenize_text(text)
                counter.update(tokens)

        # Build vocab: special tokens first, then by frequency
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        idx = len(SPECIAL_TOKENS)

        for token, freq in counter.most_common():
            if idx >= max_vocab:
                break
            if freq < min_freq:
                continue
            if token not in vocab:
                vocab[token] = idx
                idx += 1

        tokenizer = cls(vocab=vocab)
        return tokenizer

    @classmethod
    def from_texts(cls, texts: list[str],
                   min_freq: int = 1,
                   max_vocab: int = 8000) -> "TextTokenizer":
        """Build tokenizer vocabulary from a list of text strings."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(_tokenize_text(text))

        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        idx = len(SPECIAL_TOKENS)

        for token, freq in counter.most_common():
            if idx >= max_vocab:
                break
            if freq < min_freq:
                continue
            if token not in vocab:
                vocab[token] = idx
                idx += 1

        return cls(vocab=vocab)

    def __repr__(self) -> str:
        return f"TextTokenizer(vocab_size={self.vocab_size})"
