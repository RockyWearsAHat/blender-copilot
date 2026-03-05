"""Prompt → reference mesh resolver.

This module bridges text prompts to *reference meshes* stored in the
tokenized geometry datasets under data/datasets/geometry/*.jsonl.

Why: we want scoring dominated by reconstruction fidelity (e.g. Chamfer,
F-score) against a prompt-specific reference.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]+")


def normalize_prompt_text(text: str) -> str:
    """Normalize prompt text for stable matching.

    Intentionally simple: lowercases, strips punctuation, collapses
    whitespace, and removes leading articles.
    """
    t = (text or "").strip().lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    for article in ("a ", "an ", "the "):
        if t.startswith(article):
            t = t[len(article):].strip()
    return t


@dataclass(frozen=True)
class ReferenceRecord:
    prompt: str
    tokens: list[int]
    num_vertices: int | None = None
    num_faces: int | None = None
    source: str | None = None
    split: str | None = None


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


@lru_cache(maxsize=256)
def find_reference_for_prompt(
    prompt: str,
    *,
    geometry_dir: Path = Path("data/datasets/geometry"),
    splits: tuple[str, ...] = ("train", "val", "test"),
) -> Optional[ReferenceRecord]:
    """Return a reference record whose prompt matches `prompt`.

    Matching is exact on normalized prompt text.
    Returns None when no reference exists in the datasets.
    """
    wanted = normalize_prompt_text(prompt)
    if not wanted:
        return None

    geometry_dir = Path(geometry_dir)
    for split in splits:
        path = geometry_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for rec in _iter_jsonl(path):
            txt = rec.get("text")
            if not isinstance(txt, str):
                continue
            if normalize_prompt_text(txt) != wanted:
                continue
            toks = rec.get("tokens")
            if not isinstance(toks, list) or not toks:
                continue
            return ReferenceRecord(
                prompt=txt,
                tokens=[int(x) for x in toks],
                num_vertices=(int(rec["num_vertices"]) if "num_vertices" in rec else None),
                num_faces=(int(rec["num_faces"]) if "num_faces" in rec else None),
                source=(str(rec["source"]) if "source" in rec else None),
                split=str(split),
            )

    # Small convenience: try with leading article if user didn't include one.
    # This helps prompts like "cylinder" match dataset entries like "a cylinder".
    if not prompt.strip().lower().startswith(("a ", "an ", "the ")):
        return find_reference_for_prompt(
            f"a {prompt}",
            geometry_dir=geometry_dir,
            splits=splits,
        )

    return None
