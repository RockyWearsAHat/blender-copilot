from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptAugmentConfig:
    max_variants: int = 12


_TEMPLATES: tuple[str, ...] = (
    "{label}",
    "a {label}",
    "the {label}",
    "a 3d {label}",
    "a detailed {label}",
    "a low poly {label}",
    "a simple {label}",
    "{label}, 3d model",
)


def _normalize_label(label: str) -> str:
    s = (label or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def make_prompt_variants(label: str, cfg: PromptAugmentConfig | None = None) -> list[str]:
    """Create a small set of prompt variants for a base label.

    This intentionally stays lightweight and deterministic-ish so it can be
    used offline while building datasets.

    Rationale:
    - The policy's `text_features` are hashed char n-grams; seeing multiple
      near-synonymous phrasings helps generalization to user prompts that are
      "close" but not identical.
    - We keep the augmentation conservative to avoid injecting incorrect
      semantics.
    """

    cfg = cfg or PromptAugmentConfig()
    base = _normalize_label(label)
    if not base:
        return []

    variants: list[str] = []

    def _add(s: str) -> None:
        s = _normalize_label(s)
        if not s:
            return
        if s not in variants:
            variants.append(s)

    for t in _TEMPLATES:
        _add(t.format(label=base))

    # Gentle domain-ish expansions for common landscape words.
    # These do not change the noun, they only add modifiers users often type.
    if any(w in base for w in ("terrain", "landscape", "hills", "mountains", "mountain", "valley")):
        _add(f"grassy {base}")
        _add(f"rocky {base}")
        _add(f"rolling {base}")
        if "terrain" not in base:
            _add(f"{base} terrain")

    # Clamp.
    return variants[: max(0, int(cfg.max_variants))]
