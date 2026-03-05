from __future__ import annotations

import re
from dataclasses import dataclass

from policy.state import Goal, hash_text_features


@dataclass(frozen=True)
class ParsedRequest:
    """Deterministic parse result for a free-text user request.

    The parser does exactly two things:
      1. Hash the full prompt into a compact fingerprint (text_features).
         The policy model learns from training data what actions correspond
         to what fingerprint — no object knowledge is hardcoded here.
      2. Extract any explicit numeric overrides the user typed
         (e.g. "2k verts", "symmetry 0.8").

    Everything else — complexity, topology, materials, subdivision — is
    a learned behaviour, not a rule.
    """

    goal: Goal
    low_poly_style: str = "unspecified"


# Only used to extract numeric values the user explicitly typed.
_NUM_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<k>[kK])?\s*(?P<unit>verts?|vertices|faces?)\b"
)
_SYM_RE = re.compile(
    r"\b(?:symmetry|symmetric(?:ity)?)\s*[:=]?\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<pct>%|percent)?\b"
)


def infer_low_poly_style(prompt: str) -> str:
    """Deterministic low-poly style disambiguation.

    Returns one of:
      - "stylized"
      - "retro"
      - "unspecified"
    """
    text = (prompt or "").lower()
    if "low poly" not in text:
        return "unspecified"

    stylized_cues = (
        "flat shaded",
        "faceted",
        "stylized low poly",
        "low poly art",
    )
    retro_cues = (
        "ps1",
        "retro",
        "n64",
        "smooth low poly",
        "low-res texture",
    )

    if any(c in text for c in stylized_cues):
        return "stylized"
    if any(c in text for c in retro_cues):
        return "retro"
    return "stylized"


def parse_prompt_to_goal(prompt: str) -> ParsedRequest:
    """Parse a free-text user prompt into a Goal.

    This function encodes NO domain knowledge about objects, materials,
    or appropriate complexity.  All of that is learned by the policy model
    from training data.  The parser only:

      1. Hashes the full prompt into a compact fingerprint (text_features).
         The model learns what each fingerprint implies from examples it has
         seen during training — "a cube", "a low poly lamborghini", etc. all
         become vectors that the model maps to learned behaviour.

      2. Extracts explicit numeric constraints the user typed verbatim:
           "2k verts", "500 vertices", "symmetry 0.8", "symmetry: 70%"
         If absent, target_vertex_count and target_symmetry default to 0,
         meaning "unconstrained — let the model decide from text_features".
    """
    text = (prompt or "").strip()

    # -- 1. Text fingerprint (always computed) ----------------------------
    feats = hash_text_features(text)

    # -- 2. Explicit vertex/face count override ---------------------------
    target_vertex_count = 0
    m = _NUM_RE.search(text)
    if m:
        num = float(m.group("num"))
        if m.group("k"):
            num *= 1000.0
        if (m.group("unit") or "").lower().startswith("face"):
            num *= 1.5  # very rough faces→verts proxy
        target_vertex_count = max(1, int(num))

    # -- 3. Explicit symmetry override ------------------------------------
    target_symmetry = 0.0
    sm = _SYM_RE.search(text)
    if sm:
        sv = float(sm.group("val"))
        if sm.group("pct"):
            sv /= 100.0
        target_symmetry = max(0.0, min(1.0, sv))

    goal = Goal(
        target_vertex_count=target_vertex_count,
        target_symmetry=target_symmetry,
        text_features=feats,
    )
    style = infer_low_poly_style(text)
    return ParsedRequest(goal=goal, low_poly_style=style)

