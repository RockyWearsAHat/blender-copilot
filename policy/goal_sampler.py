from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from policy.prompt_parser import parse_prompt_to_goal
from policy.state import Goal, PROMPT_FEAT_DIM, hash_text_features


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


@dataclass(frozen=True)
class GoalRecord:
    goal: Goal
    text: str | None = None
    source: str | None = None
    num_vertices: int | None = None
    num_faces: int | None = None


class GoalSampler(Protocol):
    def sample(self, rng: np.random.Generator) -> Goal:
        ...


class UniformGoalSampler:
    def __init__(
        self,
        *,
        min_vertices: int = 200,
        max_vertices: int = 5000,
        min_symmetry: float = 0.4,
        max_symmetry: float = 0.9,
    ):
        self._min_v = int(min_vertices)
        self._max_v = int(max_vertices)
        self._min_s = float(min_symmetry)
        self._max_s = float(max_symmetry)

    def sample(self, rng: np.random.Generator) -> Goal:
        return Goal(
            target_vertex_count=int(rng.integers(self._min_v, self._max_v + 1)),
            target_symmetry=float(rng.uniform(self._min_s, self._max_s)),
            # No prompt text during uniform sampling; text_features stays zeros.
            text_features=np.zeros(PROMPT_FEAT_DIM, dtype=np.float32),
        )


class ListGoalSampler:
    def __init__(self, records: list[GoalRecord]):
        if not records:
            raise ValueError("ListGoalSampler requires at least 1 record")
        self._records = list(records)

    def sample(self, rng: np.random.Generator) -> Goal:
        idx = int(rng.integers(0, len(self._records)))
        return self._records[idx].goal


def load_goal_records_from_geometry_jsonl(
    *,
    path: Path,
    max_records: int = 50_000,
    min_vertices: int = 50,
    max_vertices: int = 20_000,
) -> list[GoalRecord]:
    """Load target goals from the scraped geometry dataset.

    Schema (at least in the current repo):
      {"text": str, "num_vertices": int, "num_faces": int, ...}

    We intentionally only use compact numeric targets + optional text-derived
    symmetry hint (via deterministic prompt parsing).
    """

    p = Path(path)
    records: list[GoalRecord] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= int(max_records):
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            nv = obj.get("num_vertices")
            if nv is None:
                continue
            try:
                nv_i = int(nv)
            except Exception:
                continue
            if nv_i < int(min_vertices) or nv_i > int(max_vertices):
                continue

            text = obj.get("text")
            if not isinstance(text, str):
                text = ""

            parsed = parse_prompt_to_goal(text)
            goal = Goal(
                target_vertex_count=_clamp_int(nv_i, int(min_vertices), int(max_vertices)),
                target_symmetry=float(parsed.goal.target_symmetry),
                text_features=parsed.goal.text_features,
            )

            nf = obj.get("num_faces")
            try:
                nf_i = int(nf) if nf is not None else None
            except Exception:
                nf_i = None

            src = obj.get("source")
            src_s = src if isinstance(src, str) else None

            records.append(
                GoalRecord(goal=goal, text=(text or None), source=src_s, num_vertices=nv_i, num_faces=nf_i)
            )

    if not records:
        raise RuntimeError(f"No usable goal records found in geometry jsonl: {p}")
    return records


def load_goal_records_from_mesh_cache_dir(
    *,
    cache_dir: Path,
    max_files: int = 20_000,
    min_vertices: int = 50,
    max_vertices: int = 20_000,
    min_quality_weight: float | None = None,
) -> list[GoalRecord]:
    """Load target goals from `.mesh_cache/*.pt` records."""

    d = Path(cache_dir)
    records: list[GoalRecord] = []

    pt_files = sorted(d.glob("*.pt"))
    for i, p in enumerate(pt_files):
        if i >= int(max_files):
            break
        try:
            obj = torch.load(p, map_location="cpu")
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue

        nv = obj.get("original_vert_count")
        if nv is None:
            continue
        try:
            nv_i = int(nv)
        except Exception:
            continue
        if nv_i < int(min_vertices) or nv_i > int(max_vertices):
            continue

        if min_quality_weight is not None:
            qw = obj.get("quality_weight")
            try:
                qw_f = float(qw.item() if isinstance(qw, torch.Tensor) else qw)
            except Exception:
                qw_f = None
            if qw_f is None or qw_f < float(min_quality_weight):
                continue

        text = obj.get("label")
        if not isinstance(text, str) or not text.strip():
            ws = obj.get("workflow_supervision")
            if isinstance(ws, dict):
                ti = ws.get("target_instruction")
                if isinstance(ti, str):
                    text = ti
        if not isinstance(text, str):
            text = ""

        parsed = parse_prompt_to_goal(text)
        goal = Goal(
            target_vertex_count=_clamp_int(nv_i, int(min_vertices), int(max_vertices)),
            target_symmetry=float(parsed.goal.target_symmetry),
            text_features=parsed.goal.text_features,
        )

        nf = obj.get("original_face_count")
        try:
            nf_i = int(nf) if nf is not None else None
        except Exception:
            nf_i = None

        src = obj.get("data_source")
        src_s = src if isinstance(src, str) else None

        records.append(GoalRecord(goal=goal, text=(text or None), source=src_s, num_vertices=nv_i, num_faces=nf_i))

    if not records:
        raise RuntimeError(f"No usable goal records found in mesh cache dir: {d}")
    return records
