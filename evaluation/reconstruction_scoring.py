"""Reference-based reconstruction scoring for policy rollouts.

Computes geometric similarity metrics (Chamfer, F-score, etc.) between:
  - a predicted mesh exported from Blender (OBJ)
  - a prompt-specific reference mesh (decoded from token sequences)

Also computes a simple, deterministic trajectory-quality score from the
closed-loop rollout logs (action_*.json + state_*.json).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RolloutTrajectoryScore:
    path_score: float
    breakdown: dict


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def load_obj_as_tri_mesh(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an OBJ and return (vertices, faces) with triangular faces.

    Intentionally avoids heavyweight loaders so evaluation doesn't depend on
    optional image/material libraries (e.g. Pillow).
    """

    obj_path = Path(obj_path)
    if not obj_path.exists():
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

    vertices: list[list[float]] = []
    faces_tris: list[list[int]] = []

    def _parse_face_index(tok: str, n_verts: int) -> int:
        # f tokens can be: v, v/vt, v//vn, v/vt/vn
        head = tok.split("/")[0].strip()
        if not head:
            return -1
        try:
            idx = int(head)
        except Exception:
            return -1
        # OBJ is 1-based; negative indices are relative to end.
        if idx < 0:
            idx = n_verts + idx
        else:
            idx = idx - 1
        return idx

    with obj_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except Exception:
                        continue
            elif line.startswith("f "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                idxs = [_parse_face_index(tok, len(vertices)) for tok in parts[1:]]
                idxs = [i for i in idxs if 0 <= i < len(vertices)]
                if len(idxs) < 3:
                    continue
                # Triangulate polygon by fan.
                v0 = idxs[0]
                for k in range(1, len(idxs) - 1):
                    faces_tris.append([v0, idxs[k], idxs[k + 1]])

    v = np.asarray(vertices, dtype=np.float64)
    f_arr = np.asarray(faces_tris, dtype=np.int64)
    if f_arr.ndim != 2 or (f_arr.size > 0 and f_arr.shape[1] != 3):
        f_arr = f_arr.reshape((-1, 3)) if f_arr.size else np.zeros((0, 3), dtype=np.int64)
    return v, f_arr


def normalize_vertices_unit_bbox(vertices: np.ndarray) -> np.ndarray:
    """Center + scale mesh so max bbox dimension becomes 2.0.

    This makes reference token meshes (typically normalized to [-1,1])
    comparable to Blender exports even when absolute scale differs.
    """
    if vertices.size == 0:
        return vertices
    v = np.asarray(vertices, dtype=np.float64)
    min_v = v.min(axis=0)
    max_v = v.max(axis=0)
    center = (min_v + max_v) * 0.5
    dims = (max_v - min_v)
    scale = float(max(dims.max(), 1e-9))
    # map to approx [-1, 1] range
    v = (v - center) * (2.0 / scale)
    return v


def decode_reference_tokens(tokens: list[int]) -> tuple[np.ndarray, np.ndarray]:
    from processing.mesh_tokenizer import MeshTokenizer

    tok = MeshTokenizer(vocab_size=8192, max_faces=2048)
    verts, faces = tok.decode_tokens([int(t) for t in tokens])
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    return v, f


def compute_reconstruction_metrics(
    *,
    obj_path: Path,
    reference_tokens: list[int],
    n_surface_points: int = 4096,
    normalize: bool = True,
) -> dict:
    from evaluation.metrics import evaluate_single

    v_pred, f_pred = load_obj_as_tri_mesh(obj_path)
    v_ref, f_ref = decode_reference_tokens(reference_tokens)

    if normalize:
        v_pred = normalize_vertices_unit_bbox(v_pred)
        v_ref = normalize_vertices_unit_bbox(v_ref)

    return evaluate_single(
        v_pred,
        f_pred,
        v_ref,
        f_ref,
        n_surface_points=int(n_surface_points),
    )


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def infer_expected_base_primitives_from_source(source: Optional[str]) -> tuple[Optional[set[str]], str]:
    """Infer expected base primitive(s) from dataset provenance.

    Returns (expected_set, reason). expected_set is None when unknown or
    not supported by the current action space.
    """
    if not source:
        return None, "no_source"

    s = str(source).strip().lower()
    if not s:
        return None, "empty_source"

    # Supported today by the policy action space.
    if "cylinder" in s:
        return {"ADD_CYLINDER"}, "source_cylinder"
    if "cube" in s or "box" in s or "grid" in s:
        return {"ADD_CUBE"}, "source_cubeish"

    # Many dataset generators exist (plane/terrain/torus/etc). Until the
    # policy action space includes them, don't penalize primitive choice.
    return None, "source_unsupported"


def score_rollout_trajectory(
    out_dir: Path,
    *,
    expected_base_primitives: Optional[set[str]] = None,
    expected_base_reason: str = "",
) -> RolloutTrajectoryScore:
    """Score step-by-step behavior from action/state logs.

    Goal: encourage "selection → edit" workflows and discourage exploding
    topology or wasting steps. This is *not* a Blender semantic shim; it is
    purely a scoring signal.
    """
    from policy.actions import ActionType

    out_dir = Path(out_dir)

    trace_files = sorted(out_dir.glob("trace_*.json"))
    use_trace = len(trace_files) > 0

    action_files = sorted(out_dir.glob("action_*.json"))
    state_files = sorted(out_dir.glob("state_*.json"))
    n = min(len(action_files), len(state_files))
    if n <= 0:
        return RolloutTrajectoryScore(path_score=0.0, breakdown={"reason": "no_steps"})

    if use_trace:
        n = min(n, len(trace_files))

    actions: list[dict] = []
    states: list[dict] = []
    traces: list[dict] = []
    for i in range(n):
        a = _read_json(action_files[i]) or {}
        s = _read_json(state_files[i]) or {}
        actions.append({"action_type": int(a.get("action_type", 0)), "param": int(a.get("param", 0))})
        states.append(s.get("stats", {}) if isinstance(s, dict) else {})
        if use_trace:
            traces.append(_read_json(trace_files[i]) or {})

    noop_steps = 0
    growth_faces = 0
    growth_verts = 0
    modifier_steps = 0
    subdivide_steps = 0
    selection_then_edit = 0
    modifier_stack_changes = 0
    effective_deltas: list[float] = []

    edit_ops = {
        int(ActionType.EXTRUDE),
        int(ActionType.INSET),
        int(ActionType.BEVEL),
        int(ActionType.SUBDIVIDE),
        int(ActionType.DELETE_FACE),
    }

    def _sig(st: dict) -> tuple[int, int, int, int]:
        try:
            return (
                int(st.get("vertex_count", 0)),
                int(st.get("face_count", 0)),
                int(st.get("edge_count", 0)),
                int(st.get("selected_face_count", 0)),
            )
        except Exception:
            return (0, 0, 0, 0)

    prev_sig = _sig(states[0])
    prev_mods = None
    if use_trace:
        try:
            prev_mods = traces[0].get("modifiers", {}).get("after")
        except Exception:
            prev_mods = None

    for i in range(n):
        st = states[i]
        sig = _sig(st)

        if i > 0:
            dv = sig[0] - prev_sig[0]
            df = sig[1] - prev_sig[1]
            growth_verts += max(0, dv)
            growth_faces += max(0, df)

            if use_trace:
                did_apply = bool(traces[i].get("did_apply", True))
                if not did_apply:
                    noop_steps += 1
            else:
                if sig == prev_sig:
                    noop_steps += 1

            # High-leverage step proxy: reward steps that change a lot of
            # geometry (relative to current mesh scale), but saturate.
            if use_trace:
                did_apply = bool(traces[i].get("did_apply", True))
            else:
                did_apply = (sig != prev_sig)

            if did_apply:
                prev_v, prev_f, prev_e, _prev_sel = prev_sig
                dv = abs(sig[0] - prev_v)
                df = abs(sig[1] - prev_f)
                de = abs(sig[2] - prev_e)
                denom = float(max(1, prev_v + prev_f + prev_e))
                geom_ratio = float(min(1.0, (dv + df + de) / denom))

                # BBox change ratio (use extents if present)
                try:
                    bb_prev = states[i - 1].get("bounding_box", {})
                    bb_curr = st.get("bounding_box", {})
                    prev_sum = float(abs(bb_prev.get("x", 0.0)) + abs(bb_prev.get("y", 0.0)) + abs(bb_prev.get("z", 0.0)))
                    curr_sum = float(abs(bb_curr.get("x", 0.0)) + abs(bb_curr.get("y", 0.0)) + abs(bb_curr.get("z", 0.0)))
                    bbox_ratio = float(min(1.0, abs(curr_sum - prev_sum) / max(1e-6, prev_sum)))
                except Exception:
                    bbox_ratio = 0.0

                eff = float(min(1.0, 0.75 * geom_ratio + 0.25 * bbox_ratio))
                effective_deltas.append(eff)

        a_type = int(actions[i]["action_type"])
        if a_type == int(ActionType.APPLY_MODIFIER):
            modifier_steps += 1
        if a_type == int(ActionType.SUBDIVIDE):
            subdivide_steps += 1

        if use_trace:
            try:
                mods_after = traces[i].get("modifiers", {}).get("after")
            except Exception:
                mods_after = None
            if prev_mods is not None and mods_after is not None:
                if json.dumps(prev_mods, sort_keys=True) != json.dumps(mods_after, sort_keys=True):
                    modifier_stack_changes += 1
            prev_mods = mods_after

        if i + 1 < n:
            if a_type == int(ActionType.SELECT_RANDOM_FACE):
                nxt = int(actions[i + 1]["action_type"])
                if nxt in edit_ops:
                    selection_then_edit += 1

        prev_sig = sig

    noop_ratio = noop_steps / max(1, n - 1)
    modifier_ratio = modifier_steps / max(1, n)
    subdiv_ratio = subdivide_steps / max(1, n)
    sel_edit_ratio = selection_then_edit / max(1, n - 1)

    # Growth penalty: saturating function so a few growth steps are OK.
    growth = float(growth_faces + growth_verts)
    growth_pen = growth / (growth + 2000.0)

    # --- Base mesh (primitive choice) scoring ---
    base_found = False
    base_action = None
    base_step = None
    base_implicit = False
    noops_before_base = 0

    expected = set(expected_base_primitives) if expected_base_primitives else None
    expected_reason = str(expected_base_reason or "")

    if use_trace:
        for i in range(n):
            tr = traces[i] if i < len(traces) else {}
            try:
                did_apply = bool(tr.get("did_apply", True))
            except Exception:
                did_apply = True
            try:
                a_name = str(tr.get("action", {}).get("action_name", ""))
            except Exception:
                a_name = ""

            if not base_found:
                if not did_apply:
                    noops_before_base += 1
                if did_apply and a_name in ("ADD_CUBE", "ADD_CYLINDER"):
                    base_found = True
                    base_action = a_name
                    base_step = i
                    break

    if not base_found:
        # The worker always ensures an active mesh object exists; treat that
        # implicit start as a cube base mesh if the policy never adds one.
        base_action = "ADD_CUBE"
        base_step = None
        base_implicit = True

        # Only treat implicit cube as "correct" when the expected primitive
        # explicitly includes cube (reference-driven).
        if expected is not None and "ADD_CUBE" in expected:
            base_step = 0
            noops_before_base = 0

    if expected is None:
        # Unknown/unsupported expected primitive → don't guess; be neutral.
        base_match = 1.0
    elif base_action in expected:
        base_match = 1.0
    else:
        base_match = 0.0

    if base_step is None:
        # No base primitive and cube not acceptable (e.g. cylinder prompt).
        base_delay_ratio = 1.0
    else:
        base_delay_ratio = float(base_step) / float(max(1, n - 1))

    noops_before_base_ratio = float(noops_before_base) / float(max(1, (base_step if base_step is not None else n)))

    base_mesh_score = float(base_match)
    base_mesh_score *= float(1.0 - 0.6 * _clamp01(base_delay_ratio))
    base_mesh_score *= float(1.0 - 0.5 * _clamp01(noops_before_base_ratio))
    base_mesh_score = _clamp01(base_mesh_score)

    # Combine into [0,1] quality score.
    # - reward selection→edit patterns
    # - penalize wasted steps and topology explosions
    avg_effective_delta = float(sum(effective_deltas) / max(1, len(effective_deltas)))

    raw_core = (
        0.55 * float(sel_edit_ratio)
        + 0.45 * float(1.0 - noop_ratio)
        + 0.25 * float(avg_effective_delta)
        - 0.35 * float(growth_pen)
        - 0.20 * float(modifier_ratio)
        - 0.20 * float(subdiv_ratio)
    )
    core_score = _clamp01(raw_core)

    # Base mesh affects the score as a multiplicative factor so we don't reward
    # trajectories that do nothing (core_score≈0) just because the implicit cube exists.
    if expected is None:
        base_factor = 1.0
    else:
        # Strong penalty when the expected primitive is known and we didn't
        # choose it.
        base_factor = 0.4 + 0.6 * float(base_mesh_score)  # in [0.4, 1.0]
    path_score = _clamp01(float(core_score) * float(base_factor))

    return RolloutTrajectoryScore(
        path_score=path_score,
        breakdown={
            "n_steps": int(n),
            "noop_steps": int(noop_steps),
            "noop_ratio": float(noop_ratio),
            "growth_faces": int(growth_faces),
            "growth_verts": int(growth_verts),
            "growth_penalty": float(growth_pen),
            "modifier_steps": int(modifier_steps),
            "modifier_ratio": float(modifier_ratio),
            "subdivide_steps": int(subdivide_steps),
            "subdivide_ratio": float(subdiv_ratio),
            "selection_then_edit": int(selection_then_edit),
            "selection_then_edit_ratio": float(sel_edit_ratio),
            "modifier_stack_changes": int(modifier_stack_changes),
            "avg_effective_delta": float(avg_effective_delta),
            "effective_applied_steps": int(len(effective_deltas)),
            "base_mesh": {
                "expected": (sorted(expected) if expected is not None else None),
                "expected_reason": str(expected_reason),
                "chosen": str(base_action),
                "explicit_add_found": bool(base_found),
                "implicit_start_cube": bool(base_implicit),
                "step": (int(base_step) if base_step is not None else None),
                "delay_ratio": float(base_delay_ratio),
                "noops_before": int(noops_before_base),
                "noops_before_ratio": float(noops_before_base_ratio),
                "score": float(base_mesh_score),
            },
            "used_trace": bool(use_trace),
        },
    )
