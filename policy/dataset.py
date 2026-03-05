from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from policy.actions import ActionType, Action, PARAM_BINS
from policy.env import MeshStatsEnv
from policy.state import Goal, normalize_goal, normalize_state_vector, PROMPT_FEAT_DIM, GOAL_DIM, hash_text_features


@dataclass(frozen=True)
class TrajectoryBatch:
    states: torch.Tensor   # (B, T, S)
    action_type: torch.Tensor  # (B, T)
    action_param: torch.Tensor  # (B, T)


def _teacher_policy(state_vec: np.ndarray, goal: Goal, rng: np.random.Generator) -> Action:
    # state_vec is normalized; we need raw-ish counts from it.
    # Reconstruct approximate vertex_count from normalization scale.
    vertex_count = float(state_vec[0] * 20000.0)
    symmetry = float(state_vec[8])
    selected_faces = float(state_vec[9] * 10.0)

    # For tiny targets (e.g. the default cube), prefer NOOP once we're close.
    if goal.target_vertex_count > 0 and goal.target_vertex_count <= 16:
        if abs(vertex_count - float(goal.target_vertex_count)) <= max(1.0, 0.10 * float(goal.target_vertex_count)):
            return Action(ActionType.NOOP, 0)

    if symmetry < goal.target_symmetry and rng.random() < 0.25:
        return Action(ActionType.MIRROR, int(rng.integers(0, 3)))

    # Blender mesh edit ops require an active selection.
    # If we intend to do any selection-dependent edit, select first.
    if selected_faces < 0.5 and rng.random() < 0.35:
        return Action(ActionType.SELECT_RANDOM_FACE, 0)

    # For small targets, prefer switching to a primitive instead of
    # repeatedly extruding/subdividing from the cube.
    if goal.target_vertex_count > 0 and goal.target_vertex_count <= 96 and vertex_count <= 12:
        # Map desired vertex target to cylinder vertex count (~2*verts).
        desired_verts = int(np.clip(round(float(goal.target_vertex_count) / 2.0), 12, 32))
        t = float(desired_verts - 12) / 20.0
        p = int(round(t * (PARAM_BINS - 1)))
        p = int(np.clip(p, 0, PARAM_BINS - 1))
        if rng.random() < 0.75:
            return Action(ActionType.ADD_CYLINDER, p)

    if vertex_count < goal.target_vertex_count * 0.85:
        # Grow complexity — pick a SUBDIVIDE param that targets 90 % of
        # the goal in one step so the trajectory doesn't massively overshoot
        # and then waste 50+ steps on DELETE_FACE.
        if rng.random() < 0.55:
            ratio = float(goal.target_vertex_count) / max(1.0, vertex_count)
            # mult = 1.0 + 2.0*(p/31)  →  p = (mult-1.0)/2.0 * 31
            safe_mult = float(np.clip(ratio * 0.90, 1.01, 3.0))
            safe_p = int(round((safe_mult - 1.0) / 2.0 * (PARAM_BINS - 1)))
            safe_p = int(np.clip(safe_p, 0, PARAM_BINS - 1))
            return Action(ActionType.SUBDIVIDE, safe_p)
        return Action(ActionType.EXTRUDE, int(rng.integers(0, PARAM_BINS)))

    if vertex_count > goal.target_vertex_count * 1.15:
        ratio = float(goal.target_vertex_count) / max(1.0, vertex_count)
        # DELETE_FACE removes frac in [0.05..0.50] of geometry.
        # We want: new_v ~= 1.05 * goal  ->  (1-frac) ~= 1.05*ratio
        desired = float(np.clip(1.0 - 1.05 * ratio, 0.05, 0.50))
        # frac = 0.05 + 0.45*t  ->  t = (frac-0.05)/0.45
        t = float(np.clip((desired - 0.05) / 0.45, 0.0, 1.0))
        p = int(round(t * (PARAM_BINS - 1)))
        # Deleting faces is selection-dependent; ensure we have a selection.
        if selected_faces < 0.5:
            return Action(ActionType.SELECT_RANDOM_FACE, 0)
        return Action(ActionType.DELETE_FACE, int(np.clip(p, 0, PARAM_BINS - 1)))

    # Fine tuning / shaping
    r = rng.random()
    if r < 0.25:
        return Action(ActionType.SCALE, int(rng.integers(10, 22)))
    if r < 0.45:
        if selected_faces < 0.5:
            return Action(ActionType.SELECT_RANDOM_FACE, 0)
        return Action(ActionType.BEVEL, int(rng.integers(0, PARAM_BINS)))
    if r < 0.65:
        if selected_faces < 0.5:
            return Action(ActionType.SELECT_RANDOM_FACE, 0)
        return Action(ActionType.INSET, int(rng.integers(0, PARAM_BINS)))
    if r < 0.80:
        return Action(ActionType.SELECT_RANDOM_FACE, 0)
    return Action(ActionType.APPLY_MODIFIER, int(rng.integers(0, 8)))


class SyntheticImitationStream(IterableDataset):
    """Infinite synthetic imitation data: (state -> action).

    Generates short trajectories from a deterministic compact env and a
    hand-coded teacher policy. This aligns with Phase 1 (supervised imitation)
    in ARCHITECTURE.md.
    """

    # Total feature dim = 10 (mesh stats) + GOAL_DIM (18) = 28
    STATE_DIM: int = 10 + GOAL_DIM

    def __init__(
        self,
        *,
        seed: int = 0,
        seq_len: int = 64,
        batch_size: int = 32,
        state_dim: int = 10 + GOAL_DIM,  # 28
        goal_sampler=None,
        mask_numeric_goal_prob: float = 0.0,
        trajectory_mode: str = "forward",
        reverse_prob: float = 0.5,
    ):
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._seq_len = int(seq_len)
        self._batch_size = int(batch_size)
        self._state_dim = int(state_dim)
        self._goal_sampler = goal_sampler
        self._mask_numeric_goal_prob = float(mask_numeric_goal_prob)
        self._trajectory_mode = str(trajectory_mode)
        self._reverse_prob = float(reverse_prob)

        mode = self._trajectory_mode.strip().lower()
        if mode not in {"forward", "reverse", "mixed"}:
            raise ValueError(f"Unknown trajectory_mode: {trajectory_mode!r}")

    def _maybe_mask_numeric_goal(self, model_goal: Goal) -> Goal:
        """Optionally hide numeric goal fields from the model.

        This forces the policy to learn to use `text_features` when present.
        The teacher/env still use the unmasked goal so actions remain correct.
        """

        p = float(np.clip(self._mask_numeric_goal_prob, 0.0, 1.0))
        if p <= 0.0:
            return model_goal

        tf = model_goal.text_features
        if tf is None or tf.size == 0 or float(np.sum(np.abs(tf))) <= 1e-8:
            return model_goal

        if self._rng.random() >= p:
            return model_goal

        return Goal(
            target_vertex_count=0,
            target_symmetry=0.0,
            text_features=model_goal.text_features,
        )

    def __iter__(self):
        env = MeshStatsEnv(seed=int(self._rng.integers(0, 2**31 - 1)))
        while True:
            states = np.zeros((self._batch_size, self._seq_len, self._state_dim), dtype=np.float32)
            a_type = np.zeros((self._batch_size, self._seq_len), dtype=np.int64)
            a_param = np.zeros((self._batch_size, self._seq_len), dtype=np.int64)

            for b in range(self._batch_size):
                if self._goal_sampler is None:
                    import numpy as _np
                    teacher_goal = Goal(
                        target_vertex_count=int(self._rng.integers(200, 5000)),
                        target_symmetry=float(self._rng.uniform(0.4, 0.9)),
                        text_features=_np.zeros(PROMPT_FEAT_DIM, dtype=_np.float32),
                    )
                else:
                    teacher_goal = self._goal_sampler.sample(self._rng)

                model_goal = Goal(
                    target_vertex_count=int(teacher_goal.target_vertex_count),
                    target_symmetry=float(teacher_goal.target_symmetry),
                    text_features=np.asarray(teacher_goal.text_features, dtype=np.float32),
                )
                model_goal = self._maybe_mask_numeric_goal(model_goal)

                gvec = normalize_goal(model_goal)

                mode = self._trajectory_mode.strip().lower()
                if mode == "reverse":
                    start_phase = "shrink"
                elif mode == "mixed":
                    start_phase = "shrink" if self._rng.random() < float(np.clip(self._reverse_prob, 0.0, 1.0)) else None
                else:
                    start_phase = None

                # Blender rollouts start from the default cube. To teach
                # prompt-faithfulness (text_features conditioning), start from
                # the canonical cube whenever a prompt fingerprint is present.
                tf = np.asarray(teacher_goal.text_features, dtype=np.float32)
                has_prompt = bool(tf.size > 0 and float(np.sum(np.abs(tf))) > 1e-8)
                st = env.reset(
                    goal_vertex_count=(None if has_prompt else (teacher_goal.target_vertex_count or None)),
                    start_phase=start_phase,
                )
                for t in range(self._seq_len):
                    vec = normalize_state_vector(st.as_vector())
                    act = _teacher_policy(vec, teacher_goal, self._rng).clamp()
                    feat = np.concatenate([vec, gvec], axis=0).astype(np.float32, copy=False)
                    states[b, t, :] = feat
                    a_type[b, t] = int(act.action_type)
                    a_param[b, t] = int(act.param)

                    sr = env.step(st, act, teacher_goal)
                    st = sr.state

            yield TrajectoryBatch(
                states=torch.from_numpy(states),
                action_type=torch.from_numpy(a_type),
                action_param=torch.from_numpy(a_param),
            )


def _stats_to_mesh_vec(stats: dict) -> np.ndarray:
    bbox = stats.get("bounding_box") or {}
    return np.array(
        [
            float(stats.get("vertex_count", 0)),
            float(stats.get("face_count", 0)),
            float(stats.get("edge_count", 0)),
            float(bbox.get("x", 1.0)),
            float(bbox.get("y", 1.0)),
            float(bbox.get("z", 1.0)),
            float(stats.get("avg_edge_length", 0.0)),
            float(stats.get("manifold_flag", 1.0)),
            float(stats.get("symmetry_score", 0.0)),
            float(stats.get("selected_face_count", 0.0)),
        ],
        dtype=np.float32,
    )


def _growth_ratio_to_param(pre_v: float, post_v: float) -> int:
    """Map desired growth ratio to action param bins."""
    ratio = float(max(1.0, pre_v) / max(1.0, post_v))
    # For SUBDIVIDE in env: mult = 1 + 2*(p/31)
    desired_mult = float(np.clip(ratio, 1.0, 3.0))
    p = int(round(((desired_mult - 1.0) / 2.0) * (PARAM_BINS - 1)))
    return int(np.clip(p, 0, PARAM_BINS - 1))


def _inverse_action_from_collapse_step(step: dict) -> Action:
    op = str(step.get("op") or "").strip().lower()
    pre = step.get("pre") or {}
    post = step.get("post") or {}
    pre_v = float(pre.get("vertex_count", 0.0))
    post_v = float(post.get("vertex_count", 0.0))

    if op == "remove_modifier":
        op_data = step.get("op_data") or {}
        mod_type = str(op_data.get("type") or "").upper()
        if mod_type == "MIRROR":
            return Action(ActionType.MIRROR, 0)
        # APPLY_MODIFIER choices in worker: SUBSURF / BEVEL / SOLIDIFY
        if mod_type == "BEVEL":
            return Action(ActionType.APPLY_MODIFIER, 1)
        if mod_type == "SOLIDIFY":
            return Action(ActionType.APPLY_MODIFIER, 2)
        return Action(ActionType.APPLY_MODIFIER, 0)

    if op == "unsubdivide":
        return Action(ActionType.SUBDIVIDE, _growth_ratio_to_param(pre_v, post_v))

    if op == "dissolve_limited":
        return Action(ActionType.INSET, _growth_ratio_to_param(pre_v, post_v))

    if op == "merge_by_distance":
        return Action(ActionType.BEVEL, _growth_ratio_to_param(pre_v, post_v))

    if op == "delete_loose":
        return Action(ActionType.EXTRUDE, _growth_ratio_to_param(pre_v, post_v))

    # Fallback for unknown ops: generic shape/detail increase.
    return Action(ActionType.EXTRUDE, 8)


def _action_requires_selection(action_type: int) -> bool:
    a = int(action_type)
    return a in {
        int(ActionType.EXTRUDE),
        int(ActionType.INSET),
        int(ActionType.BEVEL),
        int(ActionType.DELETE_FACE),
    }


class RealMeshBuildTraceStream(IterableDataset):
    """Infinite imitation stream from real mesh collapse traces.

    We convert each collapse trace into a forward build trajectory by reversing
    the step order and mapping each simplify op to its approximate inverse
    build op. This keeps training focused on creation/building behavior.
    """

    def __init__(
        self,
        *,
        trace_root: str | Path,
        seed: int = 0,
        seq_len: int = 64,
        batch_size: int = 32,
        state_dim: int = 10 + GOAL_DIM,
        max_traces: int = 50_000,
        mask_numeric_goal_prob: float = 0.0,
        trace_cache_size: int = 256,
    ):
        super().__init__()
        self._rng = np.random.default_rng(seed)
        self._seq_len = int(seq_len)
        self._batch_size = int(batch_size)
        self._state_dim = int(state_dim)
        self._mask_numeric_goal_prob = float(mask_numeric_goal_prob)
        self._trace_cache_size = int(max(0, trace_cache_size))
        self._trace_cache: OrderedDict[Path, list[dict]] = OrderedDict()

        root = Path(trace_root)
        if not root.exists():
            raise FileNotFoundError(f"Trace root not found: {root}")

        trace_paths = sorted(root.glob("*/trace.jsonl"))
        if max_traces > 0:
            trace_paths = trace_paths[: int(max_traces)]
        if not trace_paths:
            raise RuntimeError(f"No trace files found under {root}")
        self._traces: list[dict] = []
        for tp in trace_paths:
            # Avoid loading large trace files eagerly; just record metadata.
            label, prompt_variants = self._load_label_and_prompts(tp)
            self._traces.append(
                {
                    "trace_path": tp,
                    "label": label,
                    "prompt_variants": prompt_variants,
                }
            )
        if not self._traces:
            raise RuntimeError(f"No readable trace records found under {root}")

    def _get_trace_records_cached(self, trace_path: Path) -> list[dict]:
        if self._trace_cache_size <= 0:
            return self._load_trace_records(trace_path)

        recs = self._trace_cache.get(trace_path)
        if recs is not None:
            # Refresh LRU position.
            self._trace_cache.move_to_end(trace_path)
            return recs

        recs = self._load_trace_records(trace_path)
        self._trace_cache[trace_path] = recs
        self._trace_cache.move_to_end(trace_path)
        while len(self._trace_cache) > self._trace_cache_size:
            self._trace_cache.popitem(last=False)
        return recs

    def _maybe_mask_numeric_goal(self, model_goal: Goal) -> Goal:
        p = float(np.clip(self._mask_numeric_goal_prob, 0.0, 1.0))
        if p <= 0.0:
            return model_goal
        tf = model_goal.text_features
        if tf is None or tf.size == 0 or float(np.sum(np.abs(tf))) <= 1e-8:
            return model_goal
        if self._rng.random() >= p:
            return model_goal
        return Goal(target_vertex_count=0, target_symmetry=0.0, text_features=model_goal.text_features)

    def _load_trace_records(self, trace_path: Path) -> list[dict]:
        records: list[dict] = []
        with trace_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
        return records

    def _load_label_and_prompts(self, trace_path: Path) -> tuple[str, list[str]]:
        mesh_json = trace_path.parent / "mesh.json"
        if not mesh_json.exists():
            return "", []
        try:
            raw = mesh_json.read_text(encoding="utf-8")
            txt = raw.strip()
            if txt.startswith("```"):
                # Legacy datasets sometimes store mesh.json as a fenced Markdown code block.
                lines = txt.splitlines()
                if lines and lines[0].lstrip().startswith("```"):
                    lines = lines[1:]
                while lines and not lines[0].strip():
                    lines = lines[1:]
                if lines and lines[-1].rstrip().endswith("```"):
                    lines = lines[:-1]
                txt = "\n".join(lines).strip()
            data = json.loads(txt)
        except Exception:
            return "", []
        label = str(data.get("label") or "")
        pv = data.get("prompt_variants")
        if isinstance(pv, list):
            prompts = [str(x) for x in pv if str(x).strip()]
        else:
            prompts = []

        # Back-compat: older traces only store the label.
        if not prompts and label.strip():
            try:
                from policy.prompt_augment import make_prompt_variants

                prompts = make_prompt_variants(label)
            except Exception:
                prompts = []
        return label, prompts

    def _pick_prompt(self, label: str, prompt_variants: list[str]) -> str:
        if prompt_variants:
            try:
                return str(prompt_variants[int(self._rng.integers(0, len(prompt_variants)))])
            except Exception:
                return str(prompt_variants[0])
        return str(label or "")

    def _sample_one(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tries = 0
        while tries < 8:
            tries += 1
            tr = self._traces[int(self._rng.integers(0, len(self._traces)))]
            tp = tr["trace_path"]
            records = self._get_trace_records_cached(tp)
            if not records:
                continue

            hi = records[0].get("pre") or {}
            label = str(tr.get("label") or "")
            prompt_variants = tr.get("prompt_variants") or []
            prompt = self._pick_prompt(label, prompt_variants if isinstance(prompt_variants, list) else [])

            teacher_goal = Goal(
                target_vertex_count=int(hi.get("vertex_count", 0) or 0),
                target_symmetry=float(hi.get("symmetry_score", 0.0) or 0.0),
                text_features=hash_text_features(prompt),
            )
            model_goal = Goal(
                target_vertex_count=int(teacher_goal.target_vertex_count),
                target_symmetry=float(teacher_goal.target_symmetry),
                text_features=np.asarray(teacher_goal.text_features, dtype=np.float32),
            )
            model_goal = self._maybe_mask_numeric_goal(model_goal)
            gvec = normalize_goal(model_goal)

            # Reverse collapse trace => forward build sequence.
            rev = list(reversed(records))

            states = np.zeros((self._seq_len, self._state_dim), dtype=np.float32)
            a_type = np.zeros((self._seq_len,), dtype=np.int64)
            a_param = np.zeros((self._seq_len,), dtype=np.int64)

            t_out = 0
            for step in rev:
                if t_out >= self._seq_len:
                    break

                # Build starts from the collapsed side (step.post) and moves
                # toward original high-detail geometry (step.pre).
                mesh_vec = _stats_to_mesh_vec(step.get("post") or {})
                st_vec = normalize_state_vector(mesh_vec)
                feat = np.concatenate([st_vec, gvec], axis=0).astype(np.float32, copy=False)

                act = _inverse_action_from_collapse_step(step).clamp()

                # Many edit-mode ops require a selection. Collapse traces do not
                # include selection events, so we inject a minimal SELECT step
                # before selection-dependent actions when the trace stats show
                # no selection.
                if _action_requires_selection(act.action_type) and float(mesh_vec[9]) < 0.5:
                    states[t_out, :] = feat
                    a_type[t_out] = int(ActionType.SELECT_RANDOM_FACE)
                    a_param[t_out] = 0
                    t_out += 1
                    if t_out >= self._seq_len:
                        break

                    mesh_vec_sel = mesh_vec.copy()
                    mesh_vec_sel[9] = 1.0
                    st_vec_sel = normalize_state_vector(mesh_vec_sel)
                    feat_sel = np.concatenate([st_vec_sel, gvec], axis=0).astype(np.float32, copy=False)
                    states[t_out, :] = feat_sel
                    a_type[t_out] = int(act.action_type)
                    a_param[t_out] = int(act.param)
                    t_out += 1
                else:
                    states[t_out, :] = feat
                    a_type[t_out] = int(act.action_type)
                    a_param[t_out] = int(act.param)
                    t_out += 1

            # Pad with last valid element for temporal stability.
            if t_out > 0 and t_out < self._seq_len:
                states[t_out:, :] = states[t_out - 1, :]
                a_type[t_out:] = a_type[t_out - 1]
                a_param[t_out:] = a_param[t_out - 1]

            return states, a_type, a_param

        # Fallback: tiny neutral sample if traces are unreadable.
        states = np.zeros((self._seq_len, self._state_dim), dtype=np.float32)
        a_type = np.zeros((self._seq_len,), dtype=np.int64)
        a_param = np.zeros((self._seq_len,), dtype=np.int64)
        return states, a_type, a_param

    def __iter__(self):
        while True:
            states = np.zeros((self._batch_size, self._seq_len, self._state_dim), dtype=np.float32)
            a_type = np.zeros((self._batch_size, self._seq_len), dtype=np.int64)
            a_param = np.zeros((self._batch_size, self._seq_len), dtype=np.int64)

            for b in range(self._batch_size):
                s, at, ap = self._sample_one()
                states[b] = s
                a_type[b] = at
                a_param[b] = ap

            yield TrajectoryBatch(
                states=torch.from_numpy(states),
                action_type=torch.from_numpy(a_type),
                action_param=torch.from_numpy(a_param),
            )
