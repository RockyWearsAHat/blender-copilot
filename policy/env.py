from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy.actions import ActionType, Action, PARAM_BINS
from policy.state import MeshState, Goal
from policy.rulebook import score_state


@dataclass
class StepResult:
    state: MeshState
    reward: float
    done: bool
    invalid: bool


def _bin_to_signed_scale(param: int) -> float:
    # Map [0..31] -> [0.5..1.5]
    t = float(param) / float(max(1, PARAM_BINS - 1))
    return 0.5 + t * 1.0


def _bin_to_small_positive(param: int) -> float:
    # Map [0..31] -> [0.0..0.25]
    t = float(param) / float(max(1, PARAM_BINS - 1))
    return 0.25 * t


class MeshStatsEnv:
    """Deterministic, compact environment over mesh statistics.

    This is a training-time stand-in for bpy execution.
    It stays seedable/deterministic and enforces legality rules.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def reset(self, goal_vertex_count: int | None = None, *, start_phase: str | None = None) -> MeshState:
        """Return a starting MeshState.

        When *goal_vertex_count* is provided, the start vertex count is sampled
        from one of three phases with equal probability so that the training
        data contains a balanced mix of grow / fine-tune / shrink steps:

          grow      – start far below goal  (v in [4, 0.4 * goal])
          fine-tune – start near goal       (v in [0.7 * goal, 1.3 * goal])
          shrink    – start above goal      (v in [1.4 * goal, 2.0 * goal])

        Without a goal, start from the canonical 8-vertex primitive.
        """
        if goal_vertex_count is not None and goal_vertex_count > 0:
            g = int(goal_vertex_count)

            # For very small goals (e.g. 8-vertex cube), start from the
            # canonical primitive so the teacher can legitimately choose NOOP.
            if g <= 16:
                v, f, e = 8, 6, 12
                ael, sym, sel = 0.5, 1.0, 0
                return MeshState(
                    vertex_count=int(v),
                    face_count=int(f),
                    edge_count=int(e),
                    bbox_x=1.0,
                    bbox_y=1.0,
                    bbox_z=1.0,
                    avg_edge_length=float(ael),
                    manifold_flag=1.0,
                    symmetry_score=float(sym),
                    selected_face_count=int(sel),
                )

            phase = (start_phase or "").strip().lower()
            if phase not in {"", "auto", "grow", "fine_tune", "fine-tune", "shrink"}:
                raise ValueError(f"Unknown start_phase: {start_phase!r}")

            if phase in {"", "auto"}:
                # Default (forward) curriculum: 70% grow, 30% fine-tune.
                # We intentionally exclude shrink-start by default so DELETE_FACE
                # doesn't dominate and crowd out grow actions.
                phase = "grow" if self._rng.random() < 0.70 else "fine_tune"
            elif phase == "fine-tune":
                phase = "fine_tune"

            if phase == "grow":
                v = int(self._rng.integers(4, max(5, g * 4 // 10)))
            elif phase == "fine_tune":
                lo = max(4, g * 7 // 10)
                hi = max(lo + 1, g * 13 // 10)
                v = int(self._rng.integers(lo, hi))
            else:  # shrink
                lo = max(5, g * 14 // 10)
                hi = max(lo + 1, g * 20 // 10)
                v = int(self._rng.integers(lo, hi))
            # Scale other geometry stats proportionally from the primitive
            scale = float(v) / 8.0
            f = max(1, int(round(6 * scale)))
            e = max(1, int(round(12 * scale)))
            ael = float(max(0.05, 0.5 / max(0.1, scale ** 0.33)))
            sym = float(self._rng.uniform(0.1, 0.9))
            sel = 0
        else:
            # Canonical Blender primitive cube: 8 verts, 12 edges, 6 faces.
            v, f, e = 8, 6, 12
            ael, sym, sel = 0.5, 1.0, 0

        return MeshState(
            vertex_count=int(v),
            face_count=int(f),
            edge_count=int(e),
            bbox_x=1.0,
            bbox_y=1.0,
            bbox_z=1.0,
            avg_edge_length=float(ael),
            manifold_flag=1.0,
            symmetry_score=float(sym),
            selected_face_count=int(sel),
        )

    def step(self, state: MeshState, action: Action, goal: Goal) -> StepResult:
        action = action.clamp()
        invalid = False

        v = state.vertex_count
        f = state.face_count
        e = state.edge_count
        bx, by, bz = state.bbox_x, state.bbox_y, state.bbox_z
        ael = state.avg_edge_length
        manifold = state.manifold_flag
        sym = state.symmetry_score
        sel = state.selected_face_count

        at = ActionType(action.action_type)
        p = int(action.param)

        if at == ActionType.ADD_CUBE:
            # Treat primitive-add as switching the active mesh to a fresh primitive.
            v, f, e = 8, 6, 12
            bx, by, bz = 1.0, 1.0, 1.0
            ael = 0.5
            sym = 1.0
            sel = 0
        elif at == ActionType.ADD_CYLINDER:
            # Approximate Blender cylinder primitive stats from param.
            verts = 12 + int((float(p) / float(max(1, PARAM_BINS - 1))) * 20.0)  # 12..32
            v = int(2 * verts)
            f = int(verts + 2)
            e = int(3 * verts)
            bx, by, bz = 1.0, 1.0, 1.0
            ael = 0.4
            sym = max(sym, 0.8)
            sel = 0
        elif at == ActionType.EXTRUDE:
            if sel <= 0:
                invalid = True
            else:
                delta = int(4 + 20 * _bin_to_small_positive(p))
                v += delta
                f += delta
                e += delta * 2
                sel = 0
        elif at == ActionType.INSET:
            if sel <= 0:
                invalid = True
            else:
                # Inset increases detail but less than extrude
                delta = int(2 + 12 * _bin_to_small_positive(p))
                v += delta
                f += delta
                e += delta * 2
        elif at == ActionType.BEVEL:
            if sel <= 0:
                invalid = True
            else:
                delta = int(2 + 16 * _bin_to_small_positive(p))
                v += delta
                f += delta
                e += delta * 2
                ael = max(0.05, ael * 0.95)
        elif at == ActionType.SCALE:
            s = _bin_to_signed_scale(p)
            bx = float(np.clip(bx * s, 0.1, 3.0))
            by = float(np.clip(by * s, 0.1, 3.0))
            bz = float(np.clip(bz * s, 0.1, 3.0))
            ael = float(np.clip(ael * s, 0.02, 2.0))
        elif at == ActionType.SUBDIVIDE:
            if sel <= 0:
                invalid = True
            else:
                # Each Blender subdivide-cut roughly triples vertex count in practice.
                mult = 1.0 + 2.0 * (float(p) / float(max(1, PARAM_BINS - 1)))
                v = int(v * mult)
                f = int(f * mult)
                e = int(e * mult)
                ael = max(0.02, ael / mult)
        elif at == ActionType.DELETE_FACE:
            if sel <= 0 or f <= 0:
                invalid = True
            else:
                # Delete selected faces (approx). Clamp to keep the mesh alive.
                removed = int(min(sel, max(0, f - 4)))
                if removed <= 0:
                    invalid = True
                else:
                    frac = float(removed) / float(max(1, f))
                    f = max(4, int(f - removed))
                    v = max(4, int(round(float(v) * (1.0 - 0.6 * frac))))
                    e = max(4, int(round(float(e) * (1.0 - 0.7 * frac))))
                    sel = 0
        elif at == ActionType.SELECT_RANDOM_FACE:
            if f <= 0:
                invalid = True
                sel = 0
            else:
                # select between 1 and min(64, faces)
                sel = int(self._rng.integers(1, min(64, f) + 1))
        elif at == ActionType.MIRROR:
            # axis encoded in param bins (0/1/2)
            sym = float(np.clip(sym + 0.2, 0.0, 1.0))
        elif at == ActionType.APPLY_MODIFIER:
            # small, generic complexity bump
            v += 4
            f += 4
            e += 8
        elif at == ActionType.NOOP:
            # Intentionally do nothing.
            pass
        else:
            invalid = True

        # Manifold degrades slightly if lots of operations; keep mostly valid.
        op_penalty = (v + f) / 500000.0
        manifold = float(np.clip(manifold - op_penalty, 0.0, 1.0))

        next_state = MeshState(
            vertex_count=int(max(1, v)),
            face_count=int(max(1, f)),
            edge_count=int(max(1, e)),
            bbox_x=float(bx),
            bbox_y=float(by),
            bbox_z=float(bz),
            avg_edge_length=float(ael),
            manifold_flag=float(manifold),
            symmetry_score=float(np.clip(sym, 0.0, 1.0)),
            selected_face_count=int(max(0, sel)),
        )

        # Reward and done are computed from a deterministic "rulebook" based on
        # mesh statistics + goal. This avoids enforcing specific workflows and
        # keeps the learning signal compact.
        rb = score_state(next_state, goal)
        reward = float(rb.total)
        if invalid:
            reward -= 0.25

        # Done: require manifold + meeting symmetry target, plus (if specified)
        # being near the vertex target.
        sym_target = goal.target_symmetry if goal.target_symmetry > 0.0 else 0.5
        sym_ok = next_state.symmetry_score >= sym_target
        manifold_ok = next_state.manifold_flag >= 0.95

        if goal.target_vertex_count > 0:
            vc_err = abs(next_state.vertex_count - goal.target_vertex_count) / max(1.0, float(goal.target_vertex_count))
            vc_ok = vc_err < 0.05
        else:
            vc_ok = True

        done = bool(sym_ok and manifold_ok and vc_ok)

        return StepResult(state=next_state, reward=float(reward), done=done, invalid=bool(invalid))
