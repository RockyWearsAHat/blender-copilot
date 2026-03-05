from __future__ import annotations

from dataclasses import dataclass

from policy.state import Goal, MeshState


@dataclass(frozen=True)
class RulebookScore:
    """Deterministic, compact quality score based only on mesh stats + goal.

    This is intentionally *not* a workflow enforcer.
    It is a universal judge that can be used for:
      - self-improvement scoring (generate → score → keep-best)
      - evaluation dashboards
      - shaping rewards without hard action masks

    All terms are computed from MeshState (+ Goal), keeping the architecture
    compact and Apple-silicon friendly.
    """

    total: float
    # breakdown terms (useful for debugging / telemetry)
    manifold: float
    symmetry: float
    poly_efficiency: float
    scale_sanity: float
    edge_sanity: float


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def score_state(state: MeshState, goal: Goal) -> RulebookScore:
    """Return a scalar score + breakdown.

    Interpretation:
      - higher is better
      - score is bounded-ish but not strictly [-1, 1]

    Notes:
      - When the goal has no explicit vertex target (0), we avoid inventing
        one; instead we score efficiency with a soft "too big" penalty that
        kicks in only at very large counts.
    """

    v = int(state.vertex_count)
    f = int(state.face_count)

    # 1) Topology validity: manifoldness is a universal good.
    manifold = _clamp01(float(state.manifold_flag))

    # 2) Symmetry: if target is specified, score toward it; else reward some.
    sym = _clamp01(float(state.symmetry_score))
    sym_target = float(goal.target_symmetry) if float(goal.target_symmetry) > 0.0 else 0.5
    symmetry = sym if sym >= sym_target else (sym / max(1e-6, sym_target))
    symmetry = _clamp01(float(symmetry))

    # 3) Poly efficiency: prefer meeting the target without overshooting.
    # If goal is unconstrained, only penalize extreme runaway counts.
    poly_efficiency = 1.0
    if int(goal.target_vertex_count) > 0:
        g = max(1, int(goal.target_vertex_count))
        # no penalty up to 20% over; then increasingly harsh.
        over = max(0.0, (float(v) - 1.2 * float(g)) / float(g))
        poly_efficiency = 1.0 / (1.0 + over * over)
    else:
        # unconstrained: allow complexity, but discourage absurd meshes
        # (keeps Blender responsive; doesn't pick a style).
        if v > 200_000 or f > 200_000:
            poly_efficiency = 0.0
        elif v > 50_000 or f > 50_000:
            poly_efficiency = 0.25
        else:
            poly_efficiency = 1.0

    poly_efficiency = _clamp01(float(poly_efficiency))

    # 4) Scale sanity: bounding box should not be degenerate.
    bx = float(state.bbox_x)
    by = float(state.bbox_y)
    bz = float(state.bbox_z)
    min_dim = min(bx, by, bz)
    max_dim = max(bx, by, bz)
    if min_dim < 1e-4:
        scale_sanity = 0.0
    else:
        aspect = max_dim / max(1e-6, min_dim)
        # tolerate up to 20:1 (needles/flat panels exist), penalize beyond.
        scale_sanity = 1.0 / (1.0 + max(0.0, aspect - 20.0) / 10.0)
    scale_sanity = _clamp01(float(scale_sanity))

    # 5) Edge sanity: absurdly small average edge length is a proxy for
    # micro-triangle explosions.
    ael = float(state.avg_edge_length)
    if ael <= 0.0:
        edge_sanity = 0.0
    elif ael < 0.002:
        edge_sanity = 0.0
    elif ael < 0.01:
        edge_sanity = 0.5
    else:
        edge_sanity = 1.0
    edge_sanity = _clamp01(float(edge_sanity))

    # Total score: weights chosen to reflect universal modeling goals
    # without imposing a specific workflow.
    total = (
        0.40 * manifold
        + 0.20 * symmetry
        + 0.25 * poly_efficiency
        + 0.10 * scale_sanity
        + 0.05 * edge_sanity
    )

    return RulebookScore(
        total=float(total),
        manifold=float(manifold),
        symmetry=float(symmetry),
        poly_efficiency=float(poly_efficiency),
        scale_sanity=float(scale_sanity),
        edge_sanity=float(edge_sanity),
    )
