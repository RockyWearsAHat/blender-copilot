from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ActionType(IntEnum):
    ADD_CUBE = 0
    ADD_CYLINDER = 1
    EXTRUDE = 2
    INSET = 3
    BEVEL = 4
    SCALE = 5
    SUBDIVIDE = 6
    DELETE_FACE = 7
    SELECT_RANDOM_FACE = 8
    MIRROR = 9
    APPLY_MODIFIER = 10
    NOOP = 11


# A single discrete parameter channel (keeps the action grammar small and finite).
# Semantics are action-dependent; values not used by an action should be 0.
PARAM_BINS = 32


@dataclass(frozen=True)
class Action:
    action_type: int
    param: int = 0

    def clamp(self) -> "Action":
        a = int(self.action_type)
        p = int(self.param)
        if a < 0:
            a = 0
        if p < 0:
            p = 0
        if p >= PARAM_BINS:
            p = PARAM_BINS - 1
        return Action(a, p)
