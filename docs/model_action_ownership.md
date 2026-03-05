# Model ↔ Action Ownership Matrix

Purpose: map every training/inference capability to a responsible model path so failures can be blamed and fixed quickly.

## Rules

- Every user-visible action must map to exactly one **primary owner** model/bot.
- Supporting models may assist, but regression ownership stays with the primary owner.
- If an action is failing, debug the primary owner first.
- If adding a new action, update this file in the same PR.

## Ownership Table

| Action / Capability                                                        | Primary Owner      | Secondary Owner(s)                          | Training Data Source of Truth                          | Failure Signature                                     | First Debug Target                           |
| -------------------------------------------------------------------------- | ------------------ | ------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------- | -------------------------------------------- |
| Primitive creation (`ADD_CUBE`, `ADD_CYLINDER`, etc.)                      | `policy/modeler`   | none                                        | `master_cache` mesh/object sections (non-baked)        | Wrong topology or invalid primitive sequence          | model action logits + action decoder         |
| Topology editing (`EXTRUDE`, `INSET`, `BEVEL`, `SUBDIVIDE`, `DELETE_FACE`) | `policy/modeler`   | none                                        | `master_cache` object mesh traces (non-baked)          | Face/edge counts diverge from expected edit path      | action-type/action-param supervision         |
| Selection actions (`SELECT_RANDOM_FACE`, selection transitions)            | `policy/modeler`   | none                                        | rollout traces + state transitions                     | Editing acts on wrong region or no-op loops           | selection state encoding                     |
| Transform actions (`SCALE`, axis transforms)                               | `policy/modeler`   | `scene-builder` (layout-only)               | compact state + supervised traces                      | Correct topology but wrong scale proportions          | transform param head                         |
| Modifier intent decision (`APPLY_MODIFIER(type)`)                          | `scene-builder`    | `policy/modeler` (if simple local modifier) | file-level context (`modifiers`, `node_groups`)        | Modifier chosen in wrong stage                        | scene-level policy router                    |
| Geometry Nodes graph application/orchestration                             | `scene-builder`    | dedicated GN specialist (future)            | `node_groups`, object GN modifier refs                 | Baked/instanced outputs confused as modeler mesh      | scene-builder action policy                  |
| Material assignment / shader graph edits                                   | `shader-materials` | `scene-builder`                             | material node trees, image refs, face material indices | Correct mesh but wrong look/material slots            | material token/model head                    |
| UV/material slot consistency                                               | `shader-materials` | `policy/modeler` (topology-aware checks)    | UV layers + face material indices                      | Texture appears on wrong faces / broken mapping       | UV + material alignment checks               |
| Camera/light placement and world setup                                     | `scene-builder`    | none                                        | cameras/lights/world scene sections                    | Object reconstruction correct but scene framing wrong | scene object-type routing                    |
| Collection hierarchy / visibility orchestration                            | `scene-builder`    | none                                        | collections + object visibility fields                 | Hidden items appear or visible items disappear        | collection + visibility reconstruction logic |
| Physics/rigid-body setup                                                   | `scene-builder`    | future physics specialist                   | rigid body sections in scene payload                   | Dynamics behavior mismatch despite correct mesh       | rigid-body extraction/reapply path           |

## Triage Procedure

1. Identify failing action/capability.
2. Find row in ownership table.
3. Validate data source slice for that row in `master_cache`.
4. Validate training subset filter for that row.
5. Run owner-model smoke test on minimal subset.
6. Only escalate to secondary owner if owner path is clean.

## Required Dataset Filters by Owner

- `policy/modeler` must exclude baked GN-only geometry:
  - `geometry_is_baked == False`
  - prefer `geometry_space == RAW_LOCAL`
- `shader-materials` requires material context:
  - `materials` present OR `face_material_indices` present
- `scene-builder` requires scene context:
  - one or more of `collections`, `node_groups`, `cameras`, `lights`, `world`

## Change Log

- 2026-03-03: Initial matrix introduced for explicit ownership and blame routing.
