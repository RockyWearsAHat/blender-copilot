# Blender Copilot — Professionalization Plan

**Date:** 2026-02-18  
**Goal:** Make the system professional-grade across all major Blender workflows: modeling, scene assembly, shading/materials, rigging, animation, simulation/physics, lighting/rendering, and iterative QA.

## Implementation Status (2026-02-18)

- ✅ Composition/workflow supervision fields implemented in cache entries (`composition`, `scene_complexity_score`, `workflow_supervision`).
- ✅ Training now consumes reliability signals for real geometry (`quality_weight × label_confidence × complexity_boost`).
- ✅ Domain KPI scaffolding implemented in evaluation (`by_domain`, professional KPI gate summary).
- ✅ Gold benchmark scaffold added (`data/eval/professional_gold_set.json`).
- ✅ Tool-loop reliability shaping added in addon (repetition/dead-end detection + recovery nudges).
- ✅ Tool-loop KPI analysis script added (`scripts/analyze_tool_loops.py`).
- ⏭️ Next: run full data + A/B training runs and gate promotion by KPI outcomes.

---

## 1) North-Star Outcome

Build a model+toolchain that can reliably complete production-style Blender tasks end-to-end from natural language prompts, with measurable gains on:

- **Geometry fidelity** (complex topology + multi-object scenes)
- **Scene composition correctness** (object relationships, scale, placement)
- **Material/shader quality** (node graph realism + task fit)
- **Animation/rig quality** (functional rigs, plausible motion)
- **Render/readability quality** (lighting/composition/outputs)
- **Agent robustness** (fewer recovery loops, fewer dead-end tool calls)

---

## 2) Strategy (Core Principle)

High-poly training should be paired with **scene/composition labels and workflow supervision**, not just larger face counts.

### Why this matters

- Higher face counts improve geometric capacity, but do not by themselves teach:
  - object roles in a scene,
  - animation semantics,
  - shader graph intent,
  - production sequencing.
- We need **task-structured, multi-modal supervision** that mirrors real Blender work.

---

## 3) Capability Targets by Domain

## 3.1 Modeling + Scene Assembly

- Multi-object generation from one prompt with coherent hierarchy and scale.
- Topology quality sufficient for downstream editing (reasonable manifoldness, fewer degenerates).
- Better part preservation on complex assets (furniture, interiors, mech assemblies, characters).

## 3.2 Materials + Shading

- Produce material intent labels and map them to plausible node setups.
- Improve physically plausible defaults (roughness/metallic/transmission ranges).
- Better material assignment per object region and orientation.

## 3.3 Rigging + Animation

- Generate/adjust armatures and basic control patterns.
- Produce timeline/keyframe edits that satisfy prompt constraints.
- Increase success rate of animation tool loops (inspect timeline -> adjust -> verify).

## 3.4 Simulation + Effects

- More reliable setup for rigid body/cloth/particles for common cases.
- Better parameter defaults (stability first, then style).

## 3.5 Lighting + Rendering

- Better camera placement and scene readability in generated outputs.
- Higher consistency in render setup for “presentation-ready” results.

---

## 4) Data Plan (What to Train On)

## 4.1 Geometry + Composition Corpus

- Keep current sources and prioritize high-value categories:
  - interiors, character+props, vehicles, machinery, environment kits.
- Retain high-poly originals and add composition-level labels:
  - examples: `modern office desk setup`, `stylized medieval tavern corner`, `robot arm workstation assembly`.
- Include decomposition labels for complex items:
  - primary object type + key semantic parts.

## 4.2 Workflow-Supervision Corpus

Create training-ready examples for task execution traces:

- **Modeling traces:** object creation/editing/modifier sequence.
- **Shading traces:** node creation/connection edits with intent text.
- **Rigging traces:** armature creation, parenting, constraints.
- **Animation traces:** keyframe insert/edit + timeline diagnostics.
- **Render traces:** camera/light/compositor adjustments.

Each trace should include:

- initial state summary,
- target instruction,
- action sequence,
- final state summary/check.

## 4.3 Preference/Feedback Data (RLHF/DPO)

- Pair successful vs failed tool sequences for the same goal.
- Prioritize failure modes:
  - overlong/unfocused tool loops,
  - invalid object references,
  - incorrect mode/state transitions,
  - non-converging animation/material iterations.

---

## 5) Model + Training Plan

## Phase A — Data Upgrade + Labeling (Immediate)

1. Raise high-poly usage while preserving stability:
   - Increase face/token budget gradually (A/B tested).
2. Add scene/composition labels to high-poly entries.
3. Add workflow traces for modeling/shading/animation/rigging.
4. Add stricter quality gates for invalid/underspecified scenes.

**Deliverable:** new cache snapshots with composition + workflow supervision fields.

## Phase B — Multi-Task Curriculum (Core)

Train with weighted task mixture:

- geometry
- image-conditioned geometry
- contrastive text-image
- materials
- modifiers
- workflow policy heads (tool-sequence quality)

Curriculum:

- start with simple/medium scenes,
- ramp to complex high-poly + multi-object compositions,
- maintain a stable simple-shape proportion to avoid regression.

**Deliverable:** checkpoint series with per-domain eval curves.

## Phase C — Agent Reliability + Tool Mastery

1. Add reward shaping for task completion quality, not just completion declaration.
2. Penalize dead-end or repetitive tool calls.
3. Strengthen plan-first execution pattern:
   - decompose -> execute -> inspect -> refine -> finish.
4. Evaluate against role-specific suites:
   - modeler suite, shader suite, animator suite, rigger suite, lighting suite.

**Deliverable:** “professional interaction mode” checkpoint and prompt package.

---

## 6) Evaluation Plan (Must-Have KPIs)

## 6.1 Geometry/Scene KPIs

- Mesh validity score (mean + p10).
- Part retention score on complex prompts.
- Scene composition score (object count/placement/scale correctness).
- High-poly fidelity metrics vs references (sampled categories).

## 6.2 Workflow KPIs

- Tool-loop success rate by domain:
  - modeling, shading, rigging, animation, render setup.
- Mean tool calls to successful completion.
- Recovery rate from intermediate errors.
- Percentage of runs requiring manual intervention.

## 6.3 Quality-of-Result KPIs

- Human rating (1–5) per domain for production readiness.
- Preference win-rate against baseline checkpoint.

**Promotion gate:** New checkpoint must beat baseline in at least 4/5 domain suites and not regress simple-shape reliability.

---

## 7) Infrastructure + Run Plan

## 7.1 Data Runs

- Short validation runs for each source update.
- Scheduled full runs with high-poly + composition labeling enabled.

## 7.2 Training Runs

- Baseline run (current settings).
- High-poly + composition run (A/B).
- Workflow-augmented run.
- RLHF/DPO reliability run.

## 7.3 Artifact Tracking

For every run record:

- config snapshot,
- data snapshot hash,
- checkpoint path,
- eval bundle,
- failure taxonomy summary.

---

## 8) Execution Order (Recommended)

1. Finalize composition/workflow label schema.
2. Rebuild a high-quality mixed dataset (simple + complex).
3. Run baseline vs high-poly+composition A/B.
4. Add workflow traces and retrain multi-task.
5. Add RLHF/DPO reliability pass.
6. Promote only if KPI gates pass.

---

## 9) Immediate Next Steps (This Week)

1. Implement/confirm composition label fields in cache entries.
2. Build a small “gold” benchmark set per Blender domain.
3. Run first A/B training pair and collect per-domain evals.
4. Review failure taxonomy and tune task weights.

---

## 10) Success Definition

The system is considered **professional-grade** when it can repeatedly complete common production tasks across modeling, shading, rigging, animation, and render setup with low manual rescue, high scene fidelity, and stable multi-step tool behavior.
