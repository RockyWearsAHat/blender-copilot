# Comprehensive Plan — Modeling-First Blender Copilot (Geometry + Non‑Destructive Refinement + Noise/Textures/Displacement)

Date: 2026-02-28

This document is an implementation-oriented plan for a **serious, trainable Blender modeling system** that:

- **Generates and refines** assets via artist-like workflows (not one-shot raw mesh prediction).
- Prioritizes **geometry first**, then expands to **materials/textures/displacement** in a modular, trainable way.
- Stays compatible with the project’s canonical constraints in `.github/ARCHITECTURE.md`:
  - reduce entropy first
  - inject structure (grammar + legality + determinism)
  - scale last
  - trainable locally on Apple Silicon (M3)

---

## 0) Goals, Non‑Goals, and “Team Flow”

### Goals

- **Non-destructive by default**: the output should be editable and “team friendly” (artist can tweak parameters, art director can ask for changes, technical artist can bake later).
- **Closed-loop refinement**: the system can iterate: inspect → adjust → re-check.
- **Clear style semantics**: “low poly” must be interpreted consistently, with a bias-aware disambiguation strategy.
- **Noise/texture/displacement map generation**: first as deterministic procedural primitives + bake options; later as trainable mapping from intent → parameters.

### Non‑Goals (for the baseline)

- No monolithic “one model does everything” giant network.
- No raw vertex-array mesh inputs to the model.
- No diffusion mesh generation.
- No “LLM assistant” as the core modeling brain.

### “Team Flow” principle

- The system produces **assets that are easy to hand off**:
  - modifiers remain editable
  - materials are node-based and parameterized
  - displacement uses maps/nodes with exposed controls
  - baking is an explicit, optional, late-stage step

---

## 1) System Modules (Architecture-Compliant)

### 1.1 Blender Environment Layer (deterministic executor)

Responsibilities:

- Apply deterministic `bpy` operations.
- Extract compact, numeric state.
- Enforce legality (invalid action masking).
- Seedable randomness for reproducibility.

Core requirement:

- A rollout can be **replayed deterministically** given `(seed, initial scene, action sequence)`.

### 1.2 State Representation Layer (compact, low entropy)

Baseline geometry state features (allowed by architecture):

- `vertex_count`, `edge_count`, `face_count`
- `bounding_box_xyz` (min/max or extents)
- `avg_edge_length`, `surface_area`, `volume_estimate` (approx)
- `manifold_flag`, `self_intersection_proxy`
- `symmetry_score` (cheap heuristic)
- `selected_face_count` (if selection is part of the grammar)

Additional _low-entropy_ features needed for non-destructive workflows:

- `modifier_stack_summary` (types + coarse parameter buckets; not raw arrays)
- `shading_mode_flag` (flat vs smooth)
- `triangulated_ratio` (or boolean if triangulate modifier present)
- `decimate_ratio_bucket` (if decimate modifier present)

Texture/material/displacement state features (still compact):

- active material count + per-material **intent tags** (text labels, small vocab)
- node graph summary (counts/types; plus a few key params for procedural groups)
- displacement: map present? procedural node present? amplitude bucket? scale bucket?
- image-map stats (mean/std, min/max, histogram bins) **only when baking is used**

### 1.3 Action Grammar (finite + masked)

We treat modeling as a **constrained program**. Actions are typed, parameterized, and legality-checked.

Baseline action categories (keep under ~30 initial ops):

**Geometry / Modeling**

- `RESET_SCENE`
- `ADD_PLANE`, `ADD_CUBE`, `ADD_CYLINDER`
- `SUBDIVIDE(level_bucket)`
- `APPLY_SCALE_ROTATION`
- `ADD_MODIFIER(type)` with constrained parameter sub-actions:
  - `SET_SUBSURF(level_bucket)` (optional, not default for low poly)
  - `SET_DISPLACE_STRENGTH(bucket)`
  - `SET_DISPLACE_MIDLEVEL(bucket)`
  - `SET_DECIMATE_RATIO(bucket)`
  - `ADD_TRIANGULATE` (as modifier)
  - `ADD_MIRROR(axis)`
- `SHADE_FLAT`, `SHADE_SMOOTH`

**Selection / Editing (only if needed)**

- `SELECT_RANDOM_FACE`
- `EXTRUDE(distance_bucket)`
- `INSET(amount_bucket)`
- `BEVEL(amount_bucket)`

**Materials / Textures / Displacement**

- `CREATE_MATERIAL(intent_tag)`
- `ASSIGN_MATERIAL(obj_or_slot)`
- `ADD_NODE_GROUP(group_type)` (e.g., procedural noise group)
- `SET_NOISE_PARAMS(param_bucket_set)`
- `CONNECT_DISPLACEMENT`
- `BAKE_MAP(map_type, resolution_bucket)` (optional)

Legality examples:

- Don’t allow `SUBDIVIDE` on non-mesh objects.
- Disallow `DECIMATE` ratio that would exceed a face budget floor.
- If user wants stylized low poly: disallow `SHADE_SMOOTH` unless explicitly requested.

---

## 2) “Generate + Refine” as Closed-Loop Policy (Not an LLM)

### 2.1 Core interaction loop

1. Parse intent → pick a **recipe family** (e.g., mountains, rocks, buildings).
2. Execute a short initial program (3–10 actions).
3. Inspect state + cheap metrics.
4. Apply refinement actions (1–8 actions).
5. Stop when the goal metrics pass or a step budget is hit.

This is compatible with:

- supervised imitation (from scripted traces)
- self-improvement (sample → score → keep)

---

## 2.3) Workflow Reconstruction (Don’t hand-author thousands of demos)

We should NOT rely on hard-coded, single “recipes” as the long-term way to make assets.
Instead, we reconstruct _plausible Blender step sequences_ from existing scenes/objects so training learns from real work without you personally modeling thousands of examples.

### Why reconstruction is the right primitive

- `.blend` files contain a lot of “how it was made” even without edit history:
  - **modifier stacks** (Subsurf/Displace/Decimate/Mirror/Boolean/Solidify…)
  - **geometry node groups** and **shader node graphs**
  - object decomposition (separate cliff mesh + scatter rocks + vegetation)
- If we can turn that into action sequences, we get supervision for:
  - _technique selection_ (displacement terrain vs constructed cliffs)
  - _non-destructive workflows_ (modifiers/nodes as first-class)
  - _Blender-native intent_ (the way artists actually structure scenes)

### Reality check: we can’t recover the exact edit history

Blender doesn’t store a canonical “undo history” in a `.blend`.
So reconstruction targets:

1. **Plausible** step sequences (not the exact original ones)
2. **Deterministic** sequences (replayable from seed)
3. **Outcome-consistent** sequences (match the final asset on measurable summaries)

This is still extremely valuable for policy training.

### Reconstruction outputs: an action-plan dataset, not a single recipe

For a given object/scene we produce **multiple candidate plans** (hypotheses), score them against the extracted target, and keep the best few.
This avoids baking in one “defined way” to do anything.

### Reconstruction pipeline (high level)

1. **Extract target description** from `.blend`:

- compact mesh stats (counts, bbox, edge stats, manifold proxy)
- modifier stack (types + key params)
- shading flags (flat vs smooth; autosmooth; sharp edges)
- node graphs / node groups summaries for displacement/material intent

2. **Choose a technique family** (coarse classifier; deterministic rules at first):

- Terrain/heightfield-like (displace-friendly)
- Constructive cliffs (blockout + cuts/booleans/extrudes)
- Stylized low poly (facets + flat shading + low face budget)
- Retro low poly (smooth shading + low-res textures)

3. **Synthesize candidate action plans** using the finite action grammar:

- Start from primitives likely to explain the bbox/aspect ratios.
- Add modifiers/nodes in an order consistent with Blender practice.
- Use parameter buckets snapped from extracted modifier params.

4. **Replay + score** each candidate:

- Execute plan headlessly in Blender.
- Compare to target using _low-entropy_ metrics:
  - bbox extents similarity
  - face/vert budget similarity
  - manifold proxy
  - “facetness” proxies (for stylized low poly)
  - displacement presence + parameter ranges

5. **Emit training trace**:

- `initial_state`, per-step `(state, action)`, and `final_state`
- keep only top-k plans per object

### Key principle: learn techniques, not objects

This scales because we’re not writing “mountain code”.
We’re building:

- a small, legal action language
- a reconstruction + scoring system
- a dataset generator that turns arbitrary Blender assets into action supervision

Once that exists, adding more data means “add more `.blend` files”, not “model more examples by hand”.

### Why some mountains work with displacement and cliffs don’t

Reconstruction helps the policy learn the empirical truth:

- Displacement/noise works for rolling terrain or soft mountainous silhouettes.
- Cliffs often require **intentional macro geometry** (planes, ledges, strata breaks), then optional micro displacement.

So the learned behavior becomes:

- if the target has large planar facets / ledges / overhangs → constructive family
- if the target is smooth continuous terrain → displacement family

---

### 2.2 First-class “recipe”: Mountains (artist workflow)

A baseline, non-destructive mountain workflow should look like:

1. `RESET_SCENE`
2. `ADD_PLANE`
3. `SUBDIVIDE(level_bucket)`
4. `CREATE_MATERIAL("rock" | "snow" | "stylized")` (optional early)
5. Add displacement non-destructively:
   - `ADD_MODIFIER(DISPLACE)`
   - `ADD_NODE_GROUP(PROCEDURAL_HEIGHT)` or add procedural texture to the displace texture slot
   - `SET_NOISE_PARAMS(...)`
   - `SET_DISPLACE_STRENGTH(bucket)`
6. Control poly budget:
   - `ADD_MODIFIER(DECIMATE)` + `SET_DECIMATE_RATIO(bucket)`
   - optionally `ADD_TRIANGULATE`
7. Style:
   - `SHADE_FLAT` (stylized default) OR `SHADE_SMOOTH` (PS1/retro default)

Refinement examples:

- If silhouette is too smooth: increase noise detail/roughness bucket.
- If too noisy: reduce strength or increase noise scale bucket.
- If faces too high: increase decimate ratio or reduce subdiv.

---

## 3) Noise / Procedural Maps / Texture Generation (Power Tool)

Artists continuously rely on noise-driven workflows:

- height/displacement
- albedo variation
- roughness breakup
- normal details

The plan below treats procedural noise as a **core primitive** and makes it trainable without requiring big image models.

### 3.1 Phase 0 (immediate): Deterministic procedural “noise library”

We should support generating noise in two interchangeable ways:

**A) Blender-native procedural nodes** (preferred for non-destructive editing)

- Use Blender’s built-in nodes (Noise Texture, Musgrave, Voronoi, Wave, Brick, etc.).
- Wrap them in a small number of **canonical node groups** so the action grammar remains small.

Suggested canonical node groups:

- `PROC_HEIGHT_BASIC` (height map for displacement)
- `PROC_ROCK_ALBEDO`
- `PROC_ROUGHNESS_BREAKUP`
- `PROC_SNOW_MASK` (slope + height driven)

Each group exposes a minimal parameter surface:

- `seed`
- `scale`
- `detail`
- `roughness`
- `distortion`
- `warp_strength` (optional)

**B) Python-generated images** (useful for baking, exporting, and training data)

- Generate height/albedo/roughness maps as small images (e.g., 128–512).
- Use simple noise functions and fBm to keep it fast.

Implementation note (library choices):

- Prefer **no new heavy dependencies**.
- Acceptable lightweight choices:
  - pure-Python/simple implementations for Perlin/value noise
  - `numpy` for fast array ops (already common in ML stacks)
  - optional: `opensimplex` for Simplex noise (small dependency)

What “generate noise” means operationally:

- deterministic given `(seed, params)`
- optional tileable mode (periodic noise)
- produces maps with controlled range and histogram

### 3.2 Phase 1 (trainable, low-entropy): Learn intent → noise parameters

Instead of training an image generator early, learn:

- prompt/style tags → {noise type + parameter buckets}

Training data:

- sample noise params → generate the procedural output → store summary stats + the params
- label with synthetic text templates (e.g., “jagged rocky height”, “soft dunes”, “stylized faceted terrain”)

Model:

- small transformer/MLP head that predicts parameter buckets
- optionally conditioned on geometry state (to match face budgets)

### 3.3 Phase 2 (self-improvement): Sample/score/filter for maps

Self-improvement loop:

- sample multiple parameter sets
- score maps using cheap metrics (see evaluation)
- keep top-k

### 3.4 Baking as an explicit, optional step

Non-destructive default:

- keep procedural nodes live.

When baking is requested:

- `BAKE_MAP(height|albedo|roughness|normal, resolution_bucket)`
- set correct color space (e.g., Non-Color for height/roughness/normal)

---

## 4) Low Poly Style Spec (Bias-Aware)

“Low poly” is ambiguous; two common meanings:

### 4.1 Stylized “Low Poly” (default interpretation)

This is the common modern low-poly look:

- low face count
- **flat shading**
- often **triangulated**
- crisp facets / readable silhouette

Operational defaults for stylized low poly:

- face budget target (bucketed): e.g., 200 / 500 / 1k / 2k faces
- `SHADE_FLAT`
- `ADD_TRIANGULATE` (modifier) unless explicitly forbidden
- optional `DECIMATE` to hit budget

### 4.2 PS1 / Retro “Low Poly”

This is low poly but typically:

- low face count
- **smooth shading** (or limited smoothing)
- low-res textures / vertex colors

Operational defaults for PS1 low poly:

- same face budget buckets
- `SHADE_SMOOTH`
- triangulation optional (many retro pipelines end up triangulated anyway, but don’t force the faceted look)
- texture resolution buckets: 32/64/128 (if textures are involved)

### 4.3 Disambiguation rules (avoid hidden personal bias)

We should not silently encode one person’s preference as “the truth”.

Proposed deterministic disambiguation:

- If prompt contains any of: `flat shaded`, `faceted`, `stylized low poly`, `low poly art` → choose **Stylized Low Poly**.
- If prompt contains any of: `PS1`, `retro`, `N64`, `smooth low poly`, `low-res texture` → choose **PS1/Retro Low Poly**.
- If prompt says only “low poly” with no other cues:
  - choose **Stylized Low Poly** as a project default _but_
  - embed a lightweight “override hint” into the system prompt augmentation (not a UI change):
    - e.g., append: “(If you meant PS1/retro low poly, say ‘PS1 low poly’ or ‘smooth low poly’.)”

This makes the default explicit, reproducible, and easy to override.

### 4.4 Style is part of the state/goal

Style is not just aesthetics; it maps to measurable constraints:

- shading mode flag matches requested style
- face budget achieved
- triangulation/facet visibility consistent with chosen style

---

## 5) Evaluation (How We Measure Progress Without Hand-Waving)

### 5.1 Geometry KPIs

- validity / manifoldness proxy
- degenerates rate proxy
- non-self-intersection proxy
- face budget control (mean + p90 over prompts)
- determinism (replay produces same metrics)

### 5.2 Low-poly style KPIs

- stylized: percent outputs that are flat shaded AND meet face budget
- PS1: percent outputs that are smooth shaded AND meet face budget
- “facet visibility” proxy: distribution of face normals / sharp edge ratio

### 5.3 Displacement / noise KPIs

For procedural height maps or baked maps:

- range control: height min/max within expected bucket
- histogram sanity: avoid all-gray or clipped maps
- tileability score (if enabled)
- silhouette improvement proxy (bounding box / surface variation increases without invalidity)

### 5.4 Non-destructive KPIs

- percent outputs where modifiers remain unapplied by default
- percent outputs with exposed parameter controls (noise scale/strength present)

---

## 5.5 Prior Art Patterns That Work Fast on Weak Machines (What to Copy)

This section is here because “best results in least time on not-very-powerful machines” is mostly about picking the right _structure_, not bigger models.

### Pattern A — Program induction beats raw geometry

Canonical idea: learn a **short program in a constrained language** that produces the asset, instead of predicting raw vertices.

- Constructive Solid Geometry program induction (e.g., CSGNet) shows you can predict **modeling instructions** and even train without ground-truth programs using policy-gradient + render-based feedback.
- What we copy for Blender:
  - keep a small, typed action grammar
  - keep the state low-entropy
  - use deterministic execution + legality masks
  - use imitation/reconstruction wherever possible (cheaper than RL)

### Pattern B — Make “learning signal” cheap before it’s expressive

Instead of chasing photoreal pixels early, score the thing you care about with cheap proxies.

- Our cheap proxies: compact mesh stats + legality + style constraints.
- Only later add more expensive signals (render similarity), and only at low resolution.

### Pattern C — Data flywheels beat fancy algorithms

The fastest path to visible progress on weak machines is usually:

1. generate lots of deterministic traces cheaply
2. train a small model (teacher forcing)
3. self-improve by sampling multiple candidates + keeping top-k

This repo already supports that pattern via collapse traces + inversion + closed-loop rollouts.

### Pattern D — If you must learn from pixels, prioritize sample-efficiency tricks

Pixel-based RL has known ingredients for sample efficiency (e.g., strong data augmentation like DrQ-v2, or world models like DreamerV3), but in Blender the bottleneck is _render time_, not just GPU.

What we copy for Blender (if/when we do vision):

- tiny render targets (e.g., 64–128 px)
- heavy caching + reuse
- strong augmentations (crop/shift/brightness)
- curriculum: silhouette/depth before shaded color

---

## 6) Training Plan (Staged, Fast Feedback)

### Phase A — Scripted traces (imitation)

Deliverable: a library of deterministic recipes that produce action traces.

- mountains (plane→subdivide→displace→decimate→triangulate→shade)
- rocks
- cliffs
- simple buildings

Each trace records:

- prompt / intent tags
- seed
- action sequence
- state summaries after each step
- final score (KPIs)

### Phase B — Self-improvement data flywheel

- Run rollouts from current policy
- Sample action alternatives within legality masks
- Score + filter into a replay buffer

### Phase C — Optional lightweight RL

Only if needed after imitation + self-improvement saturate.
Use low-variance shaping rewards aligned to KPIs.

### Phase D (optional, Phase 3+) — Render-and-Compare supervision from images

NOTE: This is intentionally deferred. `.github/ARCHITECTURE.md` emphasizes compact numeric state first; image-based evaluation is a later add-on once geometry is stable.

Goal:

- Allow training / self-improvement against a **reference image** by rendering candidate outputs and computing a cheap similarity score.

Key constraints (to keep it fast + trainable locally):

- Eevee-only (or workbench) with a fixed, deterministic scene setup
- low resolution (64–128 px) and short step budgets
- compare _silhouettes/edges/depth_ first; only later compare shaded RGB

Resolution / “1:1 match” policy (important)

- We only get a meaningful pixel loss when the candidate render and reference image are aligned in **resolution + framing**.
- There are two valid ways to do this:
  1. **Render at reference resolution** (most faithful, slowest)
  2. **Normalize both to a scoring resolution** (fastest; still meaningful if done deterministically)

Recommended for local training:

- Choose a small set of scoring resolutions: e.g., `64` and `128` (square), and compute a **multi-scale score**.
- Normalize the reference image to the scoring resolution deterministically:
  - preserve aspect ratio
  - resize to fit inside the target canvas
  - letterbox/pad remaining pixels with a constant value
  - optionally apply a deterministic center-crop mode for datasets that expect it
- Render the candidate at the same scoring resolution(s) with the same camera/template.

Recommended for evaluation-only “true 1:1” tests:

- Render at the reference image’s resolution and aspect ratio, then compute the same metrics.
- Cache (reference_preprocessed, candidate_renders, masks/edges) so repeated scoring is cheap.

Scoring ladder (cheap → expensive):

1. **Silhouette IoU** (binary mask match)
2. **Edge IoU / edge F1** (Canny/Sobel edges)
3. Depth-map L1 (if stable)
4. RGB SSIM / L1 (only when lighting/materials are also in-scope)

How it plugs in:

- Use the existing “sample many candidates → score → keep top-k” pattern:
  - for each prompt + reference image, sample multiple action plans (policy or beam search)
  - render each candidate from a locked camera
  - compute similarity score
  - keep best plans as additional supervision (like reconstruction self-improvement)

Why this works even without making the policy image-conditioned at first:

- The reference image score can be used to _select_ better rollouts (data curation) while the policy still learns from compact state + prompt tags.
- If/when image conditioning is added, keep it compact: a small CNN that outputs a <=256-d embedding (not raw pixels into a big transformer).

---

## 7) Implementation Checklist (Concrete Next Steps)

### 7.1 Environment + legality

- Add/verify deterministic seeding for procedural textures.
- Implement legality masks for: subdiv/decimate/shading/triangulate.

### 7.2 Canonical procedural node groups

- Create the small set of canonical node groups listed above.
- Ensure parameters are exposed and named consistently (so actions map cleanly).

### 7.3 Action vocabulary + parameter buckets

- Define bucket ranges for:
  - subdiv level
  - decimate ratio
  - displacement strength
  - noise scale/detail/roughness
  - bake resolution

### 7.4 Evaluation suite extensions

- Add a “low-poly style” eval suite:
  - stylized low poly prompts
  - PS1/retro low poly prompts
- Add displacement/noise eval prompts:
  - “rocky terrain”, “soft dunes”, “jagged mountains”, etc.
- Phase 3+ (optional): add a render-similarity eval harness for silhouette/edges at 64–128 px.

### 7.5 UX alignment (no extra complexity)

- Keep the UI simple; use existing direct generation + policy/agent entry points.
- Avoid adding new panels/toggles early; use prompt disambiguation rules first.

### 7.6 Workflow reconstruction (the “programmer path”)

- Add a batch job that turns extracted `.blend` JSON into **candidate action plans** + scores.
- Add a headless replay step to validate candidates by comparing compact metrics.
- Export the final supervision as JSONL traces usable by the policy training loop.

If we later extend reconstruction to use reference images:

- Keep it as an additional scoring term (not a replacement for geometry KPIs).
- Prefer silhouette/edge scores and cache them aggressively.

Deliverable definition:

- Given `N` processed assets, produce `M` valid reconstructed traces with:
  - deterministic replay
  - legality compliance
  - outcome metrics within tolerances

---

## 8) Database Hygiene (Data Cleanup + Integrity)

Training progress is only meaningful if the underlying dataset is consistent and auditable.

Plan (high level):

- Define integrity invariants per artifact layer: raw → processed → cache → derived datasets.
- Run audits regularly (label entropy, dedup rate, mesh complexity distribution).
- Prefer “quarantine + rebuild” over hard deletion.
- Version datasets via lightweight manifests so checkpoints can be tied to a dataset snapshot.

Execution playbook (commands + step-by-step):

- See [docs/TRAINING_DATA_HYGIENE.md](docs/TRAINING_DATA_HYGIENE.md)

---

## 10) How This Plan Maps to the Current Codebase (What Already Exists)

This repo already implements a large chunk of the “programmer path” for reconstructing steps — not by guessing a single recipe, but by generating **deterministic traces** from arbitrary meshes and converting them into “build” supervision.

### 10.1 Existing step-reconstruction mechanism: collapse → invert → build supervision

**A) Collapse trace generator (inside Blender, deterministic)**

- `processing/collapse_trace_worker.py`
  - Inputs:
    - `--in-blend` + `--object-name` (collapse a real object from a `.blend`), OR
    - `--mesh-json` (vertices/faces from the cache/synthetic generator)
  - Output: `trace.jsonl` with per-step `{pre stats, op, post stats}`
  - Determinism: operations are deterministic (remove modifiers from end, then unsubdivide/dissolve/merge/delete_loose)

**B) Inversion into forward “build” actions (outside Blender)**

- `policy/dataset.py` implements:
  - `_inverse_action_from_collapse_step(step)` mapping collapse ops → approximate inverse build ops
  - `RealMeshBuildTraceStream` which:
    - loads `*/trace.jsonl`
    - reverses steps
    - emits compact `(state → action)` training batches

This is already a scalable answer to: “I don’t want to model thousands of demos.”
It turns _any_ mesh into action supervision using deterministic collapse traces.

### 10.2 Existing trace dataset generators (offline)

**Real meshes from cache → collapse traces**

- `scripts/generate_collapse_traces_from_cache.py`
  - Reads `.pt` samples from `data/processed/.mesh_cache`
  - Decodes mesh tokens → vertices/faces
  - Calls Blender headless with `processing/collapse_trace_worker.py`
  - Writes `data/datasets/collapse_traces/<mesh_id>/trace.jsonl`

**Synthetic terrain → collapse traces (prompt grounding)**

- `scripts/generate_collapse_traces_synthetic_terrain.py`
  - Generates heightmap-like quad terrains in Python
  - Calls `processing/collapse_trace_worker.py`
  - Produces traces with prompt variants that include terrain tokens

### 10.3 Existing policy training wiring (already supports trace mixing)

- `training/train_policy.py`
  - Trains the compact policy transformer per `.github/ARCHITECTURE.md`
  - Supports mixing:
    - `SyntheticImitationStream` (teacher in compact env)
    - `RealMeshBuildTraceStream` (from collapse traces)
  - Config hook: `data.collapse_trace_source` (path + mix_prob)

### 10.4 Existing real-Blender closed-loop execution (policy acts on real Blender state)

**Long-running Blender executor**

- `processing/blender_policy_worker.py`
  - Reads `action_XXXX.json` from a work dir
  - Applies action in Blender
  - Writes `state_XXXX.json` with compact mesh stats

**Controller/inference loop (torch in venv, Blender stays alive)**

- `scripts/rollout_policy_closed_loop.py`
  - Samples actions from the policy
  - Writes action files / reads state files
  - Optional “low poly bias” mask to prevent topology explosion

**Self-improvement with reconstruction scoring**

- `training/self_improve_policy_reconstruction.py`
  - Rollouts in Blender per prompt
  - Scores vs reference mesh (Chamfer/F-score)
  - Keeps best trajectories → imitation training

### 10.5 Current action grammar status (important constraint)

Right now the policy action space is:

- `policy/actions.py::ActionType` (12 actions)
- Executed in Blender by `processing/blender_policy_worker.py` and `processing/execute_policy_plan.py`

This action set does NOT yet include explicit:

- `DISPLACE`, `DECIMATE`, `TRIANGULATE`, `SHADE_FLAT/SMOOTH`, node-graph edits

So the correct implementation path is staged:

1. Use collapse-trace reconstruction + current action set to build robust “modeling muscle memory” (selection discipline, non-exploding edits, determinism).
2. Add new actions _only after_ the baseline is stable and measured.

---

## 11) Proper Implementation Path (Repo-Concrete Milestones)

### Milestone 1 — Step reconstruction without manual demos (already feasible)

1. Generate traces:
   - real cache: `scripts/generate_collapse_traces_from_cache.py`
   - terrain grounding: `scripts/generate_collapse_traces_synthetic_terrain.py`
2. Train policy with trace mixing:
   - configure `config.policy_*.yaml` with `data.collapse_trace_source.path = data/datasets/collapse_traces`
3. Validate with real Blender rollouts:
   - `scripts/rollout_policy_closed_loop.py` producing `mesh.obj` and `stats_final.json`

This directly operationalizes: “reconstruct steps for modeling things” using real assets.

### Milestone 2 — Reconstruction from real `.blend` objects (next, still deterministic)

Use `processing/collapse_trace_worker.py --in-blend ... --object-name ...` to produce traces directly from `.blend` objects.

Batching strategy:

- iterate over mesh objects in a `.blend` (or a curated subset)
- write `out_dir/<blend_stem>__<object_name>/trace.jsonl`

This avoids needing token decode and leverages Blender’s evaluated mesh.

### Milestone 3 — Non-destructive reconstruction (modifier/node stacks)

Once baseline is stable:

1. Extend reconstruction inputs using the existing extractor:
   - `processing/blend_extractor.py` already extracts:
     - modifier stacks (config-driven)
     - node groups / materials (summaries)
2. Add a “modifier/node hypothesis generator” that proposes _multiple_ plans:
   - plan A: constructive macro + micro detail
   - plan B: displacement-first (terrain)
   - plan C: stylized low poly (flat shaded)
3. Replay+score candidates using:
   - `processing/execute_policy_plan.py` (simple single-object executor)
   - or the worker/controller loop for longer runs

Important: this stage likely requires adding new `ActionType`s (DISPLACE/DECIMATE/SHADE/…)
and teaching the worker how to apply them.

---

---

## 8) Key Risks + Guardrails

- **Non-determinism in Blender**: enforce seed propagation; avoid ops that behave inconsistently across runs.
- **Overfitting to a single “low poly” definition**: keep disambiguation rules explicit; measure both stylized and PS1 suites.
- **Texture scope creep**: start with procedural node groups + parameter prediction before any learned image generator.
- **Performance**: keep bake resolutions small by default; avoid heavy texture baking loops on M3.

---

## 9) Minimal “Next 7 Days” Execution Plan

1. Implement canonical procedural displacement group + parameters.
2. Add mountain trace generator (scripted) producing action sequences and metrics.
3. Add low-poly style eval suite (stylized + PS1).
4. Train a tiny intent→noise-params predictor (bucket classifier) and integrate into the policy as a sub-head or as a pre-step.
5. Run A/B eval vs baseline checkpoint and verify:
   - style correctness up
   - face budget stability up
   - validity non-regression
