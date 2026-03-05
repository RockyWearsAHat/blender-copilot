# Blender AI Development Instructions

# Copilot Instructions – Blender AI Project

## MANDATORY RULE

All AI-related design and implementation decisions MUST follow
the system architecture defined in:

ARCHITECTURE.md

And MUST follow model ownership + blame routing defined in:

docs/model_action_ownership.md

If any generated code or proposal conflicts with #file:.github/ARCHITECTURE.md ,
ARCHITECTURE.md overrides.

Copilot must reference ARCHITECTURE.md when:

- Designing model architecture
- Proposing scaling changes
- Suggesting training strategies
- Modifying action space
- Expanding state representation
- Recommending RL or diffusion approaches

Copilot must reference docs/model_action_ownership.md when:

- Assigning responsibility for action failures
- Debugging model-vs-data pipeline regressions
- Adding or changing action capabilities
- Defining training subset filters per model owner
- Reviewing reconstruction mismatches involving visibility, modifiers, or materials

---

## Architectural Enforcement Block

Before generating any AI-related code,
Copilot must internally verify:

1. Does this reduce entropy?
2. Does this inject structured inductive bias?
3. Is this trainable on Apple Silicon (M3)?
4. Does this avoid unnecessary scaling?
5. Does this follow the modular system design?
6. Is the model under 50M parameters unless justified?
7. Is raw mesh input being avoided?

If the answer to any of these is NO,
Copilot must revise the proposal.

---

## Project Goal

Build a structured AI system that learns to generate and manipulate Blender meshes using a constrained action space and efficient training loops.

The goal is NOT:

- Training a large language model
- Raw mesh-to-mesh diffusion
- Full scene understanding

The goal IS:

- Learning structured modeling policies
- Generating valid, manifold low-to-moderate complexity meshes
- Producing deterministic Blender Python programs
- Achieving visible results within hours of training on Apple Silicon (M3)

---

## System Architecture

The system must be modular and structured as follows:

1. Data Layer
2. Environment Layer (Blender wrapper)
3. Model Layer (Policy Transformer)
4. Training Loop
5. Evaluation + Validation

No monolithic design.

---

## 1️⃣ Data Representation Rules

DO NOT:

- Use raw vertex arrays directly as model input initially.
- Feed entire mesh topology graphs into a large transformer.
- Use high-dimensional dense geometry tensors.

DO:

- Extract structured mesh statistics:
  - vertex_count
  - face_count
  - edge_count
  - bounding_box (x, y, z)
  - symmetry_score
  - manifold_flag
  - selected_face_count
  - avg_edge_length
  - surface_area
  - volume_estimate

State must remain compact and numerical.

---

## 2️⃣ Action Space Constraints

The action space must be finite and parameterized.

Example allowed actions:

- ADD_CUBE
- ADD_CYLINDER
- EXTRUDE(distance)
- INSET(amount)
- BEVEL(amount)
- SCALE(x, y, z)
- SUBDIVIDE(level)
- DELETE_FACE
- SELECT_RANDOM_FACE
- MIRROR(axis)
- APPLY_MODIFIER(type)

Each action must:

- Return updated state
- Validate topology
- Flag invalid operations

Keep action space under 30 operations initially.

---

## 3️⃣ Model Design Constraints

Target model size:

- 10M–50M parameters
- Max 6 transformer layers
- Hidden size 256–512
- Sequence length <= 128

Training objective:

- Next-action prediction (cross entropy)

Avoid:

- Large LLMs
- Diffusion models
- Graph transformers (initial phase)

Use:

- Small transformer encoder-decoder OR causal transformer
- PyTorch with Metal backend (mps)

---

## 4️⃣ Training Strategy

Training phases:

### Phase 1 — Supervised Imitation

- Use existing modeling logs or procedural scripts
- Train on state → action pairs
- Use teacher forcing

### Phase 2 — Self-Improvement

- Generate sequences
- Score using mesh metrics
- Keep high-scoring samples
- Add to replay buffer

### Phase 3 — Optional RL Fine-Tuning

Reward example:

reward =

- abs(vertex_count - target_vertex_count)

* symmetry_score
* manifold_bonus

- self_intersection_penalty

Avoid high-variance RL initially.

---

## 5️⃣ Efficiency Rules

- Batch size 16–32
- Mixed precision (fp16)
- Gradient clipping
- AdamW optimizer
- Learning rate warmup

Training should produce visible improvement within 1–3 hours.

If training is slow:

- Reduce model size
- Reduce sequence length
- Reduce state dimensionality

---

## 6️⃣ Evaluation Metrics

Model quality is determined by:

- Valid topology percentage
- Average vertex count stability
- Symmetry metric
- Non-self-intersection rate
- Program length consistency

Not subjective visual judgment initially.

---

## 7️⃣ Code Generation Mode

If generating Blender Python scripts:

- Use deterministic API calls
- Avoid UI interaction
- Operate strictly through bpy

Generated programs must:

- Be replayable
- Be idempotent
- Reset scene before execution

---

## 8️⃣ What Copilot Should Prioritize

When generating code:

- Favor modular structure
- Favor typed dataclasses
- Favor reproducibility
- Favor deterministic randomness (set seeds)
- Favor minimal dependencies
- Optimize for Apple Silicon

Avoid:

- Overengineering
- Massive frameworks
- Cloud training code

---

## 9️⃣ Future Expansion (Do Not Implement Yet)

- Graph neural networks for topology reasoning
- Vision-based evaluation
- Diffusion-based mesh refinement
- Multi-agent planners

These are phase 3+ ideas.

---

## Summary

This project is a structured policy learning system for Blender.

It must:

- Be trainable locally
- Show measurable improvement within hours
- Remain computationally bounded
- Scale incrementally

Keep it efficient.
Keep it modular.
Keep the action space constrained.

---

## Ownership & Blame Enforcement

All action/capability regressions must be routed through `docs/model_action_ownership.md`.

Copilot must:

1. Identify the failing action/capability and its primary owner.
2. Validate owner-specific dataset filters before proposing model changes.
3. Attribute failure to primary owner first; only escalate to secondary owners after evidence.
4. Update `docs/model_action_ownership.md` in the same PR when adding actions or changing ownership boundaries.

Do not ship AI/action changes without ownership mapping alignment.
