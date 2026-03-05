# Blender AI – System Architecture

## Core Principle

We do NOT attempt to beat scaling laws by brute force.
We bend scaling laws by reducing entropy, constraining structure,
and injecting inductive bias.

This system must be trainable on Apple Silicon (M3) and show
measurable improvement within hours.

---

# 1. Scaling Doctrine

Scaling laws apply when:

- Dense transformers
- Next-token prediction
- Broad unstructured data

We avoid this regime by:

1. Narrowing domain distribution (Blender modeling only)
2. Using structured action grammar
3. Using synthetic data
4. Using dense supervised gradients
5. Adding constraints at architecture level

We reduce entropy before increasing parameters.

---

# 2. System Design Philosophy

The system is modular.

It is NOT:

- One monolithic neural network
- Raw mesh-to-mesh generator
- Diffusion model
- General-purpose LLM

It IS:

- Policy learning over structured modeling grammar
- Deterministic environment execution
- Reward-driven refinement
- Constrained program generation

---

# 3. System Modules

## A. Blender Environment Layer

Responsibilities:

- Apply deterministic bpy operations
- Extract structured mesh statistics
- Validate topology
- Enforce action legality

Must:

- Be fully deterministic
- Be seedable
- Support replay

---

## B. State Representation Layer

State must remain compact.

Allowed inputs:

- vertex_count
- face_count
- edge_count
- bounding_box
- avg_edge_length
- manifold_flag
- symmetry_score
- volume_estimate
- selected_face_count

DO NOT:

- Feed raw vertex arrays
- Feed full adjacency matrices initially
- Use high-dimensional geometric tensors

Goal: Low-entropy state compression.

---

## C. Action Grammar

Action space must remain finite and masked.

Max initial action count: 30.

Each action:

- Has constrained parameters
- Returns updated state
- Can be invalidated by legality rules

Grammar > raw generation.

---

## D. Policy Model

Initial constraints:

- 10M–50M parameters
- <= 6 transformer layers
- Hidden size <= 512
- Sequence length <= 128
- Causal transformer or encoder-decoder
- Mixed precision training

We scale only if:

- Metrics plateau
- Model capacity is proven insufficient

Never scale preemptively.

---

## E. Training Strategy

Phase 1:
Supervised imitation on synthetic procedural data.

Phase 2:
Self-improvement loop:

- Generate
- Score
- Filter
- Replay

Phase 3:
Optional lightweight policy gradient refinement.

We avoid high-variance RL early.

---

# 4. Efficiency Doctrine

We optimize for:

- Sample efficiency
- Inductive bias
- Constraint encoding
- Deterministic validation
- Structured reward

We do NOT optimize for:

- Maximum parameter count
- LLM-scale training
- Diffusion-based generation

---

# 5. Metrics (Ground Truth Evaluation)

Quality is defined by:

- Valid topology rate
- Non-self-intersection rate
- Vertex budget control
- Symmetry score
- Surface continuity
- Program determinism

Not subjective aesthetic judgement initially.

---

# 6. Expansion Rules

We only expand complexity if:

1. Baseline system is stable
2. Metrics plateau
3. Bottleneck is identified
4. Hardware constraints are respected

Future possible expansions:

- Graph neural networks
- Visual embedding models
- Planner/executor split
- Adapter-based LLM planning layer

Not before baseline works.

---

# 7. Golden Rule

Reduce entropy first.
Inject structure second.
Scale last.
