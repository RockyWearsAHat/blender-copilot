# Project Control Panel (Read This First)

This is the shortest accurate map of how this repo works today.

## 1) What actually happens when you run training

Entry point:

- `python run.py train --config <config.yaml> --name <run_name>`

Execution path:

1. `run.py` (`cmd_train`) loads config and starts logging/monitor wiring.
2. It calls `training/train_unified.py:train(config, args)`.
3. `train_unified.py` builds model + tokenizers + task streams.
4. `InfiniteMultiTaskSampler` alternates batches across enabled tasks.
5. Main loop runs forever (until max steps, Ctrl+C, or stop condition), saving checkpoints every `save_every`.

In your current retrain config (`config.retrain_v1024_focal.yaml`):

- `enable_materials: false`
- `enable_modifiers: false`
- `enable_contrastive: false`
- `enable_image_to_mesh: false`

So practically, training is geometry-focused, mainly:

- synthetic geometry stream
- real geometry stream (if processed JSON/cache exists)

## 2) Why this feels bloated

Primary causes are not model size; they are orchestration + artifacts:

- Many historical configs in root (`config*.yaml`) make run intent ambiguous.
- One very large orchestrator file (`run.py`) handles many unrelated workflows.
- One very large trainer (`training/train_unified.py`) contains many optional paths.
- Artifact growth is huge (`data/`, `checkpoints/`, `_trash/`), making state hard to reason about.

## 3) Practical operating mode (MVP)

Use this repo in one lane unless you are explicitly experimenting:

1. **Data lane**: keep only curated sources needed for current experiment.
2. **Train lane**: run one config profile only.
3. **Serve lane**: serve from one known checkpoint alias.

Recommended default lane for local Apple Silicon iteration:

- config: `config.retrain_v1024_focal.yaml`
- command: `python run.py train --config config.retrain_v1024_focal.yaml --name retrain_v1024_focal`
- checkpoint pointer: `checkpoints/_active/train_latest.pt`

## 3b) Milestone 1 (geometry-first policy lane)

If you want the reconstruction-first policy workflow (collapse traces → policy imitation → real Blender rollout), use these commands:

1. Generate real collapse traces from cache:
   - `python run.py trace-cache --blender /Applications/Blender.app/Contents/MacOS/Blender --max-files 1000 --skip-existing`
2. (Optional) Add terrain-focused traces for better prompt grounding:
   - `python run.py trace-terrain --blender /Applications/Blender.app/Contents/MacOS/Blender --n 200 --skip-existing`
3. Train compact policy with trace mixing:
   - `python run.py train-policy --policy-config config.policy_m3_quick_traces.yaml --device auto --max-steps 2000`
4. Validate with real Blender closed-loop rollout:
   - `python run.py rollout-policy --ckpt checkpoints/policy_goal_sel10/latest.pt --out-dir data/eval/rollouts/m1_smoke --steps 64 --blender /Applications/Blender.app/Contents/MacOS/Blender --low-poly-bias --prompt "stylized low poly rock"`

This lane stays aligned with architecture constraints: compact state, finite action grammar, deterministic replay, and local M3 feasibility.

## 4) Safe de-bloat sequence

Run in this order:

1. Preview cleanup:
   - `python run.py clean --checkpoints --wandb --eval`
2. Archive old training artifacts (safe rollback path):
   - `python run.py clean --checkpoints --wandb --eval --apply`
3. If disk is still tight, include cache + renders:
   - `python run.py clean --checkpoints --cache --renders --wandb --eval --apply`
4. Permanent delete only when sure:
   - add `--rm`

## 5) Architecture guardrail (important)

Per `ARCHITECTURE.md` and `.github/ARCHITECTURE.md`, keep the system:

- compact state representation
- constrained action/policy learning
- efficient local training on M3
- model scale bounded unless metrics prove need to expand

If a change increases entropy/complexity without measurable metric gain, do not merge it.

## 6) Weekly hygiene checklist

- Keep only 1 active training config for day-to-day work.
- Move old experiment configs into a dated archive folder.
- Keep only active + best checkpoints for each run family.
- Prune `_trash/` after confirming archives are no longer needed.
- Track one dashboard metric set: topology validity, face-count stability, symmetry.

## 7) Training data integrity (the "database" plan)

When training quality feels weird, assume data integrity first.

- Execution playbook: see [docs/TRAINING_DATA_HYGIENE.md](docs/TRAINING_DATA_HYGIENE.md)
- Fast checks:
  - `python scripts/audit_cache.py`
  - `python scripts/data_quality_report.py`
- Safe cleanup (archives first):
  - `python run.py clean --checkpoints --cache --renders --wandb --eval --apply`
