# Training Data Hygiene Plan (Database Cleanup + Integrity)

Date: 2026-02-28

This is an execution-oriented plan to:

1. clean up the repo “database” (all data artifacts under `data/` + related cache/render outputs)
2. ensure **integrity** of the training data for **current modeling (geometry) training**, while also keeping artifacts in a form that is reusable for future training (policy traces, materials, image matching, etc.).

The guiding principle is: **never delete irreplaceable raw data without a manifest**, and never train on a dataset you can’t audit.

---

## 0) What “the database” is in this repo

Think of it as a pipeline of increasingly derived artifacts:

- `data/raw/`
  - downloaded original files (`.blend`, `.glb`, `.obj`, `.zip`, etc.) + `*.meta.json`
  - global dedup index: `data/raw/.global_hashes.json`

- `data/processed/`
  - extracted/normalized JSON representations (often per source)
  - unified training cache: `data/processed/.mesh_cache/*.pt`

- `data/datasets/`
  - JSONL datasets for specific tasks (geometry prompts, collapse traces, etc.)

- `data/renders/`
  - rendered images and manifests for meshes/scenes

- `data/feedback/`
  - addon feedback logs that can become future training data

**Current geometry-focused training** (e.g., `config.retrain_v1024_focal.yaml`) is primarily sensitive to:

- `data/processed/.mesh_cache/*.pt`
- and/or `data/processed/<source>/*.json` via cache building in `training/train_unified.py`

---

## 1) Integrity invariants (what must be true)

### 1.1 Raw layer invariants (`data/raw/`)

- Every downloaded file has a sibling metadata file when possible (URL, source, license hints, checksum).
- Global dedup registry exists and is not corrupted: `data/raw/.global_hashes.json` parses as JSON list.
- Files that claim to be `.blend` are actually blend files (magic bytes check).

### 1.2 Processed JSON invariants (`data/processed/<source>/*.json`)

At minimum, processed JSONs used for training should be:

- parseable JSON
- not “marker-only” or metadata-only (`*.meta.json` are intentionally skipped by training)
- for objects:
  - have a mesh representation that can be tokenized (verts/faces or already-tokenized payload depending on source)
  - have stable provenance fields (source, optional source_file, optional source_url)

### 1.3 Cache invariants (`data/processed/.mesh_cache/*.pt`) — **critical for training**

A cache entry is trainable if it has:

- `mesh_tokens`: integer tensor, length >= 3, length <= configured max
- `text_ids` + `text_mask`: shapes consistent with configured `max_text_length`
- no NaNs/Infs in any numeric tensors used for training
- counts in a plausible range (faces/verts), and matches token length budget
- a non-empty label signal (either `label` or a fallback instruction)
- stable provenance: `data_source` and/or `source_file` where possible

### 1.4 Derived artifact invariants (future-proofing)

- Collapse traces: each `*/trace.jsonl` is valid JSONL; each line has `{pre, op, post}` at minimum.
- Renders: each render dir has a manifest JSON with the parameters used to render (engine, resolution, samples, camera).
- If embedding images into cache, store:
  - the image tensor
  - the exact render settings
  - the preprocessing settings (so image matching is reproducible)

---

## 2) Audit-first workflow (don’t clean blindly)

Run audits to understand what you have before deleting/rebuilding.

### 2.1 Quick cache audit

- `python scripts/audit_cache.py`
- `python scripts/audit_cache_deep.py`

### 2.2 Full data quality report (recommended)

- `python scripts/data_quality_report.py`

This report already covers several critical things for modeling training:

- label entropy / over-repeated labels
- estimated dedup rate
- mesh complexity distribution
- quality_weight distribution

### 2.3 Interactive validation in Blender (one-click approve/reject)

If you believe “the only real QA is looking at it in Blender”, use the validator panel.

**Preferred (live-cache) workflow — validates exactly what training reads**

1. Review inside Blender:

- In addon preferences, set:
  - **Policy Project Root** = this repo folder
  - **Policy Python** = your venv python (e.g. `.../blender-copilot/.venv/bin/python`)
- Open the sidebar panel: **AI Copilot → Dataset Validator**
- Set **Validation Queue** to either:
  - the repo root folder, OR
  - `data/processed/.mesh_cache/`
- Enable **Fresh Scope Only**
- Set **Fresh Window (hours)** (e.g. `24`–`72`) so review focuses on newly regenerated/newly pulled items
- Click **Load Queue**
- For each item you can edit **Label** and **Tags**, then:
  - **Approve + Next**
  - **Reject + Next** (sets `quality_weight=0.0` in the source cache so it stops contributing to training)
  - **Skip + Next**

2. What gets written:

- Decisions are appended under `data/validation_queue_live/reviews.jsonl`.
- Cache edits are applied directly back into the originating `.pt` cache entry (via external venv python — Blender stays torch-free).
- For display, the validator will materialize an `items/<item_id>.json` file under `data/validation_queue_live/` (so Blender can load vertices/faces without torch).

Notes:

- Re-opening and clicking **Load Queue** will skip items already present in `reviews.jsonl`.
- In **Fresh Scope Only** mode, only reviews within the fresh window are considered, so old review history does not hide newly regenerated content.

**Optional (subset/legacy) export workflow**

If you want a curated subset (e.g. only a certain range/quality), you can still export a queue folder:

```bash
./.venv/bin/python scripts/export_validation_queue.py \
  --out data/validation_queue \
  --max-items 500 \
  --min-quality-weight 0.0
```

Then set **Validation Queue** to `data/validation_queue/` and review as above.

---

## 3) Cleaning plan — current modeling training (geometry) priority

### Step 1 — Make cleanup safe (archive first)

Use the built-in cleanup workflow to archive old artifacts before deletion:

- `python run.py clean --checkpoints --wandb --eval --apply`

If disk is tight:

- `python run.py clean --checkpoints --cache --renders --wandb --eval --apply`

Only add permanent deletion (`--rm`) after verifying the archive.

### Step 2 — Fix cache integrity issues (don’t delete everything)

The training stack already ignores tiny marker cache files (<200 bytes). For actual corrupted caches:

- Identify cache entries that fail to `torch.load()` or lack `mesh_tokens`.
- Delete only those `.pt` files so they are rebuilt.

Primary rebuilding tool:

- `python scripts/rebuild_cache.py`

Useful modes (pick based on what the audits show):

- cap repeated labels to reduce overfitting:
  - `python scripts/rebuild_cache.py --max-per-label 100 --cap-existing`
- backfill source attribution on older cache entries:
  - `python scripts/rebuild_cache.py --fix-attribution`
- if labels are low-quality:
  - `python scripts/rebuild_cache.py --relabel-only`

### Step 3 — Dedup strategy (two layers)

- **Raw-level dedup** (cheap, source-agnostic): SHA-256 via `data/raw/.global_hashes.json`.
- **Cache-level dedup** (training-level): structural hashes of mesh tokens (fast proxy) + label caps.

Important: don’t rely on labels for dedup — labels can be noisy.

### Step 4 — Quarantine bad samples instead of hard-delete

Create a quarantine folder for anything suspicious so it’s reversible:

- `data/_quarantine/mesh_cache/`
- `data/_quarantine/processed_json/`

Move, don’t delete, until the pipeline is stable.

---

## 4) Ensuring integrity for future training (keep artifacts usable later)

### 4.1 Add a manifest habit (lightweight “DB integrity”)

For each dataset build/cleanup pass, write a manifest directory:

- `data/_manifests/YYYYMMDD_HHMMSS/`
  - `summary.json` (counts per source, counts per artifact type)
  - `mesh_cache_index.jsonl` (one line per cache key: label, source, face estimate, quality_weight, hashes)
  - `notes.md` (what you changed and why)

This is what turns “a folder of files” into a database.

### 4.2 Record schema versions

When cache schemas evolve, add:

- `schema_version` fields inside cache items
- and/or a top-level `DATASET_VERSION` file in `data/processed/.mesh_cache/`

The goal is reproducibility: a run can log “trained on dataset version X”.

### 4.3 Keep provenance for later (materials, images, policy traces)

Even if current training ignores these, preserve them when available:

- render settings + camera metadata (see `scripts/render_scenes.py`)
- embedded images where available (mesh renders / scene renders)
- modifier/material summaries (when `scripts/build_training_jsonl.py` emits them)
- collapse traces under `data/datasets/collapse_traces/...`

---

## 5) Image-matching readiness (after base modeling is trained)

### 5.1 Resolution alignment (the “1:1” requirement)

For render-and-compare scoring to be meaningful, candidate and reference must match **resolution + framing**.

Two supported modes:

- **True 1:1**: render at the reference image’s exact size/aspect, then score.
- **Normalized scoring** (recommended for training): resize both deterministically to a small scoring canvas (multi-scale 64/128), then score.

Normalization rules should be deterministic:

- preserve aspect ratio
- letterbox/pad to target canvas
- optionally a deterministic center-crop mode

Cache the preprocessed reference image and derived masks/edges so repeated scoring is cheap.

### 5.2 Use web images primarily as evaluation first

Start with evaluation-only from web images (license-safe). Once the pipeline is stable, consider training on curated sets with clear usage rights.

---

## 6) Suggested cadence (keeps the DB from rotting)

Weekly:

- run `python scripts/data_quality_report.py`
- cap labels if a few labels dominate
- prune only what’s proven broken or redundant

Monthly:

- rebuild a fresh cache slice from scratch to verify the pipeline still works end-to-end
- write a manifest for the dataset version used in your best checkpoint
