# Blender Copilot — Production Runbook

Date: 2026-02-18

This runbook is the execution checklist for running production data pull, cache build, training, and promotion.

## 0) Definition of "Production Ready"

The stack is production ready when all items below pass:

1. Pipeline health checks pass (CLI + module import + config budgets).
2. Data pull completes without fatal source failures.
3. Cache build runs end-to-end and yields non-empty trainable cache.
4. Training starts cleanly and produces checkpoints/metrics.
5. KPI gates pass on geometric + domain + tool-loop reliability metrics.

## 1) Environment & Preconditions

1. Activate env:

```bash
source .venv/bin/activate
```

2. Verify core status:

```bash
python run.py status
```

3. Confirm production budgets are active:

```bash
python -c "import scripts.data_pipeline as dp; import scripts.rebuild_cache as rc; print('training_max_tokens', dp._training_max_tokens()); mt,_=dp._get_tokenizers(); print('cache_mesh_tok_max_faces', mt.max_faces); print('rebuild_MAX_FACES', rc.MAX_FACES, 'TARGET_DECIMATE', rc.TARGET_DECIMATE)"
```

Expected: high training max tokens (72002), rebuild max faces around 8000.

## 2) Production Data Pull

Run full live pull (no source-size discard in production path):

```bash
python run.py data --keep-raw
```

Optional scoped run by source:

```bash
python run.py data --sources open3dlab blendswap objaverse objaverse_xl --keep-raw
```

If reprocessing existing raw assets only (no network):

```bash
python run.py data --local
```

## 3) Production Cache Build

Build/refresh training cache from processed data:

```bash
python run.py build
```

If you need direct rebuild cache control:

```bash
python scripts/rebuild_cache.py
```

Quick smoke only:

```bash
python scripts/rebuild_cache.py --dry-run --max-files 10
```

## 4) Production Training

Start local training:

```bash
python run.py train
```

Cloud GPU route:

```bash
python run.py lambda-train HOST
```

Checkpoint sync back to local:

```bash
bash cloud/sync_checkpoints.sh HOST --continuous
```

## 5) Evaluation & Promotion Gates

Run geometric eval:

```bash
python run.py eval
```

Inspect domain KPI outputs in eval JSONs (includes professional gate summary).

Analyze tool-loop reliability from chat logs:

```bash
python scripts/analyze_tool_loops.py
```

Promote checkpoint only if all pass:

1. Geometry non-regression vs baseline.
2. Domain-level KPI gate marked promotion-ready.
3. Tool-loop dead-end and repetitive-call rates improved or stable.
4. No critical regressions on simple-shape reliability.

## 6) Operational Safety Rules

1. Do not re-introduce raw-file-size drop gates for production pulls.
2. Keep high-detail assets in acquisition/cache; only decimate when training sequence budget requires it.
3. Keep Qwen-VL as runtime decision layer; keep CLIP as auxiliary training/eval signal.
4. Use iterative generation policy for M3-class local inference: decompose → generate → inspect → refine.

## 7) Quick Triage

1. Data pull failures by source: rerun with `--sources` subset and inspect source-specific logs.
2. Cache anomalies: run `python scripts/rebuild_cache.py --dry-run --max-files 50`.
3. Training OOM: lower batch size / increase grad accumulation in config or cloud env overrides.
4. KPI regression: compare current eval bundle against previous promoted checkpoint before retraining.

## 8) Current Production Defaults (Expected)

1. `tokenization.max_faces = 8000`
2. `unified.geometry.max_seq_length = 72002`
3. `unified.embed_dim = 1024` and `unified.geometry.num_layers = 16`
4. `unified.enable_contrastive = true`
5. `unified.enable_image_to_mesh = true`
6. Data pipeline intake preserves large/high-detail `.blend` files.

## 8.1) High-Fidelity Guardrails (Do Not Relax)

1. Do not reduce production `max_faces`/`max_seq_length` to low-poly budgets for H100/A100-80GB runs.
2. Keep cloud training sequence cap aligned with full-detail profile on >=80GB GPUs.
3. Any smaller-GPU fallback profile must be explicitly labeled as non-production.

## 9) One-Command Daily Sequence

```bash
python run.py data --keep-raw && python run.py build && python run.py train
```

Use this only when you want a full pull→cache→train cycle in one go.
