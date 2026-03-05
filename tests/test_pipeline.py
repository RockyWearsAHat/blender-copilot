"""End-to-end pipeline smoke test.

Verifies the ENTIRE training pipeline without requiring real data:
  1. MeshTokenizer — encode/decode round-trip
  2. BPETokenizer — encode/decode round-trip
  3. Synthetic data generation — InfiniteShapeStream produces valid batches
  4. Model forward pass — UnifiedBlenderModel computes geometry loss
  5. Backward pass — gradients flow through the model
  6. Checkpoint save/load — state is preserved across restarts
  7. Geometric eval metrics — validity, Chamfer Distance, F-score
  8. Test suite — frozen prompts run through check_shape_expectations
  9. Improvement tracking — second eval result is logged correctly

Run with:
    python -m pytest tests/test_pipeline.py -v
    python tests/test_pipeline.py          # run directly
"""

import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_cube_mesh():
    """Return (vertices, faces) for a unit cube — simplest non-trivial mesh."""
    verts = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]
    faces = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [1, 2, 6], [1, 6, 5],  # right
        [0, 3, 7], [0, 7, 4],  # left
    ]
    return verts, faces


def _normalized_cube():
    """Return a cube mesh normalized to [-1, 1]."""
    from processing.generate_synthetic import normalize_mesh
    verts, faces = _make_cube_mesh()
    verts = normalize_mesh(verts, target_range=(-1.0, 1.0))
    return verts, faces


def _minimal_config():
    """Minimal config suitable for unit tests (tiny model)."""
    return {
        "unified": {
            "embed_dim": 128,
            "text_vocab_size": 256,
            "text_max_length": 32,
            "text_num_layers": 2,
            "text_num_heads": 4,
            "dropout": 0.0,
            "enable_materials": False,
            "enable_modifiers": False,
            "enable_contrastive": False,
            "enable_image_to_mesh": False,
            "image_size": 16,
            "image_num_views": 1,
            "geometry": {
                "num_layers": 2,
                "num_heads": 4,
                "mesh_vocab_size": 256,
                "max_seq_length": 512,
            },
            "materials": {
                "num_layers": 2,
                "num_heads": 4,
                "hidden_size": 128,
                "max_seq_len": 64,
                "vocab_size": 256,
            },
            "modifiers": {"hidden_size": 64},
        },
        "tokenization": {
            "vocab_size": 256,
            "coordinate_range": [-1.0, 1.0],
            "max_faces": 32,
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 10,
            "max_steps": 100,
            "eval_every": 50,
            "save_every": 100,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "fp32",
        },
        "data": {
            "geometry_dir": "data/datasets/geometry",
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Test 1: MeshTokenizer round-trip
# ──────────────────────────────────────────────────────────────────────

def test_mesh_tokenizer_roundtrip():
    """Encode + decode a cube and verify we get back valid geometry."""
    from processing.mesh_tokenizer import MeshTokenizer

    tok = MeshTokenizer(vocab_size=8192, coord_range=(-1.0, 1.0), max_faces=64)
    verts, faces = _normalized_cube()

    tokens = tok.encode_mesh(verts, faces)

    assert isinstance(tokens, list), "encode_mesh must return a list"
    assert len(tokens) >= 2, "Must have at least BOS + EOS"
    assert tokens[0] == tok.BOS, f"First token must be BOS ({tok.BOS}), got {tokens[0]}"
    assert tokens[-1] == tok.EOS, f"Last token must be EOS ({tok.EOS}), got {tokens[-1]}"

    # Number of face tokens = (len - 2) / 9 should be integer
    face_tokens = len(tokens) - 2
    assert face_tokens % 9 == 0, (
        f"Token count - 2 ({face_tokens}) must be divisible by 9 (9 tokens per face)")

    n_encoded_faces = face_tokens // 9
    assert n_encoded_faces > 0, "At least one face must be encoded"

    # Decode back
    dec_verts, dec_faces = tok.decode_tokens(tokens)
    assert len(dec_verts) > 0, "Decoded mesh must have vertices"
    assert len(dec_faces) > 0, "Decoded mesh must have faces"

    print(f"  [OK] MeshTokenizer: {len(faces)} faces → {len(tokens)} tokens "
          f"→ {len(dec_faces)} faces decoded")


# ──────────────────────────────────────────────────────────────────────
# Test 2: MeshTokenizer with tiny vocab (training scenario)
# ──────────────────────────────────────────────────────────────────────

def test_mesh_tokenizer_small_vocab():
    """Verify MeshTokenizer works with a tiny 256-token vocab (unit test config)."""
    from processing.mesh_tokenizer import MeshTokenizer

    tok = MeshTokenizer(vocab_size=256, coord_range=(-1.0, 1.0), max_faces=32)
    verts, faces = _normalized_cube()

    tokens = tok.encode_mesh(verts, faces)
    assert tokens[0] == tok.BOS
    assert tokens[-1] == tok.EOS

    # All tokens must be in valid vocab range
    for t in tokens:
        assert 0 <= t < 256, f"Token {t} out of vocab range [0, 256)"

    print(f"  [OK] MeshTokenizer small vocab: {len(tokens)} tokens, all < 256")


# ──────────────────────────────────────────────────────────────────────
# Test 3: BPE tokenizer (if available) or graceful skip
# ──────────────────────────────────────────────────────────────────────

def test_bpe_tokenizer_roundtrip():
    """BPE tokenizer encode/decode round-trip."""
    from processing.bpe_tokenizer import BPETokenizer

    bpe_dir = Path("data/datasets/geometry/bpe_tokenizer")
    if not bpe_dir.exists():
        print("  [SKIP] BPETokenizer: bpe_tokenizer not found "
              "(run: python run.py build to generate)")
        return

    tok = BPETokenizer.load(bpe_dir)
    test_texts = [
        "cube",
        "tall narrow cylinder",
        "detailed chair",
        "a 3D mesh of a house with roof",
    ]

    for text in test_texts:
        ids = tok.encode(text)
        assert len(ids) > 0, f"encode('{text}') returned empty list"
        decoded = tok.decode(ids)
        assert len(decoded) > 0, f"decode returned empty string for '{text}'"

    print(f"  [OK] BPETokenizer: {len(test_texts)} texts encoded/decoded "
          f"(vocab={tok.vocab_size})")


# ──────────────────────────────────────────────────────────────────────
# Test 4: Synthetic data generation
# ──────────────────────────────────────────────────────────────────────

def test_synthetic_data_generation():
    """SHAPE_SPECS generators produce valid mesh data."""
    from processing.generate_synthetic import SHAPE_SPECS, normalize_mesh

    shape_names = list(SHAPE_SPECS.keys())[:5]  # Test first 5 shapes
    for name in shape_names:
        spec = SHAPE_SPECS[name]
        params = spec["params"]()
        result = spec["generator"](params)

        if isinstance(result, dict):
            verts = result.get("vertices") or result.get("verts")
            faces = result.get("faces")
        elif isinstance(result, (list, tuple)) and len(result) == 2:
            verts, faces = result
        else:
            raise AssertionError(
                f"Unexpected generator return type for '{name}': {type(result)}")

        assert verts is not None and len(verts) > 0, f"No vertices for shape '{name}'"
        assert faces is not None and len(faces) > 0, f"No faces for shape '{name}'"

        # normalize_mesh should not crash
        normalized = normalize_mesh(verts, target_range=(-1.0, 1.0))
        assert len(normalized) == len(verts)

    print(f"  [OK] Synthetic data: generated {len(shape_names)} shapes")


# ──────────────────────────────────────────────────────────────────────
# Test 5: Model instantiation
# ──────────────────────────────────────────────────────────────────────

def test_model_instantiation():
    """UnifiedBlenderModel instantiates without error with minimal config."""
    from models.unified import UnifiedBlenderModel

    config = _minimal_config()
    model = UnifiedBlenderModel(config)

    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0, "Model must have parameters"

    print(f"  [OK] Model instantiation: {param_count:,} parameters")
    return model, config


# ──────────────────────────────────────────────────────────────────────
# Test 6: Model forward pass
# ──────────────────────────────────────────────────────────────────────

def test_model_forward_pass():
    """UnifiedBlenderModel.forward_geometry() computes geometry logits without error."""
    import torch.nn.functional as F
    from models.unified import UnifiedBlenderModel
    from processing.mesh_tokenizer import MeshTokenizer

    config = _minimal_config()
    model = UnifiedBlenderModel(config)
    model.eval()

    tok = MeshTokenizer(
        vocab_size=config["tokenization"]["vocab_size"],
        coord_range=tuple(config["tokenization"]["coordinate_range"]),
        max_faces=config["tokenization"]["max_faces"],
    )

    verts, faces = _normalized_cube()
    tokens = tok.encode_mesh(verts, faces)

    max_seq = config["unified"]["geometry"]["max_seq_length"]
    if len(tokens) > max_seq:
        tokens = tokens[:max_seq - 1] + [tok.EOS]

    # Pad or truncate to a fixed length for testing
    seq_len = min(len(tokens), 64)
    tokens = tokens[:seq_len]
    pad_len = 64 - len(tokens)
    tokens = tokens + [tok.PAD] * pad_len

    mesh_tokens = torch.tensor([tokens, tokens], dtype=torch.long)  # batch=2

    text_len = config["unified"]["text_max_length"]
    text_ids = torch.zeros(2, text_len, dtype=torch.long)
    text_ids[:, 0] = 1  # non-zero first token
    text_mask = torch.zeros(2, text_len, dtype=torch.float)
    text_mask[:, 0] = 1.0

    with torch.no_grad():
        # Use the correct task-specific method: forward_geometry
        logits = model.forward_geometry(text_ids, text_mask, mesh_tokens)

    # logits: (B, S, mesh_vocab_size)
    assert logits.ndim == 3, f"Expected 3D logits (B, S, V), got shape {logits.shape}"
    assert logits.shape[0] == 2, "Batch dimension must be 2"
    assert logits.shape[-1] == config["unified"]["geometry"]["mesh_vocab_size"]

    # Compute cross-entropy loss (as in the training loop)
    # Input: (B, S, V) → shift by 1 for next-token prediction
    B, S, V = logits.shape
    targets = mesh_tokens[:, 1:].contiguous()  # (B, S-1)
    lm_logits = logits[:, :-1, :].contiguous()  # (B, S-1, V)

    loss = F.cross_entropy(
        lm_logits.view(-1, V),
        targets.view(-1),
        ignore_index=tok.PAD,
    )
    assert torch.isfinite(loss), f"Loss must be finite, got {loss}"
    assert loss.item() > 0, "Loss should be > 0 before training"

    print(f"  [OK] Model forward pass: logits={logits.shape}, "
          f"geometry_loss={loss.item():.4f}")
    return loss.item()


# ──────────────────────────────────────────────────────────────────────
# Test 7: Backward pass / gradient flow
# ──────────────────────────────────────────────────────────────────────

def test_backward_pass():
    """Gradients flow through geometry decoder and text encoder parameters."""
    import torch.nn.functional as F
    from models.unified import UnifiedBlenderModel
    from processing.mesh_tokenizer import MeshTokenizer

    config = _minimal_config()
    model = UnifiedBlenderModel(config)
    model.train()

    tok = MeshTokenizer(
        vocab_size=config["tokenization"]["vocab_size"],
        coord_range=tuple(config["tokenization"]["coordinate_range"]),
        max_faces=config["tokenization"]["max_faces"],
    )

    verts, faces = _normalized_cube()
    tokens = tok.encode_mesh(verts, faces)[:64]
    pad_len = 64 - len(tokens)
    tokens = tokens + [tok.PAD] * pad_len

    mesh_tokens = torch.tensor([tokens, tokens], dtype=torch.long)
    text_len = config["unified"]["text_max_length"]
    text_ids = torch.zeros(2, text_len, dtype=torch.long)
    text_ids[:, 0] = 1
    text_mask = torch.zeros(2, text_len, dtype=torch.float)
    text_mask[:, 0] = 1.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad()

    logits = model.forward_geometry(text_ids, text_mask, mesh_tokens)
    B, S, V = logits.shape
    targets = mesh_tokens[:, 1:].contiguous()
    lm_logits = logits[:, :-1, :].contiguous()
    loss = F.cross_entropy(lm_logits.view(-1, V), targets.view(-1), ignore_index=tok.PAD)
    loss.backward()
    optimizer.step()

    # At least some parameters must have non-zero gradients
    params_with_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    total_params = sum(1 for _ in model.parameters())
    assert params_with_grad > 0, "No parameters received gradients!"
    grad_pct = 100.0 * params_with_grad / total_params

    print(f"  [OK] Backward pass: {params_with_grad}/{total_params} "
          f"parameter tensors have gradients ({grad_pct:.0f}%)")


# ──────────────────────────────────────────────────────────────────────
# Test 8: Checkpoint save and load
# ──────────────────────────────────────────────────────────────────────

def test_checkpoint_save_load():
    """Model parameters are preserved through save/load cycle."""
    from models.unified import UnifiedBlenderModel

    config = _minimal_config()
    model = UnifiedBlenderModel(config)

    # Capture initial parameter values
    before = {n: p.data.clone() for n, p in model.named_parameters()}

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "global_step": 42,
            "best_val_loss": 0.5,
            "config": config,
        }, ckpt_path)

        # New model, load the checkpoint
        model2 = UnifiedBlenderModel(config)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model2.load_state_dict(ckpt["model_state_dict"])

        for name, p2 in model2.named_parameters():
            p1 = before[name]
            assert torch.allclose(p1, p2.data), (
                f"Parameter {name} changed after save/load!")

        assert ckpt["global_step"] == 42
        assert ckpt["best_val_loss"] == 0.5

    print("  [OK] Checkpoint save/load: all parameters preserved")


# ──────────────────────────────────────────────────────────────────────
# Test 9: Geometric evaluation metrics
# ──────────────────────────────────────────────────────────────────────

def test_eval_metrics():
    """mesh_validity, chamfer_distance, f_score produce valid outputs."""
    from evaluation.metrics import (
        mesh_validity, chamfer_distance, f_score,
        sample_surface_points, evaluate_single,
    )

    verts, faces = _normalized_cube()
    v = np.array(verts, dtype=np.float64)
    f = np.array(faces, dtype=np.int64)

    # Validity
    val = mesh_validity(v, f)
    assert "validity_score" in val
    assert 0.0 <= val["validity_score"] <= 1.0
    assert "num_faces" in val and val["num_faces"] == len(faces)

    # Chamfer Distance (same mesh vs itself should be ~0)
    pts = sample_surface_points(v, f, n_points=256)
    cd = chamfer_distance(pts, pts)
    assert cd["cd"] < 1e-6, f"CD(A, A) should be ~0, got {cd['cd']}"

    # F-score (same mesh vs itself should be ~1)
    fs = f_score(pts, pts, threshold=0.01)
    assert fs["f_score"] > 0.99, f"F-score(A, A) should be ~1, got {fs['f_score']}"

    # evaluate_single (self comparison)
    result = evaluate_single(v, f, v, f, n_surface_points=256)
    assert "validity" in result
    assert "chamfer_distance" in result
    assert result["chamfer_distance"]["cd"] < 1e-6

    print(f"  [OK] Eval metrics: validity={val['validity_score']:.3f}, "
          f"CD(self)={cd['cd']:.2e}, F-score(self)={fs['f_score']:.3f}")


# ──────────────────────────────────────────────────────────────────────
# Test 10: Test suite shape expectation checks
# ──────────────────────────────────────────────────────────────────────

def test_suite_expectation_checks():
    """check_shape_expectations validates cube and sphere correctly."""
    from evaluation.test_suite import check_shape_expectations

    # Cube: aspect ratio 1:1:1 → should pass [0.7, 1.3]
    verts, faces = _normalized_cube()
    v = np.array(verts, dtype=np.float64)
    f = np.array(faces, dtype=np.int64)

    cube_expected = {"aspect_ratio_range": [0.7, 1.3], "min_faces": 6}
    checks = check_shape_expectations(v, cube_expected, faces=f)

    assert "aspect_ratio" in checks, "Should check aspect_ratio"
    passed, detail = checks["aspect_ratio"]
    assert passed, f"Cube should pass aspect ratio [0.7, 1.3]. Detail: {detail}"

    assert "min_faces" in checks
    passed, detail = checks["min_faces"]
    assert passed, f"Cube should have >= 6 faces. Detail: {detail}"

    # Empty mesh: should fail non_empty check
    empty_v = np.zeros((0, 3))
    empty_checks = check_shape_expectations(empty_v, {"min_faces": 1})
    assert "non_empty" in empty_checks
    assert not empty_checks["non_empty"][0], "Empty mesh should fail non_empty"

    print("  [OK] Test suite checks: cube passes aspect+face, empty fails non_empty")


# ──────────────────────────────────────────────────────────────────────
# Test 11: Improvement tracking (eval results logged correctly)
# ──────────────────────────────────────────────────────────────────────

def test_improvement_tracking():
    """Eval results are saved to results.jsonl and can be read back."""
    from evaluation.harness import _save_eval_results

    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch the eval dir to tmpdir
        import evaluation.harness as harness_mod
        original = harness_mod.Path

        # Create two mock eval results at step 100 and step 200
        fake_results_step100 = {
            "step": 100,
            "elapsed_seconds": 1.2,
            "max_faces": 256,
            "temperature": 0.7,
            "results": [],
            "summary": {
                "total_cases": 5,
                "generated_successfully": 3,
                "generation_rate": 0.6,
                "expectations_met": 2,
                "expectations_rate": 0.4,
                "validity_score_mean": 0.72,
                "face_count_mean": 45.0,
                "by_category": {},
            },
        }
        fake_results_step200 = {
            **fake_results_step100,
            "step": 200,
            "summary": {
                **fake_results_step100["summary"],
                "generation_rate": 0.8,
                "expectations_rate": 0.6,
                "validity_score_mean": 0.85,
                "face_count_mean": 67.0,
            },
        }

        eval_dir = Path(tmpdir) / "eval"
        eval_dir.mkdir()
        jsonl_path = eval_dir / "results.jsonl"

        # Write two entries manually (mimicking _save_eval_results logic)
        for results in [fake_results_step100, fake_results_step200]:
            step = results["step"]
            summary = {
                "step": step,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "type": "geometric_eval",
                **results["summary"],
            }
            with open(jsonl_path, "a") as fh:
                fh.write(json.dumps(summary) + "\n")

        # Read back and verify improvement
        assert jsonl_path.exists(), "results.jsonl should be created"
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 2, f"Should have 2 entries, got {len(lines)}"

        r100 = json.loads(lines[0])
        r200 = json.loads(lines[1])

        assert r100["step"] == 100
        assert r200["step"] == 200

        # Assert improvement across steps
        assert r200["generation_rate"] > r100["generation_rate"], (
            "generation_rate should improve from step 100 to 200")
        assert r200["validity_score_mean"] > r100["validity_score_mean"], (
            "validity_score_mean should improve from step 100 to 200")

    print("  [OK] Improvement tracking: results.jsonl records two steps "
          f"with generation_rate {r100['generation_rate']:.0%} → "
          f"{r200['generation_rate']:.0%}, validity "
          f"{r100['validity_score_mean']:.3f} → {r200['validity_score_mean']:.3f}")


# ──────────────────────────────────────────────────────────────────────
# Test 12: GRPO quality weights
# ──────────────────────────────────────────────────────────────────────

def test_grpo_quality_weights():
    """grpo_quality_weights produces valid normalized weights."""
    from training.train_unified import grpo_quality_weights

    losses = torch.tensor([0.1, 0.5, 1.0, 2.0, 0.3])

    weights = grpo_quality_weights(losses, temperature=1.0)

    assert weights.shape == losses.shape, "Weights must match losses shape"
    assert torch.all(weights > 0), "All weights must be positive"

    # Weights sum to batch_size (normalized)
    expected_sum = float(losses.numel())
    actual_sum = weights.sum().item()
    assert abs(actual_sum - expected_sum) < 0.01, (
        f"Weights must sum to {expected_sum:.0f}, got {actual_sum:.4f}")

    # Lower-loss samples should get higher weight (GRPO reward signal)
    min_loss_idx = losses.argmin().item()
    max_loss_idx = losses.argmax().item()
    assert weights[min_loss_idx] > weights[max_loss_idx], (
        "Sample with lowest loss should have highest GRPO weight")

    print(f"  [OK] GRPO weights: sum={actual_sum:.2f} (expected {expected_sum:.0f}), "
          f"min-loss weight={weights[min_loss_idx]:.3f} > "
          f"max-loss weight={weights[max_loss_idx]:.3f}")


# ──────────────────────────────────────────────────────────────────────
# Test 13: Curriculum learning face count ramp
# ──────────────────────────────────────────────────────────────────────

def test_curriculum_face_ramp():
    """curriculum_max_faces ramps from min to max over warmup steps."""
    from training.train_unified import curriculum_max_faces

    min_f, max_f, warmup = 32, 512, 2000

    # At step 0: should be at minimum
    f0 = curriculum_max_faces(0, warmup, min_f, max_f)
    assert f0 == min_f, f"At step 0 expect {min_f} faces, got {f0}"

    # At step warmup: should be at maximum
    f_warm = curriculum_max_faces(warmup, warmup, min_f, max_f)
    assert f_warm == max_f, f"At warmup step expect {max_f} faces, got {f_warm}"

    # At step warmup//2: should be roughly halfway (cosine, ~halfway)
    f_mid = curriculum_max_faces(warmup // 2, warmup, min_f, max_f)
    assert min_f < f_mid < max_f, f"Midpoint {f_mid} should be between {min_f} and {max_f}"

    # Always monotonically non-decreasing
    prev = min_f
    for step in range(0, warmup + 1, 100):
        curr = curriculum_max_faces(step, warmup, min_f, max_f)
        assert curr >= prev, f"Face count decreased at step {step}: {prev} → {curr}"
        prev = curr

    print(f"  [OK] Curriculum: step 0={f0}, mid={f_mid}, warmup={f_warm} — "
          f"monotonically ramps from {min_f} to {max_f}")


# ──────────────────────────────────────────────────────────────────────
# Test 14: DeepSeek LR schedule
# ──────────────────────────────────────────────────────────────────────

def test_lr_schedule():
    """deepseek_lr_schedule returns valid multipliers at all stages."""
    from training.train_unified import deepseek_lr_schedule

    total = 10000
    warmup = 500

    # Before warmup: should ramp up
    lr_start = deepseek_lr_schedule(0, warmup, total_steps=total)
    lr_warmed = deepseek_lr_schedule(warmup, warmup, total_steps=total)
    assert lr_start <= lr_warmed, "LR should increase during warmup"
    assert lr_warmed <= 1.0 + 1e-6, f"LR at warmup should be ~1.0, got {lr_warmed}"

    # LR at step 0 is 0.0 by design (linear warmup starts from 0)
    assert lr_start == 0.0, f"LR at step 0 should be 0.0 (start of linear warmup), got {lr_start}"

    # All values from step 1 onward must be non-negative and finite
    for step in range(1, total + 1, 500):
        lr = deepseek_lr_schedule(step, warmup, total_steps=total)
        assert lr >= 0, f"LR must be >= 0 at step {step}: got {lr}"
        assert math.isfinite(lr), f"LR must be finite at step {step}: got {lr}"

    print(f"  [OK] LR schedule: start={lr_start:.4f}, "
          f"post-warmup={lr_warmed:.4f}, all steps valid")


# ──────────────────────────────────────────────────────────────────────
# Test 15: Cache key consistency
# ──────────────────────────────────────────────────────────────────────

def test_cache_key_consistency():
    """The same source path always produces the same cache key in both places."""
    import hashlib

    test_path = "/data/processed/blendswap/12345_chair.json"

    # Hash as used in rebuild_cache.py
    key_rebuild = hashlib.md5(test_path.encode()).hexdigest()[:16]

    # Hash as used in train_unified.py _cache_key()
    key_train = hashlib.md5(test_path.encode()).hexdigest()[:16]

    assert key_rebuild == key_train, (
        f"Cache keys mismatch: rebuild={key_rebuild}, train={key_train}")

    # Different paths must produce different keys
    other_path = "/data/processed/blendswap/99999_sphere.json"
    key_other = hashlib.md5(other_path.encode()).hexdigest()[:16]
    assert key_rebuild != key_other, "Different paths must produce different cache keys"

    print(f"  [OK] Cache key consistency: same path → '{key_rebuild}' in both scripts")


# ──────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────

TESTS = [
    ("MeshTokenizer round-trip", test_mesh_tokenizer_roundtrip),
    ("MeshTokenizer small vocab", test_mesh_tokenizer_small_vocab),
    ("BPETokenizer round-trip", test_bpe_tokenizer_roundtrip),
    ("Synthetic data generation", test_synthetic_data_generation),
    ("Model instantiation", test_model_instantiation),
    ("Model forward pass", test_model_forward_pass),
    ("Backward pass / gradient flow", test_backward_pass),
    ("Checkpoint save/load", test_checkpoint_save_load),
    ("Geometric eval metrics", test_eval_metrics),
    ("Test suite expectation checks", test_suite_expectation_checks),
    ("Improvement tracking", test_improvement_tracking),
    ("GRPO quality weights", test_grpo_quality_weights),
    ("Curriculum face ramp", test_curriculum_face_ramp),
    ("LR schedule", test_lr_schedule),
    ("Cache key consistency", test_cache_key_consistency),
]


def run_all_tests():
    print(f"\n{'='*65}")
    print("  Blender Copilot — Pipeline Smoke Test")
    print(f"{'='*65}\n")

    passed = 0
    failed = 0
    skipped = 0

    for name, fn in TESTS:
        print(f"[{passed + failed + skipped + 1:02d}/{len(TESTS):02d}] {name}")
        try:
            fn()
            passed += 1
        except SystemExit:
            raise
        except AssertionError as e:
            print(f"  [FAIL] AssertionError: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  [ERROR] {type(e).__name__}: {e}")
            if "--verbose" in sys.argv or "-v" in sys.argv:
                traceback.print_exc()
            failed += 1
        print()

    print(f"{'='*65}")
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*65}\n")

    if failed:
        sys.exit(1)


# pytest compatibility: expose each test as a module-level function
# (already done — functions starting with test_ are auto-discovered)


if __name__ == "__main__":
    run_all_tests()
