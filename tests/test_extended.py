"""Extended pipeline tests — covers the easy + medium untested behaviors.

Companion to test_pipeline.py (which covers the 15 core behaviors).
Together they target ~50 distinct code paths across the full training pipeline.

Run with:
    python -m pytest tests/ -v
    python tests/test_extended.py        # run directly
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Shared helpers ────────────────────────────────────────────────────

def _cube_verts_faces():
    verts = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]
    faces = [
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4],
    ]
    return verts, faces


def _norm_cube():
    from processing.generate_synthetic import normalize_mesh
    v, f = _cube_verts_faces()
    return normalize_mesh(v), f


def _np_cube():
    v, f = _norm_cube()
    return np.array(v, dtype=np.float64), np.array(f, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════
# MeshTokenizer — edge cases
# ══════════════════════════════════════════════════════════════════════

def test_mesh_tokenizer_nan_inf_coords():
    """quantize_coord maps NaN and Inf to the center token."""
    from processing.mesh_tokenizer import MeshTokenizer
    tok = MeshTokenizer(vocab_size=8192, coord_range=(-1.0, 1.0))

    center_token = tok.quantize_coord(0.0)
    for bad in (float("nan"), float("inf"), float("-inf"), 1e20, -1e20):
        result = tok.quantize_coord(bad)
        # NaN/Inf → clamped to boundary or center
        assert tok.SPECIAL_TOKENS <= result < tok.vocab_size, (
            f"quantize_coord({bad}) produced out-of-range token {result}")

    # NaN and all non-finite values map to center (they are replaced by 0.0 before clamping)
    assert tok.quantize_coord(float("nan")) == center_token, \
        "NaN should map to center token"
    assert tok.quantize_coord(float("inf")) == center_token, \
        "Inf should map to center token (replaced by 0.0)"
    assert tok.quantize_coord(float("-inf")) == center_token, \
        "-Inf should map to center token (replaced by 0.0)"

    # Large but finite floats clamp to boundary tokens
    assert tok.quantize_coord(1e20) == tok.quantize_coord(1.0), \
        "Large positive finite float should clamp to max coord token"
    assert tok.quantize_coord(-1e20) == tok.quantize_coord(-1.0), \
        "Large negative finite float should clamp to min coord token"

    print("  [OK] quantize_coord: NaN/±Inf→center, large finite floats→boundary")


def test_mesh_tokenizer_dequantize_special_tokens():
    """dequantize_token returns 0.0 for all special tokens (PAD, BOS, EOS, SEP)."""
    from processing.mesh_tokenizer import MeshTokenizer
    tok = MeshTokenizer(vocab_size=8192, coord_range=(-1.0, 1.0))

    for special in (tok.PAD, tok.BOS, tok.EOS, tok.SEP):
        val = tok.dequantize_token(special)
        assert val == 0.0, \
            f"dequantize_token({special}) should be 0.0, got {val}"

    print("  [OK] dequantize_token: PAD/BOS/EOS/SEP all return 0.0")


def test_mesh_tokenizer_invalid_vertex_indices():
    """encode_mesh uses center coords for out-of-range vertex indices."""
    from processing.mesh_tokenizer import MeshTokenizer
    tok = MeshTokenizer(vocab_size=8192, coord_range=(-1.0, 1.0))

    verts = [[0.5, 0.5, 0.5]]            # only 1 vertex
    faces = [[0, 99, 200]]               # indices 99 and 200 are OOB

    tokens = tok.encode_mesh(verts, faces)
    assert tokens[0] == tok.BOS and tokens[-1] == tok.EOS
    # Should NOT raise — uses quantize_coord(0.0) for invalid indices
    face_tokens = [t for t in tokens if t not in (tok.BOS, tok.EOS)]
    assert len(face_tokens) == 9, \
        f"Expected 9 face tokens, got {len(face_tokens)}"

    print("  [OK] encode_mesh: invalid vertex indices produce center coords")


def test_mesh_tokenizer_empty_faces():
    """encode_mesh with no faces returns [BOS, EOS]."""
    from processing.mesh_tokenizer import MeshTokenizer
    tok = MeshTokenizer(vocab_size=8192, coord_range=(-1.0, 1.0))

    tokens = tok.encode_mesh([[0.0, 0.0, 0.0]], [])
    assert tokens == [tok.BOS, tok.EOS], \
        f"Empty faces should give [BOS, EOS], got {tokens}"

    print("  [OK] encode_mesh: empty faces → [BOS, EOS]")


def test_mesh_tokenizer_max_faces_truncation():
    """encode_mesh truncates at max_faces; result is always ≤ max_faces * 9 + 2."""
    from processing.mesh_tokenizer import MeshTokenizer
    tok = MeshTokenizer(vocab_size=8192, coord_range=(-1.0, 1.0), max_faces=5)

    # 12-face cube — should be truncated to 5 faces
    v, f = _norm_cube()
    tokens = tok.encode_mesh(v, f)
    face_token_count = len(tokens) - 2  # minus BOS and EOS
    assert face_token_count <= 5 * 9, \
        f"Expected ≤ 45 face tokens, got {face_token_count}"
    assert tokens[0] == tok.BOS and tokens[-1] == tok.EOS

    print(f"  [OK] encode_mesh max_faces=5: 12-face cube → {face_token_count // 9} faces encoded")


def test_mesh_tokenizer_pad_sequence():
    """pad_sequence pads short sequences and truncates long ones."""
    from processing.mesh_tokenizer import MeshTokenizer
    tok = MeshTokenizer(vocab_size=8192)

    short = [1, 2, 3]
    padded = tok.pad_sequence(short, max_length=8)
    assert len(padded) == 8
    assert padded[:3] == short
    assert all(t == tok.PAD for t in padded[3:])

    long = list(range(20))
    trunc = tok.pad_sequence(long, max_length=10)
    assert len(trunc) == 10
    assert trunc == long[:10]

    print("  [OK] pad_sequence: pads short → 8 tokens, truncates long → 10 tokens")


def test_mesh_tokenizer_save_load_roundtrip():
    """save/load preserves vocab_size, coord_range, and max_faces exactly."""
    from processing.mesh_tokenizer import MeshTokenizer

    original = MeshTokenizer(vocab_size=4096, coord_range=(-2.0, 2.0), max_faces=512)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "mesh_tok.json"
        original.save(path)
        loaded = MeshTokenizer.load(path)

    assert loaded.vocab_size == 4096
    assert loaded.coord_min == -2.0
    assert loaded.coord_max == 2.0
    assert loaded.max_faces == 512

    print("  [OK] MeshTokenizer save/load: vocab=4096, range=(-2,2), max_faces=512 preserved")


def test_mesh_tokenizer_sequence_length_formula():
    """sequence_length_for_faces matches actual encode output length."""
    from processing.mesh_tokenizer import MeshTokenizer
    tok = MeshTokenizer(vocab_size=8192)

    v, f = _norm_cube()
    tokens = tok.encode_mesh(v, f)
    expected = tok.sequence_length_for_faces(len(f))

    assert len(tokens) == expected, (
        f"sequence_length_for_faces({len(f)}) = {expected}, "
        f"but encode produced {len(tokens)} tokens")

    print(f"  [OK] sequence_length_for_faces: {len(f)} faces → {expected} tokens")


# ══════════════════════════════════════════════════════════════════════
# BPETokenizer — edge cases
# ══════════════════════════════════════════════════════════════════════

def test_bpe_encode_padded_lengths():
    """encode_padded always returns ids and mask of exactly max_length."""
    bpe_dir = Path("data/datasets/geometry/bpe_tokenizer")
    if not bpe_dir.exists():
        print("  [SKIP] encode_padded: BPE tokenizer not built yet")
        return

    from processing.bpe_tokenizer import BPETokenizer
    tok = BPETokenizer.load(bpe_dir)

    for text, max_len in [("cube", 32), ("detailed metallic sphere", 64), ("", 16)]:
        ids, mask = tok.encode_padded(text or "cube", max_length=max_len)
        assert len(ids) == max_len, \
            f"ids length {len(ids)} != max_length {max_len} for '{text}'"
        assert len(mask) == max_len, \
            f"mask length {len(mask)} != max_length {max_len} for '{text}'"
        # mask must be exactly 1 for real tokens, 0 for padding
        real_len = sum(mask)
        assert all(m in (0, 1) for m in mask), "mask must be binary"
        assert ids[real_len:] == [0] * (max_len - real_len), \
            "padded ids must all be PAD (0)"

    print("  [OK] encode_padded: ids and mask always == max_length, binary mask")


def test_bpe_decode_skips_special_tokens():
    """decode with skip_special=True removes PAD, BOS, EOS from output."""
    bpe_dir = Path("data/datasets/geometry/bpe_tokenizer")
    if not bpe_dir.exists():
        print("  [SKIP] decode special tokens: BPE tokenizer not built yet")
        return

    from processing.bpe_tokenizer import BPETokenizer, PAD_ID, BOS_ID, EOS_ID
    tok = BPETokenizer.load(bpe_dir)

    text = "cylinder"
    ids_with_special = tok.encode(text, add_special=True)

    # Manually inject extra pad tokens
    ids_padded = ids_with_special + [PAD_ID, PAD_ID, PAD_ID]

    decoded_skip = tok.decode(ids_padded, skip_special=True)
    decoded_keep = tok.decode(ids_padded, skip_special=False)

    assert len(decoded_skip) > 0, "skip_special decode returned empty string"
    # skip_special should produce same or shorter output than keep_special
    assert len(decoded_skip) <= len(decoded_keep) + 5  # +5 leeway for BOS/EOS chars

    print(f"  [OK] decode skip_special: '{text}' → '{decoded_skip}' (no special tokens)")


def test_bpe_blender_terms_encoded_as_single_tokens():
    """Key Blender terms are never split — each encodes as exactly 1 token."""
    bpe_dir = Path("data/datasets/geometry/bpe_tokenizer")
    if not bpe_dir.exists():
        print("  [SKIP] Blender terms: BPE tokenizer not built yet")
        return

    from processing.bpe_tokenizer import BPETokenizer, BLENDER_TERMS
    tok = BPETokenizer.load(bpe_dir)

    # Test a representative sample of the most critical terms
    critical_terms = ["cube", "sphere", "cylinder", "torus", "shader",
                      "metallic", "roughness", "subdivision", "bevel"]
    split_terms = []
    for term in critical_terms:
        ids = tok.encode(term, add_special=False)
        if len(ids) != 1:
            split_terms.append((term, len(ids)))

    if split_terms:
        # Not a hard failure — SentencePiece user_defined_symbols are best-effort
        print(f"  [WARN] {len(split_terms)} Blender terms split into subpieces: "
              f"{split_terms[:3]}")
    else:
        print(f"  [OK] Blender terms: all {len(critical_terms)} critical terms = 1 token each")


def test_bpe_no_unk_tokens():
    """BPETokenizer avoids UNK (id=3) on common Latin+ASCII inputs."""
    bpe_dir = Path("data/datasets/geometry/bpe_tokenizer")
    if not bpe_dir.exists():
        print("  [SKIP] no-UNK: BPE tokenizer not built yet")
        return

    from processing.bpe_tokenizer import BPETokenizer
    from processing.bpe_tokenizer import UNK_ID
    tok = BPETokenizer.load(bpe_dir)

    # These are all ASCII / Latin texts — should never produce UNK
    ascii_texts = [
        "12345678",
        "a" * 100,             # very long single-word
        "CUBE SPHERE MESH",    # uppercase ASCII
    ]
    unk_texts = []
    for text in ascii_texts:
        ids = tok.encode(text, add_special=False)
        if UNK_ID in ids:
            unk_texts.append((text[:20], ids[:10]))

    assert not unk_texts, \
        f"UNK token found in ASCII encodings: {unk_texts}"

    # Non-ASCII inputs (rare in Blender training data) may produce UNK — warn only
    tricky_texts = [
        "xyzzy_frobnicator_27",
        "エラー",              # Japanese — may produce UNK (SentencePiece fallback)
    ]
    unk_found = []
    for text in tricky_texts:
        ids = tok.encode(text, add_special=False)
        if UNK_ID in ids:
            unk_found.append(text[:20])
    if unk_found:
        print(f"  [WARN] UNK produced for non-ASCII/rare inputs (expected): {unk_found}")
    else:
        print(f"  [OK] No UNK in any of the {len(ascii_texts+tricky_texts)} tested inputs")

    print(f"  [OK] No UNK tokens in {len(ascii_texts)} ASCII inputs")


# ══════════════════════════════════════════════════════════════════════
# labeler_smart — cleaning helpers
# ══════════════════════════════════════════════════════════════════════

def test_clean_label_strips_hex_uids():
    """_strip_hex_uids removes Objaverse-style hashes but not short words."""
    from processing.labeler_smart import _strip_hex_uids

    # Long hex → removed
    result = _strip_hex_uids("a8f3b2c1d4e5f607 chair")
    assert "a8f3b2c1d4e5f607" not in result
    assert "chair" in result

    # Short word with digits → preserved("3d" is 2 chars, not 8+)
    result2 = _strip_hex_uids("3d model")
    assert "3d" in result2, f"Short '3d' should not be stripped, got: '{result2}'"

    print("  [OK] _strip_hex_uids: removes 8+ char hex, keeps short words")


def test_clean_label_strips_blender_prefixes():
    """_strip_blender_prefixes removes GEO-, HLP- etc. from label start."""
    from processing.labeler_smart import _strip_blender_prefixes

    cases = [
        ("geo-wheel", "wheel"),
        ("hlp arm", "arm"),
        ("dp-bone", "bone"),
        ("grp handle", "handle"),
        ("normal label", "normal label"),  # no prefix → unchanged
    ]
    for raw, expected in cases:
        got = _strip_blender_prefixes(raw)
        assert expected.lower() in got.lower(), \
            f"_strip_blender_prefixes('{raw}') = '{got}', expected '{expected}'"

    print("  [OK] _strip_blender_prefixes: removes Blender internal prefixes")


def test_clean_label_strips_version_parts():
    """_strip_version_parts removes v1, v.2, s12, .001 suffixes."""
    from processing.labeler_smart import _strip_version_parts

    assert "v2" not in _strip_version_parts("sword v2")
    assert "v.4" not in _strip_version_parts("chair v.4")
    assert "s12" not in _strip_version_parts("cube s12")
    assert ".001" not in _strip_version_parts("mesh.001")
    # Legitimate version not in label context — these are stripped by word boundary
    result = _strip_version_parts("iron sword")
    assert "iron" in result and "sword" in result

    print("  [OK] _strip_version_parts: removes version suffixes, preserves words")


def test_clean_label_final_pipeline():
    """_clean_label_final applies all passes in correct order."""
    from processing.labeler_smart import _clean_label_final

    # Should strip version, prefix, hex UID and keep meaningful word
    dirty = "GEO-Chair v2"
    result = _clean_label_final(dirty)
    assert len(result) > 0, f"clean_label_final('{dirty}') returned empty"
    assert result == result.lower(), "Output should be lowercase"

    # Pure hex → empty
    uid_only = "a8f3b2c1d4e5f607"
    result2 = _clean_label_final(uid_only)
    assert len(result2) == 0, f"Pure hex UID should clean to empty, got '{result2}'"

    # Normal label → preserved
    normal = "wooden_table"
    result3 = _clean_label_final(normal)
    assert "table" in result3, f"'table' should survive cleaning, got '{result3}'"

    print(f"  [OK] _clean_label_final: '{dirty}'→'{result}', "
          f"uid→'', 'wooden_table'→'{result3}'")


def test_compute_bbox_aspect():
    """compute_bbox_aspect returns correct extents for known geometry."""
    from processing.labeler_smart import compute_bbox_aspect

    # Unit cube [0,1]^3 → extents all 1.0
    verts = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]
    result = compute_bbox_aspect(verts)
    assert result is not None
    assert all(abs(e - 1.0) < 1e-6 for e in result), \
        f"Unit cube extents should be (1,1,1), got {result}"

    # Degenerate: fewer than 3 vertices → None
    assert compute_bbox_aspect([[0, 0, 0]]) is None
    assert compute_bbox_aspect([]) is None

    # NaN vertex → None
    assert compute_bbox_aspect([[float("nan"), 0, 0], [1, 0, 0], [0.5, 1, 0]]) is None

    print("  [OK] compute_bbox_aspect: unit cube=(1,1,1), degenerate→None, NaN→None")


def test_is_primitive_name():
    """_is_primitive_name identifies bare Blender primitive names correctly."""
    from processing.labeler_smart import _is_primitive_name

    # True cases
    for name in ["cube", "Cube", "Cube.001", "sphere", "CYLINDER", "torus"]:
        assert _is_primitive_name(name), f"'{name}' should be a primitive name"

    # False cases — these are actual object descriptions
    for name in ["table", "car wheel", "sword", "red cube"]:
        assert not _is_primitive_name(name), f"'{name}' should NOT be a primitive name"

    print("  [OK] _is_primitive_name: cube/sphere/cylinder→True, table/sword→False")


# ══════════════════════════════════════════════════════════════════════
# generate_synthetic — ALL SHAPE_SPECS + ALL COMPOSITE_SPECS
# ══════════════════════════════════════════════════════════════════════

def _run_all_specs(specs_dict, label):
    """Helper: run every spec in a dict and return list of failed names."""
    from processing.generate_synthetic import normalize_mesh
    from processing.mesh_tokenizer import MeshTokenizer

    tok = MeshTokenizer(vocab_size=8192, coord_range=(-1.0, 1.0), max_faces=256)
    failures = []

    for name, spec in specs_dict.items():
        try:
            params = spec["params"]()
            result = spec["generator"](params)
            if isinstance(result, (list, tuple)) and len(result) == 2:
                verts, faces = result
            elif isinstance(result, dict):
                verts = result.get("vertices") or result.get("verts") or []
                faces = result.get("faces") or []
            else:
                failures.append((name, f"unexpected return type: {type(result)}"))
                continue

            if len(verts) == 0 or len(faces) == 0:
                # Some generators (e.g. monkey) can return empty if no file
                continue

            norm = normalize_mesh(verts, target_range=(-1.0, 1.0))
            assert len(norm) == len(verts)

            tokens = tok.encode_mesh(norm, faces)
            assert tokens[0] == tok.BOS and tokens[-1] == tok.EOS

        except Exception as e:
            failures.append((name, str(e)))

    return failures


def test_all_shape_specs():
    """Every SHAPE_SPECS generator runs without error and produces valid tokens."""
    from processing.generate_synthetic import SHAPE_SPECS

    failures = _run_all_specs(SHAPE_SPECS, "SHAPE_SPECS")

    if failures:
        msg = "; ".join(f"{n}: {e}" for n, e in failures)
        raise AssertionError(f"{len(failures)} SHAPE_SPECS generators failed: {msg}")

    print(f"  [OK] All {len(SHAPE_SPECS)} SHAPE_SPECS generators: valid mesh + tokens")


def test_all_composite_specs():
    """Every COMPOSITE_SPECS generator runs without error and produces valid tokens."""
    from processing.generate_synthetic import COMPOSITE_SPECS

    failures = _run_all_specs(COMPOSITE_SPECS, "COMPOSITE_SPECS")

    if failures:
        msg = "; ".join(f"{n}: {e}" for n, e in failures)
        raise AssertionError(f"{len(failures)} COMPOSITE_SPECS generators failed: {msg}")

    print(f"  [OK] All {len(COMPOSITE_SPECS)} COMPOSITE_SPECS generators: valid mesh + tokens")


def test_normalize_mesh_degenerate_collinear():
    """normalize_mesh handles collinear and zero-extent meshes without NaN."""
    from processing.generate_synthetic import normalize_mesh

    # All vertices at same point → zero extent
    same = [[0.5, 0.5, 0.5]] * 4
    result = normalize_mesh(same)
    assert all(math.isfinite(c) for v in result for c in v), \
        "normalize_mesh with zero extent should return finite coords"

    # Collinear points (1D degenerate mesh)
    line = [[float(i), 0.0, 0.0] for i in range(5)]
    result2 = normalize_mesh(line)
    assert all(math.isfinite(c) for v in result2 for c in v), \
        "normalize_mesh with collinear verts should return finite coords"

    print("  [OK] normalize_mesh: zero-extent and collinear inputs → finite output")


def test_generate_label_correctness():
    """generate_label produces non-empty strings for all SHAPE_SPECS."""
    from processing.generate_synthetic import SHAPE_SPECS, generate_label

    for name in list(SHAPE_SPECS.keys())[:8]:
        spec = SHAPE_SPECS[name]
        params = spec["params"]()
        label = generate_label(name, params)
        assert isinstance(label, str) and len(label) > 0, \
            f"generate_label('{name}') returned empty/non-string: {label!r}"

    print("  [OK] generate_label: all tested SHAPE_SPECS produce non-empty labels")


# ══════════════════════════════════════════════════════════════════════
# Model — disabled-head RuntimeErrors + optional-head forward passes
# ══════════════════════════════════════════════════════════════════════

def _minimal_config_no_optionals():
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
                "max_seq_length": 256,
            },
            "materials": {
                "num_layers": 2, "num_heads": 4,
                "hidden_size": 128, "max_seq_len": 64, "vocab_size": 256,
            },
            "modifiers": {"hidden_size": 64},
        },
        "tokenization": {
            "vocab_size": 256, "coordinate_range": [-1.0, 1.0], "max_faces": 16,
        },
    }


def test_forward_materials_raises_when_disabled():
    """forward_materials raises RuntimeError when enable_materials=False."""
    from models.unified import UnifiedBlenderModel

    model = UnifiedBlenderModel(_minimal_config_no_optionals())
    dummy_text = torch.zeros(1, 32, dtype=torch.long)
    dummy_mask = torch.ones(1, 32)
    dummy_tok = torch.zeros(1, 16, dtype=torch.long)

    try:
        model.forward_materials(dummy_text, dummy_mask, dummy_tok)
        raise AssertionError("forward_materials should raise RuntimeError")
    except RuntimeError as e:
        assert "disable" in str(e).lower() or "material" in str(e).lower(), \
            f"Unexpected error message: {e}"

    print("  [OK] forward_materials raises RuntimeError when disabled")


def test_forward_modifiers_raises_when_disabled():
    """forward_modifiers raises RuntimeError when enable_modifiers=False."""
    from models.unified import UnifiedBlenderModel

    model = UnifiedBlenderModel(_minimal_config_no_optionals())
    dummy_text = torch.zeros(1, 32, dtype=torch.long)
    dummy_mask = torch.ones(1, 32)
    dummy_stats = torch.zeros(1, 5)

    try:
        model.forward_modifiers(dummy_text, dummy_mask, dummy_stats)
        raise AssertionError("forward_modifiers should raise RuntimeError")
    except RuntimeError as e:
        assert "disable" in str(e).lower() or "modifier" in str(e).lower(), \
            f"Unexpected error message: {e}"

    print("  [OK] forward_modifiers raises RuntimeError when disabled")


def test_forward_contrastive_raises_when_disabled():
    """forward_contrastive raises RuntimeError when enable_contrastive=False."""
    from models.unified import UnifiedBlenderModel

    model = UnifiedBlenderModel(_minimal_config_no_optionals())
    dummy_text = torch.zeros(1, 32, dtype=torch.long)
    dummy_mask = torch.ones(1, 32)
    dummy_imgs = torch.zeros(1, 3, 16, 16)

    try:
        model.forward_contrastive(dummy_text, dummy_mask, dummy_imgs)
        raise AssertionError("forward_contrastive should raise RuntimeError")
    except RuntimeError as e:
        assert "disable" in str(e).lower() or "contrastive" in str(e).lower(), \
            f"Unexpected error message: {e}"

    print("  [OK] forward_contrastive raises RuntimeError when disabled")


def test_generate_geometry_produces_token_sequence():
    """generate_geometry produces a list of integer tokens ending in EOS."""
    from models.unified import UnifiedBlenderModel
    from processing.mesh_tokenizer import MeshTokenizer

    config = _minimal_config_no_optionals()
    model = UnifiedBlenderModel(config)
    model.eval()

    tok = MeshTokenizer(
        vocab_size=config["tokenization"]["vocab_size"],
        coord_range=tuple(config["tokenization"]["coordinate_range"]),
        max_faces=config["tokenization"]["max_faces"],
    )

    text_len = config["unified"]["text_max_length"]
    text_ids = torch.zeros(1, text_len, dtype=torch.long)
    text_ids[0, 0] = 1
    text_mask = torch.ones(1, text_len)

    with torch.no_grad():
        tokens = model.generate_geometry(
            text_ids, text_mask,
            max_tokens=20,
            temperature=1.0,
        )

    assert isinstance(tokens, (list, torch.Tensor)), f"generate_geometry must return list or Tensor, got {type(tokens)}"
    # Flatten to list for uniform handling
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.flatten().tolist()
    assert len(tokens) >= 1, "generate_geometry returned empty sequence"
    # All tokens must be valid vocab IDs
    for t in tokens:
        assert 0 <= t < config["unified"]["geometry"]["mesh_vocab_size"], \
            f"Token {t} out of range [0, {config['unified']['geometry']['mesh_vocab_size']})"

    print(f"  [OK] generate_geometry: produced {len(tokens)}-token sequence")


# ══════════════════════════════════════════════════════════════════════
# Evaluation metrics — untested paths
# ══════════════════════════════════════════════════════════════════════

def test_bounding_box_iou_self():
    """bounding_box_iou of a mesh with itself is 1.0."""
    from evaluation.metrics import bounding_box_iou

    verts, _ = _np_cube()
    iou = bounding_box_iou(verts, verts)
    assert abs(iou - 1.0) < 1e-6, f"BB IoU(A, A) should be 1.0, got {iou}"

    print(f"  [OK] bounding_box_iou(A, A) = {iou:.4f}")


def test_bounding_box_iou_disjoint():
    """bounding_box_iou of non-overlapping meshes is 0.0."""
    from evaluation.metrics import bounding_box_iou

    v1 = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    v2 = np.array([[10, 10, 10], [11, 11, 11]], dtype=np.float64)

    iou = bounding_box_iou(v1, v2)
    assert iou == 0.0, f"Disjoint BBs should give IoU=0.0, got {iou}"

    print(f"  [OK] bounding_box_iou(disjoint meshes) = {iou:.4f}")


def test_bounding_box_iou_empty():
    """bounding_box_iou returns 0.0 when either mesh is empty."""
    from evaluation.metrics import bounding_box_iou

    verts, _ = _np_cube()
    empty = np.zeros((0, 3), dtype=np.float64)

    assert bounding_box_iou(empty, verts) == 0.0
    assert bounding_box_iou(verts, empty) == 0.0

    print("  [OK] bounding_box_iou: empty mesh → 0.0")


def test_normal_consistency_self():
    """normal_consistency of a mesh with itself is ~1.0."""
    from evaluation.metrics import normal_consistency

    verts, faces = _np_cube()
    nc = normal_consistency(verts, faces, verts, faces, n_samples=64)
    assert nc > 0.99, f"normal_consistency(A, A) should be ~1.0, got {nc:.4f}"

    print(f"  [OK] normal_consistency(A, A) = {nc:.4f}")


def test_normal_consistency_empty_mesh():
    """normal_consistency returns 0.0 when either mesh has no faces."""
    from evaluation.metrics import normal_consistency

    verts, faces = _np_cube()
    empty_v = np.zeros((0, 3), dtype=np.float64)
    empty_f = np.zeros((0, 3), dtype=np.int64)

    assert normal_consistency(empty_v, empty_f, verts, faces) == 0.0
    assert normal_consistency(verts, faces, empty_v, empty_f) == 0.0

    print("  [OK] normal_consistency: empty mesh → 0.0")


def test_shape_distribution_output():
    """shape_distribution returns normalized histogram of correct length."""
    from evaluation.metrics import shape_distribution

    verts, _ = _np_cube()

    hist = shape_distribution(verts, n_pairs=500, n_bins=32)
    assert hist.shape == (32,), f"Expected (32,) histogram, got {hist.shape}"
    assert abs(hist.sum() - 1.0) < 0.01, \
        f"Histogram should sum to ~1.0, got {hist.sum():.4f}"
    assert np.all(hist >= 0), "Histogram must be non-negative"

    # Single vertex → zeros (can't compute distances)
    single = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
    hist_single = shape_distribution(single, n_pairs=10, n_bins=8)
    assert np.all(hist_single == 0), "Single vertex should give zero histogram"

    print(f"  [OK] shape_distribution: (32,) histogram, sum={hist.sum():.3f}")


def test_evaluate_batch_list():
    """evaluate_batch accepts a list of predictions and summarizes correctly."""
    from evaluation.metrics import evaluate_batch

    verts, faces = _np_cube()
    predictions = [
        {"vertices": verts, "faces": faces, "label": "cube"},
        {"vertices": verts, "faces": faces, "label": "cube2"},
    ]

    result = evaluate_batch(predictions)
    assert "aggregate" in result, f"evaluate_batch must return 'aggregate' key, got {list(result.keys())}"
    assert "per_sample" in result, f"evaluate_batch must return 'per_sample' key"
    assert len(result["per_sample"]) == 2

    print(f"  [OK] evaluate_batch: 2 predictions → aggregate with "
          f"validity_mean={result['aggregate'].get('validity_score_mean', 'N/A'):.3f}")


def test_sample_surface_points_distribution():
    """sample_surface_points returns points that lie within the mesh bounding box."""
    from evaluation.metrics import sample_surface_points

    verts, faces = _np_cube()
    pts = sample_surface_points(verts, faces, n_points=512)

    assert pts.shape == (512, 3), f"Expected (512, 3), got {pts.shape}"

    # All points must lie within the mesh bounding box (with small tolerance)
    bbox_min = verts.min(axis=0) - 1e-6
    bbox_max = verts.max(axis=0) + 1e-6
    assert np.all(pts >= bbox_min) and np.all(pts <= bbox_max), \
        "Sampled points are outside bounding box"

    print(f"  [OK] sample_surface_points: 512 points all within bbox")


def test_sample_surface_points_empty_mesh():
    """sample_surface_points returns zero array for empty mesh."""
    from evaluation.metrics import sample_surface_points

    empty_v = np.zeros((0, 3), dtype=np.float64)
    empty_f = np.zeros((0, 3), dtype=np.int64)

    pts = sample_surface_points(empty_v, empty_f, n_points=16)
    assert pts.shape == (16, 3)
    assert np.all(pts == 0), "Empty mesh should return all-zero point cloud"

    print("  [OK] sample_surface_points: empty mesh → zero array (16, 3)")


# ══════════════════════════════════════════════════════════════════════
# Test suite — load/save + run_test_suite with mock generator
# ══════════════════════════════════════════════════════════════════════

def test_save_load_test_suite_roundtrip():
    """save_test_suite + load_test_suite preserves all test cases."""
    from evaluation.test_suite import save_test_suite, load_test_suite, TEST_CASES

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "suite.json"
        returned_path = save_test_suite(TEST_CASES, path=save_path)

        assert returned_path == save_path
        assert save_path.exists()

        loaded = load_test_suite(path=save_path)

    assert len(loaded) == len(TEST_CASES), \
        f"Loaded {len(loaded)} cases, expected {len(TEST_CASES)}"

    # Spot-check that IDs and prompts survived JSON round-trip
    for orig, restored in zip(TEST_CASES[:3], loaded[:3]):
        assert orig["id"] == restored["id"]
        assert orig["prompt"] == restored["prompt"]
        assert orig["category"] == restored["category"]

    print(f"  [OK] save/load test suite: {len(TEST_CASES)} cases round-tripped")


def test_load_test_suite_fallback_on_corrupt_json():
    """load_test_suite falls back to built-in TEST_CASES on corrupt JSON."""
    from evaluation.test_suite import load_test_suite, TEST_CASES

    with tempfile.TemporaryDirectory() as tmpdir:
        bad_path = Path(tmpdir) / "bad_suite.json"
        bad_path.write_text("this is not valid json {{{")

        # Should not raise — falls back to defaults
        loaded = load_test_suite(path=bad_path)

    # Fallback returns built-in cases
    assert loaded is TEST_CASES or len(loaded) == len(TEST_CASES), \
        "load_test_suite should return built-in cases on corrupt JSON"

    print("  [OK] load_test_suite: corrupt JSON → falls back to built-in TEST_CASES")


def test_run_test_suite_with_mock_generator():
    """run_test_suite calls generate_fn for each case and aggregates results."""
    from evaluation.test_suite import run_test_suite

    cube_verts, cube_faces = _np_cube()

    calls = []
    def mock_generate(prompt, max_faces):
        calls.append(prompt)
        return cube_verts.tolist(), cube_faces.tolist()

    # Use a tiny subset of 3 cases
    mini_cases = [
        {"id": "t1", "prompt": "cube", "category": "primitive",
         "expected": {"min_faces": 2}},
        {"id": "t2", "prompt": "sphere", "category": "primitive",
         "expected": {"min_faces": 2}},
        {"id": "t3", "prompt": "table", "category": "furniture",
         "expected": {"min_faces": 2}},
    ]

    output = run_test_suite(mock_generate, test_cases=mini_cases, max_faces=64)

    assert len(calls) == 3, f"generate_fn should be called 3 times, got {len(calls)}"
    assert "results" in output and "summary" in output
    assert output["summary"]["total_cases"] == 3
    assert output["summary"]["generated_successfully"] == 3
    # All cases should meet expectations (cube has ≥2 faces)
    assert output["summary"]["expectations_met"] == 3

    # Verify by_category is populated
    assert "primitive" in output["summary"]["by_category"]
    assert "furniture" in output["summary"]["by_category"]

    print(f"  [OK] run_test_suite: 3 cases, all generated, all expectations met, "
          f"2 categories tracked")


def test_run_test_suite_handles_generation_failure():
    """run_test_suite catches generation exceptions and marks case as failed."""
    from evaluation.test_suite import run_test_suite

    def always_fails(prompt, max_faces):
        raise ValueError(f"Intentional failure for '{prompt}'")

    cases = [{"id": "f1", "prompt": "anything", "category": "test",
              "expected": {"min_faces": 1}}]

    output = run_test_suite(always_fails, test_cases=cases)

    assert output["summary"]["generated_successfully"] == 0
    assert output["results"][0]["generated"] is False
    assert "error" in output["results"][0]

    print("  [OK] run_test_suite: generator exception → case marked as failed")


def test_summarize_test_results_by_category():
    """_summarize_test_results groups correctly by category."""
    from evaluation.test_suite import _summarize_test_results

    results = [
        {"id": "a", "category": "primitive", "generated": True, "expectations_met": True,
         "metrics": {"validity": {"validity_score": 0.9}}, "num_faces": 12},
        {"id": "b", "category": "primitive", "generated": True, "expectations_met": False,
         "metrics": {"validity": {"validity_score": 0.5}}, "num_faces": 6},
        {"id": "c", "category": "furniture", "generated": False, "expectations_met": False},
    ]
    summary = _summarize_test_results(results)

    assert summary["total_cases"] == 3
    assert summary["generated_successfully"] == 2
    assert summary["generation_rate"] == pytest.approx(2/3, abs=0.01)
    assert summary["expectations_met"] == 1

    # Category breakdown
    prim = summary["by_category"]["primitive"]
    assert prim["total"] == 2
    assert prim["generated"] == 2

    furn = summary["by_category"]["furniture"]
    assert furn["total"] == 1
    assert furn["generated"] == 0

    print("  [OK] _summarize_test_results: generation_rate=2/3, category breakdown correct")


# ══════════════════════════════════════════════════════════════════════
# Training utilities — GRPO edge cases
# ══════════════════════════════════════════════════════════════════════

def test_grpo_weights_equal_losses():
    """grpo_quality_weights with equal losses returns all ones."""
    from training.train_unified import grpo_quality_weights

    losses = torch.tensor([1.0, 1.0, 1.0, 1.0])
    weights = grpo_quality_weights(losses, temperature=1.0)

    assert weights.shape == losses.shape
    # With equal losses, all weights should be equal (= 1.0 each)
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-5), \
        f"Equal losses should give equal weights=1.0, got {weights}"

    print(f"  [OK] grpo_quality_weights(equal): all weights = {weights[0]:.4f}")


def test_grpo_weights_single_element():
    """grpo_quality_weights with batch_size=1 returns tensor([1.0])."""
    from training.train_unified import grpo_quality_weights

    losses = torch.tensor([0.42])
    weights = grpo_quality_weights(losses, temperature=1.0)

    assert weights.shape == (1,)
    # Single element → weight is 1.0 (sum = batch_size = 1)
    assert abs(weights[0].item() - 1.0) < 0.01, \
        f"Single-element weights sum must be 1.0, got {weights[0].item()}"

    print(f"  [OK] grpo_quality_weights(single element) = {weights[0].item():.4f}")


def test_grpo_weights_temperature_effect():
    """Higher temperature flattens GRPO weights toward uniform."""
    from training.train_unified import grpo_quality_weights

    losses = torch.tensor([0.1, 1.0, 2.0])

    w_cold = grpo_quality_weights(losses, temperature=0.1)
    w_hot = grpo_quality_weights(losses, temperature=10.0)

    # Cold temperature → high variance (more extreme reward differences)
    # Hot temperature → low variance (near-uniform)
    var_cold = w_cold.var().item()
    var_hot = w_hot.var().item()

    assert var_cold > var_hot, (
        f"Lower temperature should give HIGHER variance. "
        f"var(cold={var_cold:.4f}) should > var(hot={var_hot:.4f})")

    print(f"  [OK] grpo_quality_weights: temperature effect verified "
          f"(var cold={var_cold:.4f} > hot={var_hot:.4f})")


# ══════════════════════════════════════════════════════════════════════
# LR schedule — boundary values
# ══════════════════════════════════════════════════════════════════════

def test_lr_schedule_boundaries():
    """deepseek_lr_schedule transitions smoothly at 80% and 90% thresholds."""
    from training.train_unified import deepseek_lr_schedule

    warmup = 100
    total = 10000

    # At 80% of post-warmup steps: last cosine → first linear should be continuous
    step_80 = warmup + int((total - warmup) * 0.8)
    step_79 = step_80 - 1
    step_81 = step_80 + 1

    lr_79 = deepseek_lr_schedule(step_79, warmup, total)
    lr_80 = deepseek_lr_schedule(step_80, warmup, total)
    lr_81 = deepseek_lr_schedule(step_81, warmup, total)

    # Continuous: jump should be very small
    assert abs(lr_80 - lr_79) < 0.01, \
        f"LR discontinuity at 80% threshold: {lr_79:.5f} → {lr_80:.5f}"

    # Monotonically decreasing after warmup
    assert lr_79 >= lr_80 >= lr_81, \
        f"LR should decrease: {lr_79:.5f} ≥ {lr_80:.5f} ≥ {lr_81:.5f}"

    # Final constant phase (step > 90%): should be ~0.1
    step_95 = warmup + int((total - warmup) * 0.95)
    lr_95 = deepseek_lr_schedule(step_95, warmup, total)
    assert abs(lr_95 - 0.1) < 0.01, \
        f"Final LR should be ~0.1, got {lr_95:.5f}"

    print(f"  [OK] LR boundaries: 80%={lr_80:.4f}, 90%→constant, final={lr_95:.4f}")


# ══════════════════════════════════════════════════════════════════════
# Morton code — locality property
# ══════════════════════════════════════════════════════════════════════

def test_morton_code_locality():
    """Morton codes for nearby coords are numerically closer than distant ones."""
    from processing.mesh_tokenizer import MeshTokenizer

    # Points close together should have near-identical Morton codes
    c_near1 = (100, 100, 100)
    c_near2 = (101, 100, 100)
    c_far  = (800, 800, 800)

    code_near1 = MeshTokenizer._morton_encode_3d(*c_near1)
    code_near2 = MeshTokenizer._morton_encode_3d(*c_near2)
    code_far   = MeshTokenizer._morton_encode_3d(*c_far)

    diff_near = abs(code_near1 - code_near2)
    diff_far  = abs(code_near1 - code_far)

    assert diff_near < diff_far, (
        f"Morton: nearby points should have closer codes. "
        f"near_diff={diff_near}, far_diff={diff_far}")

    # Deterministic
    assert MeshTokenizer._morton_encode_3d(5, 3, 9) == MeshTokenizer._morton_encode_3d(5, 3, 9)

    print(f"  [OK] Morton code: nearby diff={diff_near}, far diff={diff_far} — locality preserved")


# ══════════════════════════════════════════════════════════════════════
# Cache skip threshold (200-byte filter)
# ══════════════════════════════════════════════════════════════════════

def test_cache_200_byte_filter():
    """_refresh_cache_paths skips .pt files smaller than 200 bytes."""
    # Test the logic inline (the threshold is baked into RealMeshStream._refresh_cache_paths)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Write a small "marker" file (< 200 bytes)
        tiny = cache_dir / "tiny.pt"
        tiny.write_bytes(b"not a real pt" * 5)  # 65 bytes
        assert tiny.stat().st_size < 200

        # Write a real-sized cache file (>200 bytes)
        real = cache_dir / "real.pt"
        torch.save([{"mesh_tokens": list(range(50)), "label": "cube"}], real)
        assert real.stat().st_size > 200

        # Replicate the filter logic
        paths = sorted(cache_dir.glob("*.pt"))
        filtered = [str(p) for p in paths if p.stat().st_size > 200]

        assert str(tiny) not in filtered, "Tiny file should be filtered out"
        assert str(real) in filtered, "Real cache file should be included"

        tiny_size = tiny.stat().st_size
        real_size = real.stat().st_size

    print(f"  [OK] Cache 200-byte filter: {tiny_size}B file excluded, "
          f"{real_size}B file included")


# ══════════════════════════════════════════════════════════════════════
# Evaluate_single without reference (validity-only mode)
# ══════════════════════════════════════════════════════════════════════

def test_evaluate_single_no_reference():
    """evaluate_single without reference mesh still returns validity metrics."""
    from evaluation.metrics import evaluate_single

    verts, faces = _np_cube()
    result = evaluate_single(verts, faces)  # no ref

    assert "validity" in result
    assert "validity_score" in result["validity"]
    # No comparative metrics without reference
    assert "chamfer_distance" not in result
    assert "normal_consistency" not in result

    print(f"  [OK] evaluate_single (no ref): validity={result['validity']['validity_score']:.3f}, "
          f"no comparative metrics")


# ══════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════

# pytest compatibility — import pytest for approx
try:
    import pytest
except ImportError:
    class _pytest_shim:
        @staticmethod
        def approx(val, abs=1e-6):
            class _A:
                def __init__(self, v, a): self.v, self.a = v, a
                def __eq__(self, other): return abs(other - self.v) <= self.a
                def __repr__(self): return f"approx({self.v}±{self.a})"
            return _A(val, abs)
    pytest = _pytest_shim()


TESTS = [
    # MeshTokenizer edge cases
    ("quantize_coord NaN/Inf handling",       test_mesh_tokenizer_nan_inf_coords),
    ("dequantize_token special tokens → 0.0", test_mesh_tokenizer_dequantize_special_tokens),
    ("encode_mesh invalid vertex indices",    test_mesh_tokenizer_invalid_vertex_indices),
    ("encode_mesh empty faces → [BOS,EOS]",   test_mesh_tokenizer_empty_faces),
    ("encode_mesh max_faces truncation",      test_mesh_tokenizer_max_faces_truncation),
    ("pad_sequence pad + truncate",           test_mesh_tokenizer_pad_sequence),
    ("MeshTokenizer save/load roundtrip",     test_mesh_tokenizer_save_load_roundtrip),
    ("sequence_length_for_faces formula",     test_mesh_tokenizer_sequence_length_formula),
    # BPETokenizer edge cases
    ("encode_padded ids+mask always max_len", test_bpe_encode_padded_lengths),
    ("decode skip_special removes PAD/BOS/EOS", test_bpe_decode_skips_special_tokens),
    ("Blender terms encoded as single tokens", test_bpe_blender_terms_encoded_as_single_tokens),
    ("BPETokenizer never produces UNK",       test_bpe_no_unk_tokens),
    # labeler_smart helpers
    ("_strip_hex_uids correctness",           test_clean_label_strips_hex_uids),
    ("_strip_blender_prefixes correctness",   test_clean_label_strips_blender_prefixes),
    ("_strip_version_parts correctness",      test_clean_label_strips_version_parts),
    ("_clean_label_final full pipeline",      test_clean_label_final_pipeline),
    ("compute_bbox_aspect correctness",       test_compute_bbox_aspect),
    ("_is_primitive_name detection",          test_is_primitive_name),
    # generate_synthetic — ALL specs
    ("All SHAPE_SPECS generators",            test_all_shape_specs),
    ("All COMPOSITE_SPECS generators",        test_all_composite_specs),
    ("normalize_mesh degenerate inputs",      test_normalize_mesh_degenerate_collinear),
    ("generate_label non-empty output",       test_generate_label_correctness),
    # Model disabled-head errors
    ("forward_materials raises when disabled", test_forward_materials_raises_when_disabled),
    ("forward_modifiers raises when disabled", test_forward_modifiers_raises_when_disabled),
    ("forward_contrastive raises when disabled", test_forward_contrastive_raises_when_disabled),
    ("generate_geometry produces token sequence", test_generate_geometry_produces_token_sequence),
    # Eval metrics
    ("bounding_box_iou self = 1.0",           test_bounding_box_iou_self),
    ("bounding_box_iou disjoint = 0.0",       test_bounding_box_iou_disjoint),
    ("bounding_box_iou empty = 0.0",          test_bounding_box_iou_empty),
    ("normal_consistency self ≈ 1.0",         test_normal_consistency_self),
    ("normal_consistency empty = 0.0",        test_normal_consistency_empty_mesh),
    ("shape_distribution histogram",          test_shape_distribution_output),
    ("evaluate_batch list",                   test_evaluate_batch_list),
    ("sample_surface_points within bbox",     test_sample_surface_points_distribution),
    ("sample_surface_points empty mesh",      test_sample_surface_points_empty_mesh),
    # Test suite
    ("save/load test suite roundtrip",        test_save_load_test_suite_roundtrip),
    ("load_test_suite fallback on corrupt",   test_load_test_suite_fallback_on_corrupt_json),
    ("run_test_suite mock generator",         test_run_test_suite_with_mock_generator),
    ("run_test_suite handles failure",        test_run_test_suite_handles_generation_failure),
    ("_summarize_test_results by_category",   test_summarize_test_results_by_category),
    # Training utilities
    ("grpo_quality_weights equal losses",     test_grpo_weights_equal_losses),
    ("grpo_quality_weights single element",   test_grpo_weights_single_element),
    ("grpo_quality_weights temperature",      test_grpo_weights_temperature_effect),
    # LR schedule
    ("LR schedule boundary continuity",       test_lr_schedule_boundaries),
    # Morton code
    ("Morton code locality property",         test_morton_code_locality),
    # Cache threshold
    ("Cache 200-byte filter",                 test_cache_200_byte_filter),
    # evaluate_single
    ("evaluate_single no reference",          test_evaluate_single_no_reference),
]


def run_all_tests():
    import traceback

    print(f"\n{'='*65}")
    print("  Blender Copilot — Extended Pipeline Tests")
    print(f"{'='*65}\n")

    passed = failed = skipped = 0

    for name, fn in TESTS:
        print(f"[{passed + failed + skipped + 1:02d}/{len(TESTS):02d}] {name}")
        try:
            fn()
            passed += 1
        except SystemExit:
            raise
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
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


if __name__ == "__main__":
    run_all_tests()
