#!/usr/bin/env python3
"""Deep inspection of cache samples: mesh_tokens, label quality, actual structure."""
import torch
import numpy as np
import random
from pathlib import Path
from collections import Counter

cache_dir = Path(__file__).parent.parent / "data" / "processed" / ".mesh_cache"
pt_files = sorted(cache_dir.glob("*.pt"))

print(f"=== DEEP SAMPLE INSPECTION ===\n")

# 1. Inspect structure of first file
print("--- Structure of first .pt file ---")
d0 = torch.load(pt_files[0], map_location="cpu", weights_only=False)
if isinstance(d0, list):
    print(f"  Type: list of {len(d0)} samples")
    s0 = d0[0]
else:
    print(f"  Type: single dict")
    s0 = d0
print(f"  Keys: {list(s0.keys())}")
for k, v in s0.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: Tensor shape={v.shape} dtype={v.dtype} min={v.min().item():.4f} max={v.max().item():.4f}")
    elif isinstance(v, str):
        print(f"    {k}: str len={len(v)} = \"{v[:80]}\"")
    elif isinstance(v, (int, float)):
        print(f"    {k}: {type(v).__name__} = {v}")
    elif isinstance(v, dict):
        print(f"    {k}: dict keys={list(v.keys())[:10]}")
    elif isinstance(v, list):
        print(f"    {k}: list len={len(v)}")
    else:
        print(f"    {k}: {type(v).__name__}")

# 2. Check mesh_tokens distribution across random files
print(f"\n--- mesh_tokens inspection (20 random files) ---")
random.seed(123)
chosen_files = random.sample(pt_files, min(20, len(pt_files)))

mesh_token_lengths = []
text_id_lengths = []
quality_weights = []
all_labels_sample = []

for f in chosen_files:
    data = torch.load(f, map_location="cpu", weights_only=False)
    samples = data if isinstance(data, list) else [data]
    for s in samples:
        mt = s.get("mesh_tokens")
        if mt is not None and isinstance(mt, torch.Tensor):
            mesh_token_lengths.append(len(mt))
        ti = s.get("text_ids")
        if ti is not None and isinstance(ti, torch.Tensor):
            text_id_lengths.append(len(ti))
        qw = s.get("quality_weight")
        if qw is not None:
            if isinstance(qw, torch.Tensor):
                quality_weights.append(qw.item())
            else:
                quality_weights.append(float(qw))
        all_labels_sample.append(s.get("label", ""))

if mesh_token_lengths:
    mtl = np.array(mesh_token_lengths)
    print(f"  mesh_tokens lengths: min={mtl.min()} max={mtl.max()} mean={mtl.mean():.0f} median={np.median(mtl):.0f}")
    # Back-calculate face counts: each face = 9 tokens (3 verts * 3 coords) + 1 face separator
    # Actually: mesh_token format is [BOS, v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, FACE_SEP, ...]
    # So face_count = (len - 2) / 10 approximately (BOS + EOS + per-face overhead)
    print(f"  Estimated face counts from token lengths:")
    for l in sorted(set(mesh_token_lengths))[:10]:
        faces_approx = (l - 2) // 10
        print(f"    tokens={l} -> ~{faces_approx} faces")

if text_id_lengths:
    til = np.array(text_id_lengths)
    print(f"  text_ids lengths: min={til.min()} max={til.max()} mean={til.mean():.0f}")

if quality_weights:
    qw = np.array(quality_weights)
    print(f"  quality_weight: min={qw.min():.3f} max={qw.max():.3f} mean={qw.mean():.3f}")

# 3. Full scan for mesh token stats
print(f"\n--- Full mesh_tokens scan (all {len(pt_files)} files) ---")
all_mt_lens = []
nan_mesh = 0
zero_mesh = 0
total = 0
labels_all = []
for f in pt_files:
    data = torch.load(f, map_location="cpu", weights_only=False)
    samples = data if isinstance(data, list) else [data]
    for s in samples:
        total += 1
        mt = s.get("mesh_tokens")
        if mt is not None and isinstance(mt, torch.Tensor):
            all_mt_lens.append(len(mt))
            if torch.isnan(mt.float()).any() or torch.isinf(mt.float()).any():
                nan_mesh += 1
            if len(mt) == 0:
                zero_mesh += 1
        else:
            zero_mesh += 1
        labels_all.append(s.get("label", ""))

print(f"  Total samples: {total}")
if all_mt_lens:
    aml = np.array(all_mt_lens)
    print(f"  With mesh_tokens: {len(aml)}")
    print(f"  NaN/Inf in mesh_tokens: {nan_mesh}")
    print(f"  Zero-length mesh_tokens: {zero_mesh}")
    print(f"  Lengths: min={aml.min()} max={aml.max()} mean={aml.mean():.0f} median={np.median(aml):.0f}")
    
    # Face count estimates  
    face_estimates = (aml - 2) / 10  # rough
    print(f"  Estimated faces: min={face_estimates.min():.0f} max={face_estimates.max():.0f} mean={face_estimates.mean():.0f} median={np.median(face_estimates):.0f}")
    
    for thresh in [100, 500, 1000, 5000, 10000, 20000, 36002]:
        cnt = (aml <= thresh).sum()
        print(f"  <= {thresh} tokens: {cnt} ({100*cnt/len(aml):.1f}%)")

# 4. Label problem analysis
print(f"\n--- Label problem analysis ---")
lbl_counter = Counter(l.strip().lower() for l in labels_all if l.strip())
# Labels that appear > 50 times are suspicious duplicates
dupe_labels = {lbl: cnt for lbl, cnt in lbl_counter.items() if cnt > 50}
print(f"  Labels appearing >50 times (scene-source duplication):")
total_dupes = 0
for lbl, cnt in sorted(dupe_labels.items(), key=lambda x: -x[1]):
    print(f"    [{cnt}x] \"{lbl[:70]}\"")
    total_dupes += cnt
print(f"  Total samples with highly-duplicated labels: {total_dupes}")
print(f"  Unique samples (label appears <=50 times): {total - total_dupes}")

# Check for meaningless labels
bad_patterns = ["cube", "plane", "sphere", "cylinder", "circle"]
pattern_count = 0
for lbl in labels_all:
    clean = lbl.strip().lower()
    if clean in bad_patterns:
        pattern_count += 1
print(f"  Pure geometric primitive labels (cube/plane/sphere/etc): {pattern_count}")
