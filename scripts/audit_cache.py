#!/usr/bin/env python3
"""Audit all cache .pt files for quality, label distribution, mesh integrity."""
import torch
import numpy as np
import random
from pathlib import Path
from collections import Counter

cache_dir = Path(__file__).parent.parent / "data" / "processed" / ".mesh_cache"
pt_files = sorted(cache_dir.glob("*.pt"))
print(f"Total .pt cache files: {len(pt_files)}")

total_samples = 0
all_labels = []
face_counts = []
token_counts = []
has_scene_context = 0
source_counter = Counter()
nan_count = 0
empty_token_count = 0
all_samples_flat = []

for f in pt_files:
    data = torch.load(f, map_location="cpu", weights_only=False)
    samples = data if isinstance(data, list) else [data]
    total_samples += len(samples)
    for s in samples:
        all_samples_flat.append((f.name, s))
        label = s.get("label", s.get("text", ""))
        all_labels.append(label)
        tokens = s.get("tokens")
        if tokens is not None:
            if isinstance(tokens, torch.Tensor):
                tlen = len(tokens)
                token_counts.append(tlen)
                if torch.isnan(tokens.float()).any() or torch.isinf(tokens.float()).any():
                    nan_count += 1
            elif isinstance(tokens, list):
                token_counts.append(len(tokens))
            if tlen == 0:
                empty_token_count += 1
        else:
            empty_token_count += 1
        fc = s.get("num_faces", s.get("face_count", 0))
        if fc:
            face_counts.append(fc)
        if s.get("scene_context"):
            has_scene_context += 1
        src = s.get("source", "unknown")
        source_counter[src] += 1

print(f"Total trainable samples: {total_samples}")
print(f"Samples with scene_context: {has_scene_context} ({100*has_scene_context/total_samples:.1f}%)")
print(f"NaN/Inf in tokens: {nan_count}")
print(f"Empty/missing tokens: {empty_token_count}")
print()

print("=== Per-source sample counts ===")
for src, cnt in source_counter.most_common():
    pct = 100 * cnt / total_samples
    print(f"  {src}: {cnt} ({pct:.1f}%)")
print()

print("=== Face count distribution ===")
if face_counts:
    fc = np.array(face_counts)
    print(f"  Count: {len(fc)}")
    print(f"  Min: {fc.min()}, Max: {fc.max()}, Mean: {fc.mean():.0f}, Median: {np.median(fc):.0f}")
    for thresh in [50, 100, 500, 1000, 2000, 3000, 4000]:
        print(f"  <= {thresh} faces: {(fc <= thresh).sum()} ({100*(fc <= thresh).sum()/len(fc):.1f}%)")
print()

print("=== Token count distribution ===")
if token_counts:
    tc = np.array(token_counts)
    print(f"  Count: {len(tc)}")
    print(f"  Min: {tc.min()}, Max: {tc.max()}, Mean: {tc.mean():.0f}, Median: {np.median(tc):.0f}")
print()

print("=== Label quality ===")
lens = [len(l) for l in all_labels]
print(f"  Min length: {min(lens)}, Max: {max(lens)}, Mean: {sum(lens)/len(lens):.0f}")
empty = sum(1 for l in all_labels if not l.strip())
print(f"  Empty labels: {empty}")
short = sum(1 for l in all_labels if 0 < len(l.strip()) < 3)
print(f"  Very short (1-2 chars): {short}")
vague_words = {"object", "mesh", "thing", "item", "model", "untitled", "default", "cube"}
vague = sum(1 for l in all_labels if l.strip().lower() in vague_words)
print(f"  Vague single-word labels: {vague}")

label_counter = Counter(l.strip().lower() for l in all_labels if l.strip())
print(f"  Unique labels: {len(label_counter)}")
print(f"  Top 20 most common labels:")
for lbl, cnt in label_counter.most_common(20):
    print(f"    [{cnt}x] \"{lbl}\"")
print()

print("=== Random sample inspection (20 samples) ===")
random.seed(42)
chosen = random.sample(all_samples_flat, min(20, len(all_samples_flat)))
for i, (fname, s) in enumerate(chosen):
    label = s.get("label", s.get("text", ""))
    tokens = s.get("tokens")
    tlen = len(tokens) if tokens is not None else 0
    fc = s.get("num_faces", s.get("face_count", 0))
    src = s.get("source", "?")
    sc = "yes" if s.get("scene_context") else "no"
    keys = list(s.keys())
    print(f"  [{i+1}] file={fname} src={src} label=\"{label[:60]}\" faces={fc} tokens={tlen} scene_ctx={sc}")
    print(f"       keys={keys}")
