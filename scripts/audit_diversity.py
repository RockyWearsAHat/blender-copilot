#!/usr/bin/env python3
"""Verify that 'duplicate' labels actually have diverse geometry."""
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

cache_dir = Path(__file__).parent.parent / "data" / "processed" / ".mesh_cache"
pt_files = sorted(cache_dir.glob("*.pt"))

label_to_tokens = defaultdict(list)

print("Loading all samples...")
total = 0
for f in pt_files:
    data = torch.load(f, map_location="cpu", weights_only=False)
    samples = data if isinstance(data, list) else [data]
    for s in samples:
        total += 1
        label = s.get("label", "").strip().lower()
        mt = s.get("mesh_tokens")
        if mt is not None and isinstance(mt, torch.Tensor):
            label_to_tokens[label].append((len(mt), hash(tuple(mt.tolist()[:50]))))  # hash first 50 tokens for speed

print(f"Total samples: {total}")
print(f"Unique labels: {len(label_to_tokens)}")
print()

# Check top duplicated labels for geometric diversity
print("=== Geometric diversity of top labels ===")
for label, entries in sorted(label_to_tokens.items(), key=lambda x: -len(x[1]))[:15]:
    count = len(entries)
    if count < 10:
        break
    lengths = [e[0] for e in entries]
    hashes = set(e[1] for e in entries)
    unique_ratio = len(hashes) / count
    print(f"  [{count}x] \"{label[:60]}\"")
    print(f"    token lengths: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.0f} std={np.std(lengths):.0f}")
    print(f"    unique mesh hashes: {len(hashes)}/{count} ({100*unique_ratio:.0f}%)")
    print()

# Global dedup check: how many truly identical meshes exist?
print("=== Global deduplication check ===")
all_hashes = []
for entries in label_to_tokens.values():
    for _, h in entries:
        all_hashes.append(h)
unique_hashes = len(set(all_hashes))
print(f"  Total samples: {len(all_hashes)}")
print(f"  Unique mesh hashes (first 50 tokens): {unique_hashes}")
print(f"  Potential duplicates: {len(all_hashes) - unique_hashes} ({100*(len(all_hashes) - unique_hashes)/len(all_hashes):.1f}%)")

# Check empty/very short labels
print()
print("=== Problem labels ===")
for label, entries in label_to_tokens.items():
    if len(label) == 0:
        print(f"  EMPTY LABEL: {len(entries)} samples")
    elif len(label) <= 2:
        print(f"  SHORT LABEL \"{label}\": {len(entries)} samples")
