#!/usr/bin/env python3
"""Show samples from each quality tier."""
import torch, os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.restructure_training_data import score_label_quality

CACHE = Path("data/processed/.mesh_cache")
files = sorted(CACHE.glob("*.pt"))

by_reason = {}
for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    items = d if isinstance(d, list) else [d]
    for s in items:
        label = s.get("label", "")
        score, reason = score_label_quality(label)
        if reason not in by_reason:
            by_reason[reason] = []
        if len(by_reason[reason]) < 15:
            by_reason[reason].append((score, label[:90]))

for reason in sorted(by_reason, key=lambda r: -len(by_reason[r])):
    entries = by_reason[reason]
    score = entries[0][0]
    print(f"\n=== {reason} (score={score:.2f}, count~{len(entries)}+) ===")
    for _, label in entries[:10]:
        print(f"  {label}")
