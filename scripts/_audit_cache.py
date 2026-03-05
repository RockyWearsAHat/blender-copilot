#!/usr/bin/env python3
"""Quick audit of restructured cache quality."""
import torch, os, random, re
from collections import Counter
from pathlib import Path

cache = Path("data/processed/.mesh_cache")
files = sorted(cache.glob("*.pt"))

labels = []
sources = []
tiers = []
issues = []

for f in files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    items = d if isinstance(d, list) else [d]
    for s in items:
        label = s.get("label", "")
        src = s.get("data_source", "?")
        tier = s.get("quality_tier", "?")
        labels.append(label)
        sources.append(src)
        tiers.append(tier)

        words = set(re.split(r'[\s,._-]+', label.lower()))
        clean_words = {w for w in words if w and len(w) > 1}

        # Check various quality issues
        if len(label.strip()) < 3:
            issues.append(("TOO_SHORT", label))
        elif not any(c.isalpha() for c in label):
            issues.append(("NO_ALPHA", label))
        # Blender internal names
        elif re.search(r'\b(cube|sphere|cylinder|plane|mesh)\.\d{3}\b', label.lower()):
            issues.append(("BLENDER_NAME", label))
        # Single foreign word
        elif len(clean_words) == 1:
            w = list(clean_words)[0]
            if len(w) > 3 and not w.isascii():
                issues.append(("NON_ASCII_WORD", label))

print(f"Total samples: {len(labels)}")
print(f"Tiers: {dict(Counter(tiers))}")
print(f"Sources: {dict(Counter(sources))}")
print(f"Issues found: {len(issues)}")
for reason, cnt in Counter(r for r, _ in issues).most_common():
    print(f"  {reason}: {cnt}")
print()

# Show random sample of 30 labels
random.seed(42)
sample_idx = random.sample(range(len(labels)), min(30, len(labels)))
print("Random 30 labels (with source):")
for i in sample_idx:
    print(f"  [{tiers[i]:6s}] [{sources[i]:16s}] {labels[i][:100]}")

# Show worst labels
print("\nIssue samples:")
for reason, label in issues[:15]:
    print(f"  [{reason}] {label[:100]}")
