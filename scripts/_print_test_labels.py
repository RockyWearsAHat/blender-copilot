#!/usr/bin/env python3
"""Print labels from the test batch for review."""
import torch, pathlib

data = torch.load("data/training_cache/default/batch_00000.pt", weights_only=False)

# Introspect the structure
if isinstance(data, dict):
    items = data.get("items", data.get("data", []))
elif isinstance(data, list):
    items = data
else:
    print(f"Unknown format: {type(data)}")
    items = []

print(f"Total items: {len(items)}\n")
for i, it in enumerate(items):
    if isinstance(it, dict):
        label = it.get("text_label", it.get("label", "?"))
        src = it.get("source_file", "?")
        nv = it.get("vertex_count", "?")
        nf = it.get("face_count", "?")
        print(f"  [{i:2d}] label={label!r:45s} verts={nv}  faces={nf}  src={src}")
    else:
        keys = list(it.keys()) if hasattr(it, "keys") else dir(it)[:10]
        print(f"  [{i:2d}] type={type(it).__name__}  keys={keys}")
