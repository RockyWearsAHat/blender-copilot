#!/usr/bin/env python3
"""Check blendswap 31621 master cache."""
import torch, json
from pathlib import Path

batch = torch.load("data/training_cache/test_v3/batch_00000.pt", weights_only=False)
items = batch if isinstance(batch, list) else batch.get("items", batch.get("data", []))

for i, it in enumerate(items):
    if not isinstance(it, dict):
        continue
    ref = it.get("master_cache_ref", {})
    if "31621" in str(ref.get("source_file", "")):
        print(f"Item {i}: {it.get('label', '?')}")
        print(f"  ref: {json.dumps(ref, indent=2)}")
        cache_path = Path("data/master_cache") / ref["cache_rel_path"]
        if cache_path.exists():
            master = torch.load(cache_path, weights_only=False)
            objects = master.get("objects", [])
            print(f"  {len(objects)} objects:")
            for oi, obj in enumerate(objects[:10]):
                name = obj.get("name", "?")
                t = obj.get("transforms", {})
                loc = t.get("location", [0,0,0])
                dims = obj.get("dimensions", [0,0,0])
                has_v = "vertices" in obj and (hasattr(obj["vertices"], "__len__") and len(obj["vertices"]) > 0)
                print(f"    [{oi:2d}] {name:30s} loc={loc}  dims={dims}  has_mesh={has_v}")
        break
