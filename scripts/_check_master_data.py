#!/usr/bin/env python3
"""Check what normalization/transform data is in master cache."""
import torch, json

# Load the candy_bounce master cache entry 
# Find it from the test batch
batch = torch.load("data/training_cache/test_v3/batch_00000.pt", weights_only=False)
items = batch if isinstance(batch, list) else batch.get("items", batch.get("data", []))

# Find a candy_bounce item
for i, it in enumerate(items):
    if isinstance(it, dict):
        ref = it.get("master_cache_ref", {})
        if "candy_bounce" in str(ref.get("source_file", "")):
            print(f"Item {i}: {it.get('label', '?')}")
            print(f"  master_cache_ref: {json.dumps(ref, indent=2)}")

            # Load master cache entry
            from pathlib import Path
            cache_path = Path("data/master_cache") / ref["cache_rel_path"]
            if cache_path.exists():
                master = torch.load(cache_path, weights_only=False)
                objects = master.get("objects", [])
                print(f"  Master has {len(objects)} objects")
                for oi, obj in enumerate(objects[:5]):  # First 5
                    name = obj.get("name", "?")
                    transforms = obj.get("transforms", {})
                    dims = obj.get("dimensions", [])
                    norm_c = obj.get("normalization_center")
                    norm_s = obj.get("normalization_scale")
                    has_mesh = "vertices" in obj
                    print(f"  [{oi}] {name}")
                    print(f"      transforms.location: {transforms.get('location', '?')}")
                    print(f"      transforms.scale: {transforms.get('scale', '?')}")
                    print(f"      dimensions: {dims}")
                    print(f"      normalization_center: {norm_c}")
                    print(f"      normalization_scale: {norm_s}")
                    if has_mesh:
                        v = obj["vertices"]
                        import numpy as np
                        v_np = v.numpy() if hasattr(v, "numpy") else np.array(v)
                        if v_np.size > 0:
                            print(f"      vertex range: min={v_np.min(axis=0).tolist()}, max={v_np.max(axis=0).tolist()}")
            break
