"""Check what training prompts actually look like."""
import json, sys, os
from pathlib import Path

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
os.chdir(root)

base = Path("data/processed/objaverse")
count = 0
for f in sorted(base.glob("*.json"))[:50]:
    try:
        d = json.loads(f.read_text())
        objs = d.get("objects", [])
        for o in objs:
            label = o.get("text_label", "") or d.get("metadata", {}).get("name", "")
            if label:
                print(repr(label))
                count += 1
                if count >= 15:
                    break
    except Exception:
        pass
    if count >= 15:
        break

# Also show what smart labeling produces for comparison
print("\n--- Smart label examples (from training pipeline) ---")
from processing.labeler_smart import generate_smart_label, compute_bbox_aspect
import numpy as np

# Fake a cube mesh for labeling
verts = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                   [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]], dtype=float)
bbox_aspect = compute_bbox_aspect(verts)
label = generate_smart_label(
    obj_name="Cube", material_names=[], modifier_types=[],
    num_faces=12, num_verts=8, bbox_aspect=bbox_aspect,
    file_label="", metadata_name="", metadata_desc="", 
    metadata_tags=[], metadata_categories="",
)
print(f"Cube  -> {label!r}")

# Now add the shape descriptor tokens
from processing.mesh_geometry_score import compute_signature, shape_descriptor_tokens
sig = compute_signature(verts)
desc = shape_descriptor_tokens(sig)
print(f"Cube descriptors: {desc!r}")
print(f"Full training prompt: {(label + desc)!r}")
