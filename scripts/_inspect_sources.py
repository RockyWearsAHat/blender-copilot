"""Inspect extracted source JSONs to understand data richness."""
import json
from pathlib import Path

BASE = Path(__file__).parent.parent
PROC = BASE / "data" / "processed"

for src in ['blendswap', 'blender_official', 'smutbase', 'github', 'open3dlab', 'objaverse']:
    d = PROC / src
    if not d.exists():
        continue
    files = list(d.glob('*.json'))[:3]
    for f in files[:1]:
        try:
            data = json.loads(f.read_text())
            objs = data.get('objects', [data])
            obj = objs[0] if objs else data
            mesh = obj.get('mesh', {})
            mats = obj.get('materials', [])
            fi = mesh.get('face_material_indices', mesh.get('material_indices', []))
            uvs = mesh.get('uv_layers', mesh.get('uv_map', mesh.get('uvs', [])))
            mat_keys = list(mats[0].keys()) if mats else []
            print(f"\n{src}/{f.name[:40]}")
            print(f"  mesh_keys:     {list(mesh.keys())}")
            print(f"  mats:          {len(mats)}  keys={mat_keys}")
            if mats:
                print(f"  mat[0]:        {str(mats[0])[:200]}")
            print(f"  face_mat_idx:  {len(fi)}")
            print(f"  uvs:           {type(uvs).__name__} len={len(uvs) if uvs else 0}")
        except Exception as e:
            print(f"{src}: ERROR {e}")
