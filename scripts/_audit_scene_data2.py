"""Audit cameras, lights, physics, and texture data in master cache."""
import torch, pathlib, json

def pp(v, n=200):
    return str(json.dumps(v, default=str))[:n]

# candy_bounce
mc1 = pathlib.Path('data/master_cache/blender_official/5d82173d71ec0bf6.pt')
d1 = torch.load(mc1, weights_only=False)

print('=== CAMERA OBJECT (candy_bounce) ===')
for o in d1.get('objects', []):
    if o.get('type') == 'CAMERA':
        for k, v in o.items():
            print(f'  {k}: {pp(v, 150)}')

# mesh_extra on bouncing objects (physics)
print('\n=== MESH_EXTRA (physics) ===')
for o in d1.get('objects', []):
    me = o.get('mesh_extra', {})
    if me:
        print(f"  {o['name']}: {pp(me, 300)}")

# blendswap 31621 - check for lights
mc2 = pathlib.Path('data/master_cache/blendswap/fdebd38e872ba34b.pt')
d2 = torch.load(mc2, weights_only=False)

print('\n=== 31621 ALL OBJECT TYPES ===')
for o in d2.get('objects', []):
    t = o.get('type', 'MESH')
    if t != 'MESH':
        print(f"  {o['name']}: type={t}")
        for k, v in o.items():
            if k not in ('vertices', 'faces', 'normals', 'face_material_indices', 'face_smooth', 'uv_layers'):
                print(f"    {k}: {pp(v, 120)}")

print('\n=== 31621 WORLD ===')
print(pp(d2.get('world', {}), 400))

# Check what the source JSON has for lights
print('\n=== SOURCE JSON NON-MESH OBJECTS ===')
for src_file in [
    'data/processed/blender_official/candy_bounce.json',
    'data/processed/blendswap/31621.json',
]:
    p = pathlib.Path(src_file)
    if p.exists():
        raw = json.loads(p.read_text())
        print(f'\n{p.name}:')
        for o in raw.get('objects', []):
            t = o.get('type', 'MESH')
            if t != 'MESH':
                keys = list(o.keys())
                print(f'  NON-MESH [{t}] {o.get("name")}: keys={keys}')
                light = o.get('light', {})
                cam = o.get('camera', {})
                if light:
                    print(f'    light: {pp(light, 200)}')
                if cam:
                    print(f'    camera: {pp(cam, 200)}')

# Check rigid body physics in source JSON
print('\n=== PHYSICS (rigid body) IN SOURCE ===')
p = pathlib.Path('data/processed/blender_official/candy_bounce.json')
if p.exists():
    raw = json.loads(p.read_text())
    for o in raw.get('objects', []):
        rb = o.get('rigid_body') or o.get('physics') or o.get('constraints')
        if rb:
            print(f"  {o['name']}: {pp(rb, 200)}")
    # Check all keys of first object
    objs = raw.get('objects', [])
    if objs:
        print(f'\n  All keys in first obj: {sorted(objs[0].keys())}')
