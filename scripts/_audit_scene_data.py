"""Audit what scene data is stored in master cache for scene reconstruction."""
import torch, pathlib, json, sys

def pp(d, max_len=120):
    return str(json.dumps(d, default=str))[:max_len]

mc = pathlib.Path('data/master_cache/blender_official/5d82173d71ec0bf6.pt')
d = torch.load(mc, weights_only=False)

# Image pixel data?
print('=== IMAGES ===')
for name, v in list(d.get('images', {}).items()):
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(f'  [{name}] {kk}: {type(vv).__name__} = {str(vv)[:100]}')
    print()

# Objects - check cameras and lights
print('=== ALL OBJECTS (name + type) ===')
for o in d.get('objects', []):
    print(f"  {o.get('name'):30s} type={o.get('type')}")

# Modifiers
print('\n=== MODIFIERS ===')
for o in d.get('objects', []):
    mods = o.get('modifiers', [])
    if mods:
        for m in mods:
            print(f"  {o['name']}: {pp(m, 200)}")

# Material texture nodes
print('\n=== MATERIAL TEXTURE NODES ===')
for o in d.get('objects', []):
    for mat in o.get('materials', []):
        for n in mat.get('nodes', []):
            if 'TEX' in str(n.get('type', '')):
                print(f"  {o['name']}/{mat.get('name','?')}: {pp(n, 200)}")

# Collections (look for lights, cameras)
print('\n=== COLLECTIONS ===')
coll = d.get('collections', {})
print(pp(coll, 600))

# Non-mesh objects in blend_extractor source JSON
print('\n=== SOURCE JSON CONTENT ===')
# find the source JSON
src = d.get('source_file', d.get('_source_file', ''))
if src and 'candy_bounce' in src:
    # look for the json
    for p in pathlib.Path('data/processed').rglob('candy_bounce*.json'):
        raw = json.loads(p.read_text())
        top_keys = list(raw.keys())
        print(f'{p.name}: {top_keys}')
        # Check for cameras/lights
        objs = raw.get('objects', [])
        for o in objs:
            t = o.get('type', 'MESH')
            if t != 'MESH':
                print(f"  NON-MESH: {o.get('name')} type={t} keys={list(o.keys())[:10]}")
        break
