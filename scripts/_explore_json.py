#!/usr/bin/env python3
"""Temporary script to explore extracted JSON data structure."""
import json
import os
import glob

# Look at the dog rigging file
with open('data/processed/blender_official/dog.rigging.json') as f:
    data = json.load(f)

print("=== TOP LEVEL ===")
print("Keys:", list(data.keys()))
print("fps:", data.get('fps'))
print("frame_start:", data.get('frame_start'))
print("frame_end:", data.get('frame_end'))
print("Num objects:", len(data.get('objects', [])))
print("Orphan actions:", len(data.get('orphan_actions', [])))

types = {}
for o in data['objects']:
    t = o.get('type', '?')
    types[t] = types.get(t, 0) + 1
print("Object types:", types)

# Look at armature
print("\n=== ARMATURE ===")
for obj in data['objects']:
    if obj.get('type') == 'ARMATURE' and 'armature' in obj:
        arm = obj['armature']
        print("Name:", obj['name'])
        print("Armature keys:", list(arm.keys()))
        if 'bones' in arm:
            print("Num bones:", len(arm['bones']))
            for b in arm['bones'][:3]:
                print("  Bone:", b.get('name'), "keys:", list(b.keys()))
                if 'constraints' in b:
                    for c in b['constraints'][:2]:
                        print("    Constraint:", c.get('type'), c.get('name'))
        if 'drivers' in obj:
            print("Drivers:", len(obj['drivers']))
        break

# Look at mesh with materials
print("\n=== MESH WITH MATERIALS ===")
for obj in data['objects']:
    if obj.get('type') == 'MESH' and 'mesh' in obj:
        m = obj['mesh']
        print("Name:", obj['name'])
        print("Mesh keys:", list(m.keys()))
        if 'materials' in m:
            print("Num materials:", len(m['materials']))
            for mat in m['materials'][:1]:
                print("  Material:", mat.get('name'), "keys:", list(mat.keys()))
                if 'nodes' in mat:
                    print("  Num nodes:", len(mat['nodes']))
                    for n in mat['nodes'][:5]:
                        print("    Node:", n.get('type'), n.get('name'), "keys:", list(n.keys()))
        if 'shape_keys' in obj and obj['shape_keys']:
            sk = obj['shape_keys']
            print("Shape keys type:", type(sk).__name__)
            if isinstance(sk, dict):
                print("Shape keys dict keys:", list(sk.keys()))
                for k, v in list(sk.items())[:3]:
                    print("  SK:", k, "->", type(v).__name__, str(v)[:200])
            elif isinstance(sk, list):
                print("Shape keys count:", len(sk))
                for s in sk[:3]:
                    print("  SK:", s)
        break

# Look at orphan actions (animation data)
print("\n=== ORPHAN ACTIONS ===")
if 'orphan_actions' in data:
    for a in data['orphan_actions'][:2]:
        print("Action:", a.get('name'), "keys:", list(a.keys()))
        if 'fcurves' in a:
            print("  Num fcurves:", len(a['fcurves']))
            for fc in a['fcurves'][:3]:
                print("  FCurve:", fc.get('data_path'), "array_index:", fc.get('array_index'))
                if 'keyframes' in fc:
                    print("    Keyframes:", len(fc['keyframes']), "first:", fc['keyframes'][:2])

# Count total data across ALL processed dirs
print("\n=== TOTAL DATA SURVEY ===")
source_dirs = [
    'data/processed/objaverse',
    'data/processed/blender_official',
    'data/processed/blendswap',
    'data/processed/smutbase',
    'data/processed/github',
]
total = 0
with_anim = 0
with_armature = 0
with_materials = 0
with_shape_keys = 0
with_physics = 0
with_modifiers = 0

for sd in source_dirs:
    if not os.path.exists(sd):
        continue
    for jf in glob.glob(os.path.join(sd, '**/*.json'), recursive=True):
        total += 1
        try:
            with open(jf) as f:
                d = json.load(f)
            has_anim = False
            has_arm = False
            has_mat = False
            has_sk = False
            has_phys = False
            has_mod = False
            for o in d.get('objects', []):
                if o.get('type') == 'ARMATURE':
                    has_arm = True
                if 'animation' in o or 'fcurves' in o:
                    has_anim = True
                if o.get('type') == 'MESH' and 'mesh' in o:
                    mesh = o['mesh']
                    if mesh.get('materials'):
                        has_mat = True
                    if mesh.get('modifiers'):
                        has_mod = True
                # Also check object-level materials
                if o.get('materials'):
                    has_mat = True
                if o.get('material_slots'):
                    has_mat = True
                if o.get('shape_keys'):
                    has_sk = True
                if o.get('physics'):
                    has_phys = True
                if o.get('modifiers'):
                    has_mod = True
            if d.get('orphan_actions'):
                has_anim = True
            if d.get('orphan_materials'):
                has_mat = True
            if has_anim: with_anim += 1
            if has_arm: with_armature += 1
            if has_mat: with_materials += 1
            if has_sk: with_shape_keys += 1
            if has_phys: with_physics += 1
            if has_mod: with_modifiers += 1
        except:
            pass

print(f"Total JSON files: {total}")
print(f"With animation: {with_anim}")
print(f"With armature: {with_armature}")
print(f"With materials: {with_materials}")
print(f"With shape keys: {with_shape_keys}")
print(f"With physics: {with_physics}")
print(f"With modifiers: {with_modifiers}")

# Now check where materials actually live
print("\n=== MATERIAL LOCATION CHECK ===")
jf = glob.glob('data/processed/objaverse/**/*.json', recursive=True)[0]
with open(jf) as f:
    d = json.load(f)
for o in d.get('objects', []):
    if o.get('type') == 'MESH':
        print("Object keys:", sorted(o.keys()))
        if 'mesh' in o:
            print("Mesh keys:", sorted(o['mesh'].keys()))
        break
print("Orphan materials:", len(d.get('orphan_materials', [])))
if d.get('orphan_materials'):
    print("First orphan mat keys:", sorted(d['orphan_materials'][0].keys()))

# Check a few more files for materials
print("\n=== MATERIAL EXAMPLES ===")
mat_count = 0
for sd in source_dirs:
    if not os.path.exists(sd):
        continue
    for jf in glob.glob(os.path.join(sd, '**/*.json'), recursive=True):
        if mat_count >= 3:
            break
        try:
            with open(jf) as f:
                d = json.load(f)
            for o in d.get('objects', []):
                if o.get('materials'):
                    print(f"  {os.path.basename(jf)}: obj.materials found, keys: {list(o['materials'][0].keys()) if o['materials'] else 'empty'}")
                    mat_count += 1
                    break
            if d.get('orphan_materials'):
                print(f"  {os.path.basename(jf)}: orphan_materials found, count: {len(d['orphan_materials'])}")
                mat_count += 1
        except:
            pass
