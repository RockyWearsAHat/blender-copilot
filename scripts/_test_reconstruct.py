#!/usr/bin/env python3
"""Quick test of --reconstruct-scene on the test batch."""
import subprocess, json, sys

cmd = [
    ".venv/bin/python", "scripts/validator_fetch_item.py",
    "--cache-dir", "data/training_cache/test_v3",
    "--work-dir", "data/validation_queue_live",
    "--reconstruct-scene",
    "--cache-pt", "data/training_cache/test_v3/batch_00000.pt",
    "--item-index", "1",  # plank beveled from blendswap 31621 (multi-object house)
]

proc = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
if proc.returncode != 0:
    print("STDERR:", proc.stderr[:500])
    sys.exit(1)

result = json.loads(proc.stdout)
print(f"OK: {result.get('ok')}")
print(f"Source: {result.get('source_file')}")
print(f"Total objects: {result.get('total_objects')}")
print(f"Current object index: {result.get('current_object_index')}")
print(f"Scene JSON: {result.get('scene_json')}")
for o in result.get("objects", []):
    marker = " <-- CURRENT" if o.get("is_current_item") else ""
    print(f"  [{o['object_index']}] {o['name']!r:30s} verts={o['vertex_count']}  faces={o['face_count']}  label={o.get('label','')!r}{marker}")
