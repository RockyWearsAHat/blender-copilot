"""Test complete scene reconstruction including cameras, lights, world, images."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from scripts.validator_fetch_item import _reconstruct_scene

work = pathlib.Path('data/validation_queue_live')
cache = pathlib.Path('data/training_cache/test_v3/batch_00000.pt')

# candy_bounce (has camera + world + HDRI images)
print("=== candy_bounce (item 8) ===")
out = _reconstruct_scene(work_dir=work, cache_pt=cache, item_index=8)
sp = pathlib.Path(out.get('scene_json', ''))
sd = json.loads(sp.read_text())
print(f"  Meshes: {len(sd.get('objects', []))}")
print(f"  Cameras: {len(sd.get('cameras', []))}")
print(f"  Lights: {len(sd.get('lights', []))}")
print(f"  World: {'yes (nodes=' + str(len(sd.get('world', {}).get('nodes', []))) + ')' if sd.get('world') else 'no'}")
print(f"  Images: {list(sd.get('images', {}).keys())}")
for c in sd.get('cameras', []):
    cam = c.get('camera', {})
    print(f"  > Camera '{c['name']}': lens={cam.get('lens')} type={cam.get('type')} active={c.get('is_active')}")
    print(f"    loc={c.get('transforms', {}).get('location')} rot={c.get('transforms', {}).get('rotation_euler')}")
for img_name, img_info in sd.get('images', {}).items():
    has_thumb = bool(img_info.get('thumbnail'))
    size = img_info.get('thumbnail_bytes', 0)
    print(f"  > Image '{img_name}': thumb={has_thumb} ({size} bytes) cs={img_info.get('colorspace')}")
stdout_summary = out
print(f"  stdout: ok={stdout_summary.get('ok')} cams={stdout_summary.get('total_cameras')} lights={stdout_summary.get('total_lights')} images={stdout_summary.get('has_images')}")

print()
print("=== blendswap 31621 (item 1) ===")
out2 = _reconstruct_scene(work_dir=work, cache_pt=cache, item_index=1)
sp2 = pathlib.Path(out2.get('scene_json', ''))
sd2 = json.loads(sp2.read_text())
print(f"  Meshes: {len(sd2.get('objects', []))}")
print(f"  Lights: {len(sd2.get('lights', []))}")
for l in sd2.get('lights', []):
    ld = l.get('light', {})
    print(f"  > Light '{l['name']}': type={ld.get('type')} energy={ld.get('energy')} color={ld.get('color')}")
    print(f"    loc={l.get('transforms', {}).get('location')}")
print(f"  World bg_color in nodes: ", end="")
for nd in sd2.get('world', {}).get('nodes', []):
    if nd.get('type') == 'BACKGROUND':
        print(nd.get('inputs', {}).get('Color', '?'))
        break
print(f"  stdout: cams={out2.get('total_cameras')} lights={out2.get('total_lights')}")
