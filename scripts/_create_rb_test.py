"""Blender script: create a minimal rigid-body test scene and extract it.

Run as:
    blender --background --python scripts/_create_rb_test.py -- \
            --output data/processed/test_rb/

Produces:
  • One ACTIVE rigid-body sphere (mass=2.5, friction=0.5, restitution=0.8)
  • One PASSIVE ground plane (mass=0, friction=0.7)
  • A basic world background
  • A camera and sun lamp

After extraction, rigid_body data should appear in the JSON objects.
"""

import sys
import json
from pathlib import Path

# ── parse args (after "--") ───────────────────────────────────────────────────
_argv = sys.argv
if "--" in _argv:
    _argv = _argv[_argv.index("--") + 1:]
else:
    _argv = []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args(_argv)

import bpy  # noqa: E402  (only available inside Blender)

# ── clean slate ───────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)

# ── Ground plane (PASSIVE rigid body) ────────────────────────────────────────
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
bpy.ops.rigidbody.world_add()
bpy.ops.rigidbody.object_add()
ground.rigid_body.type = "PASSIVE"
ground.rigid_body.friction = 0.7
ground.rigid_body.restitution = 0.3

mat_ground = bpy.data.materials.new("GroundMat")
mat_ground.use_nodes = True
mat_ground.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.3, 0.6, 0.3, 1.0)
ground.data.materials.append(mat_ground)

# ── Bouncing sphere (ACTIVE rigid body) ──────────────────────────────────────
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0, 0, 3))
sphere = bpy.context.active_object
sphere.name = "BouncingSphere"
bpy.ops.rigidbody.object_add()
sphere.rigid_body.type = "ACTIVE"
sphere.rigid_body.mass = 2.5
sphere.rigid_body.friction = 0.5
sphere.rigid_body.restitution = 0.8
sphere.rigid_body.use_margin = True
sphere.rigid_body.collision_margin = 0.02

mat_sphere = bpy.data.materials.new("SphereMat")
mat_sphere.use_nodes = True
mat_sphere.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.9, 0.2, 0.2, 1.0)
mat_sphere.node_tree.nodes["Principled BSDF"].inputs["Metallic"].default_value = 0.1
sphere.data.materials.append(mat_sphere)

# ── Second cube (ACTIVE rigid body, with BEVEL modifier) ─────────────────────
bpy.ops.mesh.primitive_cube_add(size=0.8, location=(1.5, 0, 5))
cube = bpy.context.active_object
cube.name = "BouncingCube"
bpy.ops.rigidbody.object_add()
cube.rigid_body.type = "ACTIVE"
cube.rigid_body.mass = 1.0
cube.rigid_body.friction = 0.4
cube.rigid_body.restitution = 0.6

bevel = cube.modifiers.new("Bevel", "BEVEL")
bevel.width = 0.05
bevel.segments = 2

mat_cube = bpy.data.materials.new("CubeMat")
mat_cube.use_nodes = True
mat_cube.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.2, 0.4, 0.9, 1.0)
cube.data.materials.append(mat_cube)

# ── Camera ────────────────────────────────────────────────────────────────────
bpy.ops.object.camera_add(location=(7, -7, 5))
cam = bpy.context.active_object
cam.rotation_euler = (1.1, 0, 0.8)
bpy.context.scene.camera = cam

# ── Sun lamp ─────────────────────────────────────────────────────────────────
bpy.ops.object.light_add(type="SUN", location=(5, 5, 10))
sun = bpy.context.active_object
sun.data.energy = 3.0

# ── Save blend + extract ──────────────────────────────────────────────────────
out_dir = Path(args.output)
out_dir.mkdir(parents=True, exist_ok=True)

blend_out = out_dir / "rigid_body_test.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(blend_out))
print(f"Saved: {blend_out}")

# Now run the actual extractor on this saved file
extractor = Path(__file__).resolve().parent.parent / "processing" / "blend_extractor.py"
import subprocess
result = subprocess.run(
    [sys.executable, str(extractor), "--", "--input", str(blend_out), "--output", str(out_dir)],
    capture_output=True, text=True,
)
# Note: Can't call blend_extractor from within Blender (separate subprocess is fine
#        but blend_extractor also needs to be run under Blender).
# Instead just print the blend path; the parent script will extract it.
print(f"BLEND_PATH:{blend_out}")
