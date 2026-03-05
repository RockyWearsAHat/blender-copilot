"""Blender headless scene renderer — renders .blend files faithfully.

Called via:
    blender <file.blend> --background --python processing/blender_scene_render.py -- \
        --output renders/scene/ [--size 0] [--max-samples 0]

Philosophy (DO NOT VIOLATE):
  - The .blend file's own camera, lighting, materials, compositor, world and
    render engine are NEVER changed.  The scene renders exactly as its author
    intended.
  - We only ever touch QUALITY settings (resolution cap, sample cap) so the
    file renders fast enough for pipeline use.
  - Exception 1 (no camera): if the scene truly has no camera of any kind,
    we add a framing camera so the render is not empty.  This framing camera
    is a clean last-resort — we prefer the scene's own camera.
  - Exception 2 (materials-only): if the scene has zero mesh objects but
    has materials, we create a small showcase grid of primitives (one per
    material) so the render is meaningful for training.
    - Exception 3 (no lights): if there are no renderable lights in the scene,
        we add a single neutral key light so full renders do not collapse into
        flat gray/black outputs.

Outputs:
  {out}/{scene_id}_full.png          — 1 full quality render (native engine, camera, lighting)
  {out}/{scene_id}_view00.png … _view13.png
                                     — 14 Workbench/Material-Preview orbiting
                                       viewport shots (fast, no lighting needed)
  {out}/{scene_id}_manifest.json     — metadata
"""

import sys
import os
import json
import math
import argparse
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector, Matrix


# ── Same 14 orbiting angles used by blender_render.py ─────────────────────
VIEWPORT_VIEWS = [
    (  0.0,   0.0, "front"),
    (180.0,   0.0, "back"),
    ( 90.0,   0.0, "right"),
    (270.0,   0.0, "left"),
    (  0.0,  89.9, "top"),
    (  0.0, -89.9, "bottom"),
    ( 45.0,  45.0, "upper_front_right"),
    (135.0,  45.0, "upper_back_right"),
    (225.0,  45.0, "upper_back_left"),
    (315.0,  45.0, "upper_front_left"),
    ( 45.0, -20.0, "lower_front_right"),
    (135.0, -20.0, "lower_back_right"),
    (225.0, -20.0, "lower_back_left"),
    (315.0, -20.0, "lower_front_left"),
]

# ── Default shapes for material showcase (alternates) ─────────────────────
_SHOWCASE_SHAPES = ["UV_SPHERE", "CUBE", "CYLINDER", "UV_SPHERE", "CUBE",
                    "CYLINDER", "UV_SPHERE", "CUBE", "CYLINDER", "UV_SPHERE"]


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--output",        required=True)
    p.add_argument("--scene-id",      default=None)
    p.add_argument("--size",          type=int, default=0,
                   help="Max full-render dimension; 0 preserves native scene resolution")
    p.add_argument("--max-samples",   type=int, default=0,
                   help="Cycles/EEVEE sample cap; 0 preserves scene defaults")
    p.add_argument("--vp-size",       type=int, default=1024,
                   help="Viewport render resolution")
    p.add_argument("--skip-full",     action="store_true")
    p.add_argument("--skip-viewport", action="store_true")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Scene geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_visible_mesh_objects():
    """Return all enabled, visible mesh objects that have geometry."""
    objs = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.hide_render or obj.hide_viewport:
            continue
        if not obj.data or not obj.data.polygons:
            continue
        objs.append(obj)
    return objs


def get_scene_bbox(mesh_objs):
    """World-space bounding box focused on interesting content.

    Filters out large flat ground-plane / landscape geometry before computing
    the framing box so orbiting cameras zoom in on actual objects rather than
    including vast empty floor planes that shrink everything to a speck.
    Falls back to all objects if filtering removes everything.
    """
    if not mesh_objs:
        return Vector((0, 0, 0)), 1.0

    # Per-object extents in world space
    per_obj = []
    for obj in mesh_objs:
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        mn = Vector([min(v[i] for v in corners) for i in range(3)])
        mx = Vector([max(v[i] for v in corners) for i in range(3)])
        dx = max(mx.x - mn.x, 0.001)
        dy = max(mx.y - mn.y, 0.001)
        dz = max(mx.z - mn.z, 0.001)
        per_obj.append({"mn": mn, "mx": mx, "dx": dx, "dy": dy, "dz": dz,
                        "xy_area": dx * dy, "xy_max": max(dx, dy)})

    # Median XY footprint — used to detect abnormally large flat objects
    xy_areas = sorted(o["xy_area"] for o in per_obj)
    median_xy = xy_areas[len(xy_areas) // 2]

    def _is_ground_plane(o):
        """True for very flat objects that are also much larger than peers."""
        z_ratio    = o["dz"] / o["xy_max"]          # height / horizontal span
        area_ratio = o["xy_area"] / max(median_xy, 0.001)
        return z_ratio < 0.08 and area_ratio > 4.0

    useful = [o for o in per_obj if not _is_ground_plane(o)]
    if not useful:          # everything looked like a ground plane — use all
        useful = per_obj

    all_corners = []
    for o in useful:
        all_corners.extend([o["mn"], o["mx"]])
    mn = Vector([min(v[i] for v in all_corners) for i in range(3)])
    mx = Vector([max(v[i] for v in all_corners) for i in range(3)])
    center = (mn + mx) / 2
    extent = max((mx - mn).x, (mx - mn).y, (mx - mn).z, 0.01)
    return center, extent


# ─────────────────────────────────────────────────────────────────────────────
# Framing camera — ONLY added when scene has no camera at all
# ─────────────────────────────────────────────────────────────────────────────

def add_framing_camera(scene, mesh_objs, view=(30.0, 35.0)):
    """Add a camera that frames all mesh_objs.  Returns the camera object.
    ONLY called when the scene contains zero camera objects.
    """
    center, extent = get_scene_bbox(mesh_objs)
    dist = extent * 1.8
    az   = math.radians(view[0])
    el   = math.radians(view[1])
    x = center.x + dist * math.cos(el) * math.cos(az)
    y = center.y + dist * math.cos(el) * math.sin(az)
    z = center.z + dist * math.sin(el)

    cam_data            = bpy.data.cameras.new("AutoFramingCamera")
    cam_data.lens       = 50
    cam_data.clip_start = 0.01
    cam_data.clip_end   = 10000.0
    cam_obj = bpy.data.objects.new("AutoFramingCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    cam_obj.location = (x, y, z)
    direction = center - Vector((x, y, z))
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam_obj
    print(f"  [framing camera] placed at ({x:.2f}, {y:.2f}, {z:.2f}) → "
          f"center {[round(float(v), 2) for v in center]}")
    return cam_obj


def add_framing_light(scene, mesh_objs, camera_obj=None):
    """Add one neutral key light when scene has no renderable lights."""
    center, extent = get_scene_bbox(mesh_objs)
    extent = max(extent, 0.5)

    light_data = bpy.data.lights.new("AutoFramingKey", type="SUN")
    light_data.energy = 3.0
    if hasattr(light_data, "angle"):
        light_data.angle = math.radians(5.0)

    light_obj = bpy.data.objects.new("AutoFramingKey", light_data)
    scene.collection.objects.link(light_obj)

    if camera_obj is not None:
        light_obj.location = camera_obj.location
    else:
        light_obj.location = (
            center.x + extent * 1.5,
            center.y - extent * 1.5,
            center.z + extent * 2.0,
        )

    direction = center - Vector(light_obj.location)
    if direction.length > 1e-6:
        light_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    print(f"  [fallback light] added {light_obj.name} (SUN, energy={light_data.energy:.2f})")
    return light_obj


# ─────────────────────────────────────────────────────────────────────────────
# Materials-only showcase
# ─────────────────────────────────────────────────────────────────────────────

def create_material_showcase(scene):
    """When scene has NO meshes but HAS materials, create a grid of primitives.
    Returns the list of created objects.
    """
    mats = [m for m in bpy.data.materials if not m.name.startswith(".")]
    if not mats:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
        return [bpy.context.active_object]

    print(f"  [showcase] {len(mats)} materials → creating showcase geometry")
    created = []
    spacing = 2.8
    cols    = min(len(mats), 6)
    rows    = math.ceil(len(mats) / cols)

    for i, mat in enumerate(mats):
        row = i // cols
        col = i % cols
        x   = col * spacing - (cols - 1) * spacing / 2
        y   = row * spacing - (rows - 1) * spacing / 2

        shape = _SHOWCASE_SHAPES[i % len(_SHOWCASE_SHAPES)]
        if shape == "UV_SPHERE":
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=1.0, location=(x, y, 0), segments=32, ring_count=16)
        elif shape == "CUBE":
            bpy.ops.mesh.primitive_cube_add(size=1.8, location=(x, y, 0))
        else:
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.9, depth=1.8, location=(x, y, 0))

        obj = bpy.context.active_object
        if shape != "CUBE":
            for poly in obj.data.polygons:
                poly.use_smooth = True
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        obj.name = f"Showcase_{mat.name[:30]}"
        created.append(obj)
        print(f"    {obj.name}  ←  {mat.name}")

    return created


# ─────────────────────────────────────────────────────────────────────────────
# Quality caps — the ONLY scene settings we ever modify
# ─────────────────────────────────────────────────────────────────────────────

def cap_resolution(max_size):
    scene = bpy.context.scene
    pct   = scene.render.resolution_percentage / 100.0
    ax    = int(scene.render.resolution_x * pct)
    ay    = int(scene.render.resolution_y * pct)
    if max_size <= 0:
        scene.render.resolution_percentage = 100
        return ax, ay
    if ax <= max_size and ay <= max_size:
        scene.render.resolution_percentage = 100
        return ax, ay
    scale = max_size / max(ax, ay)
    scene.render.resolution_x          = max(int(ax * scale), 1)
    scene.render.resolution_y          = max(int(ay * scale), 1)
    scene.render.resolution_percentage = 100
    return scene.render.resolution_x, scene.render.resolution_y


def cap_samples(max_samples):
    scene  = bpy.context.scene
    engine = scene.render.engine
    scene.render.use_persistent_data = True

    if max_samples <= 0:
        max_samples = 0

    if engine == "CYCLES":
        if max_samples > 0 and hasattr(scene.cycles, "samples"):
            scene.cycles.samples = min(scene.cycles.samples, max_samples)
        scene.cycles.use_denoising = True
        if max_samples > 0 and hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold    = max(0.05, 1.0 / max_samples)
        if sys.platform == "darwin":
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.compute_device_type = "METAL"
                prefs.get_devices()
                scene.cycles.device = "GPU"
            except Exception:
                pass
    elif "EEVEE" in engine:
        e = scene.eevee
        if max_samples > 0 and hasattr(e, "taa_render_samples"):
            e.taa_render_samples = min(e.taa_render_samples, max_samples)


# ─────────────────────────────────────────────────────────────────────────────
# 14 Workbench orbiting renders
# ─────────────────────────────────────────────────────────────────────────────

def place_orbiting_cam(cam_obj, center, extent, azimuth_deg, elevation_deg):
    dist  = extent * 1.8
    az    = math.radians(azimuth_deg)
    el    = math.radians(elevation_deg)
    x     = center.x + dist * math.cos(el) * math.cos(az)
    y     = center.y + dist * math.cos(el) * math.sin(az)
    z     = center.z + dist * math.sin(el)
    cam_obj.location       = (x, y, z)
    direction              = center - Vector((x, y, z))
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_14_viewport_views(scene, out_dir, scene_id, vp_size):
    """Render 14 Workbench Material-Preview orbiting views.
    Creates & removes a temporary camera.  Fully restores original engine."""
    mesh_objs = get_visible_mesh_objects()
    if not mesh_objs:
        return []

    center, extent = get_scene_bbox(mesh_objs)

    # Save state
    orig_engine   = scene.render.engine
    orig_rx       = scene.render.resolution_x
    orig_ry       = scene.render.resolution_y
    orig_pct      = scene.render.resolution_percentage
    orig_filepath = scene.render.filepath
    orig_camera   = scene.camera

    # Temp camera
    cd  = bpy.data.cameras.new("_VP_TempCam")
    cd.lens, cd.clip_start, cd.clip_end = 50, 0.01, 10000.0
    cam = bpy.data.objects.new("_VP_TempCam", cd)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # Switch to Workbench
    for eng in ("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = eng
            break
        except Exception:
            continue

    scene.render.resolution_x          = vp_size
    scene.render.resolution_y          = vp_size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGBA"
    scene.render.film_transparent            = True

    if scene.render.engine == "BLENDER_WORKBENCH":
        try:
            sh = scene.display.shading
            sh.light, sh.color_type         = "STUDIO", "MATERIAL"
            sh.show_shadows                  = False
            sh.show_cavity                   = False
            sh.show_specular_highlight        = True
            sh.show_object_outline            = False
        except Exception as exc:
            print(f"  Workbench shading warning: {exc}")

    print(f"  [viewport] {scene.render.engine}  {vp_size}x{vp_size}  (Material Preview)")

    results = []
    for idx, (az, el, view_label) in enumerate(VIEWPORT_VIEWS):
        place_orbiting_cam(cam, center, extent, az, el)
        vp_file = str(out_dir / f"{scene_id}_view{idx:02d}.png")
        scene.render.filepath = vp_file
        bpy.ops.render.render(write_still=True)
        print(f"  [{idx+1:02d}/14] {view_label:<22} → {Path(vp_file).name}")
        results.append({
            "view_index": idx,
            "label":      view_label,
            "filename":   Path(vp_file).name,
            "filepath":   vp_file,
            "azimuth":    az,
            "elevation":  el,
        })

    # Restore
    scene.camera = orig_camera
    bpy.data.objects.remove(cam, do_unlink=True)
    bpy.data.cameras.remove(cd)
    try:
        scene.render.engine = orig_engine
    except Exception:
        pass
    scene.render.resolution_x          = orig_rx
    scene.render.resolution_y          = orig_ry
    scene.render.resolution_percentage = orig_pct
    scene.render.filepath               = orig_filepath

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_camera_info(camera_obj):
    cam  = camera_obj.data
    info = {
        "name":            camera_obj.name,
        "location":        list(camera_obj.location),
        "rotation_euler":  [math.degrees(r) for r in camera_obj.rotation_euler],
        "lens":            cam.lens,
        "sensor_width":    cam.sensor_width,
        "clip_start":      cam.clip_start,
        "clip_end":        cam.clip_end,
        "type":            cam.type,
        "auto_framing":    camera_obj.name == "AutoFramingCamera",
    }
    if cam.type == "ORTHO":
        info["ortho_scale"] = cam.ortho_scale
    if cam.dof.use_dof:
        info["dof"] = {
            "focus_distance": cam.dof.focus_distance,
            "aperture_fstop": cam.dof.aperture_fstop,
        }
    return info


def get_light_info():
    lights = []
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        light = obj.data
        info  = {
            "name":     obj.name,    "type":     light.type,
            "location": list(obj.location),
            "energy":   light.energy, "color":   list(light.color),
        }
        if light.type == "AREA":
            info["size"] = light.size
        lights.append(info)
    return lights


def get_world_info():
    world = bpy.context.scene.world
    if not world:
        return None
    info = {"name": world.name}
    if world.use_nodes and world.node_tree:
        for n in world.node_tree.nodes:
            if n.type == "BACKGROUND":
                try:
                    info["bg_color"]    = list(n.inputs["Color"].default_value)
                    info["bg_strength"] = n.inputs["Strength"].default_value
                except Exception:
                    pass
            if n.type == "TEX_ENVIRONMENT":
                info["has_hdri"] = True
    return info


def get_compositor_info():
    scene = bpy.context.scene
    if not scene.use_nodes or not scene.node_tree:
        return None
    types = [n.type for n in scene.node_tree.nodes]
    return {
        "node_count":          len(types),
        "node_types":          list(set(types)),
        "has_glare":           "GLARE" in types,
        "has_blur":            "BLUR" in types,
        "has_color_balance":   "COLORBALANCE" in types,
        "has_lens_distortion": "LENSDIST" in types,
    }


def get_scene_objects_summary():
    s = {"total": 0, "meshes": 0, "lights": 0, "cameras": 0,
         "empties": 0, "armatures": 0, "curves": 0, "other": 0}
    for obj in bpy.context.scene.objects:
        s["total"] += 1
        t = obj.type
        if   t == "MESH":      s["meshes"]    += 1
        elif t == "LIGHT":     s["lights"]    += 1
        elif t == "CAMERA":    s["cameras"]   += 1
        elif t == "EMPTY":     s["empties"]   += 1
        elif t == "ARMATURE":  s["armatures"] += 1
        elif t == "CURVE":     s["curves"]    += 1
        else:                  s["other"]     += 1
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    scene      = bpy.context.scene
    blend_path = bpy.data.filepath
    scene_id   = args.scene_id or Path(blend_path).stem

    print("=" * 60)
    print("Scene Render Pipeline")
    print(f"  Blend  : {blend_path}")
    print(f"  ID     : {scene_id}")
    print(f"  Engine : {scene.render.engine}")
    print(f"  Output : {args.output}")
    print("=" * 60)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Materials-only detection ────────────────────────────────────
    mesh_objs           = get_visible_mesh_objects()
    materials_showcase  = False

    if not mesh_objs:
        print("  No visible mesh objects.")
        all_mats = [m for m in bpy.data.materials if not m.name.startswith(".")]
        if all_mats:
            showcase_objs  = create_material_showcase(scene)
            mesh_objs      = showcase_objs
            materials_showcase = True
        else:
            print("  No meshes and no materials — nothing to render")
            manifest = {"scene_id": scene_id, "status": "empty_scene",
                        "blend_file": blend_path, "full_render": None, "renders": []}
            (out_dir / f"{scene_id}_manifest.json").write_text(
                json.dumps(manifest, indent=2))
            sys.exit(0)

    # ── Step 2: Camera ───────────────────────────────────────────────────────
    # Use the scene's own camera first.  Re-link if needed.  Only add a new
    # framing camera if there is genuinely no camera object anywhere in the scene.
    cam_obj = scene.camera
    if cam_obj is None:
        for obj in bpy.data.objects:
            if obj.type == "CAMERA" and obj.name in scene.objects:
                scene.camera = obj
                cam_obj = obj
                print(f"  Re-linked existing camera: {obj.name}")
                break

    if cam_obj is None:
        print("  No camera found — adding auto-framing camera")
        cam_obj = add_framing_camera(scene, mesh_objs)
    else:
        print(f"  Camera : {cam_obj.name}  (native, preserved)")

    # Ensure at least one renderable light for full render readability.
    render_lights = [
        obj for obj in scene.objects
        if obj.type == "LIGHT" and not obj.hide_render
    ]
    if not render_lights:
        add_framing_light(scene, mesh_objs, camera_obj=cam_obj)

    # ── Step 3: Full render ──────────────────────────────────────────────────
    full_render_info = None
    if not args.skip_full:
        res_x, res_y = cap_resolution(args.size)
        cap_samples(args.max_samples)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode  = "RGBA"

        print(f"  Full render: {res_x}x{res_y}  engine={scene.render.engine}"
              f"  max_samples={args.max_samples}")

        full_file = str(out_dir / f"{scene_id}_full.png")
        scene.render.filepath = full_file
        bpy.ops.render.render(write_still=True)
        print(f"  Written: {Path(full_file).name}")

        full_render_info = {
            "filename": Path(full_file).name,
            "filepath": full_file,
            "engine":   scene.render.engine,
            "width":    res_x,
            "height":   res_y,
            "samples":  args.max_samples,
        }

    # ── Step 4: 14 Workbench viewport renders ───────────────────────────────
    viewport_renders = []
    if not args.skip_viewport:
        viewport_renders = render_14_viewport_views(
            scene, out_dir, scene_id, args.vp_size)

    # ── Step 5: Manifest ─────────────────────────────────────────────────────
    manifest = {
        "scene_id":           scene_id,
        "status":             "success",
        "blend_file":         blend_path,
        "materials_showcase": materials_showcase,
        "full_render":        full_render_info,
        "renders":            viewport_renders,
        "camera":             get_camera_info(cam_obj),
        "lights":             get_light_info(),
        "world":              get_world_info(),
        "compositor":         get_compositor_info(),
        "scene_objects":      get_scene_objects_summary(),
        "frame":              scene.frame_current,
    }
    manifest_path = out_dir / f"{scene_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"  Manifest: {manifest_path}")
    print(f"  Done!  full={full_render_info is not None}  viewport={len(viewport_renders)}")


if __name__ == "__main__":
    main()
