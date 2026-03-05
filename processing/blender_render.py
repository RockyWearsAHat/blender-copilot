"""Blender headless renderer — high-quality multi-view renders of meshes.

Called via:
    blender --background --python processing/blender_render.py -- \
        --input mesh.json --output renders/

Produces:
  - 1 full-quality render  (EEVEE, up to 2560x1440, 128 samples, 3/4 overhead)
  - 14 viewport-style renders (Workbench Material Preview — near-instant, no
    lighting setup needed, shows exact material colors as in Blender's viewport):
      6 orthographic axis views: front, back, right, left, top, bottom
      8 diagonal views: 4 upper (45° elev) + 4 lower (-20° elev) at 45°/135°/225°/315°

Input JSON format:
    {
        "vertices": [[x, y, z], ...],
        "faces": [[v0, v1, v2], ...],
        "label": "optional label",
        "mesh_id": "optional_id",
        "materials": [...]  (optional, list of {name, base_color, roughness, metallic})
    }

Output:
    {output_dir}/{mesh_id}_full.png          — full quality render
    {output_dir}/{mesh_id}_view00.png        — front
    {output_dir}/{mesh_id}_view01.png        — back
    ...
    {output_dir}/{mesh_id}_view13.png        — lower diagonal
    {output_dir}/{mesh_id}_manifest.json
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


# ── 14 fixed viewport camera angles ──────────────────────────────────────
# (azimuth_deg, elevation_deg, label)
# Azimuth 0° = camera on +Y axis looking at -Y (front in Blender convention)
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

# Full render angle — classic 3/4 overhead
FULL_RENDER_VIEW = (30.0, 35.0, "full")

# Max resolution cap
MAX_WIDTH  = 2560
MAX_HEIGHT = 1440


def parse_args():
    """Parse arguments after Blender's -- separator."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Render multi-view images of a mesh using Blender"
    )
    parser.add_argument("--input",   required=True, help="Path to input mesh JSON")
    parser.add_argument("--output",  required=True, help="Output directory for renders")
    parser.add_argument("--width",   type=int, default=2560, help="Full render width  (capped at 2560)")
    parser.add_argument("--height",  type=int, default=1440, help="Full render height (capped at 1440)")
    parser.add_argument("--vp-width",  type=int, default=512,  help="Viewport render width (Workbench, near-instant)")
    parser.add_argument("--vp-height", type=int, default=512,  help="Viewport render height (Workbench, near-instant)")
    parser.add_argument("--samples",    type=int, default=128, help="Full render samples")
    parser.add_argument("--vp-samples", type=int, default=8,   help="Ignored for Workbench viewport renders")
    parser.add_argument("--engine", default="BLENDER_EEVEE_NEXT",
                        choices=["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"],
                        help="Render engine for full render")
    parser.add_argument("--skip-full",     action="store_true", help="Skip the full quality render")
    parser.add_argument("--skip-viewport", action="store_true", help="Skip the 14 viewport renders")
    return parser.parse_args(argv)


def clear_scene():
    """Remove all objects, lights, cameras from the scene."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials,
                  bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            block.remove(item)


def get_collection():
    return bpy.context.scene.collection


# ── Render engine setup ───────────────────────────────────────────────────

def _resolve_engine(requested):
    """Try to set engine; fall back gracefully."""
    scene = bpy.context.scene
    for candidate in (requested, "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = candidate
            return candidate
        except Exception:
            continue
    return scene.render.engine


def setup_renderer_full(engine, width, height, samples):
    """Configure for the full quality render."""
    scene = bpy.context.scene
    actual = _resolve_engine(engine)

    width  = min(width,  MAX_WIDTH)
    height = min(height, MAX_HEIGHT)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    if actual == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        if sys.platform == "darwin":
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.compute_device_type = "METAL"
                prefs.get_devices()
                scene.cycles.device = "GPU"
            except Exception:
                scene.cycles.device = "CPU"
    else:
        # EEVEE
        if hasattr(scene, "eevee"):
            e = scene.eevee
            for attr, val in [
                ("taa_render_samples", samples),
                ("use_gtao", True),
                ("gtao_distance", 0.2),
                ("use_bloom", False),
                ("use_ssr", True),
                ("use_ssr_refraction", False),
            ]:
                if hasattr(e, attr):
                    setattr(e, attr, val)
    print(f"  Full render engine: {actual}  {width}x{height}  {samples}smp")
    return actual


def setup_renderer_viewport(vp_width, vp_height):
    """Switch to Workbench (Material Preview) for near-instant viewport renders.

    Workbench is Blender's internal Material Preview engine — the same view
    you get when pressing Z→Material Preview in the viewport.  It uses
    built-in studio HDRI lighting, requires no scene lights, and renders
    each frame in milliseconds rather than seconds.
    """
    scene = bpy.context.scene

    # Use Workbench; fall back to EEVEE if unavailable (shouldn't happen in 3.6+)
    for eng in ("BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = eng
            actual = eng
            break
        except Exception:
            continue
    else:
        actual = scene.render.engine

    scene.render.resolution_x = vp_width
    scene.render.resolution_y = vp_height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode  = "RGBA"
    scene.render.film_transparent = True

    if actual == "BLENDER_WORKBENCH":
        shading = scene.display.shading
        # Material Preview: show actual material colours with studio HDRI
        try:
            shading.light             = "STUDIO"
            shading.color_type        = "MATERIAL"
            shading.show_shadows      = False   # no shadow = faster
            shading.show_cavity       = False
            shading.show_specular_highlight = True
            shading.show_object_outline     = False
        except Exception as exc:
            print(f"  Workbench shading config warning: {exc}")
    else:
        # EEVEE fallback — minimal samples
        if hasattr(scene, "eevee"):
            for attr, val in [("taa_render_samples", 4), ("use_bloom", False),
                              ("use_ssr", False), ("use_gtao", False)]:
                if hasattr(scene.eevee, attr):
                    setattr(scene.eevee, attr, val)

    print(f"  Viewport engine: {actual}  {vp_width}x{vp_height}  (Material Preview)")


# ── Lighting ──────────────────────────────────────────────────────────────

def setup_studio_lighting():
    """3-point studio lights + soft world background."""
    col = get_collection()

    lights = [
        ("KeyLight",  "AREA", 500,  3.0, (1.00, 0.97, 0.92), ( 3.0, -2.5,  4.5), (45, 0,  30)),
        ("FillLight", "AREA", 150,  5.0, (0.80, 0.88, 1.00), (-3.5, -1.0,  3.0), (55, 0, -40)),
        ("RimLight",  "AREA", 300,  2.0, (1.00, 1.00, 1.00), ( 0.0,  3.5,  3.5), (35, 0, 180)),
        ("BotFill",   "AREA",  40,  6.0, (0.90, 0.90, 1.00), ( 0.0,  0.0, -3.0), (90, 0,   0)),
    ]

    objs = []
    for name, ltype, energy, size, color, loc, rot_deg in lights:
        ld = bpy.data.lights.new(name=name, type=ltype)
        ld.energy = energy
        ld.size   = size
        ld.color  = color
        lo = bpy.data.objects.new(name, ld)
        col.objects.link(lo)
        lo.location = loc
        lo.rotation_euler = tuple(math.radians(d) for d in rot_deg)
        objs.append(lo)

    # Neutral warm-gray world
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()
    bg = world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value    = (0.12, 0.12, 0.13, 1.0)
    bg.inputs["Strength"].default_value = 0.6
    out = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(bg.outputs["Background"], out.inputs["Surface"])

    return objs


# ── Material ──────────────────────────────────────────────────────────────

# ── Material node reconstruction ─────────────────────────────────────────

_FALLBACK_COLORS = [
    (0.55, 0.55, 0.58, 1.0),  # neutral gray
    (0.70, 0.30, 0.20, 1.0),  # terracotta
    (0.25, 0.45, 0.65, 1.0),  # steel blue
    (0.35, 0.55, 0.30, 1.0),  # moss green
    (0.60, 0.50, 0.30, 1.0),  # warm tan
    (0.65, 0.60, 0.70, 1.0),  # lavender gray
    (0.40, 0.25, 0.15, 1.0),  # dark brown
    (0.70, 0.65, 0.55, 1.0),  # sand
]


def _build_material_from_nodes(mat_info: dict, name: str, images: dict) -> bpy.types.Material:
    """Recreate a Blender material from extracted node-tree JSON.

    Reconstructs the full Principled BSDF setup including texture nodes,
    mapping nodes, and links — exactly as exported by blend_extractor.
    Falls back to a solid Principled BSDF when no node data is present.

    Args:
        mat_info: dict from extracted JSON (has 'nodes', 'links', 'inputs' etc.)
        name:     material name
        images:   scene-level images dict keyed by image name (may have thumbnail data)
    """
    mat = bpy.data.materials.new(name=name[:60])
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    nodes_json = mat_info.get("nodes", [])
    links_json = mat_info.get("links", [])

    if not nodes_json:
        # No node data — build minimal Principled BSDF fallback
        _apply_principled_fallback(tree, mat_info)
        return mat

    created = {}  # name -> bpy node

    for nd in nodes_json:
        bl_id = nd.get("bl_idname") or nd.get("type", "")
        try:
            node = tree.nodes.new(type=bl_id)
        except Exception:
            # Unknown node type — skip but don't crash
            continue
        node.name = nd["name"]
        if "location" in nd:
            node.location = nd["location"]

        # Restore node-specific props
        if nd.get("type") == "MIX_RGB":
            node.blend_type = nd.get("blend_type", "MIX")
            node.use_clamp   = nd.get("use_clamp", False)
        elif nd.get("type") == "MATH":
            node.operation = nd.get("operation", "ADD")
        elif nd.get("type") == "TEX_IMAGE" and nd.get("image_name"):
            img_name = nd["image_name"]
            # Try to load image from scene images dict (has b64 thumbnail)
            img = bpy.data.images.get(img_name)
            if img is None and img_name in images:
                img_info = images[img_name]
                thumb = img_info.get("thumbnail")
                if thumb:
                    import base64, tempfile, os as _os
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    tmp.write(base64.b64decode(thumb))
                    tmp.close()
                    try:
                        img = bpy.data.images.load(tmp.name)
                        img.name = img_name
                    except Exception:
                        img = None
                    try:
                        _os.unlink(tmp.name)
                    except OSError:
                        pass
            if img:
                node.image = img
                cs = nd.get("image_colorspace", "sRGB")
                try:
                    node.image.colorspace_settings.name = cs
                except Exception:
                    pass
            if nd.get("interpolation"):
                try:
                    node.interpolation = nd["interpolation"]
                except Exception:
                    pass
            if nd.get("extension"):
                try:
                    node.extension = nd["extension"]
                except Exception:
                    pass

        # Restore input default values
        inputs_json = nd.get("inputs", {})
        for inp_name, val in inputs_json.items():
            if val == "LINKED":
                continue
            try:
                inp = node.inputs.get(inp_name)
                if inp is None:
                    continue
                if isinstance(val, list):
                    for i, v in enumerate(val):
                        try:
                            inp.default_value[i] = float(v)
                        except Exception:
                            pass
                else:
                    inp.default_value = float(val)
            except Exception:
                pass

        created[nd["name"]] = node

    # Restore links
    for lk in links_json:
        from_node = created.get(lk["from_node"])
        to_node   = created.get(lk["to_node"])
        if not from_node or not to_node:
            continue
        from_sock = from_node.outputs.get(lk["from_socket"])
        to_sock   = to_node.inputs.get(lk["to_socket"])
        if from_sock and to_sock:
            try:
                tree.links.new(from_sock, to_sock)
            except Exception:
                pass

    # Ensure there is at least one Material Output node
    out = tree.nodes.get("Material Output")
    if not out:
        out = tree.nodes.new("ShaderNodeOutputMaterial")

    return mat


def _apply_principled_fallback(tree, mat_info: dict, color_index: int = 0):
    """Add a Principled BSDF with best available color from mat_info."""
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    out  = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Extract Principled inputs from node JSON if present
    base_color = list(_FALLBACK_COLORS[color_index % len(_FALLBACK_COLORS)])
    roughness  = 0.45
    metallic   = 0.0

    for nd in mat_info.get("nodes", []):
        if nd.get("type") == "BSDF_PRINCIPLED":
            inp = nd.get("inputs", {})
            bc = inp.get("Base Color")
            if bc and isinstance(bc, list) and len(bc) >= 3:
                r, g, b = float(bc[0]), float(bc[1]), float(bc[2])
                # Skip pure Blender default gray (0.8, 0.8, 0.8) or empty (0.4, 0.4, 0.4)
                if not (0.38 < r < 0.42 and 0.38 < g < 0.42 and 0.38 < b < 0.42) and \
                   not (0.78 < r < 0.82 and 0.78 < g < 0.82 and 0.78 < b < 0.82):
                    alpha = float(bc[3]) if len(bc) > 3 else 1.0
                    base_color = [r, g, b, alpha]
            roughness = float(inp.get("Roughness", roughness))
            metallic  = float(inp.get("Metallic",  metallic))
            break

    bsdf.inputs["Base Color"].default_value = tuple(base_color)
    bsdf.inputs["Roughness"].default_value  = min(max(float(roughness), 0.0), 1.0)
    for inp_name in ("Metallic", "Specular IOR Level", "Specular"):
        inp = bsdf.inputs.get(inp_name)
        if inp:
            inp.default_value = float(metallic) if inp_name == "Metallic" else 0.5
            break


def build_materials(mat_infos: list, images: dict) -> list:
    """Build Blender materials from the extracted materials JSON list.

    Returns list of bpy.types.Material in slot order.
    """
    result = []
    for i, m in enumerate(mat_infos or []):
        if not isinstance(m, dict):
            result.append(_make_fallback_mat(i))
            continue
        name = m.get("name", f"Mat_{i}")
        if m.get("type") == "node_tree" and m.get("nodes"):
            mat = _build_material_from_nodes(m, name, images)
        else:
            # Simple material or missing nodes
            mat = bpy.data.materials.new(name=name[:60])
            mat.use_nodes = True
            _apply_principled_fallback(mat.node_tree, m, color_index=i)
        result.append(mat)

    if not result:
        result.append(_make_fallback_mat(0))
    return result


def _make_fallback_mat(index: int) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=f"FallbackMat_{index}")
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    _apply_principled_fallback(mat.node_tree, {}, color_index=index)
    return mat


def assign_materials_to_mesh(obj, mat_infos: list, face_material_indices: list,
                              uv_layers: dict, vertex_color_layers: dict,
                              images: dict):
    """Assign rebuilt materials + UV map + vertex colors to the mesh object."""
    mesh = obj.data
    mesh.materials.clear()

    mats = build_materials(mat_infos, images)
    for mat in mats:
        mesh.materials.append(mat)

    # Assign per-face material index
    if face_material_indices and len(face_material_indices) == len(mesh.polygons):
        max_slot = len(mats) - 1
        for poly, idx in zip(mesh.polygons, face_material_indices):
            poly.material_index = min(int(idx), max_slot)
    elif len(mats) > 1:
        # No face assignment data — leave all faces on slot 0
        pass

    # Restore UV layers
    for uv_name, coords in (uv_layers or {}).items():
        if len(coords) != len(mesh.loops):
            continue
        uv_layer = mesh.uv_layers.new(name=uv_name)
        for loop, uv in zip(mesh.loops, coords):
            uv_layer.data[loop.index].uv = uv

    # Restore vertex color layers
    vcol_src = vertex_color_layers or {}
    for vcol_name, colors in vcol_src.items():
        if len(colors) != len(mesh.loops):
            continue
        vcol = mesh.vertex_colors.new(name=vcol_name)
        for loop, c in zip(mesh.loops, colors):
            vcol.data[loop.index].color = c


# ── Mesh creation ─────────────────────────────────────────────────────────

def create_mesh_from_data(vertices, faces, name="RenderMesh",
                          face_smooth=None):
    """Build mesh. Uses per-face smooth flags from extracted data.

    If face_smooth list is provided, honours the original Blender smooth
    assignment per polygon.  Otherwise uses angle-based auto-smooth
    (30° threshold) which avoids the spiky-normal explosion caused by
    applying use_smooth=True blindly on meshes with inverted faces.
    """
    mesh = bpy.data.meshes.new(name)
    obj  = bpy.data.objects.new(name, mesh)
    get_collection().objects.link(obj)

    n = len(vertices)
    valid = [f for f in faces
             if len(f) >= 3 and all(0 <= vi < n for vi in f) and len(set(f)) == len(f)]
    mesh.from_pydata(vertices, [], valid)
    mesh.update()

    bm = bmesh.new()
    bm.from_mesh(mesh)
    degen = [f for f in bm.faces if f.calc_area() < 1e-9]
    if degen:
        bmesh.ops.delete(bm, geom=degen, context="FACES")
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0002)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    # ── Smooth shading ────────────────────────────────────────────────
    if face_smooth and len(face_smooth) == len(mesh.polygons):
        # Use original per-face smooth flags from extraction
        for poly, smooth in zip(mesh.polygons, face_smooth):
            poly.use_smooth = bool(smooth)
    else:
        # Angle-based auto-smooth: smooth faces whose shared edges are
        # within 30°. This is safe on ANY mesh topology — no spikes.
        # Set ALL polys smooth, then add EdgeSplit modifier at 30° to
        # restore sharp edges where normals diverge.
        for poly in mesh.polygons:
            poly.use_smooth = True
        # Add EdgeSplit modifier (Blender's standard auto-smooth mechanism)
        es = obj.modifiers.new(name="EdgeSplit", type="EDGE_SPLIT")
        es.split_angle = math.radians(30)
        es.use_edge_angle = True
        es.use_edge_sharp = True

    return obj


# ── Camera ────────────────────────────────────────────────────────────────

def make_camera(name="RenderCamera"):
    cd = bpy.data.cameras.new(name)
    cd.lens       = 50
    cd.clip_start = 0.01
    cd.clip_end   = 1000.0
    co = bpy.data.objects.new(name, cd)
    get_collection().objects.link(co)
    bpy.context.scene.camera = co
    return co


def place_camera(cam, center, extent, azimuth_deg, elevation_deg):
    """Position camera on a sphere around center and aim at center."""
    dist = extent * 2.8
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    cx, cy, cz = center
    x = cx + dist * math.cos(el) * math.cos(az)
    y = cy + dist * math.cos(el) * math.sin(az)
    z = cz + dist * math.sin(el)
    cam.location = (x, y, z)
    direction = Vector((cx, cy, cz)) - Vector((x, y, z))
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return (x, y, z)


def get_mesh_bounds(obj):
    """Return (center_vector, max_extent_scalar) from world-space bounding box."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector([min(v[i] for v in corners) for i in range(3)])
    mx = Vector([max(v[i] for v in corners) for i in range(3)])
    center = (mn + mx) / 2
    dims   = mx - mn
    extent = max(dims.x, dims.y, dims.z, 0.01)
    return center, extent


# ── Manifest ──────────────────────────────────────────────────────────────

def write_manifest(output_dir, mesh_id, full_render, viewport_renders,
                   label, n_verts, n_faces, args):
    manifest = {
        "mesh_id":    mesh_id,
        "label":      label,
        "n_vertices": n_verts,
        "n_faces":    n_faces,
        "full_render": full_render,
        "renders":     viewport_renders,
        "render_config": {
            "full_width":   min(args.width,  MAX_WIDTH),
            "full_height":  min(args.height, MAX_HEIGHT),
            "full_samples": args.samples,
            "vp_width":     args.vp_width,
            "vp_height":    args.vp_height,
            "vp_samples":   args.vp_samples,
            "engine":       args.engine,
        },
    }
    p = Path(output_dir) / f"{mesh_id}_manifest.json"
    with open(p, "w") as f:
        json.dump(manifest, f, indent=2)
    return str(p)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("Blender Multi-View Render Pipeline")
    print(f"  Input:      {args.input}")
    print(f"  Output:     {args.output}")
    print(f"  Full size:  {args.width}x{args.height} @ {args.samples}smp  [EEVEE]")
    print(f"  VP size:    {args.vp_width}x{args.vp_height}  [Workbench/Material-Preview, near-instant]")

    # ── Load mesh JSON ────────────────────────────────────────────────────
    with open(args.input) as f:
        data = json.load(f)

    # Support both flat format {"vertices":[], "faces":[]} and
    # scene-level format {"objects":[{"mesh":{"vertices":[], "faces":[]}}]}
    if "objects" in data and not data.get("vertices"):
        # Scene-level: merge all mesh objects, picking the largest by face count
        objects = data["objects"]
        mesh_objects = [o for o in objects
                        if o.get("type") == "MESH" and o.get("mesh", {}).get("faces")]
        if not mesh_objects:
            mesh_objects = [o for o in objects if o.get("mesh", {}).get("faces")]
        if mesh_objects:
            # Use the largest mesh for rendering
            best = max(mesh_objects, key=lambda o: len(o["mesh"]["faces"]))
            mesh = best["mesh"]
            vertices  = mesh.get("vertices", [])
            faces     = mesh.get("faces", [])
            materials = best.get("materials", data.get("materials"))
        else:
            vertices, faces, materials = [], [], None
        label   = data.get("label", "")
        mesh_id = data.get("mesh_id", Path(args.input).stem)
    else:
        vertices  = data.get("vertices", [])
        faces     = data.get("faces",    [])
        label     = data.get("label",    "")
        mesh_id   = data.get("mesh_id",  Path(args.input).stem)
        materials = data.get("materials", None)

    if not vertices or not faces:
        print("ERROR: No mesh data in input file")
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{mesh_id}_manifest.json", "w") as f:
            json.dump({"mesh_id": mesh_id, "error": "No mesh data", "renders": []}, f)
        sys.exit(1)

    print(f"  Mesh:  {len(vertices)} vertices, {len(faces)} faces")
    print(f"  Label: {label[:60]}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build scene ───────────────────────────────────────────────────────
    clear_scene()
    # Studio lights only needed for the full EEVEE render.
    # Workbench (Material Preview) viewport renders use built-in HDRI — no
    # scene lights required.  We add lights here so they're ready if the
    # full render runs first; Workbench simply ignores them.
    if not args.skip_full:
        setup_studio_lighting()

    face_material_indices = data.get("face_material_indices", [])
    uv_layers             = data.get("uv_layers", {})
    vertex_color_layers   = data.get("vertex_color_layers", {})
    face_smooth           = data.get("face_smooth", [])
    images                = data.get("images", {})

    obj = create_mesh_from_data(vertices, faces, name=mesh_id[:60],
                                face_smooth=face_smooth)
    assign_materials_to_mesh(obj, materials, face_material_indices,
                             uv_layers, vertex_color_layers, images)

    center, extent = get_mesh_bounds(obj)
    cam = make_camera()

    full_render_info  = None
    viewport_renders  = []

    # ── 1. Full quality render ────────────────────────────────────────────
    if not args.skip_full:
        print(f"\n  [FULL RENDER] {args.width}x{args.height} @ {args.samples}smp")
        setup_renderer_full(args.engine, args.width, args.height, args.samples)

        pos = place_camera(cam, center, extent,
                           FULL_RENDER_VIEW[0], FULL_RENDER_VIEW[1])

        full_filename = f"{mesh_id}_full.png"
        full_filepath = str(out_dir / full_filename)
        bpy.context.scene.render.filepath = full_filepath
        bpy.ops.render.render(write_still=True)
        print(f"  Written: {full_filename}")

        full_render_info = {
            "filename":        full_filename,
            "filepath":        full_filepath,
            "width":           min(args.width,  MAX_WIDTH),
            "height":          min(args.height, MAX_HEIGHT),
            "samples":         args.samples,
            "azimuth":         FULL_RENDER_VIEW[0],
            "elevation":       FULL_RENDER_VIEW[1],
            "camera_position": list(pos),
            "camera_target":   list(center),
        }

    # ── 2. 14 viewport-style renders (Workbench Material Preview) ────────
    if not args.skip_viewport:
        print(f"\n  [VIEWPORT RENDERS] {args.vp_width}x{args.vp_height}  Workbench/Material-Preview")
        setup_renderer_viewport(args.vp_width, args.vp_height)

        for idx, (az, el, view_label) in enumerate(VIEWPORT_VIEWS):
            pos = place_camera(cam, center, extent, az, el)

            vp_filename = f"{mesh_id}_view{idx:02d}.png"
            vp_filepath = str(out_dir / vp_filename)
            bpy.context.scene.render.filepath = vp_filepath
            bpy.ops.render.render(write_still=True)
            print(f"  [{idx+1:02d}/14] {view_label:<22} → {vp_filename}")

            viewport_renders.append({
                "view_index":      idx,
                "label":           view_label,
                "filename":        vp_filename,
                "filepath":        vp_filepath,
                "azimuth":         az,
                "elevation":       el,
                "camera_position": list(pos),
                "camera_target":   list(center),
            })

    # ── Write manifest ────────────────────────────────────────────────────
    manifest_path = write_manifest(
        str(out_dir), mesh_id, full_render_info, viewport_renders,
        label, len(vertices), len(faces), args,
    )
    print(f"\n  Manifest: {manifest_path}")
    print(f"  Done! full={full_render_info is not None}, viewport={len(viewport_renders)} views")


if __name__ == "__main__":
    main()
