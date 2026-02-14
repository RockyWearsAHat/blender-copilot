"""
Blender AI Copilot — Professional 3D Modeling Tool Library.

Comprehensive helper functions for AI-generated Blender Python code.
All functions are pre-imported into the AI code execution namespace.
The AI calls these directly — no imports needed.

Categories:
  • Object creation (box, cylinder, sphere, cone, plane, text, curve, empty)
  • BMesh modeling (extrude, inset, loop cut, bridge, dissolve, knife, bevel)
  • Architecture (wall, floor, stairs, roof)
  • Materials (PBR, glass, emission, quick-assign, texture nodes)
  • Boolean operations (cut, join, intersect)
  • Modifiers (bevel, mirror, array, solidify, subdivision, shrinkwrap,
    lattice, weighted normals, skin, remesh, decimate, displace, curve)
  • Curve tools (bezier, nurbs, path, loft)
  • UV tools (smart project, cube project, unwrap, scale islands)
  • Scene setup (lighting, camera, world, HDRI)
  • Transforms (move, rotate, scale)
  • Mesh cleanup (merge by distance, recalc normals, tris-to-quads)
  • Collections & organization
  • Utilities (get, delete, duplicate, select, shade, edge crease/sharp)
"""

import bpy  # type: ignore
import bmesh  # type: ignore
import math
import os
from mathutils import Vector, Matrix, Euler  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# Object Creation — geometry centered at location
# ═══════════════════════════════════════════════════════════════════════════

def create_box(name="Box", width=2, depth=2, height=2, location=(0, 0, 0)):
    """Create a box centered at *location*.

    *width*  = size along X axis.
    *depth*  = size along Y axis.
    *height* = size along Z axis.
    The box has 8 verts and 6 faces (a simple cube).
    For shapes needing more topology, use create_mesh() with explicit verts/faces.
    """
    w, d, h = width / 2, depth / 2, height / 2
    verts = [
        (-w, -d, -h), (w, -d, -h), (w, d, -h), (-w, d, -h),
        (-w, -d,  h), (w, -d,  h), (w, d,  h), (-w, d,  h),
    ]
    faces = [
        (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
        (2, 6, 7, 3), (0, 3, 7, 4), (1, 5, 6, 2),
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def create_plane(name="Plane", width=10, depth=10, location=(0, 0, 0)):
    """Create a flat plane centered at *location*."""
    w, d = width / 2, depth / 2
    verts = [(-w, -d, 0), (w, -d, 0), (w, d, 0), (-w, d, 0)]
    faces = [(0, 1, 2, 3)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def create_cylinder(name="Cylinder", radius=1, depth=2,
                    location=(0, 0, 0), vertices=32):
    """Create a cylinder centered at *location*.

    The cylinder's axis is along **Z** by default (standing upright).
    *radius* = XY cross-section radius.
    *depth*  = total height along Z.
    To make it lie on its side (axis along Y), call rotate_deg(obj, x=90).
    To make it lie along X, call rotate_deg(obj, z=90) then rotate_deg(obj, x=90).
    """
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False,
                          segments=vertices, radius1=radius,
                          radius2=radius, depth=depth)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def create_sphere(name="Sphere", radius=1, location=(0, 0, 0),
                  segments=32, rings=16):
    """Create a UV sphere centered at *location*."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments,
                              v_segments=rings, radius=radius)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def create_cone(name="Cone", radius1=1, radius2=0, depth=2,
                location=(0, 0, 0), vertices=32):
    """Create a cone centered at *location*. radius2=0 for a point."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=True,
                          segments=vertices, radius1=radius1,
                          radius2=radius2, depth=depth)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def create_torus(name="Torus", major_radius=1, minor_radius=0.25,
                 location=(0, 0, 0), major_segments=48, minor_segments=12):
    """Create a torus centered at *location*."""
    bm = bmesh.new()
    # Build torus manually via revolution
    for i in range(major_segments):
        angle_major = 2 * math.pi * i / major_segments
        cx = major_radius * math.cos(angle_major)
        cy = major_radius * math.sin(angle_major)
        for j in range(minor_segments):
            angle_minor = 2 * math.pi * j / minor_segments
            r = major_radius + minor_radius * math.cos(angle_minor)
            x = r * math.cos(angle_major)
            y = r * math.sin(angle_major)
            z = minor_radius * math.sin(angle_minor)
            bm.verts.new((x, y, z))
    bm.verts.ensure_lookup_table()
    for i in range(major_segments):
        for j in range(minor_segments):
            v1 = i * minor_segments + j
            v2 = i * minor_segments + (j + 1) % minor_segments
            v3 = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
            v4 = ((i + 1) % major_segments) * minor_segments + j
            bm.faces.new([bm.verts[v1], bm.verts[v2], bm.verts[v3], bm.verts[v4]])
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def create_text_obj(name="Text", text="Hello", size=1, location=(0, 0, 0),
                    extrude=0.05, font=None):
    """Create a 3D text object at *location*."""
    curve = bpy.data.curves.new(name, 'FONT')
    curve.body = text
    curve.size = size
    curve.extrude = extrude
    curve.align_x = 'CENTER'
    if font:
        curve.font = font
    obj = bpy.data.objects.new(name, curve)
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def create_empty(name="Empty", location=(0, 0, 0), display_size=1):
    """Create an empty object (useful as a parent / target)."""
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_size = display_size
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# Materials
# ═══════════════════════════════════════════════════════════════════════════


def find_material(name):
    """Find an existing material by exact name. Returns None if not found."""
    return bpy.data.materials.get(name)


def find_similar_material(color=None, roughness=None, metallic=None,
                          tolerance=0.1):
    """Find an existing material with similar properties.
    Returns the first match or None.  Checks Principled BSDF colour,
    roughness, and metallic within *tolerance*.
    """
    for mat in bpy.data.materials:
        if not mat.use_nodes or not mat.node_tree:
            continue
        for node in mat.node_tree.nodes:
            if node.type != 'BSDF_PRINCIPLED':
                continue
            if color is not None:
                bc = node.inputs.get("Base Color")
                if bc and hasattr(bc, 'default_value'):
                    c = bc.default_value
                    if (abs(c[0] - color[0]) > tolerance or
                            abs(c[1] - color[1]) > tolerance or
                            abs(c[2] - color[2]) > tolerance):
                        continue
            if roughness is not None:
                r = node.inputs.get("Roughness")
                if r and hasattr(r, 'default_value'):
                    if abs(r.default_value - roughness) > tolerance:
                        continue
            if metallic is not None:
                m = node.inputs.get("Metallic")
                if m and hasattr(m, 'default_value'):
                    if abs(m.default_value - metallic) > tolerance:
                        continue
            return mat
    return None


def get_material_inventory():
    """Return a summary string of all materials in the scene with details."""
    if not bpy.data.materials:
        return "No materials in scene."
    lines = []
    for mat in bpy.data.materials:
        users = [o.name for o in bpy.data.objects
                 if o.data and hasattr(o.data, 'materials') and mat.name in
                 [m.name for m in o.data.materials if m]]
        info = '"%s"' % mat.name
        if mat.use_nodes and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bc = node.inputs.get("Base Color")
                    if bc and hasattr(bc, 'default_value'):
                        c = bc.default_value
                        info += " color=(%.2f,%.2f,%.2f)" % (c[0], c[1], c[2])
                    r = node.inputs.get("Roughness")
                    if r and hasattr(r, 'default_value'):
                        info += " rough=%.2f" % r.default_value
                    m = node.inputs.get("Metallic")
                    if m and hasattr(m, 'default_value'):
                        info += " metal=%.2f" % m.default_value
                    break
        if users:
            info += " → used by: %s" % ", ".join(users[:5])
        else:
            info += " → UNUSED"
        lines.append("  %s" % info)
    return "\n".join(lines)


def quick_material(name="Material", color=(0.8, 0.8, 0.8),
                   roughness=0.5, metallic=0.0):
    """Create or reuse a Principled BSDF material. Reuses an existing material
    with the same name if it exists, or one with similar colour/properties."""
    # Check for exact name match first
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    # Check for similar material
    similar = find_similar_material(color, roughness, metallic, tolerance=0.05)
    if similar:
        return similar
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (
            color[0], color[1], color[2], 1.0)
        bsdf.inputs['Roughness'].default_value = roughness
        bsdf.inputs['Metallic'].default_value = metallic
    return mat


def glass_material(name="Glass", color=(0.9, 0.95, 1.0),
                   roughness=0.0, ior=1.5):
    """Create a glass material with transmission."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (
            color[0], color[1], color[2], 1.0)
        bsdf.inputs['Roughness'].default_value = roughness
        if 'Transmission Weight' in bsdf.inputs:
            bsdf.inputs['Transmission Weight'].default_value = 0.95
        elif 'Transmission' in bsdf.inputs:
            bsdf.inputs['Transmission'].default_value = 0.95
        if 'IOR' in bsdf.inputs:
            bsdf.inputs['IOR'].default_value = ior
    return mat


def emission_material(name="Emission", color=(1, 1, 1), strength=10):
    """Create an emissive/light material."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    emit = nodes.new('ShaderNodeEmission')
    emit.inputs['Color'].default_value = (color[0], color[1], color[2], 1.0)
    emit.inputs['Strength'].default_value = strength
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(emit.outputs['Emission'], output.inputs['Surface'])
    return mat


def assign_material(obj, mat):
    """Assign a material to an object (replaces existing materials)."""
    if obj and obj.data:
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def set_color(obj, r, g, b, roughness=0.5, metallic=0.0):
    """Quick one-liner: create/reuse and assign a colored material to an object."""
    mat = quick_material(obj.name + "_Mat", (r, g, b), roughness, metallic)
    assign_material(obj, mat)


# ═══════════════════════════════════════════════════════════════════════════
# Boolean Operations
# ═══════════════════════════════════════════════════════════════════════════

def _boolean_op(target, tool, operation, delete_tool=True, solver='EXACT'):
    """Internal: perform a boolean and validate the result.

    Uses EXACT solver by default (more reliable than FAST for
    clean manifold geometry). If the boolean results in 0 faces,
    it is automatically rolled back and a RuntimeError is raised
    with guidance on what likely went wrong (e.g. cutter too large).
    """
    # Snapshot face count before the boolean
    pre_faces = len(target.data.polygons)

    # Back up mesh data in case the boolean trashes it
    backup_mesh = target.data.copy()

    mod = target.modifiers.new("Bool", 'BOOLEAN')
    mod.operation = operation
    mod.object = tool
    mod.solver = solver
    tool.hide_set(True)
    tool.hide_render = True

    applied = False
    try:
        with bpy.context.temp_override(object=target):
            bpy.ops.object.modifier_apply(modifier=mod.name)
        applied = True
    except Exception:
        # If modifier_apply fails, remove the modifier
        if mod.name in target.modifiers:
            target.modifiers.remove(mod)

    # Validate: did the boolean destroy the mesh?
    post_faces = len(target.data.polygons)
    if applied and post_faces == 0 and pre_faces > 0:
        # Boolean trashed the geometry — roll back
        old_mesh = target.data
        target.data = backup_mesh
        bpy.data.meshes.remove(old_mesh)
        if delete_tool:
            bpy.data.objects.remove(tool, do_unlink=True)
        raise RuntimeError(
            "Boolean %s destroyed all faces — operation rolled back. "
            "Likely cause: the cutter was larger than the target on too "
            "many axes. Make the cutter SMALLER (only as deep as the "
            "wall thickness + small margin, and only as wide/tall as "
            "the hole you want)." % operation)

    # Clean up backup
    if backup_mesh and backup_mesh.users == 0:
        bpy.data.meshes.remove(backup_mesh)

    if delete_tool:
        bpy.data.objects.remove(tool, do_unlink=True)


def boolean_cut(target, cutter, delete_cutter=True):
    """Subtract *cutter* volume from *target* — removes the overlapping region.

    The *cutter* defines a shape to be "scooped out" of *target*.
    Everywhere the cutter overlaps the target, material is removed.
    Where they don't overlap, the target is unchanged.

    CRITICAL — the cutter must:
    • **Partially overlap** the target. If the cutter is larger than the
      target on any axis, it will cut through the entire wall and may
      leave nothing. Size the cutter to be SMALLER than the target in
      every dimension except the one you want to cut through.
    • **Be positioned where you want the hole.** The cutter's center
      should be at the center of the desired hole, not at the target's
      center.
    • **Have enough depth to go through the target wall.** E.g. if the
      target wall is 0.2m thick, the cutter needs depth > 0.2m — but
      NOT so large it extends beyond the other walls.

    Uses the EXACT solver. Auto-rolls back + raises error if 0 faces remain.

    Example — cutting a wheel arch:
        wheel_b = get_bounds(wheel)
        # Cutter radius slightly bigger than wheel, depth just enough to
        # go through the body side wall (~0.3m buffer), centered on wheel
        cutter = create_cylinder("Arch", radius=wheel_b.width/2 * 1.15,
                                 depth=0.5, location=(wheel_b.center_x,
                                 wheel_b.center_y, wheel_b.center_z))
        rotate_deg(cutter, x=90)  # align cylinder axis to Y
        boolean_cut(body, cutter)
    """
    _boolean_op(target, cutter, 'DIFFERENCE', delete_cutter)


def boolean_join(target, tool, delete_tool=True):
    """Merge *tool* volume into *target* — fuses two shapes into one solid.

    The result is the combined outer shell of both objects.
    The two objects should partially overlap for a clean merge.
    Uses the EXACT solver. Auto-rolls back if result has 0 faces.
    """
    _boolean_op(target, tool, 'UNION', delete_tool)


def boolean_intersect(target, tool, delete_tool=True):
    """Keep ONLY the volume where *target* and *tool* overlap.

    Everything outside the intersection is removed.
    Both objects must actually overlap or the result will be empty.
    Uses the EXACT solver. Auto-rolls back if result has 0 faces.
    """
    _boolean_op(target, tool, 'INTERSECT', delete_tool)


# ═══════════════════════════════════════════════════════════════════════════
# Modifiers
# ═══════════════════════════════════════════════════════════════════════════

def bevel(obj, width=0.02, segments=2):
    """Add a bevel modifier for softer edges."""
    mod = obj.modifiers.new("Bevel", 'BEVEL')
    mod.width = width
    mod.segments = segments
    return mod


def mirror(obj, axis='X', clipping=True, merge_threshold=0.001):
    """Add a mirror modifier. axis='X', 'Y', or 'Z'.

    *clipping*: if True, prevents vertices from crossing the mirror plane.
                Essential for subdivision surface modeling — keeps the
                center seam welded together. Default True.
    *merge_threshold*: merge verts within this distance of the mirror plane.

    Professional workflow: model ONE half of a symmetric object with
    mirror(obj, clipping=True), then apply_modifiers() when done.
    """
    mod = obj.modifiers.new("Mirror", 'MIRROR')
    mod.use_axis = [a in axis.upper() for a in ('X', 'Y', 'Z')]
    mod.use_clip = clipping
    mod.merge_threshold = merge_threshold
    return mod


def array(obj, count=5, offset=(2, 0, 0)):
    """Add an array modifier with *count* copies and *offset* distance."""
    mod = obj.modifiers.new("Array", 'ARRAY')
    mod.count = count
    mod.use_relative_offset = False
    mod.use_constant_offset = True
    mod.constant_offset_displace = Vector(offset)
    return mod


def solidify(obj, thickness=0.1, offset=-1):
    """Add a solidify modifier to give thickness to a surface."""
    mod = obj.modifiers.new("Solidify", 'SOLIDIFY')
    mod.thickness = thickness
    mod.offset = offset
    return mod


def subsurf(obj, levels=2, render_levels=None):
    """Add a subdivision surface modifier."""
    mod = obj.modifiers.new("Subsurf", 'SUBSURF')
    mod.levels = levels
    mod.render_levels = render_levels or levels
    return mod


def apply_modifiers(obj):
    """Apply (bake) all modifiers on an object into the mesh data.

    After this, the modifiers are gone and the mesh reflects their effect.
    Call this BEFORE boolean operations if the object has unapplied modifiers
    (like bevel, mirror, array) that need to become real geometry first.

    Note: applying a bevel on a simple 8-vert box adds very few verts.
    If you need more topology for booleans to work cleanly, use
    subdivide_mesh() to add edge loops first.
    """
    try:
        with bpy.context.temp_override(object=obj):
            for mod in list(obj.modifiers):
                bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Scene Setup
# ═══════════════════════════════════════════════════════════════════════════

def clear_scene():
    """Remove ALL objects, orphan meshes, and orphan materials."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    for cam in list(bpy.data.cameras):
        if cam.users == 0:
            bpy.data.cameras.remove(cam)
    for light in list(bpy.data.lights):
        if light.users == 0:
            bpy.data.lights.remove(light)
    for curve in list(bpy.data.curves):
        if curve.users == 0:
            bpy.data.curves.remove(curve)


def setup_sun(energy=5, rotation_deg=(50, 10, -30), color=(1, 0.95, 0.9)):
    """Add a sun lamp. *rotation_deg* is (X, Y, Z) in degrees."""
    sun_data = bpy.data.lights.new("Sun", 'SUN')
    sun_data.energy = energy
    sun_data.color = color[:3]
    sun_data.angle = math.radians(1.5)
    sun_obj = bpy.data.objects.new("Sun", sun_data)
    sun_obj.rotation_euler = tuple(math.radians(d) for d in rotation_deg)
    bpy.context.scene.collection.objects.link(sun_obj)
    return sun_obj


def setup_point_light(location=(0, 0, 5), energy=1000, color=(1, 1, 1),
                      radius=0.1):
    """Add a point light at *location*."""
    light_data = bpy.data.lights.new("PointLight", 'POINT')
    light_data.energy = energy
    light_data.color = color[:3]
    light_data.shadow_soft_size = radius
    light_obj = bpy.data.objects.new("PointLight", light_data)
    light_obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(light_obj)
    return light_obj


def setup_area_light(location=(0, 0, 5), energy=500, size=2,
                     color=(1, 1, 1)):
    """Add an area light at *location*."""
    light_data = bpy.data.lights.new("AreaLight", 'AREA')
    light_data.energy = energy
    light_data.color = color[:3]
    light_data.size = size
    light_obj = bpy.data.objects.new("AreaLight", light_data)
    light_obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(light_obj)
    return light_obj


def setup_spot_light(location=(0, 0, 5), energy=1000, spot_size_deg=45,
                     color=(1, 1, 1)):
    """Add a spot light at *location*."""
    light_data = bpy.data.lights.new("SpotLight", 'SPOT')
    light_data.energy = energy
    light_data.color = color[:3]
    light_data.spot_size = math.radians(spot_size_deg)
    light_obj = bpy.data.objects.new("SpotLight", light_data)
    light_obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(light_obj)
    return light_obj


def setup_camera(location=(15, -15, 10), look_at=(0, 0, 3), lens=35):
    """Add a camera pointed at *look_at*."""
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = lens
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    cam_obj.location = Vector(location)
    direction = Vector(look_at) - Vector(location)
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    return cam_obj


def setup_world(color=(0.05, 0.08, 0.12), strength=1.0):
    """Set the world background color."""
    scene = bpy.context.scene
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new('ShaderNodeBackground')
    bg.inputs[0].default_value = (color[0], color[1], color[2], 1.0)
    bg.inputs[1].default_value = strength
    out = nodes.new('ShaderNodeOutputWorld')
    out.location = (300, 0)
    links.new(bg.outputs['Background'], out.inputs['Surface'])


def frame_all():
    """Frame all objects in the 3D viewport."""
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    with bpy.context.temp_override(area=area, region=region):
                        bpy.ops.view3d.view_all()
                    return


# ═══════════════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════════════

def move_to(obj, x=0, y=0, z=0):
    """Set object world location (absolute, not relative)."""
    obj.location = (x, y, z)


def rotate_deg(obj, x=0, y=0, z=0):
    """Set object rotation in degrees (absolute, replaces any existing rotation).

    This SETS the rotation — it does NOT add to the current rotation.
    E.g. rotate_deg(cyl, x=90) makes the cylinder lie on its side
    (Z-axis cylinder becomes Y-axis). Calling rotate_deg(cyl, x=90)
    again does NOT rotate it another 90° — it stays at 90°.
    """
    obj.rotation_euler = (math.radians(x), math.radians(y), math.radians(z))


def scale_to(obj, x=1, y=1, z=1):
    """Set object scale."""
    obj.scale = (x, y, z)


def set_parent(child, parent, keep_transform=True):
    """Parent *child* to *parent*."""
    child.parent = parent
    if keep_transform:
        child.matrix_parent_inverse = parent.matrix_world.inverted()


# ═══════════════════════════════════════════════════════════════════════════
# Collections & Organization
# ═══════════════════════════════════════════════════════════════════════════

def new_collection(name):
    """Get or create a collection."""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def link_to_collection(obj, col_name):
    """Move an object to a named collection (creates it if needed)."""
    col = new_collection(col_name)
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def get(name):
    """Get an object by name.  Returns None if not found."""
    return bpy.data.objects.get(name)


def delete(obj_or_name):
    """Delete an object by reference or name."""
    if isinstance(obj_or_name, str):
        obj = bpy.data.objects.get(obj_or_name)
    else:
        obj = obj_or_name
    if obj:
        bpy.data.objects.remove(obj, do_unlink=True)


def duplicate(obj, offset=(0, 0, 0)):
    """Duplicate an object with optional position offset.  Returns the new object."""
    new_obj = obj.copy()
    if obj.data:
        new_obj.data = obj.data.copy()
    new_obj.location = (
        obj.location.x + offset[0],
        obj.location.y + offset[1],
        obj.location.z + offset[2],
    )
    for col in obj.users_collection:
        col.objects.link(new_obj)
    return new_obj


def select_all():
    """Select all objects."""
    for obj in bpy.data.objects:
        obj.select_set(True)


def deselect_all():
    """Deselect all objects."""
    for obj in bpy.data.objects:
        obj.select_set(False)


def select_obj(obj):
    """Select a single object and make it active."""
    deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def smooth_shade(obj):
    """Apply smooth shading to an object.

    In Blender 4.x this uses the shade_smooth operator.
    Pair with shade_auto_smooth() to get crisp hard edges on sharp angles.
    """
    if obj.data and hasattr(obj.data, 'polygons'):
        for poly in obj.data.polygons:
            poly.use_smooth = True


def flat_shade(obj):
    """Apply flat shading to an object."""
    if obj.data and hasattr(obj.data, 'polygons'):
        for poly in obj.data.polygons:
            poly.use_smooth = False


def shade_auto_smooth(obj, angle=30.0):
    """Enable auto smooth normals — smooth shade + sharp edges by angle.

    Faces meeting at angles > *angle* degrees will get a hard edge;
    all other edges get smooth shading. This is THE standard way to
    get professional-looking meshes: smooth surfaces with crisp creases.

    Works on Blender 4.x (geometry-node-based) and older (legacy).
    Pair with set_edge_sharp() to manually mark specific edges as sharp.

    Best combo: smooth_shade(obj) + shade_auto_smooth(obj, 30)
    """
    # First, ensure smooth shading
    smooth_shade(obj)

    # Blender 4.1+: auto smooth is done via modifier / geometry nodes
    # Blender 3.x/4.0: legacy use_auto_smooth attribute on mesh
    if hasattr(obj.data, 'use_auto_smooth'):
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = math.radians(angle)
    else:
        # Blender 4.1+: use the operator-based auto smooth
        try:
            prev_active = bpy.context.view_layer.objects.active
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.shade_smooth_by_angle(angle=math.radians(angle))
            bpy.context.view_layer.objects.active = prev_active
        except Exception:
            pass  # Fallback: just smooth shade
    return obj


def join_objects(objects):
    """Join a list of objects into one.  Returns the joined object."""
    if not objects:
        return None
    deselect_all()
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    try:
        bpy.ops.object.join()
    except Exception:
        pass
    return bpy.context.active_object


def scene_objects():
    """Return a list of all objects in the scene."""
    return list(bpy.data.objects)


# ═══════════════════════════════════════════════════════════════════════════
# BMesh — Direct Mesh Editing (professional modeling operations)
# ═══════════════════════════════════════════════════════════════════════════

def bmesh_edit(obj, callback):
    """Open *obj*'s mesh in a BMesh context, call *callback(bm)*, then write back.

    Example::
        def my_edit(bm):
            bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=2)
        bmesh_edit(wall, my_edit)
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    callback(bm)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def extrude_faces(obj, face_indices=None, offset=1.0, direction=None):
    """Extrude faces of *obj* outward by *offset*.

    *face_indices*: list of face indices to extrude, or None for all faces.
    *direction*: (x,y,z) direction vector; if None, uses face normals.
    Returns the obj for chaining.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    if face_indices is not None:
        faces = [bm.faces[i] for i in face_indices if i < len(bm.faces)]
    else:
        faces = list(bm.faces)

    if not faces:
        bm.free()
        return obj

    result = bmesh.ops.extrude_face_region(bm, geom=faces)
    extruded_verts = [v for v in result['geom'] if isinstance(v, bmesh.types.BMVert)]

    if direction:
        vec = Vector(direction).normalized() * offset
        bmesh.ops.translate(bm, verts=extruded_verts, vec=vec)
    else:
        # Move along average normal
        for v in extruded_verts:
            v.co += v.normal * offset

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def extrude_edges(obj, edge_indices=None, offset=(0, 0, 1)):
    """Extrude edges of *obj* to create new faces.

    *edge_indices*: list of edge indices, or None for all boundary edges.
    *offset*: translation vector for the extruded edges.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    if edge_indices is not None:
        edges = [bm.edges[i] for i in edge_indices if i < len(bm.edges)]
    else:
        edges = [e for e in bm.edges if e.is_boundary]

    if edges:
        result = bmesh.ops.extrude_edge_only(bm, edges=edges)
        extruded_verts = [v for v in result['geom'] if isinstance(v, bmesh.types.BMVert)]
        bmesh.ops.translate(bm, verts=extruded_verts, vec=Vector(offset))

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def inset_faces(obj, face_indices=None, thickness=0.1, depth=0.0,
                individual=False):
    """Inset faces of *obj* — like pressing 'i' in edit mode.

    *thickness*: how far inward the inset goes.
    *depth*: how far the inset face moves along the normal.
    *individual*: if True, each face is inset individually.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    if face_indices is not None:
        faces = [bm.faces[i] for i in face_indices if i < len(bm.faces)]
    else:
        faces = list(bm.faces)

    if faces:
        if individual:
            for face in faces:
                bmesh.ops.inset_individual(bm, faces=[face],
                                           thickness=thickness, depth=depth)
        else:
            bmesh.ops.inset_region(bm, faces=faces,
                                   thickness=thickness, depth=depth)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def loop_cut(obj, edge_index=0, cuts=1, offset=0.0):
    """Add loop cuts to *obj*.

    *edge_index*: index of an edge the loop should cross.
    *cuts*: number of cuts to add.
    Equivalent to Ctrl+R in edit mode.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    if edge_index < len(bm.edges):
        edge = bm.edges[edge_index]
        result = bmesh.ops.subdivide_edges(
            bm, edges=[edge], cuts=cuts, use_grid_fill=True)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def subdivide_mesh(obj, cuts=1, smooth=0.0):
    """Subdivide all edges of *obj* mesh.

    *cuts*: number of cuts per edge.
    *smooth*: smoothing factor (0-1).
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts,
                               smooth=smooth, use_grid_fill=True)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def bevel_edges(obj, edge_indices=None, width=0.05, segments=3):
    """Bevel specific edges of *obj* using BMesh (not the modifier).

    *edge_indices*: list of edge indices, or None for all edges.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    if edge_indices is not None:
        edges = [bm.edges[i] for i in edge_indices if i < len(bm.edges)]
    else:
        edges = list(bm.edges)

    if edges:
        bmesh.ops.bevel(bm, geom=edges, offset=width, segments=segments,
                        affect='EDGES')

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def bevel_verts(obj, vert_indices=None, width=0.05, segments=3):
    """Bevel specific vertices of *obj* using BMesh.

    *vert_indices*: list of vertex indices, or None for all.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    if vert_indices is not None:
        verts = [bm.verts[i] for i in vert_indices if i < len(bm.verts)]
    else:
        verts = list(bm.verts)

    if verts:
        bmesh.ops.bevel(bm, geom=verts, offset=width, segments=segments,
                        affect='VERTICES')

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def bridge_edge_loops(obj, loop1_edge_indices, loop2_edge_indices):
    """Bridge two edge loops to create connecting faces.

    Each loop is a list of edge indices.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    edges1 = [bm.edges[i] for i in loop1_edge_indices if i < len(bm.edges)]
    edges2 = [bm.edges[i] for i in loop2_edge_indices if i < len(bm.edges)]

    if edges1 and edges2:
        bmesh.ops.bridge_loops(bm, edges=edges1 + edges2)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def fill_face(obj, vert_indices):
    """Create a face from a list of vertex indices (like pressing F)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    verts = [bm.verts[i] for i in vert_indices if i < len(bm.verts)]
    if len(verts) >= 3:
        try:
            bm.faces.new(verts)
        except ValueError:
            pass  # Face already exists

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def grid_fill(obj, edge_indices=None):
    """Fill a closed edge loop with a grid of quads."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    if edge_indices:
        edges = [bm.edges[i] for i in edge_indices if i < len(bm.edges)]
    else:
        edges = [e for e in bm.edges if e.is_boundary]

    if edges:
        bmesh.ops.grid_fill(bm, edges=edges)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def dissolve_edges(obj, edge_indices):
    """Dissolve edges (merge adjacent faces without removing geometry)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    edges = [bm.edges[i] for i in edge_indices if i < len(bm.edges)]
    if edges:
        bmesh.ops.dissolve_edges(bm, edges=edges)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def dissolve_verts(obj, vert_indices):
    """Dissolve vertices (merge surrounding geometry cleanly)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    verts = [bm.verts[i] for i in vert_indices if i < len(bm.verts)]
    if verts:
        bmesh.ops.dissolve_verts(bm, verts=verts)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def poke_faces(obj, face_indices=None):
    """Poke (triangulate with center vert) selected faces."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    if face_indices:
        faces = [bm.faces[i] for i in face_indices if i < len(bm.faces)]
    else:
        faces = list(bm.faces)

    if faces:
        bmesh.ops.poke(bm, faces=faces)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def triangulate(obj, face_indices=None):
    """Triangulate faces of *obj*."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    if face_indices:
        faces = [bm.faces[i] for i in face_indices if i < len(bm.faces)]
    else:
        faces = list(bm.faces)

    if faces:
        bmesh.ops.triangulate(bm, faces=faces)

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def spin_mesh(obj, angle_deg=360, steps=32, axis=(0, 0, 1), center=(0, 0, 0)):
    """Spin (lathe) the mesh around an axis — great for creating
    objects of revolution (vases, columns, balusters, etc.).

    Operates on all verts of *obj*.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.spin(bm, geom=geom, angle=math.radians(angle_deg),
                   steps=steps, axis=Vector(axis), cent=Vector(center))

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def solidify_mesh(obj, thickness=0.1):
    """BMesh-level solidify — gives a surface mesh real thickness.
    Alternative to the Solidify modifier when you want direct control.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=thickness)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def offset_edges(obj, edge_indices=None, offset=0.1):
    """Create an offset edge loop by splitting and moving edges inward.
    Useful for creating window frames, panel details, etc.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    if edge_indices:
        edges = [bm.edges[i] for i in edge_indices if i < len(bm.edges)]
    else:
        edges = list(bm.edges)

    if edges:
        bmesh.ops.bevel(bm, geom=edges, offset=offset, segments=1,
                        affect='EDGES')

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def select_faces_by_normal(obj, direction=(0, 0, 1), threshold_deg=45):
    """Return a list of face indices whose normals point in *direction*
    within *threshold_deg* degrees. Useful for selecting 'top faces',
    'side faces', etc.
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    dir_vec = Vector(direction).normalized()
    threshold = math.cos(math.radians(threshold_deg))

    indices = []
    for face in bm.faces:
        dot = face.normal.dot(dir_vec)
        if dot >= threshold:
            indices.append(face.index)

    bm.free()
    return indices


def select_faces_by_area(obj, min_area=0.0, max_area=float('inf')):
    """Return face indices within a given area range."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    indices = [f.index for f in bm.faces
               if min_area <= f.calc_area() <= max_area]

    bm.free()
    return indices


def get_face_centers(obj):
    """Return list of (index, center_x, center_y, center_z) for every face."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()

    result = [(f.index, f.calc_center_median().x, f.calc_center_median().y,
               f.calc_center_median().z) for f in bm.faces]

    bm.free()
    return result


def get_mesh_stats(obj):
    """Return dict with verts, edges, faces, boundary_edges, non_manifold count."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    stats = {
        'verts': len(bm.verts),
        'edges': len(bm.edges),
        'faces': len(bm.faces),
        'boundary_edges': sum(1 for e in bm.edges if e.is_boundary),
        'non_manifold': sum(1 for e in bm.edges if not e.is_manifold),
        'tris': sum(1 for f in bm.faces if len(f.verts) == 3),
        'quads': sum(1 for f in bm.faces if len(f.verts) == 4),
        'ngons': sum(1 for f in bm.faces if len(f.verts) > 4),
    }

    bm.free()
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Mesh Cleanup & Topology
# ═══════════════════════════════════════════════════════════════════════════

def merge_by_distance(obj, distance=0.0001):
    """Merge vertices that are closer than *distance*."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def recalc_normals(obj, inside=False):
    """Recalculate face normals to point outward (or inward)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if inside:
        for f in bm.faces:
            f.normal_flip()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def tris_to_quads(obj, angle_limit=40):
    """Convert triangles to quads where possible (cleaner topology)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.join_triangles(bm, faces=bm.faces,
                              angle_face_threshold=math.radians(angle_limit),
                              angle_shape_threshold=math.radians(angle_limit))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def separate_loose(obj):
    """Separate loose mesh islands into individual objects."""
    select_obj(obj)
    try:
        with bpy.context.temp_override(object=obj):
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.separate(type='LOOSE')
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# Edge Properties — crease, sharp, seam, bevel weight
# ═══════════════════════════════════════════════════════════════════════════

def set_edge_crease(obj, edge_indices, crease=1.0):
    """Set edge crease weight for subdivision surface creasing.

    *crease*: 0.0 (smooth) to 1.0 (sharp).
    """
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    crease_layer = bm.edges.layers.float.get('crease_edge')
    if not crease_layer:
        crease_layer = bm.edges.layers.float.new('crease_edge')

    for i in edge_indices:
        if i < len(bm.edges):
            bm.edges[i][crease_layer] = crease

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def set_edge_sharp(obj, edge_indices, sharp=True):
    """Mark edges as sharp for flat shading split."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    for i in edge_indices:
        if i < len(bm.edges):
            bm.edges[i].smooth = not sharp

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def set_edge_seam(obj, edge_indices, seam=True):
    """Mark edges as UV seams."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    for i in edge_indices:
        if i < len(bm.edges):
            bm.edges[i].seam = seam

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


def set_bevel_weight(obj, edge_indices, weight=1.0):
    """Set bevel weight on edges (used by Bevel modifier in 'Weight' mode)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    bw_layer = bm.edges.layers.float.get('bevel_weight_edge')
    if not bw_layer:
        bw_layer = bm.edges.layers.float.new('bevel_weight_edge')

    for i in edge_indices:
        if i < len(bm.edges):
            bm.edges[i][bw_layer] = weight

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# Advanced Modifiers
# ═══════════════════════════════════════════════════════════════════════════

def weighted_normals(obj, weight=50, keep_sharp=True):
    """Add Weighted Normals modifier for better shading on hard-surface models."""
    mod = obj.modifiers.new("WeightedNormals", 'WEIGHTED_NORMAL')
    mod.weight = weight
    mod.keep_sharp = keep_sharp
    return mod


def remesh(obj, mode='VOXEL', voxel_size=0.05, octree_depth=6):
    """Add a Remesh modifier. mode: 'VOXEL', 'SMOOTH', 'SHARP', 'BLOCKS'."""
    mod = obj.modifiers.new("Remesh", 'REMESH')
    mod.mode = mode
    if mode == 'VOXEL':
        mod.voxel_size = voxel_size
    else:
        mod.octree_depth = octree_depth
    return mod


def decimate(obj, ratio=0.5, mode='COLLAPSE'):
    """Add a Decimate modifier to reduce polygon count.

    ⚠️ WARNING: For "low poly" style, use limited_dissolve() instead!
    decimate(COLLAPSE) randomly removes vertices and DESTROYS designed shapes
    (e.g., a hood at 0.38m becomes 0.22m). limited_dissolve() preserves
    surface shape while merging flat faces into clean large polygons.

    Use decimate only when you specifically want aggressive polygon reduction
    and don't care about preserving exact surface positions.

    *ratio*: fraction of faces to keep (0.1 = 10%, 0.25 = 25%).
    *mode*: 'COLLAPSE' (default), 'UNSUBDIV', 'DISSOLVE'.
    """
    mod = obj.modifiers.new("Decimate", 'DECIMATE')
    mod.decimate_type = mode
    if mode == 'COLLAPSE':
        mod.ratio = ratio
    return mod


def limited_dissolve(obj, angle_limit=5.0):
    """Merge nearly-coplanar faces into larger polygons — the BEST tool for
    clean low-poly aesthetics.

    Unlike decimate(COLLAPSE) which randomly collapses edges and DESTROYS
    your carefully designed curves, limited_dissolve only merges faces that
    are nearly flat relative to each other. The surface shape is PRESERVED
    while the poly count drops dramatically.

    This is how professional 3D artists create the "clean low-poly" look:
    large flat facets with crisp edges, not noisy random triangles.

    *angle_limit*: faces within this angle (degrees) of each other get merged.
        2-3° = aggressive — very large flat areas, minimal detail
        4-6° = standard low-poly — good balance (RECOMMENDED for vehicles)
        8-12° = gentle — preserves more surface curvature
        15-20° = very gentle — keeps most detail, just cleans up flat regions

    Returns *obj* for chaining.

    ★ LOW-POLY WORKFLOW (use this instead of decimate for low-poly style)::

        body = shape_from_profiles("Body", top, bot, width,
            sharpness=3.0, bottom_flat=0.5,
            num_sections=32, ring_points=24)   # build detailed
        limited_dissolve(body, angle_limit=5.0) # merge flat faces → clean low-poly
        flat_shade(body)                         # crisp angular facets

    Why this is better than decimate(COLLAPSE):
      - decimate randomly removes vertices → distorts your designed shape
        (e.g., hood at 0.38m becomes 0.22m after decimate!)
      - limited_dissolve only merges FLAT regions → shape stays accurate
      - Result has clean quads/ngons instead of noisy triangles
      - Professional low-poly models use this exact technique
    """
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.dissolve_limit(
        bm,
        angle_limit=math.radians(angle_limit),
        verts=bm.verts,
        edges=bm.edges,
    )
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def displace(obj, strength=0.1, midlevel=0.5, texture=None):
    """Add a Displace modifier for surface displacement."""
    mod = obj.modifiers.new("Displace", 'DISPLACE')
    mod.strength = strength
    mod.mid_level = midlevel
    if texture:
        mod.texture = texture
    return mod


def shrinkwrap(obj, target, mode='NEAREST_SURFACEPOINT', offset=0.0):
    """Shrinkwrap *obj* onto *target* surface."""
    mod = obj.modifiers.new("Shrinkwrap", 'SHRINKWRAP')
    mod.target = target
    mod.wrap_method = mode
    mod.offset = offset
    return mod


def lattice_deform(obj, lattice_obj):
    """Add a Lattice modifier — *lattice_obj* must be a Lattice object."""
    mod = obj.modifiers.new("Lattice", 'LATTICE')
    mod.object = lattice_obj
    return mod


def curve_modifier(obj, curve_obj, axis='POS_X'):
    """Deform *obj* along a *curve_obj*."""
    mod = obj.modifiers.new("Curve", 'CURVE')
    mod.object = curve_obj
    mod.deform_axis = axis
    return mod


def skin_modifier(obj, root_radius=(0.1, 0.1)):
    """Add a Skin modifier — turns edges into a mesh surface.

    Great for organic shapes: draw an armature-like edge structure,
    then Skin + Subdivision will create smooth geometry.
    """
    mod = obj.modifiers.new("Skin", 'SKIN')
    # Set root vertex radius
    if obj.data and hasattr(obj.data, 'skin_vertices'):
        for layer in obj.data.skin_vertices:
            for v in layer.data:
                v.radius = root_radius
    return mod


def wireframe_modifier(obj, thickness=0.02, offset=0):
    """Add a Wireframe modifier — creates wireframe mesh from edges."""
    mod = obj.modifiers.new("Wireframe", 'WIREFRAME')
    mod.thickness = thickness
    mod.offset = offset
    return mod


def screw_modifier(obj, angle_deg=360, steps=64, axis='Z',
                   screw_offset=0):
    """Add a Screw modifier — revolve a profile around an axis.

    Perfect for creating columns, vases, goblets, balusters, etc.
    """
    mod = obj.modifiers.new("Screw", 'SCREW')
    mod.angle = math.radians(angle_deg)
    mod.steps = steps
    mod.render_steps = steps
    mod.screw_offset = screw_offset
    mod.axis = axis
    return mod


def edge_split(obj, angle=30):
    """Add Edge Split modifier for crisp hard edges with smooth shading."""
    mod = obj.modifiers.new("EdgeSplit", 'EDGE_SPLIT')
    mod.split_angle = math.radians(angle)
    return mod


def cast_modifier(obj, cast_type='SPHERE', factor=1.0):
    """Add Cast modifier to deform towards a primitive shape."""
    mod = obj.modifiers.new("Cast", 'CAST')
    mod.cast_type = cast_type
    mod.factor = factor
    return mod


# ═══════════════════════════════════════════════════════════════════════════
# Curves — Bezier, NURBS, Path creation
# ═══════════════════════════════════════════════════════════════════════════

def create_bezier_curve(name="BezierCurve", points=None, location=(0, 0, 0),
                        closed=False, resolution=12, extrude=0, bevel_depth=0):
    """Create a bezier curve from a list of (x, y, z) control points.

    *extrude*: thickness in one direction (flat extrusion).
    *bevel_depth*: round bevel radius.
    """
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = resolution

    if extrude > 0:
        curve.extrude = extrude
    if bevel_depth > 0:
        curve.bevel_depth = bevel_depth

    spline = curve.splines.new('BEZIER')

    if points is None:
        points = [(0, 0, 0), (1, 1, 0), (2, 0, 0)]

    spline.bezier_points.add(len(points) - 1)
    for i, pt in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = Vector(pt)
        bp.handle_type_left = 'AUTO'
        bp.handle_type_right = 'AUTO'

    if closed:
        spline.use_cyclic_u = True

    obj = bpy.data.objects.new(name, curve)
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def create_nurbs_curve(name="NurbsCurve", points=None, location=(0, 0, 0),
                       order=4, closed=False, bevel_depth=0):
    """Create a NURBS curve from (x, y, z) control points."""
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'

    if bevel_depth > 0:
        curve.bevel_depth = bevel_depth

    spline = curve.splines.new('NURBS')

    if points is None:
        points = [(0, 0, 0), (1, 1, 0), (2, 1, 0), (3, 0, 0)]

    spline.points.add(len(points) - 1)
    for i, pt in enumerate(points):
        spline.points[i].co = (pt[0], pt[1], pt[2], 1.0)

    spline.order_u = min(order, len(points))

    if closed:
        spline.use_cyclic_u = True

    obj = bpy.data.objects.new(name, curve)
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def create_profile_from_points(name="Profile", points=None, location=(0, 0, 0),
                                closed=True):
    """Create a 2D curve profile — useful as a bevel object for another curve.

    *points*: list of (x, y) or (x, y, z) tuples.
    """
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '2D'

    spline = curve.splines.new('POLY')

    if points is None:
        # Default square profile
        points = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]

    spline.points.add(len(points) - 1)
    for i, pt in enumerate(points):
        x = pt[0]
        y = pt[1] if len(pt) > 1 else 0
        z = pt[2] if len(pt) > 2 else 0
        spline.points[i].co = (x, y, z, 1.0)

    if closed:
        spline.use_cyclic_u = True

    obj = bpy.data.objects.new(name, curve)
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def curve_to_mesh(curve_obj):
    """Convert a curve object to a mesh object.
    Returns the new mesh object (old curve is removed).
    """
    select_obj(curve_obj)
    try:
        with bpy.context.temp_override(object=curve_obj):
            bpy.ops.object.convert(target='MESH')
    except Exception:
        pass
    return curve_obj


def sweep_profile_along_curve(name="Sweep", profile_curve=None,
                               path_curve=None):
    """Sweep a 2D profile along a 3D path curve by setting the
    bevel_object. Returns the path curve with bevel applied.
    """
    if path_curve and profile_curve:
        path_curve.data.bevel_object = profile_curve
        profile_curve.hide_set(True)
    return path_curve


# ═══════════════════════════════════════════════════════════════════════════
# UV Tools
# ═══════════════════════════════════════════════════════════════════════════

def smart_uv_project(obj, angle_limit=66, island_margin=0.02):
    """Smart UV Project — good general-purpose unwrap."""
    select_obj(obj)
    try:
        with bpy.context.temp_override(object=obj):
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(angle_limit=math.radians(angle_limit),
                                      island_margin=island_margin)
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
    return obj


def cube_uv_project(obj, cube_size=1.0):
    """Cube projection UV unwrap — good for architectural geometry."""
    select_obj(obj)
    try:
        with bpy.context.temp_override(object=obj):
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.cube_project(cube_size=cube_size)
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
    return obj


def uv_unwrap(obj, method='ANGLE_BASED', margin=0.02):
    """Standard unwrap (requires UV seams to be marked first)."""
    select_obj(obj)
    try:
        with bpy.context.temp_override(object=obj):
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.unwrap(method=method, margin=margin)
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        try:
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception:
            pass
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# Advanced Materials — PBR with textures, procedural, layered
# ═══════════════════════════════════════════════════════════════════════════

def pbr_material(name="PBR", base_color=(0.8, 0.8, 0.8), roughness=0.5,
                 metallic=0.0, normal_strength=1.0, specular=0.5,
                 subsurface=0.0, clearcoat=0.0, emission=(0, 0, 0),
                 emission_strength=0.0, alpha=1.0):
    """Create a full PBR material with all Principled BSDF controls."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (
            base_color[0], base_color[1], base_color[2], 1.0)
        bsdf.inputs['Roughness'].default_value = roughness
        bsdf.inputs['Metallic'].default_value = metallic
        if 'Specular IOR Level' in bsdf.inputs:
            bsdf.inputs['Specular IOR Level'].default_value = specular
        elif 'Specular' in bsdf.inputs:
            bsdf.inputs['Specular'].default_value = specular
        if 'Alpha' in bsdf.inputs:
            bsdf.inputs['Alpha'].default_value = alpha
        # Emission
        if emission_strength > 0:
            if 'Emission Color' in bsdf.inputs:
                bsdf.inputs['Emission Color'].default_value = (
                    emission[0], emission[1], emission[2], 1.0)
            if 'Emission Strength' in bsdf.inputs:
                bsdf.inputs['Emission Strength'].default_value = emission_strength
    if alpha < 1.0:
        mat.blend_method = 'HASHED'
    return mat


def noise_texture_material(name="Procedural", base_color=(0.6, 0.5, 0.4),
                            detail_color=(0.3, 0.2, 0.15), scale=5.0,
                            roughness=0.6, bump_strength=0.2):
    """Create a procedural material using Noise Texture for variation.
    Great for concrete, stone, stucco, plaster, etc.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    output = nodes.get("Material Output")

    # Noise texture
    noise = nodes.new('ShaderNodeTexNoise')
    noise.inputs['Scale'].default_value = scale
    noise.inputs['Detail'].default_value = 8.0
    noise.location = (-600, 300)

    # Color ramp for mixing two colors
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (-300, 300)
    ramp.color_ramp.elements[0].color = (
        base_color[0], base_color[1], base_color[2], 1.0)
    ramp.color_ramp.elements[1].color = (
        detail_color[0], detail_color[1], detail_color[2], 1.0)

    links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])

    bsdf.inputs['Roughness'].default_value = roughness

    # Bump map from noise
    if bump_strength > 0:
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = bump_strength
        bump.location = (-300, 0)
        links.new(noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    return mat


def brick_texture_material(name="Brick", brick_color=(0.5, 0.2, 0.1),
                            mortar_color=(0.7, 0.7, 0.65), mortar_size=0.02,
                            scale=4.0, roughness=0.8, bump_strength=0.3):
    """Create a procedural brick material using Blender's Brick Texture node."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")

    # Texture coordinate + mapping for scale control
    coord = nodes.new('ShaderNodeTexCoord')
    coord.location = (-900, 300)
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-700, 300)
    mapping.inputs['Scale'].default_value = (scale, scale, scale)
    links.new(coord.outputs['Object'], mapping.inputs['Vector'])

    # Brick texture
    brick = nodes.new('ShaderNodeTexBrick')
    brick.location = (-400, 300)
    brick.inputs['Color1'].default_value = (
        brick_color[0], brick_color[1], brick_color[2], 1.0)
    brick.inputs['Color2'].default_value = (
        brick_color[0] * 0.8, brick_color[1] * 0.8, brick_color[2] * 0.8, 1.0)
    brick.inputs['Mortar'].default_value = (
        mortar_color[0], mortar_color[1], mortar_color[2], 1.0)
    brick.inputs['Mortar Size'].default_value = mortar_size
    brick.inputs['Scale'].default_value = scale

    links.new(mapping.outputs['Vector'], brick.inputs['Vector'])
    links.new(brick.outputs['Color'], bsdf.inputs['Base Color'])

    bsdf.inputs['Roughness'].default_value = roughness

    # Bump from brick fac
    if bump_strength > 0:
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = bump_strength
        bump.location = (-200, 0)
        links.new(brick.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    return mat


def wood_material(name="Wood", base_color=(0.35, 0.18, 0.06),
                  ring_color=(0.25, 0.12, 0.04), scale=3.0,
                  roughness=0.65, bump_strength=0.15):
    """Create a procedural wood grain material using Wave + Noise textures."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")

    # Wave texture for wood grain lines
    wave = nodes.new('ShaderNodeTexWave')
    wave.location = (-600, 300)
    wave.wave_type = 'RINGS'
    wave.inputs['Scale'].default_value = scale
    wave.inputs['Distortion'].default_value = 8.0
    wave.inputs['Detail'].default_value = 3.0

    # Color ramp
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (-300, 300)
    ramp.color_ramp.elements[0].color = (
        base_color[0], base_color[1], base_color[2], 1.0)
    ramp.color_ramp.elements[1].color = (
        ring_color[0], ring_color[1], ring_color[2], 1.0)

    links.new(wave.outputs['Fac'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])

    bsdf.inputs['Roughness'].default_value = roughness

    # Bump
    if bump_strength > 0:
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = bump_strength
        bump.location = (-300, 0)
        links.new(wave.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    return mat


# ═══════════════════════════════════════════════════════════════════════════
# Mesh from Vertices — Full Topology Control
# ═══════════════════════════════════════════════════════════════════════════

def create_mesh(name="Mesh", verts=None, edges=None, faces=None,
                location=(0, 0, 0)):
    """Create a mesh object from raw vertex/edge/face data.

    *verts*: list of (x, y, z) tuples
    *edges*: list of (v1, v2) index tuples (optional)
    *faces*: list of (v1, v2, v3, ...) index tuples

    This gives you FULL topology control — the fundamental building block
    of professional 3D modeling.
    """
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts or [], edges or [], faces or [])
    mesh.update()

    # Fix normals
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def profile_ring(x, width, height, floor_z=0.0,
                 roof_taper=0.7, bottom_taper=0.85, belt_frac=0.6):
    """Generate a shaped cross-section ring for use with loft_sections().

    Instead of specifying raw vertex coordinates, provide intuitive
    DIMENSIONS and this function generates a proper 10-point body
    profile: flat bottom, vertical sides, beltline, tapered roof.

    Use this to build any elongated body: vehicles, boats, fuselages,
    bottles, furniture legs, vases, organic shapes, etc.

    *x*: position along the length axis (front-to-back)
    *width*: full width at the widest point of this section
    *height*: total height from floor to roof peak
    *floor_z*: z of the bottom surface (default 0.0)
    *roof_taper*: roof narrowing — 0.0=pointed, 1.0=same as body (0.7)
    *bottom_taper*: floor narrowing — 0.0=keel, 1.0=flat bottom (0.85)
    *belt_frac*: beltline height as fraction of height — 0.6 typical

    Returns a list of 10 (x, y, z) tuples — one closed cross-section.

    Example::

        stations = [(-2.0, 1.4, 0.5, 0.1), (0.0, 1.9, 0.7, 0.05)]
        sections = [profile_ring(x, w, h, fz) for x, w, h, fz in stations]
        body = loft_sections("Body", sections)
    """
    hw = width / 2.0
    bhw = hw * bottom_taper
    rhw = hw * roof_taper
    belt_z = floor_z + height * belt_frac
    top_z = floor_z + height
    # Lower-body offset: 10% of height (proportional, never too small/large)
    lo = max(height * 0.10, 0.02)
    # Roof peak offset: slight dome
    ro = max(height * 0.03, 0.01)

    return [
        (x,  0.0,  floor_z),           # 0: bottom center
        (x,  bhw,  floor_z),           # 1: bottom right
        (x,  hw,   floor_z + lo),      # 2: lower body right
        (x,  hw,   belt_z),            # 3: beltline right
        (x,  rhw,  top_z),             # 4: roof edge right
        (x,  0.0,  top_z + ro),        # 5: roof peak center
        (x, -rhw,  top_z),             # 6: roof edge left
        (x, -hw,   belt_z),            # 7: beltline left
        (x, -hw,   floor_z + lo),      # 8: lower body left
        (x, -bhw,  floor_z),           # 9: bottom left
    ]


def loft_sections(name="Lofted", sections=None, closed_loop=True,
                  cap_ends=True, location=(0, 0, 0)):
    """Create a 3D mesh by lofting (skinning) cross-section rings.

    Standard technique for any shaped body: cars, boats, bottles,
    aircraft fuselages, organic shapes, vases, etc.

    *sections*: list of rings.  Each ring is a list of (x, y, z) tuples.
        ALL rings must have the same number of points.
        Points in each ring should be ordered consistently
        (e.g. counter-clockwise when looking down the +X axis).
    *closed_loop*: if True (DEFAULT), each ring is closed (last→first).
        Almost always True for body shapes. Only False for open surfaces
        like a wing or ribbon.
    *cap_ends*: if True, adds faces on the first and last rings.

    **Tip — build rings from shape parameters:**

        import math
        sections = []
        # (x_station, half_width, z_bottom, z_top)
        stations = [
            (0.0, 0.1, 0.0, 0.1),   # tip
            (0.5, 0.4, 0.0, 0.3),   # nose
            (2.0, 0.5, 0.0, 0.6),   # body
            (3.0, 0.3, 0.0, 0.4),   # tail
            (3.5, 0.05, 0.1, 0.15), # tail tip
        ]
        N = 12  # points per ring
        for x, hw, zb, zt in stations:
            ring = []
            for i in range(N):
                t = i / N
                angle = t * 2 * math.pi
                y = hw * math.sin(angle)
                z = zb + (zt - zb) * (0.5 + 0.5 * math.cos(angle))
                ring.append((x, y, z))
            sections.append(ring)
        body = loft_sections("Body", sections)
    """
    if sections is None or len(sections) < 2:
        raise ValueError("loft_sections needs at least 2 sections")

    ring_len = len(sections[0])
    for i, ring in enumerate(sections):
        if len(ring) != ring_len:
            raise ValueError(
                "Section %d has %d points, expected %d" % (i, len(ring), ring_len))

    verts = []
    faces = []

    # Flatten all vertices
    for ring in sections:
        for pt in ring:
            verts.append(tuple(pt))

    n_sections = len(sections)

    # Create quad faces between adjacent rings
    for s in range(n_sections - 1):
        for p in range(ring_len):
            if closed_loop:
                p_next = (p + 1) % ring_len
            else:
                if p >= ring_len - 1:
                    continue
                p_next = p + 1

            v0 = s * ring_len + p
            v1 = s * ring_len + p_next
            v2 = (s + 1) * ring_len + p_next
            v3 = (s + 1) * ring_len + p
            faces.append((v0, v1, v2, v3))

    # Cap ends with n-gon faces
    if cap_ends:
        # First ring
        faces.append(tuple(range(ring_len - 1, -1, -1)))
        # Last ring
        last_start = (n_sections - 1) * ring_len
        faces.append(tuple(range(last_start, last_start + ring_len)))

    mesh_data = bpy.data.meshes.new(name)
    mesh_data.from_pydata(verts, [], faces)
    mesh_data.update()

    # Fix normals
    bm = bmesh.new()
    bm.from_mesh(mesh_data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh_data)
    bm.free()

    obj = bpy.data.objects.new(name, mesh_data)
    obj.location = Vector(location)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# General-Purpose Shape Creation — profiles, outlines, revolution, extrusion
# ═══════════════════════════════════════════════════════════════════════════

def _lerp_curve(points, x):
    """Linearly interpolate a list of (x, y) control points at position *x*.

    Points don't need to be pre-sorted. Values outside the range
    are clamped to the nearest endpoint.
    """
    if not points:
        return 0.0
    pts = sorted(points, key=lambda p: p[0])
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        if pts[i][0] <= x <= pts[i + 1][0]:
            dx = pts[i + 1][0] - pts[i][0]
            if dx < 1e-12:
                return (pts[i][1] + pts[i + 1][1]) / 2.0
            t = (x - pts[i][0]) / dx
            return pts[i][1] + t * (pts[i + 1][1] - pts[i][1])
    return pts[-1][1]


def _outline_bounds_at(outline, pos, pos_idx=0, val_idx=1):
    """Find min/max bounds of a closed polygon at a given position along one axis.

    Scans all edges of the closed polygon, finds crossings at *pos*
    on the *pos_idx* axis, returns (min_val, max_val) of *val_idx*.
    Returns (None, None) if no crossings found.
    """
    crossings = []
    n = len(outline)
    for i in range(n):
        p1 = outline[i]
        p2 = outline[(i + 1) % n]
        x1, v1 = p1[pos_idx], p1[val_idx]
        x2, v2 = p2[pos_idx], p2[val_idx]
        lo, hi = min(x1, x2), max(x1, x2)
        if lo - 1e-10 <= pos <= hi + 1e-10:
            dx = x2 - x1
            if abs(dx) < 1e-10:
                crossings.extend([v1, v2])
            else:
                t = max(0.0, min(1.0, (pos - x1) / dx))
                crossings.append(v1 + t * (v2 - v1))
    if not crossings:
        return None, None
    return min(crossings), max(crossings)


def shape_from_profiles(name="Shape", top_curve=None, bottom_curve=None,
                        width_curve=None, num_sections=24, ring_points=16,
                        axis='X', sharpness=2.0, bottom_flat=0.5,
                        flat_top=0.0, location=(0, 0, 0)):
    """Create a 3D mesh from three defining curves — the general-purpose body builder.

    Works for ANY elongated shape: vehicles, boats, aircraft, bottles,
    furniture, fish, characters, architectural elements, etc.

    Three intuitive curves define the shape:
      *top_curve*:    [(pos, z), ...] — top/roof edge in side view
      *bottom_curve*: [(pos, z), ...] — bottom edge in side view
      *width_curve*:  [(pos, half_width), ...] — half-width from top view

    Each curve is a list of (position_along_axis, value) control points.
    Points are interpolated. More points = more detail.

    *sharpness*: cross-section shape (superellipse exponent)
        2.0 = elliptical (organic: fish, bottles, rockets)
        3.0 = semi-boxy (rounded vehicles, boats, fuselages)
        5.0 = angular (sports cars, supercars)
        8.0 = very angular (Lamborghini, angular supercars)
        10.0+ = nearly rectangular (trucks, buildings, furniture)

    *bottom_flat*: how flat the bottom surface is
        0.0 = fully rounded (submarine, ball)
        0.5 = partially flat (car, boat)
        1.0 = completely flat (table, box)

    *flat_top*: how flat the top surface is
        0.0 = fully rounded (default — organic shapes, bottles)
        0.5 = partially flat (rounded vehicles)
        0.7 = mostly flat (angular vehicles — flat hood/roof)
        1.0 = completely flat (boxes, buildings)

    Example (angular sports car — build detailed, then dissolve for low-poly)::

        body = shape_from_profiles("CarBody",
            top_curve=[  # Sharp transitions, not smooth ramps!
                (0.0, 0.38), (0.3, 0.35), (0.6, 0.36), (0.9, 0.38),
                (1.2, 0.40), (1.4, 0.42), (1.5, 0.55), (1.6, 0.80),
                (1.7, 1.00), (1.8, 1.08), (2.0, 1.12), (2.2, 1.10),
                (2.5, 1.05), (2.8, 0.95), (3.2, 0.82), (3.6, 0.72),
                (4.0, 0.62), (4.2, 0.55), (4.4, 0.50), (4.5, 0.45)],
            bottom_curve=[
                (0.0, 0.12), (0.3, 0.10), (0.7, 0.08), (1.2, 0.06),
                (1.8, 0.06), (2.2, 0.06), (2.8, 0.06), (3.2, 0.08),
                (3.6, 0.10), (4.0, 0.14), (4.3, 0.18), (4.5, 0.22)],
            width_curve=[  # Pinch at cabin, wide at fenders
                (0.0, 0.40), (0.3, 0.55), (0.6, 0.72), (0.9, 0.88),
                (1.2, 0.95), (1.5, 0.85), (2.0, 0.82), (2.5, 0.88),
                (3.0, 1.00), (3.3, 1.02), (3.6, 0.98), (4.0, 0.80),
                (4.3, 0.55), (4.5, 0.38)],
            sharpness=8.0, bottom_flat=0.6, flat_top=0.7,
            num_sections=20, ring_points=16)
        flat_shade(body)  # angular faceted low-poly look
        # ⚠️ Do NOT use limited_dissolve on vehicle bodies.
        # It destroys center-seam vertices → creates tent/groove artifact.
        # 20 sections × 16 ring points with flat_shade IS low-poly.

    Example (bottle)::

        bottle = shape_from_profiles("Bottle",
            top_curve=[(0, 0.3), (0.08, 0.28), (0.1, 0.15),
                       (0.12, 0.02), (0.12, 0.02)],
            bottom_curve=[(0, 0.0), (0.08, 0.0), (0.12, 0.0)],
            width_curve=[(0, 0.3), (0.08, 0.28), (0.1, 0.15),
                         (0.11, 0.02), (0.12, 0.02)],
            axis='Z', sharpness=2.0, bottom_flat=1.0)
    """
    if not top_curve or not bottom_curve or not width_curve:
        raise ValueError("shape_from_profiles requires top_curve, "
                         "bottom_curve, and width_curve")

    all_pos = ([p[0] for p in top_curve] + [p[0] for p in bottom_curve]
               + [p[0] for p in width_curve])
    pos_min, pos_max = min(all_pos), max(all_pos)
    if pos_max - pos_min < 1e-6:
        raise ValueError("shape_from_profiles: curves have zero extent")

    sections = []
    for i in range(num_sections):
        t = i / max(num_sections - 1, 1)
        pos = pos_min + t * (pos_max - pos_min)

        z_top = _lerp_curve(top_curve, pos)
        z_bot = _lerp_curve(bottom_curve, pos)
        hw = max(_lerp_curve(width_curve, pos), 0.001)
        height = max(z_top - z_bot, 0.001)
        z_center = (z_top + z_bot) / 2.0
        half_h = height / 2.0

        n = sharpness
        ring = []
        for j in range(ring_points):
            angle = 2.0 * math.pi * j / ring_points
            ca, sa = math.cos(angle), math.sin(angle)

            # Superellipse: |cos|^(2/n) * sign, |sin|^(2/n) * sign
            exp = 2.0 / n
            sy = (abs(sa) ** exp) * (1.0 if sa >= 0 else -1.0) * hw
            cz = (abs(ca) ** exp) * (1.0 if ca >= 0 else -1.0)

            # Bottom flattening
            if cz < 0:
                flat_z = z_bot
                round_z = z_center + cz * half_h
                z = flat_z + (1.0 - bottom_flat) * (round_z - flat_z)
            else:
                # Top flattening (analogous to bottom_flat)
                round_z = z_center + cz * half_h
                if flat_top > 0:
                    z = z_top + (1.0 - flat_top) * (round_z - z_top)
                else:
                    z = round_z

            if axis == 'X':
                ring.append((pos, sy, z))
            elif axis == 'Y':
                ring.append((sy, pos, z))
            else:
                ring.append((sy, z, pos))
        sections.append(ring)

    return loft_sections(name, sections, closed_loop=True,
                         cap_ends=True, location=location)


def mesh_from_outlines(name="Shape", side_outline=None, top_outline=None,
                       num_sections=30, ring_points=16, axis='X',
                       sharpness=2.0, bottom_flat=0.0, flat_top=0.0,
                       location=(0, 0, 0)):
    """Create a 3D mesh from two 2D silhouette outlines — the reference-image tracing tool.

    The most powerful way to create accurate shapes from reference images.
    Trace the subject's outline in side and top views → accurate 3D mesh.

    *side_outline*: [(pos, z), ...] — CLOSED polygon from the side view.
        Trace the full silhouette as a closed loop: top edge from front
        to back, then bottom edge from back to front.

    *top_outline*: [(pos, y), ...] — CLOSED polygon from the top view.
        Trace the full silhouette as a closed loop: one side from front
        to back, then the other side from back to front.

    Both outlines share the same 'pos' axis (the length axis).
    At each cross-section position, the Z-range comes from the side
    outline and the Y-range from the top outline.

    *sharpness*: cross-section exponent (2=round, 3=semi-boxy, 4+=boxy)
    *bottom_flat*: bottom flatness (0=round, 0.5=car-like, 1=fully flat)

    Example (tracing a car from reference photos)::

        body = mesh_from_outlines("CarBody",
            side_outline=[
                (2.2, 0.4), (1.5, 0.6), (0.5, 1.1), (-0.5, 1.1),
                (-1.5, 0.8), (-2.2, 0.5),     # top edge
                (-2.2, 0.15), (-1.0, 0.10),
                (1.0, 0.10), (2.2, 0.12),      # bottom edge
            ],
            top_outline=[
                (2.2, 0.0), (2.0, 0.4), (0.0, 0.95), (-2.2, 0.5),
                (-2.2, -0.5), (0.0, -0.95), (2.0, -0.4), (2.2, 0.0),
            ],
            sharpness=3.0, bottom_flat=0.6)

    Example (fish from reference image)::

        fish = mesh_from_outlines("Fish",
            side_outline=[
                (0.3, 0.0), (0.15, 0.08), (-0.1, 0.06), (-0.3, 0.0),
                (-0.1, -0.04), (0.15, -0.05), (0.3, 0.0),
            ],
            top_outline=[
                (0.3, 0.0), (0.1, 0.06), (-0.15, 0.04), (-0.3, 0.0),
                (-0.15, -0.04), (0.1, -0.06), (0.3, 0.0),
            ],
            sharpness=2.0, bottom_flat=0.0)
    """
    if not side_outline or not top_outline:
        raise ValueError("mesh_from_outlines requires both "
                         "side_outline and top_outline")

    side_xs = [p[0] for p in side_outline]
    top_xs = [p[0] for p in top_outline]
    pos_min = max(min(side_xs), min(top_xs))
    pos_max = min(max(side_xs), max(top_xs))
    if pos_max - pos_min < 1e-6:
        raise ValueError("mesh_from_outlines: outlines have no "
                         "overlapping range")

    sections = []
    for i in range(num_sections):
        t = i / max(num_sections - 1, 1)
        pos = pos_min + t * (pos_max - pos_min)

        z_min, z_max = _outline_bounds_at(side_outline, pos, 0, 1)
        y_min, y_max = _outline_bounds_at(top_outline, pos, 0, 1)
        if z_min is None or y_min is None:
            continue

        hw = max(abs(y_min), abs(y_max), 0.001)
        z_center = (z_max + z_min) / 2.0
        half_h = max((z_max - z_min) / 2.0, 0.001)

        n = sharpness
        ring = []
        for j in range(ring_points):
            angle = 2.0 * math.pi * j / ring_points
            ca, sa = math.cos(angle), math.sin(angle)
            exp = 2.0 / n
            sy = (abs(sa) ** exp) * (1.0 if sa >= 0 else -1.0) * hw
            cz = (abs(ca) ** exp) * (1.0 if ca >= 0 else -1.0)

            if cz < 0:
                flat_z = z_min
                round_z = z_center + cz * half_h
                z = flat_z + (1.0 - bottom_flat) * (round_z - flat_z)
            else:
                # Top flattening (analogous to bottom_flat)
                round_z = z_center + cz * half_h
                if flat_top > 0:
                    z = z_max + (1.0 - flat_top) * (round_z - z_max)
                else:
                    z = round_z

            if axis == 'X':
                ring.append((pos, sy, z))
            elif axis == 'Y':
                ring.append((sy, pos, z))
            else:
                ring.append((sy, z, pos))
        sections.append(ring)

    if len(sections) < 2:
        raise ValueError("mesh_from_outlines: not enough valid sections")

    return loft_sections(name, sections, closed_loop=True,
                         cap_ends=True, location=location)


def revolve_profile(name="Revolved", profile=None, axis='Z',
                    segments=32, angle_deg=360, location=(0, 0, 0)):
    """Create a surface of revolution from a 2D profile curve.

    Works for ANY rotationally symmetric shape: bottles, vases, columns,
    wheels, goblets, chess pieces, lamp shades, domes, chalices, barrels,
    candles, pots, spheres, rockets, etc.

    *profile*: [(radius, height), ...] — profile curve control points.
        'radius' = distance from the revolution axis.
        'height' = position along the revolution axis.
        Points are connected in order.

    *axis*: revolution axis ('X', 'Y', or 'Z'; default 'Z' for upright)
    *segments*: steps around the revolution (higher = smoother)
    *angle_deg*: degrees (360 = full, 180 = half, etc.)

    Example (wine glass)::

        glass = revolve_profile("WineGlass", profile=[
            (0.00, 0.000), (0.03, 0.000), (0.03, 0.005),
            (0.005, 0.02), (0.005, 0.08), (0.02, 0.09),
            (0.04, 0.12), (0.035, 0.15),
        ], segments=24)

    Example (wheel)::

        wheel = revolve_profile("Wheel", profile=[
            (0.00, -0.1), (0.30, -0.1), (0.35, -0.08),
            (0.35, 0.08), (0.30, 0.1), (0.00, 0.1),
        ], axis='Y', segments=16)
    """
    if not profile or len(profile) < 2:
        raise ValueError("revolve_profile needs at least 2 profile points")

    angle_rad = math.radians(angle_deg)
    full = abs(angle_deg - 360.0) < 0.1

    n_prof = len(profile)
    n_seg = segments
    verts = []
    step_count = n_seg if full else n_seg + 1

    for i in range(step_count):
        theta = angle_rad * i / n_seg
        ct, st = math.cos(theta), math.sin(theta)
        for r, h in profile:
            if axis == 'Z':
                verts.append((r * ct, r * st, h))
            elif axis == 'X':
                verts.append((h, r * ct, r * st))
            else:
                verts.append((r * ct, h, r * st))

    faces = []
    wrap = step_count if full else step_count
    for i in range(n_seg):
        i_next = (i + 1) % wrap
        for j in range(n_prof - 1):
            v0 = i * n_prof + j
            v1 = i * n_prof + j + 1
            v2 = i_next * n_prof + j + 1
            v3 = i_next * n_prof + j
            faces.append((v0, v1, v2, v3))

    obj = create_mesh(name, verts=verts, faces=faces, location=location)
    recalc_normals(obj)
    return obj


def extrude_shape(name="Extruded", outline=None, depth=0.1, axis='Z',
                  location=(0, 0, 0)):
    """Extrude a 2D outline into a 3D solid — the simplest shape tool.

    Works for any shape that can be described as a 2D cross-section
    pushed along an axis: brackets, logos, floor plans, panels, etc.

    *outline*: [(a, b), ...] — 2D points forming a closed polygon.
        Points are in the plane perpendicular to *axis*.
    *depth*: extrusion distance
    *axis*: direction to extrude ('X', 'Y', or 'Z')

    Example (L-bracket)::

        bracket = extrude_shape("Bracket", outline=[
            (0, 0), (1, 0), (1, 0.2), (0.2, 0.2), (0.2, 1), (0, 1)
        ], depth=0.1, axis='Z')

    Example (star)::

        import math
        pts = []
        for i in range(10):
            a = math.pi * 2 * i / 10 - math.pi / 2
            r = 0.5 if i % 2 == 0 else 0.2
            pts.append((r * math.cos(a), r * math.sin(a)))
        star = extrude_shape("Star", outline=pts, depth=0.05)
    """
    if not outline or len(outline) < 3:
        raise ValueError("extrude_shape needs at least 3 outline points")

    half_d = depth / 2.0
    n = len(outline)
    verts = []

    # Front face vertices
    for a, b in outline:
        if axis == 'Z':
            verts.append((a, b, -half_d))
        elif axis == 'X':
            verts.append((-half_d, a, b))
        else:
            verts.append((a, -half_d, b))

    # Back face vertices
    for a, b in outline:
        if axis == 'Z':
            verts.append((a, b, half_d))
        elif axis == 'X':
            verts.append((half_d, a, b))
        else:
            verts.append((a, half_d, b))

    faces = []
    # Front face (reversed winding)
    faces.append(tuple(range(n - 1, -1, -1)))
    # Back face
    faces.append(tuple(range(n, 2 * n)))
    # Side faces
    for i in range(n):
        i_next = (i + 1) % n
        faces.append((i, i_next, n + i_next, n + i))

    obj = create_mesh(name, verts=verts, faces=faces, location=location)
    recalc_normals(obj)
    return obj


def create_lattice(name="Lattice", location=(0, 0, 0),
                   resolution=(3, 3, 3), scale=(2, 2, 2)):
    """Create a Lattice object for deformation control."""
    lat = bpy.data.lattices.new(name)
    lat.points_u = resolution[0]
    lat.points_v = resolution[1]
    lat.points_w = resolution[2]
    obj = bpy.data.objects.new(name, lat)
    obj.location = Vector(location)
    obj.scale = Vector(scale)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# HDRI & World Lighting
# ═══════════════════════════════════════════════════════════════════════════

def setup_sky_texture(sun_elevation=30, sun_rotation=0, strength=1.0):
    """Set up a procedural sky using Blender's Sky Texture node.
    Much better than a flat background color.
    """
    scene = bpy.context.scene
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    sky = nodes.new('ShaderNodeTexSky')
    sky.sky_type = 'NISHITA'
    sky.sun_elevation = math.radians(sun_elevation)
    sky.sun_rotation = math.radians(sun_rotation)
    sky.location = (-300, 0)

    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = strength
    bg.location = (0, 0)

    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (300, 0)

    links.new(sky.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs['Background'], output.inputs['Surface'])


def setup_gradient_world(horizon_color=(0.6, 0.7, 0.8),
                          zenith_color=(0.15, 0.25, 0.5),
                          ground_color=(0.3, 0.25, 0.2),
                          strength=1.0):
    """Set up a gradient sky world background."""
    scene = bpy.context.scene
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    coord = nodes.new('ShaderNodeTexCoord')
    coord.location = (-800, 0)

    separate = nodes.new('ShaderNodeSeparateXYZ')
    separate.location = (-600, 0)
    links.new(coord.outputs['Generated'], separate.inputs['Vector'])

    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (-300, 0)
    ramp.color_ramp.elements[0].position = 0.4
    ramp.color_ramp.elements[0].color = (
        ground_color[0], ground_color[1], ground_color[2], 1.0)
    ramp.color_ramp.elements[1].position = 0.6
    ramp.color_ramp.elements[1].color = (
        horizon_color[0], horizon_color[1], horizon_color[2], 1.0)
    # Add zenith
    elem = ramp.color_ramp.elements.new(0.9)
    elem.color = (zenith_color[0], zenith_color[1], zenith_color[2], 1.0)

    links.new(separate.outputs['Z'], ramp.inputs['Fac'])

    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = strength
    bg.location = (0, 0)
    links.new(ramp.outputs['Color'], bg.inputs['Color'])

    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (300, 0)
    links.new(bg.outputs['Background'], output.inputs['Surface'])


# ═══════════════════════════════════════════════════════════════════════════
# Viewport Capture — render a snapshot for AI review
# ═══════════════════════════════════════════════════════════════════════════
# Vertex-Level Editing — true edit-mode geometry manipulation
# ═══════════════════════════════════════════════════════════════════════════

def add_vert(obj, co=(0, 0, 0)):
    """Add a single vertex to *obj* at local coordinate *co*. Returns vert index."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    v = bm.verts.new(Vector(co))
    bm.to_mesh(mesh)
    idx = v.index
    bm.free()
    mesh.update()
    return idx


def add_verts(obj, coords):
    """Add multiple vertices to *obj*. *coords* is a list of (x,y,z). Returns list of indices."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    indices = []
    for co in coords:
        v = bm.verts.new(Vector(co))
        indices.append(v.index)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return indices


def add_edge(obj, vert_idx1, vert_idx2):
    """Add an edge between two existing vertices by index."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    try:
        bm.edges.new((bm.verts[vert_idx1], bm.verts[vert_idx2]))
    except ValueError:
        pass  # Edge already exists
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def add_face(obj, vert_indices):
    """Add a face from a list of vertex indices. Alias for fill_face."""
    return fill_face(obj, vert_indices)


def move_vert(obj, vert_index, co):
    """Move a single vertex to new local coordinates *co* = (x, y, z)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    if vert_index < len(bm.verts):
        bm.verts[vert_index].co = Vector(co)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def move_verts(obj, vert_indices, offset):
    """Translate vertices by *offset* = (dx, dy, dz)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    vec = Vector(offset)
    for i in vert_indices:
        if i < len(bm.verts):
            bm.verts[i].co += vec
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def set_vert_positions(obj, index_coord_pairs):
    """Set exact positions for multiple verts. *index_coord_pairs* = [(idx, (x,y,z)), ...]."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    for idx, co in index_coord_pairs:
        if idx < len(bm.verts):
            bm.verts[idx].co = Vector(co)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def delete_verts(obj, vert_indices):
    """Delete vertices and their connected geometry."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    verts = [bm.verts[i] for i in vert_indices if i < len(bm.verts)]
    bmesh.ops.delete(bm, geom=verts, context='VERTS')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def delete_faces(obj, face_indices):
    """Delete faces (keeps vertices and edges)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    faces = [bm.faces[i] for i in face_indices if i < len(bm.faces)]
    bmesh.ops.delete(bm, geom=faces, context='FACES')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def delete_edges(obj, edge_indices):
    """Delete edges and their faces (keeps vertices)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    edges = [bm.edges[i] for i in edge_indices if i < len(bm.edges)]
    bmesh.ops.delete(bm, geom=edges, context='EDGES')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def get_vert_coords(obj):
    """Return all vertex positions as list of (index, x, y, z)."""
    mesh = obj.data
    return [(i, v.co.x, v.co.y, v.co.z) for i, v in enumerate(mesh.vertices)]


def get_edge_verts(obj):
    """Return all edges as list of (edge_index, vert1_index, vert2_index)."""
    mesh = obj.data
    return [(i, e.vertices[0], e.vertices[1]) for i, e in enumerate(mesh.edges)]


def get_face_verts(obj):
    """Return all faces as list of (face_index, [vert_indices])."""
    mesh = obj.data
    return [(i, list(f.vertices)) for i, f in enumerate(mesh.polygons)]


def merge_verts(obj, vert_indices, target='CENTER'):
    """Merge vertices together. target: 'CENTER' (average position), 'FIRST', 'LAST', or (x,y,z) coords."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    verts = [bm.verts[i] for i in vert_indices if i < len(bm.verts)]
    if len(verts) < 2:
        bm.free()
        return
    if isinstance(target, (list, tuple)):
        target_co = Vector(target)
    elif target == 'FIRST':
        target_co = verts[0].co.copy()
    elif target == 'LAST':
        target_co = verts[-1].co.copy()
    else:  # CENTER
        target_co = sum((v.co for v in verts), Vector()) / len(verts)
    bmesh.ops.pointmerge(bm, verts=verts, merge_co=target_co)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def weld_verts_by_distance(obj, vert_indices=None, distance=0.001):
    """Merge nearby vertices within *distance*. If vert_indices=None, operates on all verts."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    if vert_indices is not None:
        verts = [bm.verts[i] for i in vert_indices if i < len(bm.verts)]
    else:
        verts = list(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=verts, dist=distance)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def connect_verts(obj, vert_indices):
    """Connect vertices with edges/faces through the mesh interior (like J key)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    verts = [bm.verts[i] for i in vert_indices if i < len(bm.verts)]
    if len(verts) >= 2:
        try:
            bmesh.ops.connect_verts(bm, verts=verts)
        except Exception:
            pass
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def knife_cut(obj, cut_coords, face_indices=None):
    """Knife-project a line through geometry. *cut_coords* = list of (x,y,z) local coords defining the cut path."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    if face_indices is not None:
        faces = [bm.faces[i] for i in face_indices if i < len(bm.faces)]
    else:
        faces = list(bm.faces)

    # Build pairs for knife
    edges = []
    for i in range(len(cut_coords) - 1):
        p1 = Vector(cut_coords[i])
        p2 = Vector(cut_coords[i + 1])
        edges.append((p1, p2))

    for p1, p2 in edges:
        try:
            bmesh.ops.bisect_plane(
                bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
                plane_co=(p1 + p2) / 2,
                plane_no=(p2 - p1).cross(Vector((0, 0, 1))).normalized(),
                clear_inner=False, clear_outer=False)
        except Exception:
            pass

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def bisect(obj, plane_co=(0, 0, 0), plane_no=(0, 0, 1), clear_inner=False, clear_outer=False, fill=False):
    """Cut mesh with a plane. *plane_co* = point on plane, *plane_no* = plane normal. Optionally clear one side and/or fill the cut."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    result = bmesh.ops.bisect_plane(
        bm, geom=geom,
        plane_co=Vector(plane_co), plane_no=Vector(plane_no),
        clear_inner=clear_inner, clear_outer=clear_outer)
    if fill:
        edges = [e for e in result['geom_cut'] if isinstance(e, bmesh.types.BMEdge)]
        if edges:
            try:
                bmesh.ops.edgeloop_fill(bm, edges=edges)
            except Exception:
                pass
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def flatten_verts(obj, vert_indices, axis='Z'):
    """Flatten verts to the same coordinate on *axis* ('X', 'Y', or 'Z'). Uses average position."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    verts = [bm.verts[i] for i in vert_indices if i < len(bm.verts)]
    if not verts:
        bm.free()
        return
    ax = {'X': 0, 'Y': 1, 'Z': 2}.get(axis.upper(), 2)
    avg = sum(v.co[ax] for v in verts) / len(verts)
    for v in verts:
        v.co[ax] = avg
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def snap_verts_to(obj, vert_indices, axis='Z', value=0.0):
    """Snap vertices to an exact value on the given axis."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    ax = {'X': 0, 'Y': 1, 'Z': 2}.get(axis.upper(), 2)
    for i in vert_indices:
        if i < len(bm.verts):
            bm.verts[i].co[ax] = value
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def scale_verts(obj, vert_indices, scale, pivot='MEDIAN'):
    """Scale vertices around a pivot. *scale* = (sx, sy, sz). pivot: 'MEDIAN', 'ORIGIN', or (x,y,z)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    verts = [bm.verts[i] for i in vert_indices if i < len(bm.verts)]
    if not verts:
        bm.free()
        return
    if isinstance(pivot, (list, tuple)):
        center = Vector(pivot)
    elif pivot == 'ORIGIN':
        center = Vector((0, 0, 0))
    else:  # MEDIAN
        center = sum((v.co for v in verts), Vector()) / len(verts)
    sx, sy, sz = scale
    for v in verts:
        v.co.x = center.x + (v.co.x - center.x) * sx
        v.co.y = center.y + (v.co.y - center.y) * sy
        v.co.z = center.z + (v.co.z - center.z) * sz
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


def rotate_verts(obj, vert_indices, angle_deg, axis='Z', pivot='MEDIAN'):
    """Rotate vertices around an axis. pivot: 'MEDIAN', 'ORIGIN', or (x,y,z)."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    verts = [bm.verts[i] for i in vert_indices if i < len(bm.verts)]
    if not verts:
        bm.free()
        return
    if isinstance(pivot, (list, tuple)):
        center = Vector(pivot)
    elif pivot == 'ORIGIN':
        center = Vector((0, 0, 0))
    else:
        center = sum((v.co for v in verts), Vector()) / len(verts)
    axis_vec = {'X': Vector((1, 0, 0)), 'Y': Vector((0, 1, 0)), 'Z': Vector((0, 0, 1))}.get(axis.upper(), Vector((0, 0, 1)))
    mat = Matrix.Rotation(math.radians(angle_deg), 4, axis_vec)
    for v in verts:
        v.co = center + mat @ (v.co - center)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# ═══════════════════════════════════════════════════════════════════════════
# Object Alignment & Positioning
# ═══════════════════════════════════════════════════════════════════════════

def align_objects(objects, axis='X', mode='CENTER'):
    """Align multiple objects along an axis. mode: 'MIN', 'MAX', 'CENTER', 'DISTRIBUTE'."""
    if not objects or len(objects) < 2:
        return
    ax = {'X': 0, 'Y': 1, 'Z': 2}.get(axis.upper(), 0)

    if mode == 'DISTRIBUTE':
        # Distribute evenly between first and last
        objects_sorted = sorted(objects, key=lambda o: o.location[ax])
        if len(objects_sorted) < 3:
            return
        start = objects_sorted[0].location[ax]
        end = objects_sorted[-1].location[ax]
        step = (end - start) / (len(objects_sorted) - 1)
        for i, obj in enumerate(objects_sorted):
            loc = list(obj.location)
            loc[ax] = start + step * i
            obj.location = tuple(loc)
    else:
        if mode == 'MIN':
            target = min(o.location[ax] for o in objects)
        elif mode == 'MAX':
            target = max(o.location[ax] for o in objects)
        else:  # CENTER
            target = sum(o.location[ax] for o in objects) / len(objects)
        for obj in objects:
            loc = list(obj.location)
            loc[ax] = target
            obj.location = tuple(loc)


def snap_to_grid(obj, grid_size=1.0):
    """Snap an object's location to the nearest grid point."""
    loc = obj.location
    obj.location = (
        round(loc.x / grid_size) * grid_size,
        round(loc.y / grid_size) * grid_size,
        round(loc.z / grid_size) * grid_size,
    )


def match_location(source, target, axes='XYZ'):
    """Copy location from *target* to *source* on specified axes."""
    loc = list(source.location)
    for ax_name in axes.upper():
        idx = {'X': 0, 'Y': 1, 'Z': 2}.get(ax_name, -1)
        if idx >= 0:
            loc[idx] = target.location[idx]
    source.location = tuple(loc)


def match_dimensions(source, target, axes='XYZ'):
    """Match *source* dimensions to *target* dimensions on specified axes."""
    for ax_name in axes.upper():
        idx = {'X': 0, 'Y': 1, 'Z': 2}.get(ax_name, -1)
        if idx >= 0 and target.dimensions[idx] > 0:
            ratio = target.dimensions[idx] / source.dimensions[idx] if source.dimensions[idx] > 0 else 1
            scale = list(source.scale)
            scale[idx] *= ratio
            source.scale = tuple(scale)


def copy_transforms(source, target):
    """Copy location, rotation, and scale from *target* to *source*."""
    source.location = target.location.copy()
    source.rotation_euler = target.rotation_euler.copy()
    source.scale = target.scale.copy()


def place_on_ground(obj):
    """Move object so its bottom sits at Z=0."""
    bb = [Vector(v) for v in obj.bound_box]
    world_bb = [obj.matrix_world @ v for v in bb]
    min_z = min(v.z for v in world_bb)
    obj.location.z -= min_z


def stack_on(obj, target, gap=0.0):
    """Place *obj* on top of *target*.

    Moves *obj* so its bottom touches *target*'s top (+ optional gap).
    Also aligns obj's XY center to target's XY center.

    WARNING: This moves obj to target's XY position. If the target is
    one wheel and you want the body centered over all 4 wheels, use
    center_at() to set X/Y manually instead.
    """
    tbb = [target.matrix_world @ Vector(v) for v in target.bound_box]
    target_top = max(v.z for v in tbb)
    obb = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    obj_bottom = min(v.z for v in obb)
    obj.location.z += (target_top + gap) - obj_bottom
    obj.location.x = target.location.x
    obj.location.y = target.location.y


def center_at(obj, x=None, y=None, z=None):
    """Set specific axes of an object's location, leaving others unchanged.

    Only the axes you specify are changed. Omitted axes keep their current value.
    E.g. center_at(body, x=0, y=0) centers the body at origin XY but keeps
    its current Z height.
    """
    if x is not None:
        obj.location.x = x
    if y is not None:
        obj.location.y = y
    if z is not None:
        obj.location.z = z


def apply_transforms(obj, location=True, rotation=True, scale=True):
    """Apply transforms (freeze transforms into mesh data). Like Ctrl+A in Blender."""
    deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.transform_apply(location=location, rotation=rotation, scale=scale)
    except Exception:
        pass


def set_origin(obj, origin='GEOMETRY'):
    """Set object origin. origin: 'GEOMETRY' (center of mesh), 'CURSOR', 'BOUNDS' (center of bounds), 'BOTTOM' (bottom center)."""
    deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        if origin == 'BOTTOM':
            # Move origin to bottom center of bounding box
            bb = [Vector(v) for v in obj.bound_box]
            world_bb = [obj.matrix_world @ v for v in bb]
            bottom_z = min(v.z for v in world_bb)
            cx = sum(v.x for v in world_bb) / 8
            cy = sum(v.y for v in world_bb) / 8
            bpy.context.scene.cursor.location = (cx, cy, bottom_z)
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        elif origin == 'CURSOR':
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        elif origin == 'BOUNDS':
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        else:
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Mesh Joining & Welding
# ═══════════════════════════════════════════════════════════════════════════

def join_and_merge(objects, merge_distance=0.001):
    """Join objects into one AND merge overlapping vertices. Returns the joined object."""
    if not objects:
        return None
    result = join_objects(objects)
    if result:
        merge_by_distance(result, distance=merge_distance)
    return result


def snap_object_to(obj, target, snap='BOTTOM_TO_TOP', offset=0.0):
    """Snap one object to another. snap modes:
    'BOTTOM_TO_TOP' — obj bottom to target top
    'TOP_TO_BOTTOM' — obj top to target bottom
    'CENTER' — center to center
    'ORIGIN' — origin to origin
    """
    if snap == 'ORIGIN':
        obj.location = target.location.copy()
        obj.location.z += offset
        return

    tbb = [target.matrix_world @ Vector(v) for v in target.bound_box]
    obb = [obj.matrix_world @ Vector(v) for v in obj.bound_box]

    if snap == 'BOTTOM_TO_TOP':
        target_top = max(v.z for v in tbb)
        obj_bottom = min(v.z for v in obb)
        obj.location.z += (target_top + offset) - obj_bottom
    elif snap == 'TOP_TO_BOTTOM':
        target_bottom = min(v.z for v in tbb)
        obj_top = max(v.z for v in obb)
        obj.location.z += (target_bottom + offset) - obj_top
    elif snap == 'CENTER':
        tc = sum(tbb, Vector()) / len(tbb)
        oc = sum(obb, Vector()) / len(obb)
        obj.location += tc - oc
        obj.location.z += offset


class Bounds:
    """World-space bounding box info. Access via attributes:
       .min_x .min_y .min_z  .max_x .max_y .max_z
       .center_x .center_y .center_z
       .width  = size along X axis  (max_x - min_x)
       .depth  = size along Y axis  (max_y - min_y)
       .height = size along Z axis  (max_z - min_z)

    IMPORTANT axis mapping: width=X, depth=Y, height=Z.
    This matches create_box(width=X, depth=Y, height=Z).
    A cylinder's default axis is Z, so its bounding .height = cylinder depth,
    and .width = .depth = cylinder diameter.
    """
    __slots__ = ('min_x','min_y','min_z','max_x','max_y','max_z',
                 'center_x','center_y','center_z','width','depth','height')

    def __init__(self, min_x, min_y, min_z, max_x, max_y, max_z):
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z
        self.center_x = (min_x + max_x) / 2
        self.center_y = (min_y + max_y) / 2
        self.center_z = (min_z + max_z) / 2
        self.width  = max_x - min_x
        self.depth  = max_y - min_y
        self.height = max_z - min_z

    def __repr__(self):
        return ("Bounds(x=%.2f..%.2f, y=%.2f..%.2f, z=%.2f..%.2f, "
                "size=%.2f×%.2f×%.2f)") % (
            self.min_x, self.max_x, self.min_y, self.max_y,
            self.min_z, self.max_z, self.width, self.depth, self.height)

    # Keep dict-style access as fallback so old code doesn't crash
    def __getitem__(self, key):
        mapping = {
            'min': (self.min_x, self.min_y, self.min_z),
            'max': (self.max_x, self.max_y, self.max_z),
            'center': (self.center_x, self.center_y, self.center_z),
            'dimensions': (self.width, self.depth, self.height),
        }
        return mapping[key]


def get_bounds(obj):
    """Get world-space bounding box as a Bounds object with direct attributes:
      b = get_bounds(obj)
      b.min_x, b.max_z, b.center_y, b.width, b.height, etc.
    Returns None if obj is None or invalid."""
    if obj is None:
        return None
    # Force Blender to compute up-to-date transforms and bounding boxes
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    try:
        bb = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
        if not bb:
            loc = obj.location
            dim = obj.dimensions
            half = dim / 2
            return Bounds(loc.x - half.x, loc.y - half.y, loc.z - half.z,
                          loc.x + half.x, loc.y + half.y, loc.z + half.z)
        xs = [v.x for v in bb]
        ys = [v.y for v in bb]
        zs = [v.z for v in bb]
        return Bounds(min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))
    except Exception:
        loc = obj.location
        return Bounds(loc.x, loc.y, loc.z, loc.x, loc.y, loc.z)


# ═══════════════════════════════════════════════════════════════════════════
# Viewport
# ═══════════════════════════════════════════════════════════════════════════

def set_viewport_shading(mode='MATERIAL'):
    """Set the 3D viewport shading mode.
    mode: 'WIREFRAME', 'SOLID', 'MATERIAL', or 'RENDERED'.
    'MATERIAL' shows Material Preview (Eevee preview with materials visible).
    'RENDERED' shows full render preview.
    """
    mode_map = {
        'WIREFRAME': 'WIREFRAME',
        'SOLID': 'SOLID',
        'MATERIAL': 'MATERIAL',
        'RENDERED': 'RENDERED',
    }
    shading_type = mode_map.get(mode.upper(), 'MATERIAL')
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = shading_type
            break


def capture_viewport(filepath=None, resolution=(960, 540)):
    """Render a viewport snapshot and save to disk.
    Returns the filepath of the saved image.
    """
    import tempfile
    if filepath is None:
        filepath = os.path.join(tempfile.gettempdir(), "ai_copilot_viewport.png")

    # Store original settings
    scene = bpy.context.scene
    orig_x = scene.render.resolution_x
    orig_y = scene.render.resolution_y
    orig_pct = scene.render.resolution_percentage
    orig_format = scene.render.image_settings.file_format

    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'

    # Try OpenGL viewport render (fast)
    try:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        with bpy.context.temp_override(area=area, region=region):
                            bpy.ops.render.opengl(write_still=False)
                        break
                break

        if bpy.data.images.get('Render Result'):
            bpy.data.images['Render Result'].save_render(filepath)
    except Exception:
        pass
    finally:
        # Restore
        scene.render.resolution_x = orig_x
        scene.render.resolution_y = orig_y
        scene.render.resolution_percentage = orig_pct
        scene.render.image_settings.file_format = orig_format

    return filepath

# ═══════════════════════════════════════════════════════════════════════════
# Advanced Vertex Selection & Manipulation
# ═══════════════════════════════════════════════════════════════════════════

def find_verts_near(obj, x=None, y=None, z=None, tolerance=0.05):
    """Find vertex indices near a world-space coordinate.

    Specify one, two, or all three axes. Only specified axes are tested.
    Returns a list of vertex indices that match within *tolerance*.

    Example — find all verts near the front of a car (X≈2.0):
        front_verts = find_verts_near(body, x=2.0, tolerance=0.2)
    """
    mw = obj.matrix_world
    result = []
    for v in obj.data.vertices:
        co = mw @ v.co
        if x is not None and abs(co.x - x) > tolerance:
            continue
        if y is not None and abs(co.y - y) > tolerance:
            continue
        if z is not None and abs(co.z - z) > tolerance:
            continue
        result.append(v.index)
    return result


def find_verts_in_range(obj, axis='Z', min_val=0.0, max_val=1.0):
    """Find vertex indices within a world-space range on one axis.

    *axis* is 'X', 'Y', or 'Z'. Returns indices where
    min_val <= coord <= max_val.

    Example — find all verts in the upper half of a body:
        top_verts = find_verts_in_range(body, axis='Z', min_val=1.0, max_val=2.0)
    """
    axis_idx = {'X': 0, 'Y': 1, 'Z': 2}[axis.upper()]
    mw = obj.matrix_world
    result = []
    for v in obj.data.vertices:
        co = mw @ v.co
        if min_val <= co[axis_idx] <= max_val:
            result.append(v.index)
    return result


def proportional_translate(obj, vert_indices, offset, falloff_radius=1.0,
                           falloff='SMOOTH'):
    """Move vertices with proportional falloff affecting nearby verts.

    *vert_indices* — the "selected" verts that get the full offset.
    *offset* — (x, y, z) translation applied to selected verts.
    *falloff_radius* — how far the influence extends (in local space).
    *falloff* — 'SMOOTH', 'LINEAR', 'SHARP', 'SPHERE', or 'CONSTANT'.

    Nearby verts within falloff_radius get a fraction of the offset
    based on their distance to the nearest selected vert.

    Example — push the front hood down for a sleeker profile:
        front = find_verts_near(body, x=2.0, tolerance=0.3)
        proportional_translate(body, front, offset=(0, 0, -0.2), falloff_radius=1.5)
    """
    import bmesh as _bm
    from mathutils import Vector as _Vec

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    sel_set = set(vert_indices)
    sel_positions = [bm.verts[i].co.copy() for i in sel_set if i < len(bm.verts)]
    off = _Vec(offset)

    for v in bm.verts:
        if v.index in sel_set:
            v.co += off
        else:
            # Find minimum distance to any selected vert
            min_dist = min((v.co - sp).length for sp in sel_positions) if sel_positions else float('inf')
            if min_dist < falloff_radius and falloff_radius > 0:
                t = min_dist / falloff_radius
                if falloff == 'LINEAR':
                    factor = 1.0 - t
                elif falloff == 'SHARP':
                    factor = (1.0 - t) ** 2
                elif falloff == 'SPHERE':
                    factor = (1.0 - t * t) ** 0.5 if t < 1 else 0
                elif falloff == 'CONSTANT':
                    factor = 1.0
                else:  # SMOOTH (default)
                    factor = (1.0 - t) ** 2 * (1.0 + 2.0 * t)
                v.co += off * factor

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def smooth_verts(obj, iterations=1, factor=0.5, vert_indices=None):
    """Smooth (relax) vertex positions by averaging with neighbours.

    Like Blender's Smooth Vertex tool. Each vertex moves toward the
    average position of its connected neighbours.

    *vert_indices* — limit to these verts. None = all verts.
    *factor* — 0.0 = no smoothing, 1.0 = full average. Default 0.5.
    *iterations* — repeat the smooth pass this many times.
    """
    import bmesh as _bm

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    if vert_indices is not None:
        target_set = set(vert_indices)
    else:
        target_set = None

    for _ in range(iterations):
        new_positions = {}
        for v in bm.verts:
            if target_set is not None and v.index not in target_set:
                continue
            neighbours = [e.other_vert(v) for e in v.link_edges]
            if not neighbours:
                continue
            avg = sum((n.co for n in neighbours), v.co.copy() * 0) / len(neighbours)
            new_positions[v.index] = v.co.lerp(avg, factor)
        for idx, pos in new_positions.items():
            bm.verts[idx].co = pos

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def symmetrize_mesh(obj, axis='X', direction='POSITIVE'):
    """Mirror mesh data across an axis, deleting one side.

    *axis* — 'X', 'Y', or 'Z'.
    *direction* — 'POSITIVE' keeps +axis side, mirrors to -axis.
                  'NEGATIVE' keeps -axis side, mirrors to +axis.

    This is destructive (bakes symmetry into the mesh), unlike mirror().
    Useful after manually editing one side of a symmetric model.
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map[axis.upper()]

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    # Delete verts on the "wrong" side
    threshold = 0.0001
    to_delete = []
    for v in bm.verts:
        if direction == 'POSITIVE' and v.co[ax] < -threshold:
            to_delete.append(v)
        elif direction == 'NEGATIVE' and v.co[ax] > threshold:
            to_delete.append(v)

    import bmesh as _bm2
    _bm2.ops.delete(bm, geom=to_delete, context='VERTS')

    bm.to_mesh(obj.data)
    bm.free()

    # Now add a mirror modifier and apply it
    deselect_all()
    obj.select_set(True)
    import bpy as _bpy
    _bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("Symmetrize", 'MIRROR')
    mod.use_axis = [ax == 0, ax == 1, ax == 2]
    mod.merge_threshold = 0.001
    try:
        with _bpy.context.temp_override(object=obj):
            _bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception:
        pass
    obj.data.update()


def simple_deform(obj, method='TWIST', angle=45.0, axis='Z', factor=1.0):
    """Add a Simple Deform modifier.

    *method* — 'TWIST', 'BEND', 'TAPER', or 'STRETCH'.
    *angle* — deformation angle in degrees (for TWIST and BEND).
    *factor* — deformation factor (for TAPER and STRETCH).
    *axis* — deform axis: 'X', 'Y', or 'Z'.
    """
    mod = obj.modifiers.new("SimpleDeform", 'SIMPLE_DEFORM')
    mod.deform_method = method.upper()
    mod.deform_axis = axis.upper()
    if method.upper() in ('TWIST', 'BEND'):
        mod.angle = math.radians(angle)
    else:
        mod.factor = factor
    return mod


def create_vertex_group(obj, name, vert_indices=None, weight=1.0):
    """Create a vertex group and optionally assign vertices.

    Returns the vertex group. Use this to control modifier influence
    (e.g. limit a deform to certain verts).

    Example:
        top_verts = find_verts_in_range(body, 'Z', 1.0, 2.0)
        vg = create_vertex_group(body, "TopVerts", top_verts, weight=1.0)
    """
    vg = obj.vertex_groups.new(name=name)
    if vert_indices:
        vg.add(vert_indices, weight, 'REPLACE')
    return vg


def assign_vertex_group(obj, group_name, vert_indices, weight=1.0):
    """Add vertices to an existing vertex group."""
    vg = obj.vertex_groups.get(group_name)
    if not vg:
        vg = obj.vertex_groups.new(name=group_name)
    vg.add(vert_indices, weight, 'REPLACE')
    return vg


def shrink_fatten(obj, vert_indices=None, offset=0.1):
    """Move vertices along their normals (shrink/fatten).

    Positive offset = outward (fatten), negative = inward (shrink).
    If *vert_indices* is None, affects all verts.
    This is essential for inflating/deflating mesh regions to create
    organic shapes from flat geometry.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    if vert_indices is None:
        verts = list(bm.verts)
    else:
        verts = [bm.verts[i] for i in vert_indices if i < len(bm.verts)]

    for v in verts:
        v.co += v.normal * offset

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def extrude_region(obj, face_indices, offset=0.5):
    """Extrude a connected region of faces along their average normal.

    Unlike extrude_faces (which extrudes each face individually),
    this extrudes the region as a whole — keeping shared edges connected.
    This is how professional modelers pull out shapes from a base mesh.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    faces = [bm.faces[i] for i in face_indices if i < len(bm.faces)]
    if not faces:
        bm.free()
        return obj

    # Calculate average normal
    avg_normal = Vector((0, 0, 0))
    for f in faces:
        avg_normal += f.normal
    avg_normal.normalize()

    result = bmesh.ops.extrude_face_region(bm, geom=faces)
    new_verts = [g for g in result['geom'] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=avg_normal * offset, verts=new_verts)

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def select_face_loop(obj, face_index, axis='X'):
    """Select a loop of faces along an axis direction from a starting face.

    Returns list of face indices forming the loop. Useful for selecting
    a ring of faces around a model for extrusion or material assignment.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    if face_index >= len(bm.faces):
        bm.free()
        return []

    axis_idx = {'X': 0, 'Y': 1, 'Z': 2}.get(axis.upper(), 0)
    start = bm.faces[face_index]
    center_val = start.calc_center_median()[axis_idx]
    tolerance = max(f.calc_center_median()[axis_idx] for f in bm.faces) * 0.05 + 0.01

    result = []
    for f in bm.faces:
        if abs(f.calc_center_median()[axis_idx] - center_val) < tolerance:
            result.append(f.index)

    bm.free()
    return result


def edge_loop_from_edge(obj, edge_index):
    """Select an entire edge loop starting from one edge index.

    Returns a list of edge indices forming the loop.
    Like Alt+Click on an edge in Blender.
    """
    import bmesh as _bm

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()

    if edge_index >= len(bm.edges):
        bm.free()
        return []

    # Walk along the loop
    start = bm.edges[edge_index]
    loop_edges = set()
    loop_edges.add(start.index)

    def _walk(edge, vert, visited):
        """Walk the loop from vert along edge."""
        while True:
            visited.add(edge.index)
            # Find the other vert
            other = edge.other_vert(vert)
            # At other vert, find the "opposite" edge (across a quad)
            link = other.link_edges
            candidates = [e for e in link if e.index not in visited
                          and e != edge and len(e.link_faces) > 0]
            # For a quad mesh, the loop edge shares exactly one face with current
            next_edge = None
            for e in candidates:
                shared = set(edge.link_faces) & set(e.link_faces)
                if len(shared) == 1:
                    face = list(shared)[0]
                    if len(face.verts) == 4:
                        next_edge = e
                        break
            if next_edge is None:
                break
            loop_edges.add(next_edge.index)
            edge = next_edge
            vert = other

    for vert in start.verts:
        _walk(start, vert, set(loop_edges))

    result = sorted(loop_edges)
    bm.free()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# High-Level Geometry Editing — in-place mesh shaping & sculpting
# ═══════════════════════════════════════════════════════════════════════════

def get_mesh_profile(obj, axis='X', num_slices=10):
    """Analyze mesh cross-sections along *axis*. Returns a list of dicts:
    [{"pos": float, "width": float, "height": float, "vert_count": int}, ...]

    Useful for understanding and modifying the shape of bodies, hulls, etc.
    Use the returned data to decide where to taper, scale, or reshape.
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)
    # Width/height axes
    other_axes = [i for i in range(3) if i != ax]

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bm.verts.ensure_lookup_table()

    if not bm.verts:
        bm.free()
        return []

    coords = [v.co[ax] for v in bm.verts]
    min_val, max_val = min(coords), max(coords)
    span = max_val - min_val
    if span < 0.001:
        bm.free()
        return [{"pos": min_val, "width": 0, "height": 0, "vert_count": len(bm.verts)}]

    slices = []
    for i in range(num_slices):
        t = i / max(num_slices - 1, 1)
        pos = min_val + t * span
        tol = span / (num_slices * 2)

        nearby = [v for v in bm.verts if abs(v.co[ax] - pos) < tol]
        if not nearby:
            # Expand tolerance
            tol *= 3
            nearby = [v for v in bm.verts if abs(v.co[ax] - pos) < tol]

        if nearby:
            vals_a = [v.co[other_axes[0]] for v in nearby]
            vals_b = [v.co[other_axes[1]] for v in nearby]
            width = max(vals_a) - min(vals_a) if vals_a else 0
            height = max(vals_b) - min(vals_b) if vals_b else 0
        else:
            width, height = 0, 0

        slices.append({
            "pos": round(pos, 3),
            "width": round(width, 3),
            "height": round(height, 3),
            "vert_count": len(nearby),
        })

    bm.free()
    return slices


def get_mesh_analysis(obj):
    """Get comprehensive mesh measurements for AI inspection.

    Returns dict with: bounds, dimensions, center, vertex_count, face_count,
    and cross-section profiles along X, Y, Z.
    """
    import bmesh as _bm

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    if not bm.verts:
        bm.free()
        return {"empty": True}

    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]

    result = {
        "bounds": {
            "min": [round(min(xs), 3), round(min(ys), 3), round(min(zs), 3)],
            "max": [round(max(xs), 3), round(max(ys), 3), round(max(zs), 3)],
        },
        "dimensions": [
            round(max(xs) - min(xs), 3),
            round(max(ys) - min(ys), 3),
            round(max(zs) - min(zs), 3),
        ],
        "center": [
            round((min(xs) + max(xs)) / 2, 3),
            round((min(ys) + max(ys)) / 2, 3),
            round((min(zs) + max(zs)) / 2, 3),
        ],
        "verts": len(bm.verts),
        "faces": len(bm.faces),
    }

    bm.free()

    # Profile along all three axes — AI determines which is length/width/height
    result["profile_X"] = get_mesh_profile(obj, 'X', 8)
    result["profile_Y"] = get_mesh_profile(obj, 'Y', 8)
    result["profile_Z"] = get_mesh_profile(obj, 'Z', 6)

    # Auto-detect length axis (whichever horizontal axis has the larger span)
    x_span = result["dimensions"][0]
    y_span = result["dimensions"][1]
    result["length_axis"] = 'X' if x_span >= y_span else 'Y'
    result["width_axis"] = 'Y' if x_span >= y_span else 'X'

    return result


def taper(obj, axis='X', start_scale=1.0, end_scale=0.5,
          start_pos=None, end_pos=None, axes='YZ'):
    """Gradually scale vertices along *axis* from *start_scale* to *end_scale*.

    Like narrowing a car body toward the front or back.
    *axes* controls which axes get scaled (default 'YZ' = width and height).
    *start_pos*/*end_pos* limit the taper range (None = full mesh extent).
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)
    scale_axes = [axis_map[a] for a in axes.upper() if a in axis_map]

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    coords = [v.co[ax] for v in bm.verts]
    lo = start_pos if start_pos is not None else min(coords)
    hi = end_pos if end_pos is not None else max(coords)
    span = hi - lo
    if span < 0.0001:
        bm.free()
        return obj

    # Compute center at each slice for correct pivot
    for v in bm.verts:
        t = max(0.0, min(1.0, (v.co[ax] - lo) / span))
        s = start_scale + t * (end_scale - start_scale)
        # Scale relative to the axis (around local center per-slice)
        for sa in scale_axes:
            v.co[sa] *= s

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def sculpt_move(obj, center, radius, offset, falloff='SMOOTH'):
    """Move vertices near *center* by *offset* with smooth falloff.

    Like the Sculpt Grab brush — vertices close to center move fully,
    those near the edge of *radius* move less.
    *center* = (x, y, z) in world space.
    *offset* = (dx, dy, dz) displacement.
    *falloff* = 'SMOOTH', 'LINEAR', 'SHARP', or 'CONSTANT'.
    """
    import bmesh as _bm

    center_v = Vector(center)
    offset_v = Vector(offset)
    imat = obj.matrix_world.inverted()
    local_center = imat @ center_v
    # Scale offset to local space
    local_offset = imat.to_3x3() @ offset_v

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    # Local-space radius (approximate)
    scale_factor = sum(obj.scale) / 3.0
    local_radius = radius / max(scale_factor, 0.001)

    for v in bm.verts:
        dist = (v.co - local_center).length
        if dist > local_radius:
            continue
        t = dist / local_radius
        if falloff == 'SMOOTH':
            w = 1.0 - (3 * t * t - 2 * t * t * t)  # smoothstep
        elif falloff == 'LINEAR':
            w = 1.0 - t
        elif falloff == 'SHARP':
            w = (1.0 - t) ** 2
        elif falloff == 'CONSTANT':
            w = 1.0
        else:
            w = 1.0 - t
        v.co += local_offset * w

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def scale_section(obj, axis='X', position=0.0, tolerance=0.1,
                  scale_x=1.0, scale_y=1.0, scale_z=1.0):
    """Scale vertices at a cross-section of the mesh.

    Finds all verts within *tolerance* of *position* along *axis*,
    then scales them around the section center.
    Great for reshaping bodies: widen the hips, narrow the waist, etc.
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)
    scales = [scale_x, scale_y, scale_z]

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    section_verts = [v for v in bm.verts if abs(v.co[ax] - position) < tolerance]
    if not section_verts:
        bm.free()
        return obj

    # Find section center
    center = Vector((0, 0, 0))
    for v in section_verts:
        center += v.co
    center /= len(section_verts)

    for v in section_verts:
        for i in range(3):
            if i != ax:  # Don't scale along the slicing axis
                v.co[i] = center[i] + (v.co[i] - center[i]) * scales[i]

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def move_section(obj, axis='X', position=0.0, tolerance=0.1,
                 offset=(0, 0, 0)):
    """Move all vertices at a cross-section of the mesh.

    Finds all verts within *tolerance* of *position* along *axis*,
    then translates them by *offset*. Good for adjusting the
    profile of a body — push the roof up, pull the nose down, etc.
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    for v in bm.verts:
        if abs(v.co[ax] - position) < tolerance:
            v.co.x += offset[0]
            v.co.y += offset[1]
            v.co.z += offset[2]

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def set_profile_shape(obj, axis='X', profile=None):
    """Reshape mesh to match a target width/height profile.

    *profile* is a list of (position, width_scale, height_scale) tuples
    along *axis*. The mesh is reshaped so each cross-section matches
    the given proportions. Values are relative to the current dimensions.

    Example: set_profile_shape(body, 'X', [
        (-2.0, 0.5, 0.3),   # narrow + low at front
        (-1.0, 0.8, 0.7),   # widening
        (0.0,  1.0, 1.0),   # full width at center
        (1.0,  0.9, 0.8),   # tapering
        (2.0,  0.6, 0.4),   # narrow at rear
    ])
    """
    if not profile or len(profile) < 2:
        return obj

    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)
    other = [i for i in range(3) if i != ax]

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    if not bm.verts:
        bm.free()
        return obj

    # Sort profile by position
    profile = sorted(profile, key=lambda p: p[0])

    # Get mesh bounds along axis
    coords = [v.co[ax] for v in bm.verts]
    mesh_min, mesh_max = min(coords), max(coords)
    mesh_span = mesh_max - mesh_min
    if mesh_span < 0.001:
        bm.free()
        return obj

    # Compute current section centers and extents
    for v in bm.verts:
        pos = v.co[ax]
        # Normalize position to [0, 1] within mesh
        t_mesh = (pos - mesh_min) / mesh_span

        # Interpolate profile at this position
        p_min = profile[0][0]
        p_max = profile[-1][0]
        p_span = p_max - p_min
        if p_span < 0.001:
            continue

        # Map mesh position to profile position
        p_pos = p_min + t_mesh * p_span

        # Find surrounding profile points and interpolate
        w_scale, h_scale = 1.0, 1.0
        for j in range(len(profile) - 1):
            if profile[j][0] <= p_pos <= profile[j + 1][0]:
                seg_span = profile[j + 1][0] - profile[j][0]
                if seg_span > 0.001:
                    t = (p_pos - profile[j][0]) / seg_span
                else:
                    t = 0
                w_scale = profile[j][1] + t * (profile[j + 1][1] - profile[j][1])
                h_scale = profile[j][2] + t * (profile[j + 1][2] - profile[j][2])
                break
        else:
            # Outside profile range — use nearest endpoint
            if p_pos <= profile[0][0]:
                w_scale, h_scale = profile[0][1], profile[0][2]
            else:
                w_scale, h_scale = profile[-1][1], profile[-1][2]

        # Apply scale relative to mesh center at this cross-section
        v.co[other[0]] *= w_scale
        v.co[other[1]] *= h_scale

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def bend(obj, axis='X', angle_deg=30, center=None, bend_axis='Z'):
    """Bend mesh along *axis* by rotating verts progressively.

    Like the Simple Deform modifier in BEND mode but as a direct mesh edit.
    *axis* = the axis along which the bend progresses.
    *bend_axis* = the axis around which verts rotate.
    *angle_deg* = total bend angle from start to end.
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)
    bend_ax = axis_map.get(bend_axis.upper(), 2)

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    coords = [v.co[ax] for v in bm.verts]
    lo, hi = min(coords), max(coords)
    span = hi - lo
    if span < 0.001:
        bm.free()
        return obj

    angle_rad = math.radians(angle_deg)

    # Determine rotation plane
    rot_axes = [i for i in range(3) if i != bend_ax]

    center_pt = Vector(center) if center else Vector((0, 0, 0))

    for v in bm.verts:
        t = (v.co[ax] - lo) / span
        angle = angle_rad * (t - 0.5)  # centered bend
        ca, sa = math.cos(angle), math.sin(angle)

        # Rotate in the plane perpendicular to bend_axis
        a, b = rot_axes[0], rot_axes[1]
        dx = v.co[a] - center_pt[a]
        dy = v.co[b] - center_pt[b]
        v.co[a] = center_pt[a] + dx * ca - dy * sa
        v.co[b] = center_pt[b] + dx * sa + dy * ca

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def add_detail_cuts(obj, axis='X', num_cuts=3):
    """Add evenly-spaced loop cuts along *axis* to increase mesh resolution.

    More resolution = more verts for the AI to sculpt and reshape.
    Use before taper(), sculpt_move(), or set_profile_shape() for smoother results.
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()

    if not bm.edges:
        bm.free()
        return obj

    # Find edges that run along the specified axis
    axis_edges = []
    for e in bm.edges:
        v1, v2 = e.verts
        delta = v2.co - v1.co
        # Edge runs along axis if its component along axis is dominant
        if abs(delta[ax]) > 0.001:
            other_len = sum(abs(delta[i]) for i in range(3) if i != ax)
            if abs(delta[ax]) > other_len * 0.5:
                axis_edges.append(e)

    if not axis_edges:
        bm.free()
        return obj

    # Use bmesh bisect to add cuts at evenly-spaced positions
    coords = [v.co[ax] for v in bm.verts]
    lo, hi = min(coords), max(coords)
    span = hi - lo

    for i in range(1, num_cuts + 1):
        t = i / (num_cuts + 1)
        pos = lo + t * span
        plane_no = [0, 0, 0]
        plane_no[ax] = 1.0
        plane_co = [0, 0, 0]
        plane_co[ax] = pos

        geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
        result = bmesh.ops.bisect_plane(
            bm, geom=geom,
            plane_co=Vector(plane_co),
            plane_no=Vector(plane_no),
            clear_inner=False, clear_outer=False
        )

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def pinch(obj, axis='X', position=0.0, radius=0.5, strength=0.5, pinch_axes='YZ'):
    """Pinch (narrow) or expand vertices near a position along *axis*.

    *strength* > 0 pinches inward, < 0 expands outward.
    Like creating a waist on a body or narrowing a neck.
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)
    p_axes = [axis_map[a] for a in pinch_axes.upper() if a in axis_map]

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    for v in bm.verts:
        dist = abs(v.co[ax] - position)
        if dist > radius:
            continue
        t = dist / radius
        w = 1.0 - (3 * t * t - 2 * t * t * t)  # smoothstep falloff
        factor = 1.0 - strength * w  # <1 = pinch inward, >1 = expand
        for pa in p_axes:
            v.co[pa] *= factor

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def bulge(obj, axis='X', position=0.0, radius=0.5, strength=0.3, bulge_axes='YZ'):
    """Bulge (expand) vertices near a position along *axis*.

    Opposite of pinch — makes a section wider/taller.
    *strength* > 0 expands, like adding muscle or a fender flare.
    """
    return pinch(obj, axis, position, radius, -strength, bulge_axes)


def flatten_region(obj, center, radius, axis='Z'):
    """Flatten all vertices within *radius* of *center* to the same value on *axis*.

    Good for creating flat surfaces on organic shapes — a table top,
    a car roof panel, a flat bottom, etc.
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 2)
    center_v = Vector(center)

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bm.verts.ensure_lookup_table()

    affected = [v for v in bm.verts if (v.co - center_v).length < radius]
    if not affected:
        bm.free()
        return obj

    avg = sum(v.co[ax] for v in affected) / len(affected)
    for v in affected:
        v.co[ax] = avg

    imat = obj.matrix_world.inverted()
    bm.transform(imat)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def crease_edge_loop_at(obj, axis='X', position=0.0, tolerance=0.1, sharpness=0.3):
    """Create a visible crease/edge line at a position along *axis*.

    Finds the nearest edge loop and applies inward scaling to create
    a visible crease line — like a body line on a car, a panel gap,
    or a style line on furniture.
    *sharpness* controls how deep the crease is (0.0 = none, 1.0 = deep).
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    section_verts = [v for v in bm.verts if abs(v.co[ax] - position) < tolerance]
    if not section_verts:
        bm.free()
        return obj

    center = Vector((0, 0, 0))
    for v in section_verts:
        center += v.co
    center /= len(section_verts)

    # Pull verts slightly toward center (creates inward crease)
    for v in section_verts:
        for i in range(3):
            if i != ax:
                v.co[i] = v.co[i] + (center[i] - v.co[i]) * sharpness * 0.1

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def extrude_and_scale(obj, face_indices, extrude_offset=0.2, scale=0.8):
    """Extrude faces then scale them — the most common modeling operation.

    Used for: air intakes, windows, panel details, buttons, grilles,
    wheel arches, headlight cavities, etc.
    *extrude_offset* = how far out (positive) or in (negative).
    *scale* = how much to scale the extruded face (< 1 = inset effect).
    """
    import bmesh as _bm

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    faces = [bm.faces[i] for i in face_indices if i < len(bm.faces)]
    if not faces:
        bm.free()
        return obj

    result = bmesh.ops.extrude_discrete_faces(bm, faces=faces)
    new_faces = [f for f in result['faces']]

    for f in new_faces:
        center = f.calc_center_median()
        normal = f.normal
        for v in f.verts:
            # Extrude along normal
            v.co += normal * extrude_offset
            # Scale relative to face center
            v.co = center + (v.co - center) * scale

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def select_faces_by_position(obj, axis='X', min_val=None, max_val=None,
                              normal_axis=None, normal_sign=None):
    """Find face indices matching positional and normal criteria.

    Returns a list of face indices. Use with extrude_and_scale(),
    inset_faces(), delete_faces(), etc.

    *axis* + *min_val*/*max_val*: filter by face center position.
    *normal_axis* + *normal_sign*: filter by face normal direction.
      e.g. normal_axis='Z', normal_sign='+' = upward-facing faces.
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    result = []
    for f in bm.faces:
        center = f.calc_center_median()

        # Position filter
        if axis:
            ax = axis_map.get(axis.upper(), 0)
            if min_val is not None and center[ax] < min_val:
                continue
            if max_val is not None and center[ax] > max_val:
                continue

        # Normal filter
        if normal_axis:
            nax = axis_map.get(normal_axis.upper(), 2)
            nval = f.normal[nax]
            if normal_sign == '+' and nval < 0.3:
                continue
            if normal_sign == '-' and nval > -0.3:
                continue

        result.append(f.index)

    bm.free()
    return result


def thicken(obj, thickness=0.05, offset=-1):
    """Give a surface mesh thickness (apply solidify and bake it).

    Useful after creating flat panels (glass, trim) to give them
    real-world thickness.
    """
    mod = obj.modifiers.new("Thicken", 'SOLIDIFY')
    mod.thickness = thickness
    mod.offset = offset
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception:
        pass
    return obj


def carve_groove(obj, axis='X', position=0.0, width=0.05, depth=0.02):
    """Cut a groove/channel into the mesh at a position along *axis*.

    Creates panel lines, body creases, or decorative grooves.
    Uses boolean subtraction with a thin box cutter.
    """
    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)

    # Get mesh bounds for cutter size
    bounds = get_bounds(obj)
    dims = [bounds.width * 1.5, bounds.depth * 1.5, bounds.height * 1.5]
    dims[ax] = width

    loc = [bounds.center_x, bounds.center_y, bounds.center_z]
    loc[ax] = position

    cutter = create_box(
        name="_groove_cutter",
        width=dims[0], depth=dims[1], height=dims[2],
        location=tuple(loc)
    )

    # Scale the cutter inward slightly so it only cuts a groove, not through
    # Actually we want it to cut into the surface, so offset by depth
    # along the non-axis directions
    for i in range(3):
        if i != ax:
            dim = dims[i]
            new_dim = dim - depth * 2
            if new_dim > 0:
                scale_factor = new_dim / dim
                cutter.scale[i] = scale_factor

    boolean_cut(obj, cutter, delete_cutter=True)
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# Box-Modeling Convenience Tools
# ═══════════════════════════════════════════════════════════════════════════
# These are the PREFERRED tools for creating complex shapes.  They all take
# simple numbers, fractions, or enums — never raw coordinate arrays.
# The AI should use these instead of inventing vertex positions.

def get_section_positions(obj, axis='X', num_slices=10):
    """Return a list of vertex-position values along *axis*.

    After calling ``add_detail_cuts(obj, 'X', 10)`` use this to discover
    where the new edge loops actually sit so you can pass exact positions
    to ``scale_section`` / ``move_section`` / ``crease_edge_loop_at``.
    *num_slices* controls how many unique positions to return (evenly
    sampled from the actual vertex positions along the axis).
    Returns a sorted list of floats.
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    all_pos = sorted(set(round(v.co[ax], 5) for v in bm.verts))
    bm.free()

    if len(all_pos) <= num_slices:
        return all_pos

    # Evenly sample
    step = max(1, len(all_pos) // num_slices)
    return [all_pos[i] for i in range(0, len(all_pos), step)]


def scale_section_relative(obj, axis='X', fraction=0.5,
                           scale_width=1.0, scale_height=1.0, tolerance=None):
    """Scale a cross-section at a *relative* position along the mesh.

    *fraction* goes from 0.0 (min end) to 1.0 (max end).
    The AI doesn't need to know absolute coordinates — just say
    "at the 25% mark, make it 80% as wide and 60% as tall".
    *scale_width* scales the first perpendicular axis (usually Y).
    *scale_height* scales the second perpendicular axis (usually Z).
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)
    others = [i for i in range(3) if i != ax]

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    coords = [v.co[ax] for v in bm.verts]
    lo, hi = min(coords), max(coords)
    span = hi - lo
    if span < 0.0001:
        bm.free()
        return obj
    position = lo + fraction * span
    if tolerance is None:
        tolerance = span / 40.0  # auto — ~2.5 % of mesh length

    scales = [1.0, 1.0, 1.0]
    scales[others[0]] = scale_width
    scales[others[1]] = scale_height

    section_verts = [v for v in bm.verts if abs(v.co[ax] - position) < tolerance]
    if not section_verts:
        bm.free()
        return obj
    center = Vector((0, 0, 0))
    for v in section_verts:
        center += v.co
    center /= len(section_verts)
    for v in section_verts:
        for i in others:
            v.co[i] = center[i] + (v.co[i] - center[i]) * scales[i]

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def move_section_relative(obj, axis='X', fraction=0.5, offset=(0, 0, 0)):
    """Move a cross-section at a *relative* position along the mesh.

    *fraction* goes from 0.0 (min end) to 1.0 (max end).
    Use to push the roof up, pull the nose down, shift a section sideways.
    """
    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)

    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    coords = [v.co[ax] for v in bm.verts]
    lo, hi = min(coords), max(coords)
    span = hi - lo
    if span < 0.0001:
        bm.free()
        return obj
    position = lo + fraction * span
    tolerance = span / 40.0

    for v in bm.verts:
        if abs(v.co[ax] - position) < tolerance:
            v.co.x += offset[0]
            v.co.y += offset[1]
            v.co.z += offset[2]

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def shape_body(obj, axis='X', profile=None):
    """Reshape a primitive into a complex body using a **ratio-based** profile.

    This is the MAIN tool for creating car bodies, bottles, vases, etc.
    The AI specifies shape as relative proportions — NOT coordinates.

    *profile* is a list of (fraction, width_ratio, height_ratio) tuples:
    - *fraction*: 0.0 = start-end, 1.0 = far-end (along *axis*)
    - *width_ratio*: 1.0 = keep current width, 0.5 = half width
    - *height_ratio*: 1.0 = keep current height, 0.5 = half height

    Typical car body profile along X::

        shape_body(body, 'X', [
            (0.00, 0.40, 0.35),   # front tip — narrow and low
            (0.10, 0.75, 0.55),   # hood rise
            (0.25, 0.90, 0.65),   # A-pillar
            (0.35, 0.95, 1.00),   # windshield top → full roof height
            (0.55, 1.00, 1.00),   # cabin — widest and tallest
            (0.70, 0.95, 0.90),   # C-pillar taper begins
            (0.85, 0.80, 0.60),   # rear deck / trunk
            (1.00, 0.50, 0.40),   # tail — narrowing
        ])

    The function adds enough loop cuts automatically before reshaping.
    """
    if not profile or len(profile) < 2:
        return obj

    import bmesh as _bm

    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ax = axis_map.get(axis.upper(), 0)
    others = [i for i in range(3) if i != ax]

    # Ensure enough resolution — add cuts if needed
    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    unique_positions = len(set(round(v.co[ax], 4) for v in bm.verts))
    bm.free()
    target_cuts = max(0, len(profile) * 3 - unique_positions)
    if target_cuts > 0:
        add_detail_cuts(obj, axis, target_cuts)

    # Now reshape
    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    coords = [v.co[ax] for v in bm.verts]
    lo, hi = min(coords), max(coords)
    span = hi - lo
    if span < 0.001:
        bm.free()
        return obj

    profile = sorted(profile, key=lambda p: p[0])

    for v in bm.verts:
        frac = (v.co[ax] - lo) / span  # 0..1 along axis

        # Interpolate profile at this fraction
        w_ratio, h_ratio = 1.0, 1.0
        for j in range(len(profile) - 1):
            if profile[j][0] <= frac <= profile[j + 1][0]:
                seg = profile[j + 1][0] - profile[j][0]
                t = (frac - profile[j][0]) / seg if seg > 0.001 else 0
                w_ratio = profile[j][1] + t * (profile[j + 1][1] - profile[j][1])
                h_ratio = profile[j][2] + t * (profile[j + 1][2] - profile[j][2])
                break
        else:
            if frac <= profile[0][0]:
                w_ratio, h_ratio = profile[0][1], profile[0][2]
            else:
                w_ratio, h_ratio = profile[-1][1], profile[-1][2]

        v.co[others[0]] *= w_ratio
        v.co[others[1]] *= h_ratio

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def select_top_faces(obj, threshold_deg=30):
    """Return face indices of upward-facing faces (Z+ normals).

    Shortcut for roof, top surfaces, lids, etc.
    """
    return select_faces_by_normal(obj, direction=(0, 0, 1),
                                  threshold_deg=threshold_deg)


def select_bottom_faces(obj, threshold_deg=30):
    """Return face indices of downward-facing faces (Z- normals).

    Shortcut for undersides, floors, bases.
    """
    return select_faces_by_normal(obj, direction=(0, 0, -1),
                                  threshold_deg=threshold_deg)


def select_front_faces(obj, threshold_deg=30):
    """Return face indices of forward-facing faces (X- normals).

    For vehicles aligned along X: returns faces pointing toward the front.
    """
    return select_faces_by_normal(obj, direction=(-1, 0, 0),
                                  threshold_deg=threshold_deg)


def select_back_faces(obj, threshold_deg=30):
    """Return face indices of rearward-facing faces (X+ normals).

    For vehicles aligned along X: returns faces pointing toward the back.
    """
    return select_faces_by_normal(obj, direction=(1, 0, 0),
                                  threshold_deg=threshold_deg)


def select_left_faces(obj, threshold_deg=30):
    """Return face indices of leftward-facing faces (Y+ normals)."""
    return select_faces_by_normal(obj, direction=(0, 1, 0),
                                  threshold_deg=threshold_deg)


def select_right_faces(obj, threshold_deg=30):
    """Return face indices of rightward-facing faces (Y- normals)."""
    return select_faces_by_normal(obj, direction=(0, -1, 0),
                                  threshold_deg=threshold_deg)


def wheel(name="Wheel", radius=0.35, width=0.22, location=(0, 0, 0),
          tire_color=(0.02, 0.02, 0.02), rim_color=(0.7, 0.7, 0.7)):
    """Create a complete wheel (tire + rim) as a single object.

    This is a convenience tool — the AI just specifies size and position,
    and gets a realistic wheel with materials already applied.
    Returns the wheel object.
    """
    # Tire — torus
    tire = create_torus(
        name=name + "_tire",
        major_radius=radius,
        minor_radius=width / 2,
        major_segments=24, minor_segments=12,
        location=location,
    )
    tire_mat = quick_material(name + "_rubber",
                              color=tire_color, roughness=0.95, metallic=0.0)
    assign_material(tire, tire_mat)

    # Rim — cylinder inside the tire
    rim = create_cylinder(
        name=name + "_rim",
        radius=radius * 0.6,
        depth=width * 0.8,
        vertices=16,
        location=location,
    )
    # Rotate rim to align with tire (tire is in XZ plane by default)
    rim_mat = quick_material(name + "_rim_mat",
                             color=rim_color, roughness=0.3, metallic=0.9)
    assign_material(rim, rim_mat)

    # Join into one object
    result = join_objects([tire, rim])
    result.name = name
    return result


def headlight(name="Headlight", size=0.15, location=(0, 0, 0),
              color=(1.0, 0.95, 0.85), strength=50):
    """Create a headlight/taillight as a small emissive sphere.

    Returns the light object with emission material applied.
    """
    light = create_sphere(name=name, radius=size, location=location,
                          segments=12, rings=6)
    mat = emission_material(name + "_emit", color=color, strength=strength)
    assign_material(light, mat)
    return light


def window_glass(name="Window", width=1.0, height=0.5, depth=0.02,
                 location=(0, 0, 0), rotation_deg=(0, 0, 0),
                 color=(0.15, 0.2, 0.25)):
    """Create a glass window panel.

    A thin box with glass material, positioned and rotated as needed.
    Good for windshields, side windows, building windows, etc.
    """
    win = create_box(name=name, width=width, depth=depth, height=height,
                     location=location)
    rotate_deg(win, *rotation_deg)
    mat = glass_material(name + "_glass", color=color, roughness=0.0, ior=1.5)
    assign_material(win, mat)
    return win


def place_at_bounds(obj, target, position='TOP', offset=0.0, axis=None):
    """Place *obj* at a specific face/edge of *target*'s bounding box.

    *position*: 'TOP', 'BOTTOM', 'FRONT', 'BACK', 'LEFT', 'RIGHT',
                'FRONT_BOTTOM', 'BACK_TOP', etc. (combine with _).
    *offset*: extra distance from the surface.

    Great for placing wheels at corners, lights at edges, etc.
    """
    bounds = get_bounds(target)
    if bounds is None:
        return obj

    x = bounds.center_x
    y = bounds.center_y
    z = bounds.center_z

    pos_upper = position.upper()
    if 'TOP' in pos_upper:
        z = bounds.max_z + offset
    if 'BOTTOM' in pos_upper:
        z = bounds.min_z - offset
    if 'FRONT' in pos_upper:
        x = bounds.min_x - offset  # front = negative X
    if 'BACK' in pos_upper:
        x = bounds.max_x + offset
    if 'LEFT' in pos_upper:
        y = bounds.max_y + offset
    if 'RIGHT' in pos_upper:
        y = bounds.min_y - offset

    move_to(obj, x, y, z)
    return obj