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
from mathutils import Vector, Matrix  # type: ignore

# Re-export procedural materials so they're available in execute_code
try:
    from .materials import (make_stucco, make_brick, make_wood, make_concrete,  # noqa: F401
                            make_stone, make_glass, make_metal, make_grass)  # noqa: F401
except ImportError:
    pass


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


def _parse_color(color):
    """Normalize a color value to (R, G, B) floats in 0-1 range.
    Accepts: tuple/list of 3-4 floats, hex string '#RRGGBB', single float (grey).
    """
    if isinstance(color, str):
        c = color.lstrip('#')
        if len(c) == 6:
            return (int(c[0:2], 16) / 255.0,
                    int(c[2:4], 16) / 255.0,
                    int(c[4:6], 16) / 255.0)
        return (0.8, 0.8, 0.8)
    if isinstance(color, (int, float)):
        v = float(color)
        return (v, v, v)
    if hasattr(color, '__len__') and len(color) >= 3:
        return (float(color[0]), float(color[1]), float(color[2]))
    return (0.8, 0.8, 0.8)


def _find_principled_bsdf(node_tree):
    """Find the Principled BSDF node by type (not by name, for Blender 4.x compat)."""
    for node in node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return node
    return None


def quick_material(name="Material", color=(0.8, 0.8, 0.8),
                   roughness=0.5, metallic=0.0):
    """Create or reuse a Principled BSDF material. Reuses an existing material
    with the same name if it exists, or one with similar colour/properties.
    Color can be a tuple (R,G,B), hex string '#RRGGBB', or single float (grey).
    """
    color = _parse_color(color)
    roughness = max(0.0, min(1.0, float(roughness)))
    metallic = max(0.0, min(1.0, float(metallic)))
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    similar = find_similar_material(color, roughness, metallic, tolerance=0.05)
    if similar:
        return similar
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = _find_principled_bsdf(mat.node_tree)
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (
            color[0], color[1], color[2], 1.0)
        bsdf.inputs['Roughness'].default_value = roughness
        bsdf.inputs['Metallic'].default_value = metallic
    return mat


def glass_material(name="Glass", color=(0.9, 0.95, 1.0),
                   roughness=0.0, ior=1.5):
    """Create a glass material with transmission."""
    color = _parse_color(color)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = _find_principled_bsdf(mat.node_tree)
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


def add_material_slot(obj, mat):
    """Add a material to an object WITHOUT clearing existing materials.
    Returns the material slot index. If the material is already assigned,
    returns the existing slot index."""
    if not obj or not obj.data:
        return -1
    for i, slot in enumerate(obj.material_slots):
        if slot.material and slot.material == mat:
            return i
    obj.data.materials.append(mat)
    return len(obj.material_slots) - 1


def assign_material_to_faces(obj, mat, face_indices):
    """Assign a material to specific faces of a mesh object by face index.
    Adds the material as a new slot (preserves existing materials on other faces).
    face_indices: list of polygon indices, or 'all' for every face.
    Example: assign_material_to_faces(bowl, wood_mat, [0, 1, 2, 3])
    Example: assign_material_to_faces(bowl, clay_mat, range(20, 50))
    """
    if not obj or obj.type != 'MESH' or not obj.data:
        return {"error": "Object is None or not a mesh"}
    slot_idx = add_material_slot(obj, mat)
    if slot_idx < 0:
        return {"error": "Could not add material slot"}
    mesh = obj.data
    if face_indices == 'all':
        face_indices = range(len(mesh.polygons))
    count = 0
    for fi in face_indices:
        if 0 <= fi < len(mesh.polygons):
            mesh.polygons[fi].material_index = slot_idx
            count += 1
    mesh.update()
    return {"slot": slot_idx, "faces_assigned": count,
            "total_faces": len(mesh.polygons)}


def get_face_count(obj):
    """Return the number of faces (polygons) on a mesh object."""
    if obj and obj.type == 'MESH' and obj.data:
        return len(obj.data.polygons)
    return 0


def assign_material_by_normal(obj, mat, axis='Z', direction='UP',
                              threshold=0.5):
    """Assign a material to faces based on their normal direction.
    Useful for applying different materials to top/bottom/sides of a mesh.
    axis: 'X', 'Y', or 'Z'
    direction: 'UP' (positive) or 'DOWN' (negative) along that axis
    threshold: dot product threshold (0.5 = within 60 degrees of axis)
    Example: assign_material_by_normal(bowl, rim_mat, 'Z', 'UP', 0.7)
    """
    if not obj or obj.type != 'MESH':
        return {"error": "Object is None or not a mesh"}
    axis_map = {'X': 0, 'Y': 1, 'Z': 2}
    ai = axis_map.get(axis.upper(), 2)
    sign = 1.0 if direction.upper() == 'UP' else -1.0
    slot_idx = add_material_slot(obj, mat)
    mesh = obj.data
    mesh.calc_normals()
    count = 0
    for poly in mesh.polygons:
        if poly.normal[ai] * sign >= threshold:
            poly.material_index = slot_idx
            count += 1
    mesh.update()
    return {"slot": slot_idx, "faces_assigned": count,
            "total_faces": len(mesh.polygons)}


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

    Returns a dict with applied/failed counts so the AI can see what happened.
    """
    if not obj or not obj.modifiers:
        return {"applied": 0, "failed": 0, "remaining": 0}
    applied = 0
    failed = []
    for mod in list(obj.modifiers):
        try:
            with bpy.context.temp_override(object=obj):
                bpy.ops.object.modifier_apply(modifier=mod.name)
            applied += 1
        except Exception as e:
            failed.append("%s: %s" % (mod.name, str(e)[:100]))
    result = {"applied": applied, "failed": len(failed),
              "remaining": len(obj.modifiers)}
    if failed:
        result["errors"] = failed
    return result


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
        bmesh.ops.subdivide_edges(
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
        _faces = [bm.faces[i] for i in face_indices if i < len(bm.faces)]  # noqa: F841
    else:
        _faces = list(bm.faces)  # noqa: F841

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


def join_objects_with_cleanup(objects, merge_distance=0.001,
                              recalc_outside=True, boolean_cleanup=False):
    """Join parts and run weld/normal cleanup for decomposed generation flows."""
    if not objects:
        return None
    result = join_objects(objects)
    if result is None:
        return None
    try:
        merge_by_distance(result, distance=merge_distance)
    except Exception:
        pass
    if boolean_cleanup:
        try:
            tris_to_quads(result)
        except Exception:
            pass
    if recalc_outside:
        try:
            recalc_normals(result, inside=False)
        except Exception:
            pass
    return result


def apply_suggested_modifiers(obj, object_type="auto"):
    """Apply a lightweight object-type-based modifier preset.

    Returns a list of modifier names that were added.
    """
    if obj is None:
        return []

    t = (object_type or "auto").lower()
    applied = []

    try:
        if t == "auto":
            name = obj.name.lower()
            if any(k in name for k in ("car", "vehicle", "robot", "weapon", "engine", "hard")):
                t = "hard_surface"
            elif any(k in name for k in ("character", "head", "face", "creature", "animal", "organic")):
                t = "organic"
            elif any(k in name for k in ("building", "house", "wall", "architecture", "room")):
                t = "architectural"
            else:
                t = "default"

        if t == "organic":
            applied.append(subsurf(obj, levels=3).name)
            applied.append(bevel(obj, width=0.004, segments=2).name)
        elif t == "hard_surface":
            applied.append(bevel(obj, width=0.01, segments=2).name)
            applied.append(subsurf(obj, levels=1).name)
            mod = obj.modifiers.new("WeightedNormal", 'WEIGHTED_NORMAL')
            applied.append(mod.name)
        elif t == "architectural":
            applied.append(solidify(obj, thickness=0.05).name)
            applied.append(bevel(obj, width=0.015, segments=2).name)
        else:
            applied.append(subsurf(obj, levels=2).name)
    except Exception:
        return applied

    return applied


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
        bmesh.ops.bisect_plane(
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
    try:
        with bpy.context.temp_override(object=obj):
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


# ══════════════════════════════════════════════════════════════════════════════
# Vertex and Face Selection/Editing — For Iterative Mesh Refinement
# ══════════════════════════════════════════════════════════════════════════════

def select_vertices_by_location(obj, axis='Z', min_val=None, max_val=None):
    """Select vertices within a range on an axis.
    
    Returns list of vertex indices for use with move_vertices.
    Example: verts = select_vertices_by_location(obj, 'Z', min_val=2.0)
    """
    import bmesh as _bm
    
    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    
    axis_idx = {'X': 0, 'Y': 1, 'Z': 2}[axis.upper()]
    selected = []
    
    for i, v in enumerate(bm.verts):
        coord = v.co[axis_idx]
        if min_val is not None and coord < min_val:
            continue
        if max_val is not None and coord > max_val:
            continue
        selected.append(i)
    
    bm.free()
    return selected


def select_faces_by_normal(obj, direction=(0, 0, 1), threshold=0.7):
    """Select faces pointing in a direction (for material assignment or extrusion).
    
    Returns list of face indices.
    direction: (x, y, z) vector, e.g. (0, 0, 1) for upward-facing
    threshold: dot product threshold (0.7 = faces within ~45° of direction)
    
    Example: top_faces = select_faces_by_normal(car, (0, 0, 1), 0.8)
             assign_material_to_faces(car, paint_mat, top_faces)
    """
    import bmesh as _bm
    from mathutils import Vector
    
    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    
    dir_vec = Vector(direction).normalized()
    selected = []
    
    for i, f in enumerate(bm.faces):
        if f.normal.dot(dir_vec) >= threshold:
            selected.append(i)
    
    bm.free()
    return selected


def move_vertices(obj, vertex_indices, offset=(0, 0, 0), relative=True):
    """Move specific vertices by an offset or to absolute position.
    
    Example: verts = select_vertices_by_location(obj, 'Z', min_val=2.0)
             move_vertices(obj, verts, offset=(0, 0, 0.5))  # lift top vertices
    """
    import bmesh as _bm
    from mathutils import Vector
    
    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    
    offset_vec = Vector(offset)
    for i in vertex_indices:
        if i < len(bm.verts):
            if relative:
                bm.verts[i].co += offset_vec
            else:
                bm.verts[i].co = offset_vec
    
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def scale_vertices(obj, vertex_indices, scale=(1, 1, 1), pivot='center'):
    """Scale specific vertices around a pivot point.
    
    pivot: 'center' (center of selected verts), 'origin' (object origin), or (x,y,z) tuple
    Example: rim_verts = select_vertices_by_location(bowl, 'Z', min_val=1.5)
             scale_vertices(bowl, rim_verts, scale=(1.2, 1.2, 1.0))
    """
    import bmesh as _bm
    from mathutils import Vector
    
    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    
    # Calculate pivot
    if pivot == 'center':
        center = Vector((0, 0, 0))
        count = 0
        for i in vertex_indices:
            if i < len(bm.verts):
                center += bm.verts[i].co
                count += 1
        if count > 0:
            center /= count
    elif pivot == 'origin':
        center = Vector((0, 0, 0))
    else:
        center = Vector(pivot)
    
    scale_vec = Vector(scale)
    for i in vertex_indices:
        if i < len(bm.verts):
            v = bm.verts[i]
            offset = v.co - center
            v.co = center + Vector((offset.x * scale_vec.x, 
                                     offset.y * scale_vec.y,
                                     offset.z * scale_vec.z))
    
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def smooth_vertices(obj, vertex_indices, iterations=1, factor=0.5):
    """Smooth specific vertices by averaging with neighbors.
    
    Example: sharp_verts = select_vertices_by_location(obj, 'Z', 0.9, 1.1)
             smooth_vertices(obj, sharp_verts, iterations=2, factor=0.7)
    """
    import bmesh as _bm
    from mathutils import Vector
    
    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    
    for _ in range(iterations):
        new_positions = {}
        for i in vertex_indices:
            if i >= len(bm.verts):
                continue
            v = bm.verts[i]
            
            # Average with connected vertices
            neighbors = []
            for edge in v.link_edges:
                other = edge.other_vert(v)
                neighbors.append(other.co)
            
            if neighbors:
                avg = Vector((0, 0, 0))
                for n_co in neighbors:
                    avg += n_co
                avg /= len(neighbors)
                
                # Blend between original and average
                new_positions[i] = v.co.lerp(avg, factor)
        
        # Apply new positions
        for i, new_co in new_positions.items():
            bm.verts[i].co = new_co
    
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return obj


def get_vertex_positions(obj, vertex_indices=None):
    """Get current positions of vertices (for inspection/debugging).
    
    Returns: list of (x, y, z) tuples
    Example: verts = select_vertices_by_location(obj, 'Z', min_val=2.0)
             positions = get_vertex_positions(obj, verts)
    """
    import bmesh as _bm
    
    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    
    if vertex_indices is None:
        vertex_indices = range(len(bm.verts))
    
    positions = []
    for i in vertex_indices:
        if i < len(bm.verts):
            co = bm.verts[i].co
            positions.append((co.x, co.y, co.z))
    
    bm.free()
    return positions


# ═══════════════════════════════════════════════════════════════════════════
# ANIMATION & KEYFRAMING
# ═══════════════════════════════════════════════════════════════════════════

def set_frame_range(start=1, end=250):
    """Set the scene playback frame range.

    Example: set_frame_range(1, 120)  # 5 seconds at 24fps
    """
    import bpy
    bpy.context.scene.frame_start = start
    bpy.context.scene.frame_end = end
    return {"start": start, "end": end}


def set_current_frame(frame=1):
    """Set the current frame (playhead position).

    Example: set_current_frame(30)
    """
    import bpy
    bpy.context.scene.frame_set(frame)
    return frame


def set_fps(fps=24):
    """Set scene frames per second.

    Example: set_fps(30)
    """
    import bpy
    bpy.context.scene.render.fps = fps
    return fps


def insert_keyframe(obj, data_path='location', frame=None, index=-1):
    """Insert a keyframe on an object property.

    data_path: 'location', 'rotation_euler', 'scale', 'hide_viewport',
               'hide_render', or any animatable property path.
    frame: frame number (None = current frame).
    index: channel index (-1 = all channels, 0=X, 1=Y, 2=Z).

    Example:
        move_to(cube, 0, 0, 0)
        insert_keyframe(cube, 'location', frame=1)
        move_to(cube, 5, 0, 3)
        insert_keyframe(cube, 'location', frame=60)
    """
    import bpy
    if frame is not None:
        bpy.context.scene.frame_set(frame)
    obj.keyframe_insert(data_path=data_path, index=index,
                        frame=frame or bpy.context.scene.frame_current)
    return {"object": obj.name, "data_path": data_path,
            "frame": frame or bpy.context.scene.frame_current}


def delete_keyframe(obj, data_path='location', frame=None, index=-1):
    """Delete a keyframe from an object property.

    Example: delete_keyframe(cube, 'location', frame=30)
    """
    import bpy
    f = frame or bpy.context.scene.frame_current
    obj.keyframe_delete(data_path=data_path, index=index, frame=f)
    return {"object": obj.name, "data_path": data_path, "frame": f}


def clear_animation(obj):
    """Remove all animation data from an object.

    Example: clear_animation(cube)
    """
    obj.animation_data_clear()
    return {"object": obj.name, "status": "animation cleared"}


def animate_location(obj, keyframes):
    """Animate object location over multiple frames.

    keyframes: list of (frame, (x, y, z)) tuples.

    Example:
        animate_location(cube, [
            (1,  (0, 0, 0)),
            (30, (5, 0, 0)),
            (60, (5, 5, 3)),
        ])
    """
    import bpy
    for frame, loc in keyframes:
        obj.location = loc
        obj.keyframe_insert(data_path='location', frame=frame)
    bpy.context.scene.frame_set(keyframes[0][0])
    return {"object": obj.name, "keyframes": len(keyframes)}


def animate_rotation(obj, keyframes, mode='euler'):
    """Animate object rotation over multiple frames.

    keyframes: list of (frame, (rx, ry, rz)) tuples (degrees for euler).
    mode: 'euler' or 'quaternion'.

    Example:
        animate_rotation(fan, [
            (1,  (0, 0, 0)),
            (60, (0, 0, 360)),
        ])
    """
    import bpy
    import math
    for frame, rot in keyframes:
        if mode == 'euler':
            obj.rotation_euler = tuple(math.radians(r) for r in rot)
            obj.keyframe_insert(data_path='rotation_euler', frame=frame)
        else:
            obj.rotation_quaternion = rot
            obj.keyframe_insert(data_path='rotation_quaternion', frame=frame)
    bpy.context.scene.frame_set(keyframes[0][0])
    return {"object": obj.name, "keyframes": len(keyframes)}


def animate_scale(obj, keyframes):
    """Animate object scale over multiple frames.

    keyframes: list of (frame, (sx, sy, sz)) tuples.

    Example:
        animate_scale(ball, [(1, (1,1,1)), (30, (2,2,2)), (60, (1,1,1))])
    """
    import bpy
    for frame, sc in keyframes:
        obj.scale = sc
        obj.keyframe_insert(data_path='scale', frame=frame)
    bpy.context.scene.frame_set(keyframes[0][0])
    return {"object": obj.name, "keyframes": len(keyframes)}


def animate_value(obj, data_path, keyframes, index=-1):
    """Animate any animatable property over multiple frames.

    keyframes: list of (frame, value) tuples.

    Example:
        # Animate material emission strength
        mat = obj.active_material
        animate_value(mat, 'node_tree.nodes["Emission"].inputs[1].default_value',
                      [(1, 0), (30, 10), (60, 0)])
    """
    for frame, value in keyframes:
        try:
            parts = data_path.rsplit('.', 1)
            if len(parts) == 2:
                parent = obj.path_resolve(parts[0])
                setattr(parent, parts[1], value)
            else:
                setattr(obj, data_path, value)
        except Exception:
            exec(f"obj.{data_path} = {value!r}")
        obj.keyframe_insert(data_path=data_path, index=index, frame=frame)
    return {"object": obj.name, "data_path": data_path,
            "keyframes": len(keyframes)}


def set_interpolation(obj, interpolation='BEZIER', data_path=None):
    """Set keyframe interpolation for an object's F-Curves.

    interpolation: 'CONSTANT', 'LINEAR', 'BEZIER', 'SINE', 'QUAD',
                   'CUBIC', 'QUART', 'QUINT', 'EXPO', 'CIRC',
                   'BACK', 'BOUNCE', 'ELASTIC'.
    data_path: filter to specific property (None = all).

    Example: set_interpolation(cube, 'LINEAR')
             set_interpolation(cube, 'CONSTANT', data_path='hide_viewport')
    """
    if not obj.animation_data or not obj.animation_data.action:
        return {"error": "no animation data"}
    count = 0
    for fc in obj.animation_data.action.fcurves:
        if data_path and fc.data_path != data_path:
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = interpolation
            count += 1
    return {"object": obj.name, "interpolation": interpolation,
            "keyframes_updated": count}


def set_extrapolation(obj, extrapolation='CONSTANT', data_path=None):
    """Set F-Curve extrapolation mode (what happens beyond keyframes).

    extrapolation: 'CONSTANT' (hold last value), 'LINEAR' (continue slope),
                   'MAKE_CYCLIC' (loop).

    Example: set_extrapolation(spinner, 'MAKE_CYCLIC')
    """
    if not obj.animation_data or not obj.animation_data.action:
        return {"error": "no animation data"}
    count = 0
    for fc in obj.animation_data.action.fcurves:
        if data_path and fc.data_path != data_path:
            continue
        if extrapolation == 'MAKE_CYCLIC':
            fc.modifiers.new(type='CYCLES')
            count += 1
        else:
            fc.extrapolation = extrapolation
            count += 1
    return {"object": obj.name, "extrapolation": extrapolation,
            "curves_updated": count}


def create_action(name="Action"):
    """Create a new empty Action (animation clip).

    Example: action = create_action("WalkCycle")
    """
    import bpy
    action = bpy.data.actions.new(name=name)
    return action


def set_action(obj, action):
    """Assign an Action to an object.

    Example:
        action = create_action("Bounce")
        set_action(cube, action)
    """
    if not obj.animation_data:
        obj.animation_data_create()
    obj.animation_data.action = action
    return {"object": obj.name, "action": action.name}


def push_to_nla(obj, track_name="Track", start_frame=1):
    """Push the active action to an NLA track (non-destructive stacking).

    Example:
        push_to_nla(character, "WalkCycle", start_frame=1)
    """
    if not obj.animation_data or not obj.animation_data.action:
        return {"error": "no active action to push"}
    action = obj.animation_data.action
    track = obj.animation_data.nla_tracks.new()
    track.name = track_name
    strip = track.strips.new(action.name, int(start_frame), action)
    obj.animation_data.action = None
    return {"object": obj.name, "track": track_name,
            "action": action.name, "strip": strip.name}


def add_follow_path(obj, curve_obj, use_curve_follow=True,
                    forward_axis='FORWARD_X', up_axis='UP_Z'):
    """Make an object follow a curve path.

    Example:
        path = create_bezier_curve("Path", points=[(0,0,0),(5,0,0),(10,5,0)])
        add_follow_path(car, path)
    """
    con = obj.constraints.new(type='FOLLOW_PATH')
    con.target = curve_obj
    con.use_curve_follow = use_curve_follow
    con.forward_axis = forward_axis
    con.up_axis = up_axis
    if curve_obj.data and hasattr(curve_obj.data, 'path_duration'):
        curve_obj.data.path_duration = 100
    return {"object": obj.name, "constraint": con.name,
            "curve": curve_obj.name}


def add_marker(name="Marker", frame=None):
    """Add a timeline marker at the given frame.

    Example: add_marker("Explosion", frame=45)
    """
    import bpy
    f = frame or bpy.context.scene.frame_current
    bpy.context.scene.timeline_markers.new(name, frame=f)
    return {"name": name, "frame": f}


# ═══════════════════════════════════════════════════════════════════════════
# ARMATURE & RIGGING
# ═══════════════════════════════════════════════════════════════════════════

def create_armature(name="Armature", location=(0, 0, 0)):
    """Create an empty armature object. Add bones with add_bone().

    Example:
        arm = create_armature("CharacterRig")
        add_bone(arm, "spine", head=(0,0,0), tail=(0,0,0.5))
    """
    import bpy
    armature_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, armature_data)
    bpy.context.collection.objects.link(arm_obj)
    arm_obj.location = location
    return arm_obj


def add_bone(armature_obj, name="Bone", head=(0, 0, 0), tail=(0, 0, 1),
             parent_bone=None, connected=False, roll=0):
    """Add a bone to an armature. Must provide head and tail positions.

    parent_bone: name of parent bone (string) or None.
    connected: if True, bone's head is fused to parent's tail.

    Example:
        arm = create_armature("Rig")
        add_bone(arm, "spine",      head=(0,0,0),   tail=(0,0,0.5))
        add_bone(arm, "chest",      head=(0,0,0.5), tail=(0,0,1.0),
                 parent_bone="spine", connected=True)
        add_bone(arm, "upper_arm.L", head=(0.3,0,0.9), tail=(0.7,0,0.7),
                 parent_bone="chest")
    """
    import bpy
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        bone = armature_obj.data.edit_bones.new(name)
        bone.head = head
        bone.tail = tail
        bone.roll = roll
        if parent_bone:
            parent = armature_obj.data.edit_bones.get(parent_bone)
            if parent:
                bone.parent = parent
                bone.use_connect = connected
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    return {"armature": armature_obj.name, "bone": name,
            "head": list(head), "tail": list(tail)}


def extrude_bone(armature_obj, parent_bone_name, name="NewBone",
                 tail_offset=(0, 0, 0.5)):
    """Extrude a new bone from the tail of an existing bone.

    Example:
        extrude_bone(arm, "spine", "chest", tail_offset=(0, 0, 0.5))
    """
    import bpy
    from mathutils import Vector
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        parent = armature_obj.data.edit_bones.get(parent_bone_name)
        if not parent:
            return {"error": f"bone '{parent_bone_name}' not found"}
        bone = armature_obj.data.edit_bones.new(name)
        bone.head = parent.tail.copy()
        bone.tail = parent.tail + Vector(tail_offset)
        bone.parent = parent
        bone.use_connect = True
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    return {"armature": armature_obj.name, "bone": name}


def create_bone_chain(armature_obj, chain_name, joints, parent_bone=None):
    """Create a chain of connected bones from a list of joint positions.

    joints: list of (x, y, z) positions. N positions = N-1 bones.

    Example:
        arm = create_armature("Snake")
        create_bone_chain(arm, "segment", [
            (0,0,0), (0,0.5,0), (0,1,0), (0,1.5,0.2), (0,2,0.5)
        ])
    """
    import bpy
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bones_created = []
    try:
        prev_bone = None
        if parent_bone:
            prev_bone = armature_obj.data.edit_bones.get(parent_bone)
        for i in range(len(joints) - 1):
            bname = f"{chain_name}_{i+1:02d}"
            bone = armature_obj.data.edit_bones.new(bname)
            bone.head = joints[i]
            bone.tail = joints[i + 1]
            if prev_bone:
                bone.parent = prev_bone
                bone.use_connect = (i > 0 or parent_bone is not None)
            prev_bone = bone
            bones_created.append(bname)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    return {"armature": armature_obj.name, "bones": bones_created}


def parent_to_armature(mesh_obj, armature_obj, method='ARMATURE_AUTO'):
    """Parent a mesh to an armature with automatic weight painting.

    method: 'ARMATURE_AUTO' (automatic weights), 'ARMATURE_NAME'
            (vertex groups by name), 'ARMATURE_ENVELOPE' (by bone envelope).

    Example:
        parent_to_armature(character_mesh, rig, method='ARMATURE_AUTO')
    """
    import bpy
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.parent_set(type=method)
    return {"mesh": mesh_obj.name, "armature": armature_obj.name,
            "method": method}


def set_bone_ik(armature_obj, bone_name, target_obj=None,
                chain_length=0, pole_target=None, pole_angle=0):
    """Add an Inverse Kinematics constraint to a bone.

    chain_length: 0 = entire chain. N = N bones up the chain.

    Example:
        set_bone_ik(rig, "hand.L", target_obj=ik_target, chain_length=3)
    """
    import bpy
    import math
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    try:
        pose_bone = armature_obj.pose.bones.get(bone_name)
        if not pose_bone:
            return {"error": f"pose bone '{bone_name}' not found"}
        ik = pose_bone.constraints.new('IK')
        ik.target = target_obj
        ik.chain_count = chain_length
        if pole_target:
            ik.pole_target = pole_target
            ik.pole_angle = math.radians(pole_angle)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    return {"armature": armature_obj.name, "bone": bone_name,
            "constraint": "IK"}


def add_bone_constraint(armature_obj, bone_name, constraint_type,
                        target_obj=None, **kwargs):
    """Add a constraint to a pose bone.

    constraint_type: 'COPY_LOCATION', 'COPY_ROTATION', 'COPY_SCALE',
                     'TRACK_TO', 'DAMPED_TRACK', 'STRETCH_TO',
                     'LIMIT_ROTATION', 'LIMIT_LOCATION', 'LIMIT_SCALE',
                     'TRANSFORMATION', 'IK', etc.

    Example:
        add_bone_constraint(rig, "eye.L", "TRACK_TO",
                            target_obj=look_at_target,
                            track_axis='TRACK_NEGATIVE_Z',
                            up_axis='UP_Y')
    """
    import bpy
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    try:
        pose_bone = armature_obj.pose.bones.get(bone_name)
        if not pose_bone:
            return {"error": f"pose bone '{bone_name}' not found"}
        con = pose_bone.constraints.new(constraint_type)
        if target_obj:
            con.target = target_obj
        for k, v in kwargs.items():
            if hasattr(con, k):
                setattr(con, k, v)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    return {"armature": armature_obj.name, "bone": bone_name,
            "constraint": constraint_type}


def get_bones(armature_obj):
    """List all bones in an armature with their head/tail positions.

    Example: bones = get_bones(rig)
    """
    result = []
    for bone in armature_obj.data.bones:
        result.append({
            "name": bone.name,
            "head": [round(v, 4) for v in bone.head_local],
            "tail": [round(v, 4) for v in bone.tail_local],
            "parent": bone.parent.name if bone.parent else None,
            "connected": bone.use_connect,
            "length": round(bone.length, 4),
        })
    return result


def pose_bone(armature_obj, bone_name, location=None,
              rotation_euler=None, scale=None):
    """Set a pose bone's transform (in pose mode coords).

    rotation_euler: (rx, ry, rz) in degrees.

    Example: pose_bone(rig, "upper_arm.L", rotation_euler=(0, 0, -45))
    """
    import bpy
    import math
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    try:
        pb = armature_obj.pose.bones.get(bone_name)
        if not pb:
            return {"error": f"pose bone '{bone_name}' not found"}
        if location:
            pb.location = location
        if rotation_euler:
            pb.rotation_mode = 'XYZ'
            pb.rotation_euler = tuple(math.radians(r) for r in rotation_euler)
        if scale:
            pb.scale = scale
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    return {"armature": armature_obj.name, "bone": bone_name}


def keyframe_bone(armature_obj, bone_name, data_path='rotation_euler',
                  frame=None):
    """Insert a keyframe on a pose bone property.

    data_path: 'location', 'rotation_euler', 'rotation_quaternion', 'scale'.

    Example:
        pose_bone(rig, "upper_arm.L", rotation_euler=(0, 0, -45))
        keyframe_bone(rig, "upper_arm.L", 'rotation_euler', frame=1)
        pose_bone(rig, "upper_arm.L", rotation_euler=(0, 0, 0))
        keyframe_bone(rig, "upper_arm.L", 'rotation_euler', frame=30)
    """
    import bpy
    f = frame or bpy.context.scene.frame_current
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')
    try:
        pb = armature_obj.pose.bones.get(bone_name)
        if not pb:
            return {"error": f"pose bone '{bone_name}' not found"}
        pb.keyframe_insert(data_path=data_path, frame=f)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    return {"armature": armature_obj.name, "bone": bone_name,
            "data_path": data_path, "frame": f}


def mirror_bones(armature_obj, suffix_from='.L', suffix_to='.R'):
    """Mirror all bones from one side to the other (e.g. .L → .R).

    Example: mirror_bones(rig)  # mirrors all .L bones to .R
    """
    import bpy
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    created = []
    try:
        bpy.ops.armature.select_all(action='DESELECT')
        for bone in armature_obj.data.edit_bones:
            if bone.name.endswith(suffix_from):
                bone.select = True
                bone.select_head = True
                bone.select_tail = True
        bpy.ops.armature.symmetrize()
        for bone in armature_obj.data.edit_bones:
            if bone.name.endswith(suffix_to):
                created.append(bone.name)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    return {"armature": armature_obj.name, "mirrored": created}


# ═══════════════════════════════════════════════════════════════════════════
# SHAPE KEYS
# ═══════════════════════════════════════════════════════════════════════════

def add_shape_key(obj, name="Key", from_mix=False):
    """Add a shape key to a mesh object.

    The first call creates the 'Basis' key automatically.
    After adding, edit vertices to define the shape, then set value to blend.

    Example:
        add_shape_key(face, "Basis")     # reference shape
        add_shape_key(face, "Smile")     # now edit verts for smile pose
    """
    if not obj.data.shape_keys:
        obj.shape_key_add(name="Basis", from_mix=False)
    sk = obj.shape_key_add(name=name, from_mix=from_mix)
    return {"object": obj.name, "shape_key": sk.name,
            "index": len(obj.data.shape_keys.key_blocks) - 1}


def set_shape_key_value(obj, name, value=1.0):
    """Set a shape key's influence value (0.0 to 1.0).

    Example: set_shape_key_value(face, "Smile", 0.7)
    """
    sk = obj.data.shape_keys
    if not sk:
        return {"error": "no shape keys on object"}
    key = sk.key_blocks.get(name)
    if not key:
        return {"error": f"shape key '{name}' not found"}
    key.value = value
    return {"object": obj.name, "shape_key": name, "value": value}


def edit_shape_key(obj, name, vertex_offsets):
    """Edit shape key vertex positions by offsets from basis.

    vertex_offsets: dict of {vertex_index: (dx, dy, dz)} or
                   list of (vertex_index, (dx, dy, dz)) tuples.

    Example:
        add_shape_key(face, "Smile")
        edit_shape_key(face, "Smile", {
            10: (0, 0, 0.1),   # lift corner of mouth
            11: (0, 0, 0.1),
            20: (0, 0, -0.05), # lower chin slightly
        })
    """
    from mathutils import Vector
    sk = obj.data.shape_keys
    if not sk:
        return {"error": "no shape keys"}
    key = sk.key_blocks.get(name)
    basis = sk.key_blocks.get("Basis")
    if not key or not basis:
        return {"error": f"shape key '{name}' or Basis not found"}
    if isinstance(vertex_offsets, dict):
        vertex_offsets = list(vertex_offsets.items())
    for vi, offset in vertex_offsets:
        if vi < len(key.data):
            base_co = basis.data[vi].co
            key.data[vi].co = base_co + Vector(offset)
    return {"object": obj.name, "shape_key": name,
            "vertices_edited": len(vertex_offsets)}


def animate_shape_key(obj, name, keyframes):
    """Animate a shape key value over time.

    keyframes: list of (frame, value) tuples.

    Example:
        animate_shape_key(face, "Smile", [(1, 0), (15, 1.0), (30, 0)])
    """
    sk = obj.data.shape_keys
    if not sk:
        return {"error": "no shape keys"}
    key = sk.key_blocks.get(name)
    if not key:
        return {"error": f"shape key '{name}' not found"}
    for frame, value in keyframes:
        key.value = value
        key.keyframe_insert(data_path='value', frame=frame)
    return {"object": obj.name, "shape_key": name,
            "keyframes": len(keyframes)}


def get_shape_keys(obj):
    """List all shape keys on an object.

    Example: keys = get_shape_keys(face)
    """
    sk = obj.data.shape_keys
    if not sk:
        return []
    return [{"name": kb.name, "value": round(kb.value, 3),
             "mute": kb.mute}
            for kb in sk.key_blocks]


# ═══════════════════════════════════════════════════════════════════════════
# DRIVERS
# ═══════════════════════════════════════════════════════════════════════════

def add_driver(obj, driven_path, driver_obj, driver_path,
               expression=None, index=-1):
    """Add a driver to a property — one property controls another.

    driven_path: property being driven (e.g. 'location', 'scale').
    driver_obj: object providing the driving value.
    driver_path: property on driver_obj to read from.
    expression: optional math expression using 'var' (e.g. 'var * 2').
    index: channel index (-1 = all, 0=X, 1=Y, 2=Z).

    Example:
        # Make cube's Z scale follow empty's Z location
        add_driver(cube, 'scale', empty, 'location',
                   expression='var * 0.5', index=2)
    """
    if index >= 0:
        fc = obj.driver_add(driven_path, index)
    else:
        result = obj.driver_add(driven_path)
        fc = result if not isinstance(result, list) else result[0]
    driver = fc.driver
    if expression:
        driver.type = 'SCRIPTED'
        driver.expression = expression
    else:
        driver.type = 'AVERAGE'
    var = driver.variables.new()
    var.name = 'var'
    var.targets[0].id = driver_obj
    var.targets[0].data_path = driver_path if '.' in driver_path else driver_path
    return {"object": obj.name, "driven": driven_path,
            "driver_source": driver_obj.name}


def add_expression_driver(obj, data_path, expression, variables=None,
                          index=-1):
    """Add a scripted expression driver with multiple variables.

    variables: dict of {var_name: (object, data_path)} pairs.

    Example:
        add_expression_driver(cube, 'location', 'x + y * 0.5',
            variables={'x': (empty1, 'location.x'),
                       'y': (empty2, 'location.z')}, index=2)
    """
    if index >= 0:
        fc = obj.driver_add(data_path, index)
    else:
        result = obj.driver_add(data_path)
        fc = result if not isinstance(result, list) else result[0]
    driver = fc.driver
    driver.type = 'SCRIPTED'
    driver.expression = expression
    if variables:
        for vname, (vobj, vpath) in variables.items():
            var = driver.variables.new()
            var.name = vname
            var.targets[0].id = vobj
            var.targets[0].data_path = vpath
    return {"object": obj.name, "data_path": data_path,
            "expression": expression}


def remove_driver(obj, data_path, index=-1):
    """Remove a driver from a property.

    Example: remove_driver(cube, 'scale', index=2)
    """
    if index >= 0:
        obj.driver_remove(data_path, index)
    else:
        obj.driver_remove(data_path)
    return {"object": obj.name, "data_path": data_path, "removed": True}


# ═══════════════════════════════════════════════════════════════════════════
# OBJECT CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════

def add_constraint(obj, constraint_type, target=None, **kwargs):
    """Add a constraint to an object.

    constraint_type: 'COPY_LOCATION', 'COPY_ROTATION', 'COPY_SCALE',
                     'TRACK_TO', 'DAMPED_TRACK', 'LIMIT_LOCATION',
                     'LIMIT_ROTATION', 'LIMIT_SCALE', 'FLOOR',
                     'CHILD_OF', 'FOLLOW_PATH', 'CLAMP_TO',
                     'LOCKED_TRACK', 'STRETCH_TO', 'MAINTAIN_VOLUME',
                     'TRANSFORM', 'SHRINKWRAP'.
    target: target object (optional, depends on constraint type).
    **kwargs: any additional constraint properties.

    Example:
        add_constraint(eye, 'TRACK_TO', target=look_target,
                       track_axis='TRACK_NEGATIVE_Z', up_axis='UP_Y')
    """
    con = obj.constraints.new(type=constraint_type)
    if target:
        con.target = target
    for k, v in kwargs.items():
        if hasattr(con, k):
            setattr(con, k, v)
    return {"object": obj.name, "constraint": con.name,
            "type": constraint_type}


def remove_constraint(obj, constraint_name=None, constraint_type=None):
    """Remove a constraint by name or type.

    Example: remove_constraint(cube, constraint_name="Track To")
             remove_constraint(cube, constraint_type='TRACK_TO')
    """
    to_remove = []
    for con in obj.constraints:
        if constraint_name and con.name == constraint_name:
            to_remove.append(con)
        elif constraint_type and con.type == constraint_type:
            to_remove.append(con)
    for con in to_remove:
        obj.constraints.remove(con)
    return {"object": obj.name, "removed": len(to_remove)}


def add_copy_location(obj, target, influence=1.0, use_x=True,
                      use_y=True, use_z=True):
    """Shortcut: add Copy Location constraint.

    Example: add_copy_location(follower, leader, use_z=False)
    """
    con = obj.constraints.new(type='COPY_LOCATION')
    con.target = target
    con.influence = influence
    con.use_x = use_x
    con.use_y = use_y
    con.use_z = use_z
    return {"object": obj.name, "constraint": con.name}


def add_copy_rotation(obj, target, influence=1.0, use_x=True,
                      use_y=True, use_z=True):
    """Shortcut: add Copy Rotation constraint.

    Example: add_copy_rotation(turret, controller)
    """
    con = obj.constraints.new(type='COPY_ROTATION')
    con.target = target
    con.influence = influence
    con.use_x = use_x
    con.use_y = use_y
    con.use_z = use_z
    return {"object": obj.name, "constraint": con.name}


def add_track_to(obj, target, track_axis='TRACK_NEGATIVE_Z',
                 up_axis='UP_Y'):
    """Shortcut: add Track To constraint (object always looks at target).

    Example: add_track_to(camera, focus_point)
    """
    con = obj.constraints.new(type='TRACK_TO')
    con.target = target
    con.track_axis = track_axis
    con.up_axis = up_axis
    return {"object": obj.name, "constraint": con.name}


def add_floor_constraint(obj, target, offset=0, use_rotation=False):
    """Shortcut: add Floor constraint (object can't go below target).

    Example: add_floor_constraint(ball, ground_plane)
    """
    con = obj.constraints.new(type='FLOOR')
    con.target = target
    con.offset = offset
    con.use_rotation = use_rotation
    return {"object": obj.name, "constraint": con.name}


# ═══════════════════════════════════════════════════════════════════════════
# PARTICLE SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════

def add_particle_emitter(obj, name="Particles", count=1000, lifetime=50,
                         frame_start=1, frame_end=200, velocity=(0, 0, 1),
                         gravity=1.0, size=0.05):
    """Add an emitter particle system to an object.

    Example:
        add_particle_emitter(fountain, count=500, lifetime=30,
                             velocity=(0, 0, 5), size=0.02)
    """
    mod = obj.modifiers.new(name=name, type='PARTICLE_SYSTEM')
    ps = mod.particle_system.settings
    ps.count = count
    ps.lifetime = lifetime
    ps.frame_start = frame_start
    ps.frame_end = frame_end
    ps.normal_factor = velocity[2] if len(velocity) > 2 else 1.0
    ps.factor_random = 0.2
    ps.particle_size = size
    ps.effector_weights.gravity = gravity
    return {"object": obj.name, "particle_system": name, "count": count}


def add_hair_system(obj, name="Hair", count=500, length=0.5,
                    segments=5, seed=0):
    """Add a hair particle system to an object.

    Example:
        add_hair_system(head, count=2000, length=0.3, segments=6)
    """
    mod = obj.modifiers.new(name=name, type='PARTICLE_SYSTEM')
    ps = mod.particle_system.settings
    ps.type = 'HAIR'
    ps.count = count
    ps.hair_length = length
    ps.hair_step = segments
    ps.child_type = 'INTERPOLATED'
    ps.child_nbr = 10
    ps.rendered_child_count = 50
    if seed:
        ps.seed = seed
    return {"object": obj.name, "particle_system": name, "count": count}


def set_particle_instance(particle_obj, instance_obj, use_rotation=True,
                          use_scale=True, scale_random=0.1):
    """Set an object to be instanced on each particle.

    Example:
        leaf = create_sphere("Leaf", radius=0.02)
        add_particle_emitter(tree, count=200)
        set_particle_instance(tree, leaf)
    """
    for mod in particle_obj.modifiers:
        if mod.type == 'PARTICLE_SYSTEM':
            ps = mod.particle_system.settings
            ps.render_type = 'OBJECT'
            ps.instance_object = instance_obj
            ps.use_rotation_instance = use_rotation
            ps.use_scale_instance = use_scale
            ps.size_random = scale_random
            return {"particle_obj": particle_obj.name,
                    "instance": instance_obj.name}
    return {"error": "no particle system found on object"}


def remove_particles(obj, name=None):
    """Remove particle system(s) from an object.

    name: specific system name, or None to remove all.

    Example: remove_particles(obj)
    """
    to_remove = []
    for mod in obj.modifiers:
        if mod.type == 'PARTICLE_SYSTEM':
            if name is None or mod.name == name:
                to_remove.append(mod.name)
    for mname in to_remove:
        obj.modifiers.remove(obj.modifiers[mname])
    return {"object": obj.name, "removed": len(to_remove)}


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════════════════════════════════

def add_rigid_body(obj, body_type='ACTIVE', mass=1.0, friction=0.5,
                   bounciness=0.5, collision_shape='CONVEX_HULL'):
    """Add rigid body physics to an object.

    body_type: 'ACTIVE' (falls/collides) or 'PASSIVE' (static obstacle).
    collision_shape: 'BOX', 'SPHERE', 'CAPSULE', 'CYLINDER',
                     'CONE', 'CONVEX_HULL', 'MESH'.

    Example:
        add_rigid_body(ball, 'ACTIVE', mass=2.0, bounciness=0.8)
        add_rigid_body(floor, 'PASSIVE')
    """
    import bpy
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    rb = obj.rigid_body
    rb.type = body_type
    rb.mass = mass
    rb.friction = friction
    rb.restitution = bounciness
    rb.collision_shape = collision_shape
    return {"object": obj.name, "type": body_type, "mass": mass}


def add_cloth(obj, quality=5, mass=0.3, stiffness=15.0,
              damping=5.0, gravity=True):
    """Add cloth simulation to a mesh.

    Example: add_cloth(curtain, stiffness=5.0, damping=10.0)
    """
    mod = obj.modifiers.new(name="Cloth", type='CLOTH')
    cs = mod.settings
    cs.quality = quality
    cs.mass = mass
    cs.tension_stiffness = stiffness
    cs.tension_damping = damping
    cs.effector_weights.gravity = 1.0 if gravity else 0.0
    return {"object": obj.name, "modifier": "Cloth"}


def add_collision(obj, thickness_outer=0.02, thickness_inner=0.01,
                  damping=0.0, friction=0.0):
    """Make an object a collision surface for cloth/softbody/particles.

    Example:
        add_collision(ground)    # particles and cloth bounce off ground
    """
    mod = obj.modifiers.new(name="Collision", type='COLLISION')
    cs = mod.settings
    cs.thickness_outer = thickness_outer
    cs.thickness_inner = thickness_inner
    cs.damping = damping
    cs.cloth_friction = friction
    return {"object": obj.name, "modifier": "Collision"}


def add_soft_body(obj, mass=1.0, friction=0.5, speed=1.0,
                  goal_strength=0.7):
    """Add soft body simulation to a mesh.

    Example: add_soft_body(jelly, mass=0.5, goal_strength=0.3)
    """
    mod = obj.modifiers.new(name="Softbody", type='SOFT_BODY')
    sb = mod.settings
    sb.mass = mass
    sb.friction = friction
    sb.speed = speed
    sb.goal_spring = goal_strength
    return {"object": obj.name, "modifier": "Softbody"}


def add_force_field(obj=None, field_type='FORCE', strength=1.0,
                    location=(0, 0, 0)):
    """Add a force field (wind, vortex, turbulence, etc.).

    field_type: 'FORCE', 'WIND', 'VORTEX', 'MAGNETIC', 'HARMONIC',
                'CHARGE', 'LENNARDJ', 'TEXTURE', 'GUIDE', 'BOID',
                'TURBULENCE', 'DRAG', 'FLUID_FLOW'.

    Example:
        add_force_field(field_type='WIND', strength=5.0, location=(0,0,0))
        add_force_field(field_type='TURBULENCE', strength=2.0)
    """
    import bpy
    if obj is None:
        bpy.ops.object.empty_add(location=location)
        obj = bpy.context.active_object
        obj.name = f"Force_{field_type}"
    obj.field.type = field_type
    obj.field.strength = strength
    return {"object": obj.name, "field_type": field_type,
            "strength": strength}


def set_gravity(x=0, y=0, z=-9.81):
    """Set scene gravity.

    Example: set_gravity(0, 0, -1.62)  # Moon gravity
    """
    import bpy
    bpy.context.scene.gravity = (x, y, z)
    return {"gravity": (x, y, z)}


def bake_physics(frame_start=1, frame_end=250):
    """Bake all physics simulations in the scene.

    Example: bake_physics(1, 120)
    """
    import bpy
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    bpy.ops.ptcache.bake_all(bake=True)
    return {"status": "baked", "frames": f"{frame_start}-{frame_end}"}


def free_physics_bake():
    """Free all baked physics caches (allows re-simulation).

    Example: free_physics_bake()
    """
    import bpy
    bpy.ops.ptcache.free_bake_all()
    return {"status": "freed"}


# ═══════════════════════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════════════════════

def set_render_engine(engine='CYCLES'):
    """Set render engine.

    engine: 'CYCLES', 'BLENDER_EEVEE_NEXT' (4.0+), 'BLENDER_EEVEE' (<4.0),
            'BLENDER_WORKBENCH'.

    Example: set_render_engine('CYCLES')
    """
    import bpy
    bpy.context.scene.render.engine = engine
    return {"engine": engine}


def set_render_resolution(x=1920, y=1080, percentage=100):
    """Set render output resolution.

    Example: set_render_resolution(3840, 2160)  # 4K
    """
    import bpy
    bpy.context.scene.render.resolution_x = x
    bpy.context.scene.render.resolution_y = y
    bpy.context.scene.render.resolution_percentage = percentage
    return {"resolution": f"{x}x{y}", "percentage": percentage}


def set_render_samples(samples=128, denoise=True):
    """Set render samples (for Cycles or EEVEE).

    Example: set_render_samples(256, denoise=True)
    """
    import bpy
    engine = bpy.context.scene.render.engine
    if engine == 'CYCLES':
        bpy.context.scene.cycles.samples = samples
        bpy.context.scene.cycles.use_denoising = denoise
    else:
        bpy.context.scene.eevee.taa_render_samples = samples
    return {"samples": samples, "denoise": denoise, "engine": engine}


def set_output_path(path="//render/", file_format='PNG'):
    """Set render output path and file format.

    file_format: 'PNG', 'JPEG', 'OPEN_EXR', 'TIFF', 'BMP',
                 'FFMPEG' (for animations).

    Example: set_output_path("//output/scene_", file_format='PNG')
    """
    import bpy
    bpy.context.scene.render.filepath = path
    bpy.context.scene.render.image_settings.file_format = file_format
    return {"path": path, "format": file_format}


def render_image(filepath=None):
    """Render the current frame to an image file.

    filepath: output path. None = uses scene output settings.

    Example: render_image("//render/preview.png")
    """
    import bpy
    if filepath:
        bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    return {"filepath": bpy.context.scene.render.filepath,
            "status": "rendered"}


def render_animation(filepath=None):
    """Render the full animation.

    Example: render_animation("//render/anim_")
    """
    import bpy
    if filepath:
        bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(animation=True)
    return {"filepath": bpy.context.scene.render.filepath,
            "status": "animation rendered"}


def set_transparent_background(transparent=True):
    """Enable transparent (alpha) background for renders.

    Example: set_transparent_background(True)
    """
    import bpy
    bpy.context.scene.render.film_transparent = transparent
    if bpy.context.scene.render.image_settings.file_format == 'PNG':
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    return {"transparent": transparent}


def set_color_management(view_transform='Filmic', look='None',
                         exposure=0, gamma=1.0):
    """Set color management settings for rendering.

    view_transform: 'Standard', 'Filmic', 'AgX' (4.0+), 'Raw'.
    look: 'None', 'High Contrast', 'Medium High Contrast', etc.

    Example: set_color_management('AgX', look='High Contrast')
    """
    import bpy
    cm = bpy.context.scene.view_settings
    cm.view_transform = view_transform
    cm.look = look
    cm.exposure = exposure
    cm.gamma = gamma
    return {"view_transform": view_transform, "look": look}


# ═══════════════════════════════════════════════════════════════════════════
# ADVANCED MATERIALS & TEXTURES
# ═══════════════════════════════════════════════════════════════════════════

def image_texture_material(name="ImageMat", image_path=None,
                           roughness=0.5, metallic=0.0):
    """Create a material with an image texture.

    Example:
        mat = image_texture_material("BrickWall", "//textures/brick.jpg")
        assign_material(wall, mat)
    """
    import bpy
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    if image_path:
        tex = nodes.new('ShaderNodeTexImage')
        tex.location = (-400, 0)
        try:
            img = bpy.data.images.load(image_path)
            tex.image = img
        except Exception:
            pass
        links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        coord = nodes.new('ShaderNodeTexCoord')
        coord.location = (-800, 0)
        mapping = nodes.new('ShaderNodeMapping')
        mapping.location = (-600, 0)
        links.new(coord.outputs['UV'], mapping.inputs['Vector'])
        links.new(mapping.outputs['Vector'], tex.inputs['Vector'])
    return mat


def gradient_material(name="Gradient", color1=(0.1, 0.1, 0.8),
                      color2=(0.8, 0.1, 0.1), gradient_type='LINEAR',
                      axis='Z'):
    """Create a gradient material blending between two colors.

    gradient_type: 'LINEAR', 'QUADRATIC', 'EASING', 'DIAGONAL',
                   'SPHERICAL', 'QUADRATIC_SPHERE', 'RADIAL'.

    Example:
        mat = gradient_material("SunsetSky", (0.1,0.1,0.5), (1,0.5,0.1))
        assign_material(backdrop, mat)
    """
    import bpy
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    mix = nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.location = (0, 0)
    mix.inputs[6].default_value = (*color1, 1)
    mix.inputs[7].default_value = (*color2, 1)
    links.new(mix.outputs[2], bsdf.inputs['Base Color'])
    grad = nodes.new('ShaderNodeTexGradient')
    grad.gradient_type = gradient_type
    grad.location = (-400, 0)
    links.new(grad.outputs['Fac'], mix.inputs['Factor'])
    coord = nodes.new('ShaderNodeTexCoord')
    coord.location = (-800, 0)
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-600, 0)
    if axis == 'Z':
        mapping.inputs['Rotation'].default_value = (1.5708, 0, 0)
    elif axis == 'X':
        mapping.inputs['Rotation'].default_value = (0, 0, 1.5708)
    links.new(coord.outputs['Object'], mapping.inputs['Vector'])
    links.new(mapping.outputs['Vector'], grad.inputs['Vector'])
    return mat


def subsurface_material(name="Skin", color=(0.8, 0.5, 0.4),
                        subsurface=0.3, subsurface_color=(0.7, 0.2, 0.1),
                        roughness=0.4):
    """Create a subsurface scattering material (skin, wax, marble, etc.).

    Example:
        mat = subsurface_material("HumanSkin", color=(0.8, 0.6, 0.5))
        assign_material(character, mat)
    """
    import bpy
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if not bsdf:
        bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Subsurface Weight'].default_value = subsurface
    bsdf.inputs['Subsurface Color'].default_value = (*subsurface_color, 1)
    bsdf.inputs['Roughness'].default_value = roughness
    return mat


def add_normal_map(mat, image_path=None, strength=1.0):
    """Add a normal map to an existing material.

    Example:
        mat = quick_material("BrickWall", color=(0.5, 0.3, 0.2))
        add_normal_map(mat, "//textures/brick_normal.jpg", strength=1.5)
    """
    import bpy
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            bsdf = node
            break
    if not bsdf:
        return {"error": "no Principled BSDF found"}
    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.inputs['Strength'].default_value = strength
    normal_map.location = (bsdf.location.x - 300, bsdf.location.y - 300)
    links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
    if image_path:
        tex = nodes.new('ShaderNodeTexImage')
        tex.location = (normal_map.location.x - 300,
                        normal_map.location.y)
        try:
            img = bpy.data.images.load(image_path)
            img.colorspace_settings.name = 'Non-Color'
            tex.image = img
        except Exception:
            pass
        links.new(tex.outputs['Color'], normal_map.inputs['Color'])
    return {"material": mat.name, "normal_map": True}


def add_displacement(mat, strength=0.1, midlevel=0.5, image_path=None):
    """Add displacement mapping to a material.

    Example:
        add_displacement(terrain_mat, strength=0.5)
    """
    import bpy
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL':
            output = node
            break
    if not output:
        return {"error": "no material output found"}
    disp = nodes.new('ShaderNodeDisplacement')
    disp.inputs['Scale'].default_value = strength
    disp.inputs['Midlevel'].default_value = midlevel
    disp.location = (output.location.x - 200, output.location.y - 200)
    links.new(disp.outputs['Displacement'], output.inputs['Displacement'])
    if image_path:
        tex = nodes.new('ShaderNodeTexImage')
        tex.location = (disp.location.x - 300, disp.location.y)
        try:
            img = bpy.data.images.load(image_path)
            img.colorspace_settings.name = 'Non-Color'
            tex.image = img
        except Exception:
            pass
        links.new(tex.outputs['Color'], disp.inputs['Height'])
    else:
        noise = nodes.new('ShaderNodeTexNoise')
        noise.location = (disp.location.x - 300, disp.location.y)
        links.new(noise.outputs['Fac'], disp.inputs['Height'])
    return {"material": mat.name, "displacement": True}


def mix_shader(mat, shader1_type='BSDF_PRINCIPLED',
               shader2_type='BSDF_PRINCIPLED', factor=0.5):
    """Create a material that mixes two shaders.

    Example: mat = mix_shader(mat, factor=0.5)
    """
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL':
            output = node
            break
    mix = nodes.new('ShaderNodeMixShader')
    mix.location = (output.location.x - 200, output.location.y)
    mix.inputs['Fac'].default_value = factor
    shader2 = nodes.new(f'ShaderNode{shader2_type.replace("BSDF_", "Bsdf")}' if 'BSDF' in shader2_type else f'ShaderNode{shader2_type}')
    shader2.location = (mix.location.x - 300, mix.location.y - 200)
    existing_shader = None
    for link in links:
        if link.to_node == output and link.to_socket.name == 'Surface':
            existing_shader = link.from_node
            links.remove(link)
            break
    if existing_shader:
        links.new(existing_shader.outputs[0], mix.inputs[1])
    links.new(shader2.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs['Surface'])
    return mat


def environment_texture(image_path, strength=1.0):
    """Set an HDRI environment texture for world lighting.

    Example: environment_texture("//hdri/studio_small.exr")
    """
    import bpy
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (400, 0)
    bg = nodes.new('ShaderNodeBackground')
    bg.location = (0, 0)
    bg.inputs['Strength'].default_value = strength
    links.new(bg.outputs['Background'], output.inputs['Surface'])
    env_tex = nodes.new('ShaderNodeTexEnvironment')
    env_tex.location = (-400, 0)
    try:
        img = bpy.data.images.load(image_path)
        env_tex.image = img
    except Exception:
        pass
    links.new(env_tex.outputs['Color'], bg.inputs['Color'])
    coord = nodes.new('ShaderNodeTexCoord')
    coord.location = (-700, 0)
    links.new(coord.outputs['Generated'], env_tex.inputs['Vector'])
    return {"world": world.name, "image": image_path}


# ═══════════════════════════════════════════════════════════════════════════
# WEIGHT PAINTING & VERTEX GROUPS
# ═══════════════════════════════════════════════════════════════════════════

def set_vertex_weights(obj, group_name, vertex_indices, weight=1.0):
    """Set vertex weights for a vertex group.

    Example:
        create_vertex_group(mesh, "Head")
        head_verts = find_verts_in_range(mesh, 'Z', 1.5, 2.0)
        set_vertex_weights(mesh, "Head", head_verts, weight=1.0)
    """
    vg = obj.vertex_groups.get(group_name)
    if not vg:
        vg = obj.vertex_groups.new(name=group_name)
    vg.add(list(vertex_indices), weight, 'REPLACE')
    return {"object": obj.name, "group": group_name,
            "vertices": len(vertex_indices), "weight": weight}


def paint_weight_gradient(obj, group_name, axis='Z', min_weight=0.0,
                          max_weight=1.0):
    """Paint a linear weight gradient along an axis.

    Vertices at the minimum of the axis get min_weight, at maximum
    get max_weight. Useful for falloff effects.

    Example:
        paint_weight_gradient(curtain, "Pin", axis='Z',
                              min_weight=0.0, max_weight=1.0)
    """
    vg = obj.vertex_groups.get(group_name)
    if not vg:
        vg = obj.vertex_groups.new(name=group_name)
    axis_idx = {'X': 0, 'Y': 1, 'Z': 2}[axis.upper()]
    verts = obj.data.vertices
    if not verts:
        return {"error": "no vertices"}
    coords = [v.co[axis_idx] for v in verts]
    min_c, max_c = min(coords), max(coords)
    span = max_c - min_c
    if span < 1e-6:
        span = 1.0
    for v in verts:
        t = (v.co[axis_idx] - min_c) / span
        w = min_weight + t * (max_weight - min_weight)
        vg.add([v.index], w, 'REPLACE')
    return {"object": obj.name, "group": group_name, "axis": axis,
            "vertices": len(verts)}


def normalize_weights(obj):
    """Normalize all vertex group weights on an object.

    Example: normalize_weights(character)
    """
    import bpy
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    try:
        bpy.ops.object.vertex_group_normalize_all()
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    return {"object": obj.name, "status": "normalized"}


def get_vertex_weights(obj, group_name, vertex_indices=None):
    """Get vertex weights for a vertex group.

    Example: weights = get_vertex_weights(mesh, "Head")
    """
    vg = obj.vertex_groups.get(group_name)
    if not vg:
        return {"error": f"group '{group_name}' not found"}
    if vertex_indices is None:
        vertex_indices = range(len(obj.data.vertices))
    result = {}
    for vi in vertex_indices:
        try:
            w = vg.weight(vi)
            result[vi] = round(w, 4)
        except RuntimeError:
            pass
    return result


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITOR
# ═══════════════════════════════════════════════════════════════════════════

def setup_compositor(use_nodes=True):
    """Enable the compositor and set up basic nodes.

    Example: setup_compositor()
    """
    import bpy
    bpy.context.scene.use_nodes = use_nodes
    tree = bpy.context.scene.node_tree
    if not tree.nodes:
        rl = tree.nodes.new('CompositorNodeRLayers')
        rl.location = (0, 0)
        comp = tree.nodes.new('CompositorNodeComposite')
        comp.location = (600, 0)
        tree.links.new(rl.outputs['Image'], comp.inputs['Image'])
    return {"compositor": "enabled"}


def add_glare(glare_type='FOG_GLOW', threshold=1.0, size=6,
              strength=1.0):
    """Add a glare/bloom effect in the compositor.

    glare_type: 'BLOOM' (4.0+), 'FOG_GLOW', 'STREAKS', 'SIMPLE_STAR', 'GHOSTS'.

    Example: add_glare(glare_type='FOG_GLOW', threshold=0.8, size=8)
    """
    import bpy
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    glare = tree.nodes.new('CompositorNodeGlare')
    glare.glare_type = glare_type
    glare.threshold = threshold
    glare.size = size
    glare.mix = strength - 1.0
    rl = None
    comp = None
    for node in tree.nodes:
        if node.type == 'R_LAYERS':
            rl = node
        elif node.type == 'COMPOSITE':
            comp = node
    if rl and comp:
        for link in list(tree.links):
            if (link.from_node == rl and link.to_node == comp
                    and link.from_socket.name == 'Image'):
                tree.links.remove(link)
        glare.location = (rl.location.x + 300, rl.location.y)
        tree.links.new(rl.outputs['Image'], glare.inputs['Image'])
        tree.links.new(glare.outputs['Image'], comp.inputs['Image'])
    return {"compositor_node": "Glare", "type": glare_type}


def add_color_correction(brightness=0, contrast=0, saturation=1.0):
    """Add brightness/contrast/saturation adjustment in compositor.

    Example: add_color_correction(brightness=0.1, contrast=0.2,
                                  saturation=1.2)
    """
    import bpy
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    bc = tree.nodes.new('CompositorNodeBrightContrast')
    bc.inputs['Bright'].default_value = brightness
    bc.inputs['Contrast'].default_value = contrast
    hue_sat = tree.nodes.new('CompositorNodeHueSat')
    hue_sat.inputs['Saturation'].default_value = saturation
    rl = None
    comp = None
    for node in tree.nodes:
        if node.type == 'R_LAYERS':
            rl = node
        elif node.type == 'COMPOSITE':
            comp = node
    if rl and comp:
        for link in list(tree.links):
            if link.from_node == rl and link.to_node == comp:
                tree.links.remove(link)
        bc.location = (rl.location.x + 300, rl.location.y)
        hue_sat.location = (rl.location.x + 500, rl.location.y)
        tree.links.new(rl.outputs['Image'], bc.inputs['Image'])
        tree.links.new(bc.outputs['Image'], hue_sat.inputs['Image'])
        tree.links.new(hue_sat.outputs['Image'], comp.inputs['Image'])
    return {"brightness": brightness, "contrast": contrast,
            "saturation": saturation}


def add_depth_of_field(camera_obj=None, focus_object=None,
                       focus_distance=10.0, fstop=2.8):
    """Enable depth of field on a camera.

    Example:
        cam = get("Camera")
        add_depth_of_field(cam, focus_object=get("Character"), fstop=1.4)
    """
    import bpy
    if camera_obj is None:
        camera_obj = bpy.context.scene.camera
    if not camera_obj or camera_obj.type != 'CAMERA':
        return {"error": "no camera found"}
    cam = camera_obj.data
    cam.dof.use_dof = True
    if focus_object:
        cam.dof.focus_object = focus_object
    else:
        cam.dof.focus_distance = focus_distance
    cam.dof.aperture_fstop = fstop
    return {"camera": camera_obj.name, "fstop": fstop}


def add_vignette(amount=0.5):
    """Add a vignette effect in the compositor.

    Example: add_vignette(0.7)
    """
    import bpy
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    ellipse = tree.nodes.new('CompositorNodeEllipseMask')
    ellipse.width = 0.9
    ellipse.height = 0.9
    blur = tree.nodes.new('CompositorNodeBlur')
    blur.size_x = 200
    blur.size_y = 200
    blur.use_relative = False
    multiply = tree.nodes.new('CompositorNodeMixRGB')
    multiply.blend_type = 'MULTIPLY'
    multiply.inputs['Fac'].default_value = amount
    tree.links.new(ellipse.outputs['Mask'], blur.inputs['Image'])
    rl = None
    comp = None
    for node in tree.nodes:
        if node.type == 'R_LAYERS':
            rl = node
        elif node.type == 'COMPOSITE':
            comp = node
    if rl and comp:
        for link in list(tree.links):
            if link.to_node == comp and link.to_socket.name == 'Image':
                tree.links.remove(link)
        tree.links.new(rl.outputs['Image'], multiply.inputs[1])
        tree.links.new(blur.outputs['Image'], multiply.inputs[2])
        tree.links.new(multiply.outputs['Image'], comp.inputs['Image'])
    return {"vignette": amount}


# ═══════════════════════════════════════════════════════════════════════════
# SCENE & TIMELINE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def get_scene_info():
    """Get comprehensive scene information.

    Example: info = get_scene_info()
    """
    import bpy
    sc = bpy.context.scene
    objects = []
    for obj in sc.objects:
        info = {"name": obj.name, "type": obj.type,
                "location": [round(v, 3) for v in obj.location]}
        if obj.type == 'MESH' and obj.data:
            info["vertices"] = len(obj.data.vertices)
            info["faces"] = len(obj.data.polygons)
        if obj.type == 'ARMATURE' and obj.data:
            info["bones"] = len(obj.data.bones)
        if obj.constraints:
            info["constraints"] = [c.type for c in obj.constraints]
        if obj.particle_systems:
            info["particles"] = [ps.name for ps in obj.particle_systems]
        if obj.rigid_body:
            info["rigid_body"] = obj.rigid_body.type
        if obj.animation_data and obj.animation_data.action:
            info["action"] = obj.animation_data.action.name
        objects.append(info)
    return {
        "frame_range": [sc.frame_start, sc.frame_end],
        "fps": sc.render.fps,
        "render_engine": sc.render.engine,
        "resolution": f"{sc.render.resolution_x}x{sc.render.resolution_y}",
        "gravity": [round(v, 3) for v in sc.gravity],
        "objects": objects,
    }


def list_actions():
    """List all actions (animation clips) in the file.

    Example: actions = list_actions()
    """
    import bpy
    return [{"name": a.name,
             "frame_range": [int(a.frame_range[0]), int(a.frame_range[1])],
             "curves": len(a.fcurves)}
            for a in bpy.data.actions]


def list_materials():
    """List all materials in the file with basic info.

    Example: mats = list_materials()
    """
    import bpy
    result = []
    for mat in bpy.data.materials:
        info = {"name": mat.name, "use_nodes": mat.use_nodes}
        if mat.use_nodes and mat.node_tree:
            info["nodes"] = len(mat.node_tree.nodes)
        result.append(info)
    return result


def set_active_camera(camera_obj):
    """Set the active scene camera.

    Example: set_active_camera(get("Camera.002"))
    """
    import bpy
    bpy.context.scene.camera = camera_obj
    return {"camera": camera_obj.name}


def add_camera(name="Camera", location=(10, -10, 8), look_at=(0, 0, 0),
               lens=50, sensor_width=36):
    """Add a camera aimed at a target point.

    Example: cam = add_camera("MainCam", location=(15, -15, 10),
                              look_at=(0, 0, 2), lens=85)
    """
    import bpy
    from mathutils import Vector
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.sensor_width = sensor_width
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = location
    direction = Vector(look_at) - Vector(location)
    rot = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot.to_euler()


# ═══════════════════════════════════════════════════════════════════
# SCENE INSPECTION TOOLS  —  designed for LLM reasoning
# ═══════════════════════════════════════════════════════════════════

def inspect_timeline():
    """Return a structured timeline overview: fps, frame range, playback, and
    per-object keyframe summary with human-readable timestamps.

    Output format for each animated object:
      Object "Cube": location
        0.00s [0.0, 0.0, 0.0] → 1.00s [3.0, 0.0, 2.0] → 2.00s [5.0, 0.0, 0.0]
      Object "Cube": rotation_euler
        0.00s [0.0, 0.0, 0.0] → 1.00s [0.0, 0.0, 1.571]

    Example: tl = inspect_timeline()
    """
    import bpy
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    frame_start = scene.frame_start
    frame_end = scene.frame_end
    frame_current = scene.frame_current
    duration = (frame_end - frame_start) / fps

    result = {
        "fps": round(fps, 2),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_current": frame_current,
        "duration_seconds": round(duration, 3),
        "total_frames": frame_end - frame_start + 1,
        "objects": [],
    }

    for obj in bpy.data.objects:
        if not obj.animation_data or not obj.animation_data.action:
            continue
        action = obj.animation_data.action
        channels = {}
        for fc in action.fcurves:
            dp = fc.data_path
            idx = fc.array_index
            key = dp
            if key not in channels:
                channels[key] = {}
            for kp in fc.keyframe_points:
                frame = int(kp.co[0])
                t = round((frame - frame_start) / fps, 3)
                if frame not in channels[key]:
                    channels[key][frame] = {"time": t, "values": {}}
                channels[key][frame]["values"][idx] = round(kp.co[1], 4)

        obj_info = {"name": obj.name, "action": action.name, "channels": []}
        for dp, frames in channels.items():
            channel = {"property": dp, "keyframes": []}
            for frame in sorted(frames.keys()):
                kf = frames[frame]
                vals = kf["values"]
                val_list = [vals.get(i, 0.0) for i in range(max(vals.keys()) + 1)] if vals else []
                channel["keyframes"].append({
                    "frame": frame,
                    "time": kf["time"],
                    "values": val_list,
                })
            obj_info["channels"].append(channel)
        result["objects"].append(obj_info)

    # Also list NLA tracks
    nla_objects = []
    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.nla_tracks:
            tracks = []
            for track in obj.animation_data.nla_tracks:
                strips = []
                for strip in track.strips:
                    strips.append({
                        "name": strip.name,
                        "action": strip.action.name if strip.action else None,
                        "frame_start": round(strip.frame_start, 1),
                        "frame_end": round(strip.frame_end, 1),
                        "blend_type": strip.blend_type,
                    })
                tracks.append({"name": track.name, "mute": track.mute,
                                "strips": strips})
            nla_objects.append({"name": obj.name, "tracks": tracks})
    if nla_objects:
        result["nla"] = nla_objects

    # Format a human-readable summary for the LLM
    lines = [
        "Timeline: {fps}fps, frames {fs}-{fe} ({dur}s), current frame {fc}".format(
            fps=result["fps"], fs=frame_start, fe=frame_end,
            dur=result["duration_seconds"], fc=frame_current),
    ]
    for obj_info in result["objects"]:
        for ch in obj_info["channels"]:
            arrows = []
            for kf in ch["keyframes"]:
                arrows.append("{t}s {v}".format(t=kf["time"], v=kf["values"]))
            lines.append('  {name}: {prop}'.format(name=obj_info["name"],
                                                    prop=ch["property"]))
            lines.append('    ' + ' → '.join(arrows))
    result["summary"] = '\n'.join(lines)
    return result


def inspect_animation(obj):
    """Return detailed animation info for one object: all FCurves, keyframes
    with timestamps, interpolation type, handles, and bone poses.

    Example: anim = inspect_animation(get("Character"))
    """
    import bpy
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    fs = scene.frame_start

    result = {"name": obj.name, "animated": False}

    if not obj.animation_data:
        return result

    ad = obj.animation_data
    result["animated"] = True
    result["action"] = ad.action.name if ad.action else None
    result["influence"] = round(ad.action_influence, 3)

    if ad.action:
        channels = []
        for fc in ad.action.fcurves:
            kfs = []
            for kp in fc.keyframe_points:
                kfs.append({
                    "frame": int(kp.co[0]),
                    "time": round((kp.co[0] - fs) / fps, 3),
                    "value": round(kp.co[1], 4),
                    "interpolation": kp.interpolation,
                    "easing": kp.easing,
                    "handle_left": [round(kp.handle_left[0], 2),
                                    round(kp.handle_left[1], 4)],
                    "handle_right": [round(kp.handle_right[0], 2),
                                     round(kp.handle_right[1], 4)],
                })
            channels.append({
                "data_path": fc.data_path,
                "array_index": fc.array_index,
                "keyframes": kfs,
                "mute": fc.mute,
            })
        result["channels"] = channels

    # NLA
    if ad.nla_tracks:
        tracks = []
        for track in ad.nla_tracks:
            strips = []
            for strip in track.strips:
                strips.append({
                    "name": strip.name,
                    "action": strip.action.name if strip.action else None,
                    "frame_start": round(strip.frame_start, 1),
                    "frame_end": round(strip.frame_end, 1),
                    "scale": round(strip.scale, 3),
                    "repeat": round(strip.repeat, 3),
                    "blend_type": strip.blend_type,
                    "influence": round(strip.influence, 3),
                })
            tracks.append({"name": track.name, "mute": track.mute,
                            "strips": strips})
        result["nla_tracks"] = tracks

    # Drivers
    if ad.drivers:
        drivers = []
        for d in ad.drivers:
            drivers.append({
                "data_path": d.data_path,
                "array_index": d.array_index,
                "expression": d.driver.expression if d.driver else None,
            })
        result["drivers"] = drivers

    return result


def inspect_materials_detail(obj=None):
    """Return detailed material info for one object or all materials in scene.

    If *obj* is given, returns materials assigned to that object with slot
    indices and node tree overview.  If *obj* is None, lists all scene materials.

    Example: mats = inspect_materials_detail(get("Sword"))
             all_mats = inspect_materials_detail()
    """
    import bpy

    def _mat_info(mat):
        info = {
            "name": mat.name,
            "use_nodes": mat.use_nodes,
            "blend_method": getattr(mat, 'blend_method', 'OPAQUE'),
        }
        if mat.use_nodes and mat.node_tree:
            nodes = []
            for n in mat.node_tree.nodes:
                nd = {"type": n.type, "name": n.name, "label": n.label}
                # Extract key values from common node types
                if n.type == 'BSDF_PRINCIPLED':
                    for inp_name in ['Base Color', 'Metallic', 'Roughness',
                                     'IOR', 'Alpha', 'Emission Color',
                                     'Emission Strength', 'Subsurface Weight']:
                        inp = n.inputs.get(inp_name)
                        if inp is not None and not inp.is_linked:
                            val = inp.default_value
                            if hasattr(val, '__len__'):
                                nd[inp_name.lower().replace(' ', '_')] = [
                                    round(v, 3) for v in val]
                            else:
                                nd[inp_name.lower().replace(' ', '_')] = round(
                                    val, 3)
                elif n.type == 'TEX_IMAGE':
                    nd["image"] = n.image.name if n.image else None
                    nd["interpolation"] = n.interpolation
                elif n.type == 'MIX_RGB' or n.type == 'MIX':
                    nd["blend_type"] = n.blend_type
                elif n.type == 'MAPPING':
                    if hasattr(n, 'inputs'):
                        loc_inp = n.inputs.get('Location')
                        scale_inp = n.inputs.get('Scale')
                        if loc_inp and not loc_inp.is_linked:
                            nd["mapping_location"] = [
                                round(v, 3) for v in loc_inp.default_value]
                        if scale_inp and not scale_inp.is_linked:
                            nd["mapping_scale"] = [
                                round(v, 3) for v in scale_inp.default_value]
                nodes.append(nd)
            info["nodes"] = nodes
            # Summarize links
            links = []
            for lnk in mat.node_tree.links:
                links.append("{f}.{fo} → {t}.{ti}".format(
                    f=lnk.from_node.name, fo=lnk.from_socket.name,
                    t=lnk.to_node.name, ti=lnk.to_socket.name))
            info["links"] = links
        else:
            info["diffuse_color"] = [round(v, 3) for v in mat.diffuse_color]
        return info

    if obj is not None:
        slots = []
        for i, slot in enumerate(obj.material_slots):
            if slot.material:
                mi = _mat_info(slot.material)
                mi["slot_index"] = i
                slots.append(mi)
        return {"object": obj.name, "material_slots": slots,
                "active_index": obj.active_material_index}
    else:
        return [_mat_info(m) for m in bpy.data.materials
                if m.name != "Dots Stroke"]


def inspect_modifiers(obj):
    """Return all modifiers on an object with their key settings.

    Example: mods = inspect_modifiers(get("Sword"))
    """
    result = []
    for mod in obj.modifiers:
        info = {"name": mod.name, "type": mod.type, "show_viewport": mod.show_viewport,
                "show_render": mod.show_render}
        # Extract key settings per modifier type
        if mod.type == 'SUBSURF':
            info["levels"] = mod.levels
            info["render_levels"] = mod.render_levels
            info["subdivision_type"] = mod.subdivision_type
        elif mod.type == 'MIRROR':
            info["use_axis"] = [mod.use_axis[0], mod.use_axis[1], mod.use_axis[2]]
            info["use_clip"] = mod.use_clip
        elif mod.type == 'BEVEL':
            info["width"] = round(mod.width, 4)
            info["segments"] = mod.segments
            info["limit_method"] = mod.limit_method
        elif mod.type == 'BOOLEAN':
            info["operation"] = mod.operation
            info["object"] = mod.object.name if mod.object else None
            info["solver"] = mod.solver
        elif mod.type == 'SOLIDIFY':
            info["thickness"] = round(mod.thickness, 4)
            info["offset"] = round(mod.offset, 4)
        elif mod.type == 'ARRAY':
            info["count"] = mod.count
            info["relative_offset"] = [round(v, 4) for v in mod.relative_offset_displace]
            info["use_merge_vertices"] = mod.use_merge_vertices
        elif mod.type == 'ARMATURE':
            info["armature"] = mod.object.name if mod.object else None
        elif mod.type == 'SHRINKWRAP':
            info["target"] = mod.target.name if mod.target else None
            info["wrap_method"] = mod.wrap_method
        elif mod.type == 'DECIMATE':
            info["decimate_type"] = mod.decimate_type
            info["ratio"] = round(mod.ratio, 4)
        elif mod.type == 'PARTICLE_SYSTEM':
            ps = mod.particle_system
            if ps and ps.settings:
                info["particle_count"] = ps.settings.count
                info["particle_type"] = ps.settings.type
        elif mod.type == 'NODES':
            info["node_group"] = mod.node_group.name if mod.node_group else None
        result.append(info)
    return {"object": obj.name, "modifiers": result, "count": len(result)}


def inspect_constraints(obj):
    """Return all constraints on an object with their key settings.

    Example: cons = inspect_constraints(get("Camera"))
    """
    result = []
    for con in obj.constraints:
        info = {
            "name": con.name,
            "type": con.type,
            "mute": con.mute,
            "influence": round(con.influence, 3),
        }
        if hasattr(con, 'target') and con.target:
            info["target"] = con.target.name
        if hasattr(con, 'subtarget') and con.subtarget:
            info["subtarget"] = con.subtarget
        # Type-specific
        if con.type == 'TRACK_TO':
            info["track_axis"] = con.track_axis
            info["up_axis"] = con.up_axis
        elif con.type == 'IK':
            info["chain_count"] = con.chain_count
            info["use_rotation"] = con.use_rotation
            if con.pole_target:
                info["pole_target"] = con.pole_target.name
        elif con.type in ('COPY_LOCATION', 'COPY_ROTATION', 'COPY_SCALE'):
            info["use_x"] = con.use_x
            info["use_y"] = con.use_y
            info["use_z"] = con.use_z
        elif con.type == 'LIMIT_ROTATION':
            info["use_limit_x"] = con.use_limit_x
            info["min_x"] = round(con.min_x, 4)
            info["max_x"] = round(con.max_x, 4)
        elif con.type == 'FLOOR':
            info["floor_location"] = con.floor_location
        result.append(info)
    return {"object": obj.name, "constraints": result, "count": len(result)}


def inspect_armature(obj):
    """Return detailed armature info: bone hierarchy, pose bones with
    current transforms, constraints per bone, and IK chains.

    Example: arm = inspect_armature(get("Armature"))
    """
    if obj.type != 'ARMATURE' or not obj.data:
        return {"error": "{} is not an armature".format(obj.name)}

    armature = obj.data
    bones = []
    for bone in armature.bones:
        b = {
            "name": bone.name,
            "head": [round(v, 4) for v in bone.head_local],
            "tail": [round(v, 4) for v in bone.tail_local],
            "length": round(bone.length, 4),
            "parent": bone.parent.name if bone.parent else None,
            "children": [c.name for c in bone.children],
            "connected": bone.use_connect,
            "deform": bone.use_deform,
        }
        bones.append(b)

    # Pose bone info
    pose_info = []
    if obj.pose:
        for pb in obj.pose.bones:
            pi = {
                "name": pb.name,
                "location": [round(v, 4) for v in pb.location],
                "rotation": [round(v, 4) for v in pb.rotation_euler],
                "scale": [round(v, 4) for v in pb.scale],
                "rotation_mode": pb.rotation_mode,
            }
            # Bone constraints
            if pb.constraints:
                pi["constraints"] = []
                for con in pb.constraints:
                    ci = {"name": con.name, "type": con.type,
                          "mute": con.mute,
                          "influence": round(con.influence, 3)}
                    if hasattr(con, 'target') and con.target:
                        ci["target"] = con.target.name
                    if hasattr(con, 'subtarget') and con.subtarget:
                        ci["subtarget"] = con.subtarget
                    if con.type == 'IK':
                        ci["chain_count"] = con.chain_count
                    pi["constraints"].append(ci)
            pose_info.append(pi)

    # Build hierarchy summary
    root_bones = [b["name"] for b in bones if b["parent"] is None]
    ik_chains = []
    if obj.pose:
        for pb in obj.pose.bones:
            for con in pb.constraints:
                if con.type == 'IK':
                    chain = [pb.name]
                    parent = pb.parent
                    count = con.chain_count if con.chain_count > 0 else 99
                    while parent and len(chain) < count:
                        chain.append(parent.name)
                        parent = parent.parent
                    ik_chains.append({
                        "tip": pb.name,
                        "chain": list(reversed(chain)),
                        "target": con.target.name if con.target else None,
                    })

    return {
        "name": obj.name,
        "armature": armature.name,
        "bone_count": len(bones),
        "bones": bones,
        "pose_bones": pose_info,
        "root_bones": root_bones,
        "ik_chains": ik_chains,
    }


def inspect_physics(obj=None):
    """Return physics settings for one object or all objects with physics.

    Example: phys = inspect_physics(get("Cloth"))
             all_phys = inspect_physics()
    """
    import bpy

    def _phys_info(o):
        info = {"name": o.name, "physics": []}
        # Rigid body
        if o.rigid_body:
            rb = o.rigid_body
            info["physics"].append({
                "type": "RIGID_BODY",
                "body_type": rb.type,
                "mass": round(rb.mass, 3),
                "collision_shape": rb.collision_shape,
                "friction": round(rb.friction, 3),
                "restitution": round(rb.restitution, 3),
                "enabled": rb.enabled,
                "kinematic": rb.kinematic,
            })
        # Check modifiers for physics types
        for mod in o.modifiers:
            if mod.type == 'CLOTH':
                s = mod.settings
                info["physics"].append({
                    "type": "CLOTH",
                    "quality": s.quality,
                    "mass": round(s.mass, 3),
                    "tension_stiffness": round(s.tension_stiffness, 3),
                    "use_pressure": getattr(s, 'use_pressure', False),
                })
            elif mod.type == 'SOFT_BODY':
                s = mod.settings
                info["physics"].append({
                    "type": "SOFT_BODY",
                    "mass": round(s.mass, 3),
                    "friction": round(s.friction, 3),
                    "speed": round(s.speed, 3),
                })
            elif mod.type == 'FLUID':
                info["physics"].append({
                    "type": "FLUID",
                    "fluid_type": getattr(mod, 'fluid_type', 'NONE'),
                })
            elif mod.type == 'COLLISION':
                s = mod.settings
                info["physics"].append({
                    "type": "COLLISION",
                    "thickness_outer": round(s.thickness_outer, 4),
                    "damping": round(s.damping, 3),
                })
            elif mod.type == 'PARTICLE_SYSTEM':
                ps = mod.particle_system
                if ps and ps.settings:
                    s = ps.settings
                    info["physics"].append({
                        "type": "PARTICLES",
                        "count": s.count,
                        "particle_type": s.type,
                        "lifetime": round(s.lifetime, 1),
                        "emit_from": s.emit_from,
                    })
        return info if info["physics"] else None

    if obj is not None:
        result = _phys_info(obj)
        return result if result else {"name": obj.name, "physics": []}
    else:
        all_phys = []
        for o in bpy.data.objects:
            info = _phys_info(o)
            if info:
                all_phys.append(info)
        return {"physics_objects": all_phys, "count": len(all_phys)}


def inspect_shape_keys(obj):
    """Return all shape keys on a mesh with their values, ranges, and drivers.

    Example: sks = inspect_shape_keys(get("Face"))
    """
    if not obj.data or not obj.data.shape_keys:
        return {"name": obj.name, "shape_keys": []}

    sk = obj.data.shape_keys
    blocks = []
    for kb in sk.key_blocks:
        info = {
            "name": kb.name,
            "value": round(kb.value, 4),
            "slider_min": round(kb.slider_min, 4),
            "slider_max": round(kb.slider_max, 4),
            "mute": kb.mute,
            "relative_key": kb.relative_key.name if kb.relative_key else None,
        }
        # Check for driver
        if sk.animation_data:
            for drv in sk.animation_data.drivers:
                if kb.name in drv.data_path:
                    info["driver"] = drv.driver.expression
                    break
        blocks.append(info)

    return {
        "name": obj.name,
        "use_relative": sk.use_relative,
        "reference_key": sk.reference_key.name if sk.reference_key else None,
        "shape_keys": blocks,
        "count": len(blocks),
    }


def inspect_vertex_groups(obj):
    """Return all vertex groups on an object with vertex counts per group.

    Example: vgs = inspect_vertex_groups(get("Character"))
    """
    if not obj.type == 'MESH':
        return {"error": "{} is not a mesh".format(obj.name)}

    groups = []
    for vg in obj.vertex_groups:
        count = 0
        for v in obj.data.vertices:
            for g in v.groups:
                if g.group == vg.index:
                    count += 1
                    break
        groups.append({
            "name": vg.name,
            "index": vg.index,
            "lock_weight": vg.lock_weight,
            "vertex_count": count,
        })

    return {
        "name": obj.name,
        "vertex_groups": groups,
        "count": len(groups),
        "total_vertices": len(obj.data.vertices),
    }


def inspect_uv_maps(obj):
    """Return UV map info for a mesh object.

    Example: uvs = inspect_uv_maps(get("Sword"))
    """
    if obj.type != 'MESH' or not obj.data:
        return {"error": "{} is not a mesh".format(obj.name)}

    uv_layers = []
    for uv in obj.data.uv_layers:
        uv_layers.append({
            "name": uv.name,
            "active": uv.active,
            "active_render": uv.active_render,
        })

    return {
        "name": obj.name,
        "uv_maps": uv_layers,
        "count": len(uv_layers),
    }


def inspect_scene_hierarchy():
    """Return the full scene collection hierarchy with all objects,
    their parent-child relationships, and collection membership.

    Example: hier = inspect_scene_hierarchy()
    """
    import bpy

    def _collection_tree(col, depth=0):
        items = []
        for obj in col.objects:
            item = {
                "name": obj.name,
                "type": obj.type,
                "parent": obj.parent.name if obj.parent else None,
                "visible": obj.visible_get(),
            }
            if obj.children:
                item["children"] = [c.name for c in obj.children]
            items.append(item)
        children = []
        for child_col in col.children:
            children.append(_collection_tree(child_col, depth + 1))
        return {
            "name": col.name,
            "objects": items,
            "children": children,
        }

    scene = bpy.context.scene
    tree = _collection_tree(scene.collection)

    # Object parenting overview
    parented = []
    for obj in bpy.data.objects:
        if obj.parent:
            parented.append({
                "child": obj.name,
                "parent": obj.parent.name,
                "parent_type": obj.parent_type,
            })

    return {
        "scene": scene.name,
        "hierarchy": tree,
        "parent_relationships": parented,
        "total_objects": len(bpy.data.objects),
    }


def inspect_render_settings():
    """Return current render settings: engine, resolution, output, samples.

    Example: rs = inspect_render_settings()
    """
    import bpy
    scene = bpy.context.scene
    render = scene.render

    result = {
        "engine": render.engine,
        "resolution_x": render.resolution_x,
        "resolution_y": render.resolution_y,
        "resolution_percentage": render.resolution_percentage,
        "fps": render.fps,
        "fps_base": render.fps_base,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "film_transparent": render.film_transparent,
        "output_path": render.filepath,
        "file_format": render.image_settings.file_format,
        "color_mode": render.image_settings.color_mode,
    }

    # Engine-specific
    if render.engine == 'CYCLES':
        cycles = scene.cycles
        result["cycles"] = {
            "samples": cycles.samples,
            "preview_samples": cycles.preview_samples,
            "use_denoising": cycles.use_denoising,
            "device": cycles.device,
        }
    elif render.engine == 'BLENDER_EEVEE_NEXT' or render.engine == 'BLENDER_EEVEE':
        eevee = scene.eevee
        result["eevee"] = {
            "taa_render_samples": getattr(eevee, 'taa_render_samples', None),
            "use_bloom": getattr(eevee, 'use_bloom', None),
            "use_ssr": getattr(eevee, 'use_ssr', None),
            "use_gtao": getattr(eevee, 'use_gtao', None),
        }

    return result


def inspect_world():
    """Return world/environment settings: background, HDRI, volume.

    Example: w = inspect_world()
    """
    import bpy
    world = bpy.context.scene.world
    if not world:
        return {"world": None}

    result = {"name": world.name, "use_nodes": world.use_nodes}
    if world.use_nodes and world.node_tree:
        nodes = []
        for n in world.node_tree.nodes:
            nd = {"type": n.type, "name": n.name}
            if n.type == 'BACKGROUND':
                color_inp = n.inputs.get('Color')
                strength_inp = n.inputs.get('Strength')
                if color_inp and not color_inp.is_linked:
                    nd["color"] = [round(v, 3) for v in color_inp.default_value]
                if strength_inp and not strength_inp.is_linked:
                    nd["strength"] = round(strength_inp.default_value, 3)
            elif n.type == 'TEX_ENVIRONMENT':
                nd["image"] = n.image.name if n.image else None
            elif n.type == 'TEX_SKY':
                nd["sky_type"] = n.sky_type
                nd["sun_elevation"] = round(n.sun_elevation, 4)
                nd["sun_rotation"] = round(n.sun_rotation, 4)
            nodes.append(nd)
        result["nodes"] = nodes
    return result


# ═══════════════════════════════════════════════════════════════════
# SHADER NODE EDITING  —  generic node manipulation for Qwen
# ═══════════════════════════════════════════════════════════════════

def _ensure_node_tree(mat):
    """Internal: make sure a material has a node tree."""
    if not mat.use_nodes:
        mat.use_nodes = True
    return mat.node_tree


def _auto_arrange_tree(tree, x_spacing=250, y_spacing=80):
    """Auto-arrange nodes in a clean left-to-right horizontal layout.

    Performs a topological sort from Output/Composite backwards, then
    places nodes in columns with consistent spacing — like a real
    Blender artist would lay them out.
    """
    if not tree or not tree.nodes:
        return

    # Find sink nodes (Output Material, Composite, Viewer, etc.)
    sinks = [n for n in tree.nodes if n.type in (
        'OUTPUT_MATERIAL', 'COMPOSITE', 'VIEWER', 'OUTPUT_WORLD',
        'OUTPUT_LIGHT')]
    if not sinks:
        sinks = [n for n in tree.nodes if not any(
            lnk.from_node == n for lnk in tree.links)]
    if not sinks:
        return

    # BFS backwards from sinks to assign depth (column)
    depth = {}
    visited = set()
    queue = [(s, 0) for s in sinks]
    for node, d in queue:
        if id(node) in visited:
            if d > depth.get(id(node), 0):
                depth[id(node)] = d
            continue
        visited.add(id(node))
        depth[id(node)] = max(depth.get(id(node), 0), d)
        for inp in node.inputs:
            for lnk in inp.links:
                queue.append((lnk.from_node, d + 1))

    # Unconnected nodes get placed far left
    max_depth = max(depth.values()) if depth else 0
    for node in tree.nodes:
        if id(node) not in depth:
            max_depth += 1
            depth[id(node)] = max_depth

    # Group by column
    columns = {}
    for node in tree.nodes:
        d = depth.get(id(node), 0)
        columns.setdefault(d, []).append(node)

    # Place: rightmost column at x=0, going left
    max_col = max(columns.keys()) if columns else 0
    for col_idx, nodes_in_col in sorted(columns.items()):
        x = (max_col - col_idx) * x_spacing
        total_height = sum(max(n.height, 150) + y_spacing
                          for n in nodes_in_col) - y_spacing
        y = total_height / 2
        for node in nodes_in_col:
            node.location.x = x
            node.location.y = y
            y -= max(node.height, 150) + y_spacing


def add_shader_node(mat, node_type, name=None, location=None, **kwargs):
    """Add a shader node to a material's node tree.

    node_type: Blender node type string, e.g.:
      'ShaderNodeBsdfPrincipled', 'ShaderNodeTexNoise',
      'ShaderNodeTexImage', 'ShaderNodeMix', 'ShaderNodeBump',
      'ShaderNodeNormalMap', 'ShaderNodeTexCoord',
      'ShaderNodeMapping', 'ShaderNodeMath', 'ShaderNodeValToRGB',
      'ShaderNodeSeparateXYZ', 'ShaderNodeTexVoronoi',
      'ShaderNodeTexMusgrave', 'ShaderNodeInvert',
      'ShaderNodeMixShader', 'ShaderNodeBsdfGlass',
      'ShaderNodeBsdfDiffuse', 'ShaderNodeEmission'

    Shorthand aliases also accepted (case-insensitive):
      'principled', 'noise', 'voronoi', 'image', 'mix', 'colorramp',
      'math', 'bump', 'normalmap', 'mapping', 'texcoord', 'separate_xyz',
      'musgrave', 'invert', 'mix_shader', 'glass', 'emission',
      'diffuse', 'wave', 'gradient', 'fresnel', 'layer_weight'

    Returns the created node.

    Example:
        mat = quick_material("Rock")
        noise = add_shader_node(mat, 'noise', name='RockNoise')
        set_node_input(noise, 'Scale', 15.0)
        set_node_input(noise, 'Detail', 8.0)
        ramp = add_shader_node(mat, 'colorramp', name='ColorVariation')
        connect_nodes(mat, noise, 'Fac', ramp, 'Fac')
        bsdf = get_node(mat, 'Principled BSDF')
        connect_nodes(mat, ramp, 'Color', bsdf, 'Base Color')
        auto_arrange_nodes(mat)
    """
    _ALIASES = {
        'principled': 'ShaderNodeBsdfPrincipled',
        'noise': 'ShaderNodeTexNoise',
        'voronoi': 'ShaderNodeTexVoronoi',
        'image': 'ShaderNodeTexImage',
        'mix': 'ShaderNodeMix',
        'mix_rgb': 'ShaderNodeMixRGB',
        'colorramp': 'ShaderNodeValToRGB',
        'color_ramp': 'ShaderNodeValToRGB',
        'math': 'ShaderNodeMath',
        'bump': 'ShaderNodeBump',
        'normalmap': 'ShaderNodeNormalMap',
        'normal_map': 'ShaderNodeNormalMap',
        'mapping': 'ShaderNodeMapping',
        'texcoord': 'ShaderNodeTexCoord',
        'tex_coord': 'ShaderNodeTexCoord',
        'separate_xyz': 'ShaderNodeSeparateXYZ',
        'combine_xyz': 'ShaderNodeCombineXYZ',
        'musgrave': 'ShaderNodeTexMusgrave',
        'invert': 'ShaderNodeInvert',
        'mix_shader': 'ShaderNodeMixShader',
        'add_shader': 'ShaderNodeAddShader',
        'glass': 'ShaderNodeBsdfGlass',
        'emission': 'ShaderNodeEmission',
        'diffuse': 'ShaderNodeBsdfDiffuse',
        'transparent': 'ShaderNodeBsdfTransparent',
        'glossy': 'ShaderNodeBsdfGlossy',
        'wave': 'ShaderNodeTexWave',
        'gradient': 'ShaderNodeTexGradient',
        'checker': 'ShaderNodeTexChecker',
        'brick': 'ShaderNodeTexBrick',
        'fresnel': 'ShaderNodeFresnel',
        'layer_weight': 'ShaderNodeLayerWeight',
        'rgb': 'ShaderNodeRGB',
        'value': 'ShaderNodeValue',
        'displacement': 'ShaderNodeDisplacement',
        'vector_math': 'ShaderNodeVectorMath',
        'clamp': 'ShaderNodeClamp',
        'map_range': 'ShaderNodeMapRange',
        'ambient_occlusion': 'ShaderNodeAmbientOcclusion',
        'ao': 'ShaderNodeAmbientOcclusion',
        'output': 'ShaderNodeOutputMaterial',
    }
    tree = _ensure_node_tree(mat)
    actual_type = _ALIASES.get(node_type.lower(), node_type)
    node = tree.nodes.new(actual_type)
    if name:
        node.name = name
        node.label = name
    if location:
        node.location = location
    # Apply extra keyword settings
    for key, val in kwargs.items():
        if hasattr(node, key):
            setattr(node, key, val)
    return node


def get_node(mat, name):
    """Get a node from a material's node tree by name.

    Also searches by type label (e.g., 'Principled BSDF', 'Material Output').

    Example: bsdf = get_node(mat, 'Principled BSDF')
    """
    tree = _ensure_node_tree(mat)
    node = tree.nodes.get(name)
    if node:
        return node
    # Search by label
    for n in tree.nodes:
        if n.label == name or n.name == name:
            return n
    # Search by type display name
    name_lower = name.lower().replace(' ', '').replace('_', '')
    for n in tree.nodes:
        type_name = n.bl_label.lower().replace(' ', '').replace('_', '') if hasattr(n, 'bl_label') else ''
        if type_name == name_lower:
            return n
    return None


def set_node_input(node, input_name, value):
    """Set an input value on a shader/compositor node.

    Handles color (tuple), float, int automatically.

    Example:
        bsdf = get_node(mat, 'Principled BSDF')
        set_node_input(bsdf, 'Base Color', (0.8, 0.2, 0.1, 1))
        set_node_input(bsdf, 'Metallic', 0.9)
        set_node_input(bsdf, 'Roughness', 0.2)
    """
    inp = node.inputs.get(input_name)
    if inp is None:
        # Try case-insensitive match
        for i in node.inputs:
            if i.name.lower() == input_name.lower():
                inp = i
                break
    if inp is None:
        return {"error": "Input '{}' not found on {}".format(
            input_name, node.name)}
    inp.default_value = value
    return {"node": node.name, "input": input_name, "value": str(value)}


def get_node_input(node, input_name):
    """Read the current value of a node input.

    Example: color = get_node_input(bsdf, 'Base Color')
    """
    inp = node.inputs.get(input_name)
    if inp is None:
        for i in node.inputs:
            if i.name.lower() == input_name.lower():
                inp = i
                break
    if inp is None:
        return {"error": "Input '{}' not found".format(input_name)}
    val = inp.default_value
    if hasattr(val, '__len__'):
        return {"value": [round(v, 4) for v in val], "linked": inp.is_linked}
    return {"value": round(val, 4) if isinstance(val, float) else val,
            "linked": inp.is_linked}


def connect_nodes(mat_or_tree, from_node, from_output, to_node, to_input):
    """Connect two nodes in a material or compositor tree.

    Accepts material object or node tree directly. Nodes can be objects
    or name strings. Socket names are matched case-insensitively.

    Example:
        connect_nodes(mat, 'RockNoise', 'Fac', 'Principled BSDF', 'Roughness')
        connect_nodes(mat, noise_node, 'Color', bsdf_node, 'Base Color')
    """
    if hasattr(mat_or_tree, 'node_tree'):
        tree = mat_or_tree.node_tree
    elif hasattr(mat_or_tree, 'nodes'):
        tree = mat_or_tree
    else:
        return {"error": "Not a material or node tree"}

    # Resolve nodes by name if strings
    if isinstance(from_node, str):
        from_node = tree.nodes.get(from_node) or get_node(mat_or_tree, from_node)
    if isinstance(to_node, str):
        to_node = tree.nodes.get(to_node) or get_node(mat_or_tree, to_node)
    if not from_node or not to_node:
        return {"error": "Node not found"}

    # Find output socket
    out_socket = None
    for s in from_node.outputs:
        if s.name.lower() == from_output.lower():
            out_socket = s
            break
    if not out_socket:
        out_socket = from_node.outputs.get(from_output)
    if not out_socket and from_output.isdigit():
        idx = int(from_output)
        if idx < len(from_node.outputs):
            out_socket = from_node.outputs[idx]

    # Find input socket
    in_socket = None
    for s in to_node.inputs:
        if s.name.lower() == to_input.lower():
            in_socket = s
            break
    if not in_socket:
        in_socket = to_node.inputs.get(to_input)
    if not in_socket and to_input.isdigit():
        idx = int(to_input)
        if idx < len(to_node.inputs):
            in_socket = to_node.inputs[idx]

    if not out_socket:
        return {"error": "Output '{}' not found on {}".format(
            from_output, from_node.name)}
    if not in_socket:
        return {"error": "Input '{}' not found on {}".format(
            to_input, to_node.name)}

    # Remove existing link to this input
    for lnk in list(tree.links):
        if lnk.to_socket == in_socket:
            tree.links.remove(lnk)

    tree.links.new(out_socket, in_socket)
    return {"linked": "{}.{} → {}.{}".format(
        from_node.name, out_socket.name, to_node.name, in_socket.name)}


def disconnect_node_input(mat_or_tree, node, input_name):
    """Disconnect all links going into a specific node input.

    Example: disconnect_node_input(mat, bsdf, 'Base Color')
    """
    if hasattr(mat_or_tree, 'node_tree'):
        tree = mat_or_tree.node_tree
    else:
        tree = mat_or_tree
    if isinstance(node, str):
        node = tree.nodes.get(node)
    if not node:
        return {"error": "Node not found"}
    inp = node.inputs.get(input_name)
    if not inp:
        return {"error": "Input not found"}
    removed = 0
    for lnk in list(tree.links):
        if lnk.to_socket == inp:
            tree.links.remove(lnk)
            removed += 1
    return {"disconnected": removed}


def remove_node(mat_or_tree, node):
    """Remove a node from a material or compositor tree.

    Example: remove_node(mat, 'OldNoise')
    """
    if hasattr(mat_or_tree, 'node_tree'):
        tree = mat_or_tree.node_tree
    else:
        tree = mat_or_tree
    if isinstance(node, str):
        node = tree.nodes.get(node)
    if not node:
        return {"error": "Node not found"}
    name = node.name
    tree.nodes.remove(node)
    return {"removed": name}


def list_node_inputs(node):
    """List all inputs on a node with their types and current values.

    Example: inputs = list_node_inputs(bsdf)
    """
    result = []
    for inp in node.inputs:
        info = {"name": inp.name, "type": inp.type, "linked": inp.is_linked}
        if not inp.is_linked:
            val = inp.default_value
            if hasattr(val, '__len__'):
                info["value"] = [round(v, 4) for v in val]
            elif isinstance(val, float):
                info["value"] = round(val, 4)
            else:
                info["value"] = val
        result.append(info)
    return result


def list_node_outputs(node):
    """List all outputs on a node with their types.

    Example: outputs = list_node_outputs(noise)
    """
    return [{"name": s.name, "type": s.type} for s in node.outputs]


def auto_arrange_nodes(mat_or_tree):
    """Auto-arrange nodes in a clean horizontal left-to-right layout.

    Call this after building or modifying a node tree to keep it tidy.

    Example:
        auto_arrange_nodes(mat)     # Material tree
        auto_arrange_nodes(comp)    # Compositor tree
    """
    if hasattr(mat_or_tree, 'node_tree'):
        tree = mat_or_tree.node_tree
    elif hasattr(mat_or_tree, 'nodes'):
        tree = mat_or_tree
    else:
        return {"error": "Not a material or node tree"}
    _auto_arrange_tree(tree)
    return {"arranged": len(tree.nodes)}


def set_colorramp_stops(node, stops):
    """Configure color ramp stops.

    stops: list of (position, color) tuples.
      position: 0.0-1.0
      color: (r, g, b, a) tuple

    Example:
        ramp = add_shader_node(mat, 'colorramp')
        set_colorramp_stops(ramp, [
            (0.0, (0.1, 0.05, 0.02, 1)),   # dark brown
            (0.4, (0.4, 0.25, 0.12, 1)),   # medium brown
            (1.0, (0.7, 0.55, 0.35, 1)),   # light tan
        ])
    """
    if not hasattr(node, 'color_ramp'):
        return {"error": "Node has no color_ramp"}
    cr = node.color_ramp
    # Ensure enough stops
    while len(cr.elements) < len(stops):
        cr.elements.new(0.5)
    # Remove extras
    while len(cr.elements) > len(stops) and len(cr.elements) > 1:
        cr.elements.remove(cr.elements[-1])
    for i, (pos, color) in enumerate(stops):
        if i < len(cr.elements):
            cr.elements[i].position = pos
            cr.elements[i].color = color
    return {"stops": len(stops)}


def set_node_property(node, prop_name, value):
    """Set a property on a node (not an input socket, but the node itself).

    Useful for: blend_type, operation, gradient_type, musgrave_type, etc.

    Example:
        math_node = add_shader_node(mat, 'math')
        set_node_property(math_node, 'operation', 'MULTIPLY')

        mix_node = add_shader_node(mat, 'mix')
        set_node_property(mix_node, 'blend_type', 'OVERLAY')
    """
    if hasattr(node, prop_name):
        setattr(node, prop_name, value)
        return {"node": node.name, "property": prop_name, "value": str(value)}
    return {"error": "Property '{}' not found on {}".format(prop_name, node.name)}


def duplicate_material(mat, new_name=None):
    """Create a copy of a material with independent node tree.

    Example: mat2 = duplicate_material(find_material('Wood'), 'DarkWood')
    """
    new_mat = mat.copy()
    if new_name:
        new_mat.name = new_name
    return new_mat


def clear_material_nodes(mat):
    """Remove all nodes from a material and start fresh with Output + Principled.

    Example: clear_material_nodes(mat)
    """
    tree = _ensure_node_tree(mat)
    tree.nodes.clear()
    output = tree.nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)
    bsdf = tree.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    return {"material": mat.name, "nodes": 2}


# ═══════════════════════════════════════════════════════════════════
# COMPOSITOR NODE EDITING  —  generic node manipulation for Qwen
# ═══════════════════════════════════════════════════════════════════

def _ensure_compositor():
    """Internal: ensure compositor is enabled and return the node tree."""
    import bpy
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    # Ensure basic nodes exist
    has_rl = any(n.type == 'R_LAYERS' for n in tree.nodes)
    has_comp = any(n.type == 'COMPOSITE' for n in tree.nodes)
    if not has_rl:
        rl = tree.nodes.new('CompositorNodeRLayers')
        rl.location = (0, 0)
    if not has_comp:
        comp = tree.nodes.new('CompositorNodeComposite')
        comp.location = (800, 0)
    return tree


def add_compositor_node(node_type, name=None, location=None, **kwargs):
    """Add a node to the compositor tree.

    node_type: Blender compositor node type or shorthand alias:
      'glare', 'blur', 'bright_contrast', 'hue_sat', 'color_balance',
      'mix', 'alpha_over', 'lens_distortion', 'denoise', 'viewer',
      'ellipse_mask', 'box_mask', 'split_viewer', 'map_value',
      'math', 'separate_rgba', 'combine_rgba', 'rgb_curves',
      'defocus', 'directional_blur', 'invert', 'filter',
      'map_range', 'normalize', 'tonemap', 'file_output',
      'switch', 'translate', 'scale', 'rotate', 'flip',
      'crop', 'stabilize', 'keying', 'keying_screen',
      'color_correction', 'exposure'

    Example:
        glare = add_compositor_node('glare', name='Bloom')
        set_node_input(glare, 'Threshold', 0.8)
        set_node_property(glare, 'glare_type', 'FOG_GLOW')
    """
    _COMP_ALIASES = {
        'render_layers': 'CompositorNodeRLayers',
        'composite': 'CompositorNodeComposite',
        'viewer': 'CompositorNodeViewer',
        'split_viewer': 'CompositorNodeSplitViewer',
        'glare': 'CompositorNodeGlare',
        'blur': 'CompositorNodeBlur',
        'bright_contrast': 'CompositorNodeBrightContrast',
        'hue_sat': 'CompositorNodeHueSat',
        'color_balance': 'CompositorNodeColorBalance',
        'color_correction': 'CompositorNodeColorCorrection',
        'mix': 'CompositorNodeMixRGB',
        'alpha_over': 'CompositorNodeAlphaOver',
        'lens_distortion': 'CompositorNodeLensdist',
        'denoise': 'CompositorNodeDenoise',
        'ellipse_mask': 'CompositorNodeEllipseMask',
        'box_mask': 'CompositorNodeBoxMask',
        'math': 'CompositorNodeMath',
        'map_value': 'CompositorNodeMapValue',
        'map_range': 'CompositorNodeMapRange',
        'separate_rgba': 'CompositorNodeSepRGBA',
        'combine_rgba': 'CompositorNodeCombRGBA',
        'rgb_curves': 'CompositorNodeCurveRGB',
        'defocus': 'CompositorNodeDefocus',
        'directional_blur': 'CompositorNodeDBlur',
        'invert': 'CompositorNodeInvert',
        'filter': 'CompositorNodeFilter',
        'normalize': 'CompositorNodeNormalize',
        'tonemap': 'CompositorNodeTonemap',
        'file_output': 'CompositorNodeOutputFile',
        'switch': 'CompositorNodeSwitch',
        'translate': 'CompositorNodeTranslate',
        'scale': 'CompositorNodeScale',
        'rotate': 'CompositorNodeRotate',
        'flip': 'CompositorNodeFlip',
        'crop': 'CompositorNodeCrop',
        'stabilize': 'CompositorNodeStabilize',
        'keying': 'CompositorNodeKeying',
        'keying_screen': 'CompositorNodeKeyingScreen',
        'exposure': 'CompositorNodeExposure',
    }
    tree = _ensure_compositor()
    actual_type = _COMP_ALIASES.get(node_type.lower(), node_type)
    node = tree.nodes.new(actual_type)
    if name:
        node.name = name
        node.label = name
    if location:
        node.location = location
    for key, val in kwargs.items():
        if hasattr(node, key):
            setattr(node, key, val)
    return node


def get_compositor_node(name):
    """Get a compositor node by name.

    Example: rl = get_compositor_node('Render Layers')
    """
    import bpy
    tree = bpy.context.scene.node_tree
    if not tree:
        return None
    node = tree.nodes.get(name)
    if node:
        return node
    for n in tree.nodes:
        if n.label == name or n.name == name:
            return n
        name_lower = name.lower().replace(' ', '').replace('_', '')
        type_name = n.bl_label.lower().replace(' ', '').replace('_', '') if hasattr(n, 'bl_label') else ''
        if type_name == name_lower:
            return n
    return None


def connect_compositor_nodes(from_node, from_output, to_node, to_input):
    """Connect two compositor nodes.

    Nodes can be objects or name strings.

    Example:
        connect_compositor_nodes('Render Layers', 'Image', 'Bloom', 'Image')
        connect_compositor_nodes('Bloom', 'Image', 'Composite', 'Image')
    """
    import bpy
    tree = bpy.context.scene.node_tree
    if not tree:
        return {"error": "No compositor tree"}

    if isinstance(from_node, str):
        from_node = get_compositor_node(from_node)
    if isinstance(to_node, str):
        to_node = get_compositor_node(to_node)
    if not from_node or not to_node:
        return {"error": "Node not found"}

    out_socket = None
    for s in from_node.outputs:
        if s.name.lower() == from_output.lower():
            out_socket = s
            break
    if not out_socket:
        out_socket = from_node.outputs.get(from_output)

    in_socket = None
    for s in to_node.inputs:
        if s.name.lower() == to_input.lower():
            in_socket = s
            break
    if not in_socket:
        in_socket = to_node.inputs.get(to_input)

    if not out_socket or not in_socket:
        return {"error": "Socket not found"}

    for lnk in list(tree.links):
        if lnk.to_socket == in_socket:
            tree.links.remove(lnk)

    tree.links.new(out_socket, in_socket)
    return {"linked": "{}.{} → {}.{}".format(
        from_node.name, out_socket.name, to_node.name, in_socket.name)}


def remove_compositor_node(node):
    """Remove a node from the compositor.

    Example: remove_compositor_node('OldGlare')
    """
    import bpy
    tree = bpy.context.scene.node_tree
    if not tree:
        return {"error": "No compositor tree"}
    if isinstance(node, str):
        node = get_compositor_node(node)
    if not node:
        return {"error": "Node not found"}
    name = node.name
    tree.nodes.remove(node)
    return {"removed": name}


def auto_arrange_compositor():
    """Auto-arrange all compositor nodes in a clean horizontal layout.

    Example: auto_arrange_compositor()
    """
    import bpy
    tree = bpy.context.scene.node_tree
    if not tree:
        return {"error": "No compositor tree"}
    _auto_arrange_tree(tree)
    return {"arranged": len(tree.nodes)}


def clear_compositor():
    """Clear all compositor nodes and set up fresh Render Layers → Composite.

    Example: clear_compositor()
    """
    tree = _ensure_compositor()
    tree.nodes.clear()
    rl = tree.nodes.new('CompositorNodeRLayers')
    rl.location = (0, 0)
    comp = tree.nodes.new('CompositorNodeComposite')
    comp.location = (400, 0)
    tree.links.new(rl.outputs['Image'], comp.inputs['Image'])
    return {"compositor": "reset"}


def add_viewer_node():
    """Add a Viewer node to the compositor for preview.

    Example: add_viewer_node()
    """
    tree = _ensure_compositor()
    viewer = tree.nodes.new('CompositorNodeViewer')
    viewer.location = (800, -200)
    # Connect from last node before composite
    comp = None
    for n in tree.nodes:
        if n.type == 'COMPOSITE':
            comp = n
            break
    if comp:
        for lnk in tree.links:
            if lnk.to_node == comp and lnk.to_socket.name == 'Image':
                tree.links.new(lnk.from_socket, viewer.inputs['Image'])
                break
    return {"viewer": "added"}


def insert_compositor_node_between(new_node_type, before_node, after_node,
                                   socket_name='Image', **kwargs):
    """Insert a new compositor node between two existing nodes.

    Automatically re-links the chain. Useful for adding effects inline.

    Example:
        insert_compositor_node_between('glare', 'Render Layers', 'Composite')
    """
    import bpy
    tree = bpy.context.scene.node_tree
    if not tree:
        return {"error": "No compositor tree"}

    if isinstance(before_node, str):
        before_node = get_compositor_node(before_node)
    if isinstance(after_node, str):
        after_node = get_compositor_node(after_node)
    if not before_node or not after_node:
        return {"error": "Node not found"}

    # Find the link between them
    target_link = None
    for lnk in tree.links:
        if lnk.from_node == before_node and lnk.to_node == after_node:
            target_link = lnk
            break
    if not target_link:
        return {"error": "No direct link between these nodes"}

    out_socket = target_link.from_socket
    in_socket = target_link.to_socket

    # Create the new node
    new_node = add_compositor_node(new_node_type, **kwargs)
    tree.links.remove(target_link)

    # Find matching sockets on new node
    new_in = None
    for s in new_node.inputs:
        if s.name == socket_name or s.type == out_socket.type:
            new_in = s
            break
    if not new_in:
        new_in = new_node.inputs[0]

    new_out = None
    for s in new_node.outputs:
        if s.name == socket_name or s.type == in_socket.type:
            new_out = s
            break
    if not new_out:
        new_out = new_node.outputs[0]

    tree.links.new(out_socket, new_in)
    tree.links.new(new_out, in_socket)

    _auto_arrange_tree(tree)
    return {"inserted": new_node.name,
            "chain": "{} → {} → {}".format(
                before_node.name, new_node.name, after_node.name)}