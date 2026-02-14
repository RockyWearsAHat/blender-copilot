"""Blender Injector — converts model output directly into Blender scene objects.

This is the KEY module that replaces Python script generation.
Instead of the AI writing code, the model outputs structured mesh data
and this injector creates real Blender objects from it.

This runs INSIDE Blender (imported by the addon).
"""

import json
from typing import Any


def inject_mesh(obj_data: dict) -> "bpy.types.Object":
    """Create a Blender mesh object from structured data.

    Args:
        obj_data: Dict with keys:
            - name: object name
            - mesh: {vertices: [[x,y,z],...], faces: [[v1,v2,v3],...]}
            - materials: [{name, type, base_color, roughness, ...}, ...]
            - modifiers: [{type, ...}, ...]
            - transforms: {location, rotation_euler, scale}

    Returns:
        The created bpy.types.Object
    """
    import bpy
    import bmesh
    from mathutils import Vector

    mesh_data = obj_data.get("mesh", {})
    vertices = mesh_data.get("vertices", [])
    faces = mesh_data.get("faces", [])
    name = obj_data.get("name", "Generated")

    # Create mesh data block
    mesh = bpy.data.meshes.new(name + "_mesh")

    # Populate mesh directly — no scripts, just data
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    mesh.validate()

    # Create object
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # Apply transforms
    transforms = obj_data.get("transforms", {})
    if "location" in transforms:
        obj.location = transforms["location"]
    if "rotation_euler" in transforms:
        obj.rotation_euler = transforms["rotation_euler"]
    if "scale" in transforms:
        obj.scale = transforms["scale"]

    # Apply materials
    for mat_data in obj_data.get("materials", []):
        mat = create_material_from_data(mat_data)
        if mat:
            obj.data.materials.append(mat)

    # Apply modifiers
    for mod_data in obj_data.get("modifiers", []):
        apply_modifier_from_data(obj, mod_data)

    # Shading
    if obj_data.get("shade_smooth", True):
        for poly in mesh.polygons:
            poly.use_smooth = True

    # Post-processing
    postprocess_mesh(obj)

    return obj


def create_material_from_data(mat_data: dict) -> "bpy.types.Material | None":
    """Create a Blender material from structured data.

    Supports both simple PBR materials and full node tree definitions.
    """
    import bpy

    name = mat_data.get("name", "Material")
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    mat_type = mat_data.get("type", "PBR")

    if mat_type == "PBR" or mat_type == "simple":
        # Simple PBR — just set Principled BSDF values
        principled = nodes.get("Principled BSDF")
        if principled is None:
            # Find it or create it
            for n in nodes:
                if n.type == "BSDF_PRINCIPLED":
                    principled = n
                    break
            if principled is None:
                principled = nodes.new("ShaderNodeBsdfPrincipled")

        # Set input values
        if "base_color" in mat_data:
            color = mat_data["base_color"]
            if len(color) == 3:
                color = list(color) + [1.0]
            principled.inputs["Base Color"].default_value = color

        for prop, input_name in [
            ("roughness", "Roughness"),
            ("metallic", "Metallic"),
            ("specular", "Specular IOR Level"),
            ("alpha", "Alpha"),
        ]:
            if prop in mat_data:
                try:
                    principled.inputs[input_name].default_value = mat_data[prop]
                except (KeyError, TypeError):
                    pass

        if mat_data.get("alpha", 1.0) < 1.0:
            mat.blend_method = "BLEND" if hasattr(mat, "blend_method") else None

    elif mat_type == "node_tree":
        # Full node tree reconstruction
        _rebuild_node_tree(tree, mat_data)

    return mat


def _rebuild_node_tree(tree, mat_data: dict):
    """Reconstruct a complete shader node tree from structured data."""
    import bpy

    nodes = tree.nodes
    links = tree.links

    # Clear existing nodes
    nodes.clear()

    node_map = {}  # name → node object

    for node_def in mat_data.get("nodes", []):
        bl_idname = node_def.get("bl_idname", "")
        if not bl_idname:
            # Map type to bl_idname
            type_to_idname = {
                "BSDF_PRINCIPLED": "ShaderNodeBsdfPrincipled",
                "OUTPUT_MATERIAL": "ShaderNodeOutputMaterial",
                "TEX_IMAGE": "ShaderNodeTexImage",
                "TEX_NOISE": "ShaderNodeTexNoise",
                "TEX_BRICK": "ShaderNodeTexBrick",
                "MIX_RGB": "ShaderNodeMixRGB",
                "MAPPING": "ShaderNodeMapping",
                "TEX_COORD": "ShaderNodeTexCoord",
                "NORMAL_MAP": "ShaderNodeNormalMap",
                "BUMP": "ShaderNodeBump",
                "MATH": "ShaderNodeMath",
                "SEPARATE_XYZ": "ShaderNodeSeparateXYZ",
                "COMBINE_XYZ": "ShaderNodeCombineXYZ",
                "INVERT": "ShaderNodeInvert",
                "HUE_SAT": "ShaderNodeHueSaturation",
                "BSDF_GLOSSY": "ShaderNodeBsdfGlossy",
                "BSDF_DIFFUSE": "ShaderNodeBsdfDiffuse",
                "BSDF_GLASS": "ShaderNodeBsdfGlass",
                "EMISSION": "ShaderNodeEmission",
                "MIX_SHADER": "ShaderNodeMixShader",
                "ADD_SHADER": "ShaderNodeAddShader",
            }
            bl_idname = type_to_idname.get(node_def.get("type", ""), "")

        if not bl_idname:
            continue

        try:
            node = nodes.new(bl_idname)
        except Exception:
            continue

        node.name = node_def.get("name", node.name)
        if "location" in node_def:
            node.location = node_def["location"]

        # Set input values
        for inp_name, inp_val in node_def.get("inputs", {}).items():
            if inp_val == "LINKED":
                continue
            try:
                inp = node.inputs.get(inp_name)
                if inp:
                    inp.default_value = inp_val
            except (TypeError, ValueError):
                pass

        # Set node-specific properties
        if node_def.get("type") == "MATH" and "operation" in node_def:
            node.operation = node_def["operation"]
        if node_def.get("type") == "MIX_RGB":
            if "blend_type" in node_def:
                node.blend_type = node_def["blend_type"]

        node_map[node.name] = node

    # Rebuild links
    for link_def in mat_data.get("links", []):
        from_node = node_map.get(link_def.get("from_node"))
        to_node = node_map.get(link_def.get("to_node"))
        if from_node and to_node:
            try:
                from_socket = from_node.outputs.get(link_def.get("from_socket"))
                to_socket = to_node.inputs.get(link_def.get("to_socket"))
                if from_socket and to_socket:
                    links.new(from_socket, to_socket)
            except Exception:
                pass


def apply_modifier_from_data(obj, mod_data: dict):
    """Add a modifier to an object from structured data."""
    mod_type = mod_data.get("type", "")
    mod_name = mod_data.get("name", mod_type)

    try:
        mod = obj.modifiers.new(mod_name, mod_type)
    except Exception:
        return

    # Set modifier properties
    if mod_type == "SUBSURF":
        mod.levels = mod_data.get("levels", 2)
        mod.render_levels = mod_data.get("render_levels", mod.levels)
        if "subdivision_type" in mod_data:
            mod.subdivision_type = mod_data["subdivision_type"]

    elif mod_type == "MIRROR":
        axes = mod_data.get("use_axis", [True, False, False])
        for i, val in enumerate(axes[:3]):
            mod.use_axis[i] = val
        mod.use_clip = mod_data.get("use_clip", True)

    elif mod_type == "SOLIDIFY":
        mod.thickness = mod_data.get("thickness", 0.1)
        mod.offset = mod_data.get("offset", -1)

    elif mod_type == "BEVEL":
        mod.width = mod_data.get("width", 0.02)
        mod.segments = mod_data.get("segments", 2)
        if "limit_method" in mod_data:
            mod.limit_method = mod_data["limit_method"]

    elif mod_type == "ARRAY":
        mod.count = mod_data.get("count", 2)
        if "relative_offset_displace" in mod_data:
            for i, v in enumerate(mod_data["relative_offset_displace"][:3]):
                mod.relative_offset_displace[i] = v

    elif mod_type == "SHRINKWRAP":
        if "wrap_method" in mod_data:
            mod.wrap_method = mod_data["wrap_method"]
        mod.offset = mod_data.get("offset", 0)

    elif mod_type == "SIMPLE_DEFORM":
        if "deform_method" in mod_data:
            mod.deform_method = mod_data["deform_method"]
        if "angle" in mod_data:
            mod.angle = mod_data["angle"]
        if "deform_axis" in mod_data:
            mod.deform_axis = mod_data["deform_axis"]

    elif mod_type == "SCREW":
        if "angle" in mod_data:
            mod.angle = mod_data["angle"]
        mod.steps = mod_data.get("steps", 16)
        mod.render_steps = mod_data.get("render_steps", mod.steps)

    elif mod_type == "DECIMATE":
        if "decimate_type" in mod_data:
            mod.decimate_type = mod_data["decimate_type"]
        mod.ratio = mod_data.get("ratio", 0.5)

    elif mod_type == "WIREFRAME":
        mod.thickness = mod_data.get("thickness", 0.02)


def postprocess_mesh(obj):
    """Clean up a generated mesh — remove degenerates, fix normals."""
    import bpy
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(obj.data)

    # Remove degenerate faces
    degenerate = [f for f in bm.faces if f.calc_area() < 1e-6]
    if degenerate:
        bmesh.ops.delete(bm, geom=degenerate, context="FACES")

    # Merge by distance
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)

    # Recalculate normals
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def inject_scene(scene_data: dict) -> list:
    """Inject a complete scene (multiple objects) from model output.

    Args:
        scene_data: Dict with "objects" list, each containing mesh/material/modifier data

    Returns:
        List of created bpy.types.Object
    """
    import bpy

    created = []
    for obj_data in scene_data.get("objects", []):
        try:
            obj = inject_mesh(obj_data)
            created.append(obj)
        except Exception as e:
            print(f"Warning: Failed to inject {obj_data.get('name', '?')}: {e}")

    # Update scene
    bpy.context.view_layer.update()
    return created
