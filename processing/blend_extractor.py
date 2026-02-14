"""Extract structured training data from .blend files using Blender headless.

This script runs INSIDE Blender's Python environment (via --background mode).
It reads a .blend file and extracts:
- Mesh data (vertices, faces, normals) for each object
- Material/shader node tree definitions
- Modifier stacks
- Object hierarchy and transforms
- Scene metadata

The output is a structured JSON that represents the 3D content without
any Python scripts — just pure geometry and material data.

Usage:
    blender --background --python processing/blend_extractor.py -- \
        --input data/raw/blendswap/vehicles \
        --output data/processed/vehicles
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# This script runs inside Blender — bpy is always available
try:
    import bpy
    import bmesh
    from mathutils import Vector
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("ERROR: This script must run inside Blender.")
    print("Usage: blender --background --python processing/blend_extractor.py -- --input <dir> --output <dir>")
    sys.exit(1)


def extract_mesh_data(obj, config: dict) -> dict | None:
    """Extract mesh geometry from a Blender mesh object.

    Returns normalized vertex positions, face indices, and normals.
    The mesh is:
    1. Evaluated with modifiers applied (to get final geometry)
    2. Triangulated (consistent face format for training)
    3. Normalized to [-1, 1] range centered at origin
    """
    if obj.type != "MESH":
        return None

    mesh_config = config.get("processing", {}).get("mesh_extraction", {})
    min_verts = mesh_config.get("min_vertices", 8)
    max_verts = mesh_config.get("max_vertices", 100000)
    precision = mesh_config.get("coordinate_precision", 4)
    normalize = mesh_config.get("normalize", True)

    # Get evaluated mesh (with modifiers applied)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()

    if mesh is None or len(mesh.vertices) < min_verts:
        eval_obj.to_mesh_clear()
        return None

    if len(mesh.vertices) > max_verts:
        eval_obj.to_mesh_clear()
        return None

    # Triangulate
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()

    # Extract vertices
    vertices = []
    for v in mesh.vertices:
        # Transform to world space
        world_co = obj.matrix_world @ v.co
        vertices.append([world_co.x, world_co.y, world_co.z])

    # Extract faces (triangulated — always 3 indices)
    faces = []
    for poly in mesh.polygons:
        face = list(poly.vertices)
        if len(face) == 3:
            faces.append(face)

    # Extract normals
    normals = []
    mesh.calc_normals_split()
    for v in mesh.vertices:
        normals.append([round(v.normal.x, 3),
                        round(v.normal.y, 3),
                        round(v.normal.z, 3)])

    eval_obj.to_mesh_clear()

    if not faces:
        return None

    # Normalize to [-1, 1] centered at origin
    if normalize and vertices:
        import numpy as np
        verts_np = np.array(vertices)
        center = verts_np.mean(axis=0)
        verts_np -= center
        max_extent = np.abs(verts_np).max()
        if max_extent > 0.0001:
            verts_np /= max_extent
        vertices = [[round(float(c), precision) for c in v] for v in verts_np]
    else:
        vertices = [[round(c, precision) for c in v] for v in vertices]

    return {
        "vertices": vertices,
        "faces": faces,
        "normals": normals,
        "num_vertices": len(vertices),
        "num_faces": len(faces),
    }


def extract_material_data(mat) -> dict | None:
    """Extract material properties and node tree structure.

    Returns a structured representation of the shader node graph
    that can be used to recreate the material.
    """
    if mat is None:
        return None

    result = {
        "name": mat.name,
        "use_nodes": mat.use_nodes,
    }

    if not mat.use_nodes or not mat.node_tree:
        # Simple material — extract basic properties
        result["type"] = "simple"
        return result

    # Extract node tree
    nodes = []
    links = []
    node_tree = mat.node_tree

    for node in node_tree.nodes:
        node_data = {
            "name": node.name,
            "type": node.type,
            "bl_idname": node.bl_idname,
            "location": [round(node.location.x, 1), round(node.location.y, 1)],
        }

        # Extract node-specific properties
        if node.type == "BSDF_PRINCIPLED":
            inputs = {}
            for inp in node.inputs:
                if inp.is_linked:
                    inputs[inp.name] = "LINKED"
                elif hasattr(inp, "default_value"):
                    val = inp.default_value
                    if hasattr(val, "__len__"):
                        inputs[inp.name] = [round(float(v), 4) for v in val]
                    else:
                        inputs[inp.name] = round(float(val), 4)
            node_data["inputs"] = inputs

        elif node.type in ("TEX_IMAGE", "TEX_NOISE", "TEX_BRICK",
                            "TEX_CHECKER", "TEX_GRADIENT", "TEX_WAVE"):
            node_data["texture_type"] = node.type
            # Extract relevant properties
            if hasattr(node, "noise_dimensions"):
                node_data["noise_dimensions"] = node.noise_dimensions

        elif node.type == "MIX_RGB":
            node_data["blend_type"] = node.blend_type
            node_data["use_clamp"] = node.use_clamp

        elif node.type == "MATH":
            node_data["operation"] = node.operation

        elif node.type in ("MAPPING", "TEX_COORD", "NORMAL_MAP",
                            "BUMP", "DISPLACEMENT"):
            pass  # Just capture type

        # Extract all input default values generically
        if "inputs" not in node_data:
            inputs = {}
            for inp in node.inputs:
                if not inp.is_linked and hasattr(inp, "default_value"):
                    val = inp.default_value
                    try:
                        if hasattr(val, "__len__"):
                            inputs[inp.name] = [round(float(v), 4) for v in val]
                        else:
                            inputs[inp.name] = round(float(val), 4)
                    except (TypeError, ValueError):
                        pass
            if inputs:
                node_data["inputs"] = inputs

        nodes.append(node_data)

    # Extract links
    for link in node_tree.links:
        links.append({
            "from_node": link.from_node.name,
            "from_socket": link.from_socket.name,
            "to_node": link.to_node.name,
            "to_socket": link.to_socket.name,
        })

    result["type"] = "node_tree"
    result["nodes"] = nodes
    result["links"] = links

    return result


def extract_modifier_stack(obj) -> list[dict]:
    """Extract modifier stack configuration."""
    modifiers = []
    for mod in obj.modifiers:
        mod_data = {
            "type": mod.type,
            "name": mod.name,
            "show_viewport": mod.show_viewport,
        }

        # Extract type-specific properties
        if mod.type == "SUBSURF":
            mod_data["levels"] = mod.levels
            mod_data["render_levels"] = mod.render_levels
            mod_data["subdivision_type"] = mod.subdivision_type
        elif mod.type == "MIRROR":
            mod_data["use_axis"] = [mod.use_axis[0], mod.use_axis[1], mod.use_axis[2]]
            mod_data["use_clip"] = mod.use_clip
        elif mod.type == "SOLIDIFY":
            mod_data["thickness"] = round(mod.thickness, 4)
            mod_data["offset"] = round(mod.offset, 4)
        elif mod.type == "BEVEL":
            mod_data["width"] = round(mod.width, 4)
            mod_data["segments"] = mod.segments
            mod_data["limit_method"] = mod.limit_method
        elif mod.type == "ARRAY":
            mod_data["count"] = mod.count
            mod_data["use_relative_offset"] = mod.use_relative_offset
            if mod.use_relative_offset:
                mod_data["relative_offset_displace"] = [
                    round(v, 4) for v in mod.relative_offset_displace
                ]
        elif mod.type == "BOOLEAN":
            mod_data["operation"] = mod.operation
            mod_data["solver"] = mod.solver
        elif mod.type == "SHRINKWRAP":
            mod_data["wrap_method"] = mod.wrap_method
            mod_data["offset"] = round(mod.offset, 4)
        elif mod.type == "SIMPLE_DEFORM":
            mod_data["deform_method"] = mod.deform_method
            mod_data["angle"] = round(mod.angle, 4)
            mod_data["deform_axis"] = mod.deform_axis
        elif mod.type == "SCREW":
            mod_data["angle"] = round(mod.angle, 4)
            mod_data["steps"] = mod.steps
            mod_data["render_steps"] = mod.render_steps
            mod_data["axis"] = mod.axis
        elif mod.type == "DECIMATE":
            mod_data["decimate_type"] = mod.decimate_type
            mod_data["ratio"] = round(mod.ratio, 4)
        elif mod.type == "SKIN":
            pass  # Complex — just capture type
        elif mod.type == "WIREFRAME":
            mod_data["thickness"] = round(mod.thickness, 4)

        modifiers.append(mod_data)

    return modifiers


def extract_object_data(obj, config: dict) -> dict | None:
    """Extract complete object data: mesh + materials + modifiers + transforms."""
    if obj.type not in ("MESH", "CURVE", "SURFACE"):
        return None

    result = {
        "name": obj.name,
        "type": obj.type,
        "transforms": {
            "location": [round(v, 4) for v in obj.location],
            "rotation_euler": [round(v, 4) for v in obj.rotation_euler],
            "scale": [round(v, 4) for v in obj.scale],
        },
        "dimensions": [round(v, 4) for v in obj.dimensions],
    }

    # Parent info
    if obj.parent:
        result["parent"] = obj.parent.name

    # Mesh data
    if obj.type == "MESH":
        mesh_data = extract_mesh_data(obj, config)
        if mesh_data is None:
            return None  # Skip objects that fail quality checks
        result["mesh"] = mesh_data

    # Materials
    materials = []
    for slot in obj.material_slots:
        if slot.material:
            mat_data = extract_material_data(slot.material)
            if mat_data:
                materials.append(mat_data)
    result["materials"] = materials

    # Modifiers (unapplied ones — the stack)
    result["modifiers"] = extract_modifier_stack(obj)

    # Vertex groups
    vgroups = []
    for vg in obj.vertex_groups:
        vgroups.append({"name": vg.name, "index": vg.index})
    if vgroups:
        result["vertex_groups"] = vgroups

    # Smooth shading
    if obj.type == "MESH" and obj.data:
        shade_smooth = any(p.use_smooth for p in obj.data.polygons)
        result["shade_smooth"] = shade_smooth

    return result


def extract_scene_data(config: dict) -> dict:
    """Extract all objects from the current scene."""
    scene = bpy.context.scene

    objects = []
    for obj in scene.objects:
        try:
            obj_data = extract_object_data(obj, config)
            if obj_data:
                objects.append(obj_data)
        except Exception as e:
            print(f"  Warning: Failed to extract {obj.name}: {e}")
            continue

    # Scene-level data
    scene_data = {
        "scene_name": scene.name,
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "objects": objects,
        "object_count": len(objects),
    }

    # World/environment
    if scene.world:
        world = scene.world
        scene_data["world"] = {
            "name": world.name,
            "use_nodes": world.use_nodes,
        }

    return scene_data


def process_blend_file(blend_path: str, output_dir: str,
                       config: dict) -> bool:
    """Process a single .blend file and save extracted data."""
    blend_path = Path(blend_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = blend_path.stem
    output_file = output_dir / f"{stem}.json"

    if output_file.exists():
        return True

    try:
        # Open the .blend file
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))
        print(f"Processing: {blend_path.name}")

        # Extract all data
        scene_data = extract_scene_data(config)
        scene_data["source_file"] = str(blend_path)
        scene_data["blender_version"] = list(bpy.app.version)

        if scene_data["object_count"] == 0:
            print(f"  Skipping {blend_path.name}: no valid objects")
            return False

        # Save
        with open(output_file, "w") as f:
            json.dump(scene_data, f, indent=2)

        print(f"  Extracted {scene_data['object_count']} objects → {output_file.name}")
        return True

    except Exception as e:
        print(f"  ERROR processing {blend_path.name}: {e}")
        traceback.print_exc()
        return False


def main():
    # Parse args after "--" separator (Blender passes its own args before)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Extract data from .blend files")
    parser.add_argument("--input", required=True,
                        help="Input directory with .blend files")
    parser.add_argument("--output", required=True,
                        help="Output directory for JSON files")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (default: auto-detect)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max files to process")
    args = parser.parse_args(argv)

    # Load config
    config = {}
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml"
    )
    if os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Find all .blend files
    input_dir = Path(args.input)
    blend_files = sorted(input_dir.rglob("*.blend"))
    print(f"Found {len(blend_files)} .blend files in {input_dir}")

    if args.limit:
        blend_files = blend_files[:args.limit]

    # Process each file
    success_count = 0
    for i, bf in enumerate(blend_files):
        print(f"\n[{i+1}/{len(blend_files)}]")
        if process_blend_file(bf, args.output, config):
            success_count += 1

    print(f"\nDone! Extracted {success_count}/{len(blend_files)} files.")


if __name__ == "__main__":
    main()
