"""Local model client for the Blender Copilot addon.

This module replaces the OpenAI API calls in ai_engine.py with
requests to the local inference server (inference/server.py).

Instead of:
    User prompt → OpenAI API → Python script → execute → geometry

It does:
    User prompt → Local Model Server → mesh/material data → inject into Blender

The addon drops this file into its directory and imports it when
the user selects "Local Model" as the AI backend in preferences.

Usage in operators.py:
    from . import local_client
    result = local_client.generate(prompt, server_url="http://127.0.0.1:8420")
    local_client.inject_result(result)
"""

import json
import logging
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# Default server URL
DEFAULT_SERVER = "http://127.0.0.1:8420"


def check_server(server_url: str = DEFAULT_SERVER, timeout: float = 3.0) -> bool:
    """Check if the local inference server is running.

    Returns True if the server responds to /health.
    """
    try:
        req = urllib.request.Request(f"{server_url}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except (urllib.error.URLError, TimeoutError, Exception):
        return False


def generate_mesh(prompt: str, server_url: str = DEFAULT_SERVER,
                  temperature: float = 0.8, top_k: int = 50,
                  max_faces: int = 2048, timeout: float = 120.0) -> dict:
    """Request mesh generation from the local server.

    Args:
        prompt: Text description of the desired 3D model.
        server_url: URL of the inference server.
        temperature: Sampling temperature (higher = more creative).
        top_k: Top-k sampling parameter.
        max_faces: Maximum number of faces in generated mesh.
        timeout: Request timeout in seconds.

    Returns:
        Dict with 'objects' list, each containing mesh, materials, modifiers.
        Returns {'error': '...'} on failure.
    """
    payload = json.dumps({
        "prompt": prompt,
        "temperature": temperature,
        "top_k": top_k,
        "max_faces": max_faces,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{server_url}/generate/mesh",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": f"Server error {e.code}: {body}"}
    except urllib.error.URLError as e:
        return {"error": f"Connection error: {e.reason}. Is the server running?"}
    except TimeoutError:
        return {"error": f"Request timed out after {timeout}s"}
    except Exception as e:
        return {"error": str(e)}


def generate_material(prompt: str, server_url: str = DEFAULT_SERVER,
                      timeout: float = 30.0) -> dict:
    """Request material generation from the local server."""
    payload = json.dumps({"prompt": prompt}).encode("utf-8")

    req = urllib.request.Request(
        f"{server_url}/generate/material",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def generate_modifiers(prompt: str, server_url: str = DEFAULT_SERVER,
                       timeout: float = 30.0) -> dict:
    """Request modifier stack prediction from the local server."""
    payload = json.dumps({"prompt": prompt}).encode("utf-8")

    req = urllib.request.Request(
        f"{server_url}/generate/modifiers",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def inject_result_into_blender(result: dict) -> str:
    """Inject generation result directly into the Blender scene.

    This runs ON THE MAIN THREAD via bpy.
    Call this from within _run_on_main_thread() in operators.py.

    Args:
        result: Dict from generate_mesh() with 'objects' list.

    Returns:
        Summary string of what was created.
    """
    try:
        import bpy
    except ImportError:
        return "Error: bpy not available (not running inside Blender)"

    if "error" in result:
        return f"Error: {result['error']}"

    objects = result.get("objects", [])
    if not objects:
        return "No objects generated"

    created = []
    collection_name = "Generated"

    # Create or get collection
    if collection_name not in bpy.data.collections:
        col = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(col)
    else:
        col = bpy.data.collections[collection_name]

    for obj_data in objects:
        name = obj_data.get("name", "Generated")
        mesh_data = obj_data.get("mesh", {})

        # Create mesh
        vertices = mesh_data.get("vertices", [])
        faces = mesh_data.get("faces", [])

        if not vertices or not faces:
            continue

        # Convert to tuples
        verts = [tuple(v) for v in vertices]
        face_tuples = [tuple(f) for f in faces]

        # Create mesh data
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts, [], face_tuples)
        mesh.update()

        # Validate
        mesh.validate(verbose=False)

        # Create object
        obj = bpy.data.objects.new(name, mesh)
        col.objects.link(obj)

        # Apply transforms
        transforms = obj_data.get("transforms", {})
        if transforms:
            loc = transforms.get("location", [0, 0, 0])
            rot = transforms.get("rotation_euler", [0, 0, 0])
            scl = transforms.get("scale", [1, 1, 1])
            obj.location = tuple(loc)
            obj.rotation_euler = tuple(rot)
            obj.scale = tuple(scl)

        # Smooth shading
        if obj_data.get("shade_smooth", False):
            for poly in mesh.polygons:
                poly.use_smooth = True

        # Materials
        for mat_data in obj_data.get("materials", []):
            mat = _create_material(mat_data)
            if mat:
                obj.data.materials.append(mat)

        # Modifiers
        for mod_data in obj_data.get("modifiers", []):
            _apply_modifier(obj, mod_data)

        # Post-processing
        _postprocess(mesh)

        created.append(f"{name} ({len(verts)} verts, {len(face_tuples)} faces)")

    # Select last created
    if created:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    gen_time = result.get("generation_time", 0)
    summary = f"Created {len(created)} object(s) in {gen_time}s:\n"
    summary += "\n".join(f"  • {c}" for c in created)
    return summary


def _create_material(mat_data: dict):
    """Create a Blender material from structured data."""
    try:
        import bpy

        name = mat_data.get("name", "Material")
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        tree = mat.node_tree
        nodes = tree.nodes
        links = tree.links

        # Clear default nodes
        nodes.clear()

        # Add output + principled BSDF as minimum
        output = nodes.new("ShaderNodeOutputMaterial")
        output.location = (300, 0)
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (0, 0)
        links.new(bsdf.outputs[0], output.inputs[0])

        # Set PBR values if provided
        pbr = mat_data.get("pbr", {})
        if pbr:
            base_color = pbr.get("base_color", [0.8, 0.8, 0.8, 1.0])
            if len(base_color) == 3:
                base_color = list(base_color) + [1.0]
            bsdf.inputs["Base Color"].default_value = base_color
            if "metallic" in pbr:
                bsdf.inputs["Metallic"].default_value = pbr["metallic"]
            if "roughness" in pbr:
                bsdf.inputs["Roughness"].default_value = pbr["roughness"]

        return mat
    except Exception as e:
        logger.error(f"Material creation failed: {e}")
        return None


def _apply_modifier(obj, mod_data: dict):
    """Apply a modifier to a Blender object from structured data."""
    try:
        import bpy

        mod_type = mod_data.get("type", "")
        if not mod_type:
            return

        params = mod_data.get("params", {})
        mod = obj.modifiers.new(name=mod_type, type=mod_type)

        for key, value in params.items():
            if hasattr(mod, key):
                try:
                    setattr(mod, key, value)
                except (TypeError, AttributeError):
                    pass
    except Exception as e:
        logger.error(f"Modifier application failed: {e}")


def _postprocess(mesh):
    """Clean up generated mesh."""
    try:
        import bmesh

        bm = bmesh.new()
        bm.from_mesh(mesh)

        # Remove loose vertices
        loose = [v for v in bm.verts if not v.link_faces]
        for v in loose:
            bm.verts.remove(v)

        # Merge by distance
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)

        # Recalculate normals
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
    except Exception:
        pass
