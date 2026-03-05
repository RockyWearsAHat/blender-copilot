"""
Tool Definitions -- Slim hybrid tool-calling for Blender Copilot.

9 tools: generate_mesh (primary), execute_code (secondary),
inspect_scene, inspect_object, get_object_bounds, inspect_timeline,
inspect_animation, capture_viewport, declare_complete.
"""

import json
import logging
import traceback
import urllib.request
import ssl

logger = logging.getLogger(__name__)


def _obj_info(obj):
    if obj is None:
        return {"error": "object is None"}
    info = {
        "name": obj.name,
        "type": obj.type,
        "location": [round(v, 4) for v in obj.location],
        "dimensions": [round(v, 4) for v in obj.dimensions],
    }
    if obj.type == "MESH" and obj.data:
        info["vertices"] = len(obj.data.vertices)
        info["faces"] = len(obj.data.polygons)
    if obj.active_material:
        info["material"] = obj.active_material.name
    return info


def _bounds_info(bounds_obj):
    if bounds_obj is None:
        return {"error": "no bounds"}
    return {
        "min": [round(bounds_obj.min_x, 4), round(bounds_obj.min_y, 4),
                round(bounds_obj.min_z, 4)],
        "max": [round(bounds_obj.max_x, 4), round(bounds_obj.max_y, 4),
                round(bounds_obj.max_z, 4)],
        "center": [round(bounds_obj.center_x, 4), round(bounds_obj.center_y, 4),
                   round(bounds_obj.center_z, 4)],
        "width_x": round(bounds_obj.width, 4),
        "depth_y": round(bounds_obj.depth, 4),
        "height_z": round(bounds_obj.height, 4),
    }


def execute_tool(func_name, args):
    from . import blender_tools as bt
    try:
        return _dispatch_tool(bt, func_name, args)
    except Exception as e:
        tb = traceback.format_exc()
        return {"error": str(e), "traceback": tb[:800]}


def _dispatch_tool(bt, name, args):
    if name == "execute_code":
        return _exec_code(bt, args.get("code", ""))
    if name == "generate_mesh":
        return _generate_mesh_tool(bt, args)
    if name == "generate_mesh_from_image":
        return _generate_mesh_from_image_tool(bt, args)
    if name == "inspect_scene":
        return _inspect_scene()
    if name == "inspect_object":
        return _inspect_object(bt, args["name"])
    if name == "get_object_bounds":
        return _get_object_bounds(bt, args["name"])
    if name == "inspect_timeline":
        return bt.inspect_timeline()
    if name == "inspect_animation":
        obj = bt.get(args["name"])
        if obj is None:
            return {"error": "Object not found: %s" % args["name"]}
        return bt.inspect_animation(obj)
    if name == "capture_viewport":
        return _capture_viewport(bt)
    if name == "declare_complete":
        return {"status": "COMPLETE", "summary": args.get("summary", "")}
    return {"error": "Unknown tool: %s" % name}


# ── Mesh server URL (set by the tool loop before execution) ─────────
_mesh_server_url = ""


def set_mesh_server_url(url):
    global _mesh_server_url
    _mesh_server_url = url


def _generate_mesh_tool(bt, args):
    """Call the trained mesh model to generate geometry, then create it in Blender.
    NOTE: When called via execute_tool from main thread, this works but blocks UI.
    Prefer using fetch_mesh_from_server + create_mesh_object split from ai_engine.
    """
    mesh_data = fetch_mesh_from_server(args)
    if "error" in mesh_data:
        return mesh_data
    return create_mesh_object(args, mesh_data)


def fetch_mesh_from_server(args):
    """HTTP-only: fetch mesh data from the trained model server.
    Safe to call from a background thread (no bpy calls).
    Returns the raw mesh_data dict or {"error": ...}.
    """
    prompt = args.get("prompt", "")
    temperature = args.get("temperature", 0.6)
    max_faces = args.get("max_faces", 128)
    top_k = args.get("top_k", 40)
    top_p = args.get("top_p", 0.95)
    cfg_scale = args.get("cfg_scale", 2.0)
    reference_image = args.get("reference_image")

    if not _mesh_server_url:
        return {"error": "Mesh server not running. Start with: python run.py serve"}

    if not prompt and not reference_image:
        return {"error": "prompt is required"}

    ptxt = str(prompt).lower()
    if "low poly" in ptxt:
        max_faces = min(int(max_faces), 96)

    try:
        payload_obj = {
            "prompt": prompt,
            "temperature": temperature,
            "max_faces": max_faces,
            "top_k": top_k,
            "top_p": top_p,
            "cfg_scale": cfg_scale,
        }
        endpoint = "%s/generate/mesh" % _mesh_server_url
        if reference_image:
            payload_obj["reference_image"] = reference_image
        payload = json.dumps(payload_obj).encode("utf-8")

        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            mesh_data = json.loads(resp.read().decode("utf-8"))

    except urllib.error.URLError as e:
        return {"error": "Cannot connect to mesh server at %s: %s" % (_mesh_server_url, str(e))}
    except Exception as e:
        return {"error": "Mesh server request failed: %s" % str(e)}

    if "error" in mesh_data:
        return {"error": "Mesh server: %s" % mesh_data["error"]}

    if not mesh_data.get("objects") or not mesh_data["objects"][0].get("mesh"):
        return {"error": "Mesh server returned empty result"}

    obj_data = mesh_data["objects"][0]
    mesh_info = obj_data["mesh"]
    vertices = mesh_info.get("vertices", [])
    faces = mesh_info.get("faces", [])

    if not vertices or not faces:
        return {"error": "Generated mesh has no geometry (%d verts, %d faces)" % (
            len(vertices), len(faces))}

    return mesh_data


def create_mesh_object(args, mesh_data):
    """Blender-only: create a mesh object from server response data.
    MUST run on the main thread (uses bpy/bmesh).
    """
    import bpy
    import bmesh
    from . import blender_tools

    name = args.get("name", "")
    prompt = args.get("prompt", "")
    location = args.get("location", [0, 0, 0])
    scale = args.get("scale", 1.0)

    if not name:
        name = prompt.replace(" ", "_")[:30] if prompt else "GeneratedMesh"

    obj_data = mesh_data["objects"][0]
    mesh_info = obj_data["mesh"]
    vertices = mesh_info["vertices"]
    faces = mesh_info["faces"]

    mesh = bpy.data.meshes.new(name=name)
    obj = bpy.data.objects.new(name=name, object_data=mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    try:
        bm_verts = []
        for v in vertices:
            bm_verts.append(bm.verts.new((
                v[0] * scale, v[1] * scale, v[2] * scale)))
        bm.verts.ensure_lookup_table()

        seen_faces = set()
        created_faces = 0
        for face in faces:
            face_key = tuple(sorted(face))
            if face_key in seen_faces:
                continue
            seen_faces.add(face_key)
            if any(vi >= len(bm_verts) for vi in face):
                continue
            try:
                bm.faces.new([bm_verts[vi] for vi in face])
                created_faces += 1
            except ValueError:
                continue

        bm.to_mesh(mesh)
    finally:
        bm.free()

    mesh.update()

    for face in mesh.polygons:
        face.use_smooth = True

    obj.location = (location[0], location[1], location[2])

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
    except Exception:
        pass

    # Smart post-process: apply a lightweight modifier preset by inferred type.
    try:
        p = prompt.lower()
        inferred = "default"
        if any(k in p for k in ("character", "creature", "animal", "organic", "head")):
            inferred = "organic"
        elif any(k in p for k in ("car", "vehicle", "robot", "engine", "weapon", "hard-surface")):
            inferred = "hard_surface"
        elif any(k in p for k in ("building", "house", "architecture", "room", "wall")):
            inferred = "architectural"
        mods = blender_tools.apply_suggested_modifiers(obj, inferred)
    except Exception:
        mods = []

    gen_time = mesh_data.get("generation_time", 0)
    token_count = mesh_data.get("token_count", 0)

    logger.info("generate_mesh: created '%s' with %d verts, %d faces (%.1fs, %d tokens)",
                name, len(vertices), created_faces, gen_time, token_count)

    return {
        "status": "created",
        "name": obj.name,
        "vertices": len(vertices),
        "faces": created_faces,
        "generation_time": gen_time,
        "suggested_modifiers": mods,
        "location": list(obj.location),
        "dimensions": [round(d, 4) for d in obj.dimensions],
    }


def _generate_mesh_from_image_tool(bt, args):
    """Capture the current viewport and send it to the mesh server for image-to-3D."""
    import tempfile
    import base64
    import os

    # Step 1: capture viewport
    try:
        path = bt.capture_viewport(
            filepath=os.path.join(
                tempfile.gettempdir(), "copilot_i2m_capture.png"),
            width=256, height=256)
        if not path or not os.path.exists(path):
            return {"error": "Viewport capture failed — no image produced"}
        with open(path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        return {"error": "Viewport capture failed: %s" % str(e)}

    # Step 2: call server
    mesh_data = fetch_mesh_from_image_server({
        "image": image_b64,
        "prompt": args.get("prompt", ""),
        "temperature": args.get("temperature", 0.6),
        "max_faces": args.get("max_faces", 2048),
        "cfg_scale": args.get("cfg_scale", 2.0),
    })
    if "error" in mesh_data:
        return mesh_data

    # Step 3: create mesh in Blender
    return create_mesh_object(args, mesh_data)


def fetch_mesh_from_image_server(args):
    """HTTP call to /generate/mesh-from-image. Safe for background threads."""
    if not _mesh_server_url:
        return {"error": "Mesh server not running. Start with: python run.py serve"}

    try:
        payload = json.dumps({
            "image": args.get("image", ""),
            "prompt": args.get("prompt", ""),
            "temperature": args.get("temperature", 0.6),
            "max_faces": args.get("max_faces", 2048),
            "cfg_scale": args.get("cfg_scale", 2.0),
        }).encode("utf-8")

        req = urllib.request.Request(
            "%s/generate/mesh-from-image" % _mesh_server_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            mesh_data = json.loads(resp.read().decode("utf-8"))

    except urllib.error.URLError as e:
        return {"error": "Cannot connect to mesh server at %s: %s" % (
            _mesh_server_url, str(e))}
    except Exception as e:
        return {"error": "Image-to-mesh request failed: %s" % str(e)}

    if "error" in mesh_data:
        return {"error": "Mesh server: %s" % mesh_data["error"]}

    if not mesh_data.get("objects") or not mesh_data["objects"][0].get("mesh"):
        return {"error": "Mesh server returned empty result for image"}

    obj_data = mesh_data["objects"][0]
    mesh_info = obj_data["mesh"]
    if not mesh_info.get("vertices") or not mesh_info.get("faces"):
        return {"error": "Generated image-to-mesh has no geometry"}

    return mesh_data


def _exec_code(bt, code):
    import bpy  # type: ignore
    import math
    import bmesh  # type: ignore
    from mathutils import Vector, Matrix, Euler  # type: ignore

    namespace = {
        "__builtins__": __builtins__,
        "bpy": bpy,
        "bmesh": bmesh,
        "math": math,
        "Vector": Vector,
        "Matrix": Matrix,
        "Euler": Euler,
    }
    for attr in dir(bt):
        if not attr.startswith("_"):
            namespace[attr] = getattr(bt, attr)

    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

    exec(code, namespace)
    return _inspect_scene()


def _inspect_scene():
    import bpy  # type: ignore
    objs = []
    for obj in bpy.data.objects:
        objs.append(_obj_info(obj))
    mats = [mat.name for mat in bpy.data.materials if mat.name != "Dots Stroke"]
    return {"objects": objs, "materials": mats, "count": len(objs)}


def _inspect_object(bt, name):
    obj = bt.get(name)
    if obj is None:
        return {"error": "Object not found: %s" % name}
    info = _obj_info(obj)
    bounds = bt.get_bounds(obj)
    if bounds:
        info["bounds"] = _bounds_info(bounds)
    return info


def _get_object_bounds(bt, name):
    obj = bt.get(name)
    if obj is None:
        return {"error": "Object not found: %s" % name}
    bounds = bt.get_bounds(obj)
    if bounds is None:
        return {"error": "Cannot compute bounds for %s" % name}
    info = _bounds_info(bounds)
    info["name"] = name
    info["note"] = ("width_x = X extent (for a car: LENGTH), "
                    "depth_y = Y extent (for a car: LATERAL WIDTH), "
                    "height_z = Z extent (HEIGHT)")
    return info


def _capture_viewport(bt):
    import tempfile
    import base64
    import os
    try:
        path = bt.capture_viewport(
            filepath=os.path.join(tempfile.gettempdir(), "copilot_viewport.png"),
            width=1024, height=768)
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return {"image": "data:image/png;base64," + b64}
    except Exception as e:
        return {"error": "Viewport capture failed: %s" % str(e)}
    return {"error": "No viewport image available"}


def get_tool_definitions():
    return _TOOL_DEFS


_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "generate_mesh_from_image",
            "description": "Capture the current 3D viewport and reconstruct a 3D mesh from what you see. The model uses its image encoder to condition on the rendered view and generate matching geometry. Useful when the user shows a reference object in the scene or describes something already visible. Combine with an optional prompt to guide the generation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Optional text hint describing the object in the viewport (e.g. 'a wooden chair'). Helps the model focus on the key semantic. Omit to use pure image conditioning.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Name for the created Blender object. Auto-generated from prompt if omitted.",
                    },
                    "location": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Position [x, y, z] in meters. Default [0,0,0].",
                    },
                    "max_faces": {
                        "type": "integer",
                        "description": "Maximum faces to generate. Default 128 for faster, more stable outputs.",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature. Default 0.6. Use lower (0.3-0.5) for faithful reconstruction, higher (0.7-0.9) for creative interpretation.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_mesh",
            "description": "PRIMARY TOOL — Generate a 3D mesh using the trained AI model. Describe what you want and the model generates real geometry (vertices, faces). Use this for ALL object creation. Returns the created object name, vertex/face count, and dimensions. You can then use execute_code to add materials, position, and modify the generated mesh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Description of the 3D object to generate. Be specific: 'a wooden chair', 'a sports car', 'a coffee mug', 'a medieval sword'. Simple nouns work best.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Name for the Blender object (e.g. 'Chair', 'CarBody'). Auto-generated from prompt if omitted.",
                    },
                    "location": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Position [x, y, z] in meters. Default [0,0,0].",
                    },
                    "scale": {
                        "type": "number",
                        "description": "Scale multiplier. Default 1.0.",
                    },
                    "max_faces": {
                        "type": "integer",
                        "description": "Maximum faces to generate. Default 128 for stability and speed. Use 300-800 only when you need more detail.",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature. Default 0.6. Use 0.3-0.5 for precise geometry, 0.7-0.9 for creative/varied shapes.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Top-k sampling. Default 40. Higher can increase variety.",
                    },
                    "top_p": {
                        "type": "number",
                        "description": "Nucleus sampling. Default 0.95.",
                    },
                    "reference_image": {
                        "type": "string",
                        "description": "Optional base64 image (or data URL) for image-conditioned generation. When provided, server routes to image-to-mesh conditioning while still applying prompt guidance.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in Blender. Use for: adding materials, positioning objects, modifiers (subsurf, mirror, bevel), shaping (scale_section, taper), boolean operations, and any post-processing of generated meshes. All blender_tools functions pre-imported. Do NOT use this to create objects from scratch — use generate_mesh instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code. All blender_tools functions available directly.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_scene",
            "description": "Get overview of all objects and materials in the scene.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_object",
            "description": "Get detailed info about one object: dimensions, bounds, location, material.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Object name"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_object_bounds",
            "description": "Get bounding box: min/max coords, center, extents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Object name"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_timeline",
            "description": "Get timeline overview: fps, frame range, duration, and per-object keyframe summary with timestamps. Shows each animated property as: Object 'name': property  0.0s [values] → 1.2s [values] → ... Perfect for understanding the current animation state before modifying it.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_animation",
            "description": "Get detailed animation data for one object: all FCurves, keyframes with timestamps, interpolation, handles, NLA tracks, and drivers. Use after inspect_timeline to drill into a specific object's animation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Object name"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_viewport",
            "description": "Take a screenshot of the 3D viewport for visual verification.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "declare_complete",
            "description": "Declare the model COMPLETE. Call only after all objects have materials and look correct.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Summary of the completed model"},
                },
                "required": ["summary"],
            },
        },
    },
]
