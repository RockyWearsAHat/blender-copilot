"""
Tool Definitions -- Slim hybrid tool-calling for Blender Copilot.

Only 6 tools: execute_code (primary), inspect_scene, inspect_object,
get_object_bounds, capture_viewport, declare_complete.
"""

import traceback


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
    if name == "inspect_scene":
        return _inspect_scene()
    if name == "inspect_object":
        return _inspect_object(bt, args["name"])
    if name == "get_object_bounds":
        return _get_object_bounds(bt, args["name"])
    if name == "capture_viewport":
        return _capture_viewport(bt)
    if name == "declare_complete":
        return {"status": "COMPLETE", "summary": args.get("summary", "")}
    return {"error": "Unknown tool: %s" % name}


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
            "name": "execute_code",
            "description": "Execute Python code in Blender. PRIMARY tool for ALL creation, modification, material assignment. All blender_tools functions pre-imported (call directly). bpy, bmesh, math, Vector, Matrix, Euler available. Write COMPLETE bulk code blocks. Returns scene state after execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code. All blender_tools functions available directly. Write bulk operations.",
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
            "description": "Get detailed info about one object: dimensions, bounds, location, material. AXIS MAP: width_x=X (car LENGTH), depth_y=Y (car LATERAL WIDTH), height_z=Z (HEIGHT).",
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
            "description": "Get bounding box: min/max coords, center, extents. AXIS MAP: width_x=X (car LENGTH), depth_y=Y (car LATERAL WIDTH), height_z=Z. Use depth_y/2 for left/right, NOT width_x/2.",
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
            "description": "Take a screenshot of the 3D viewport. Returns base64 image. Use to visually verify your work after major changes.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "declare_complete",
            "description": "Declare the model COMPLETE. Call when model has all key parts, materials, and is recognizable.",
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
