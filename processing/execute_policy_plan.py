"""Execute a policy action-plan in Blender headless.

Run via:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python processing/execute_policy_plan.py -- \
    --plan data/eval/plan.json --out-dir data/eval/rollouts/run1

This script is deliberately torch-free.

Outputs:
- out_dir/scene.blend
- out_dir/mesh.obj
- out_dir/stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy  # type: ignore
import bmesh  # type: ignore
from mathutils import Vector  # type: ignore
import math


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Execute a policy plan inside Blender")
    p.add_argument("--plan", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--obj-name", type=str, default="PolicyObject")
    p.add_argument("--apply-modifiers", action="store_true", help="Apply modifiers as they are added")
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            try:
                block.remove(item)
            except Exception:
                pass


def _ensure_active_mesh_object(obj_name: str):
    obj = bpy.context.view_layer.objects.active
    if obj is not None and obj.type == "MESH":
        return obj

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.view_layer.objects.active
    if obj is None:
        raise RuntimeError("Failed to create active mesh object")
    obj.name = obj_name
    obj.data.name = obj_name + "_mesh"
    return obj


def _enter_object_mode():
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass


def _enter_edit_mode(obj):
    _enter_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_mode(type="FACE")


def _select_all_faces():
    bpy.ops.mesh.select_all(action="SELECT")


def _selected_face_count(obj) -> int:
    """Return selected face count in edit-mode.

    Blender mesh operators act on the active selection; we avoid implicit
    selection so semantics match Blender.
    """
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        return int(sum(1 for f in bm.faces if f.select))
    except Exception:
        return 0


def _select_random_face(obj, rng):
    _enter_edit_mode(obj)
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    if len(bm.faces) == 0:
        return 0
    for f in bm.faces:
        f.select = False
    idx = int(rng.randrange(0, len(bm.faces)))
    bm.faces[idx].select = True
    bmesh.update_edit_mesh(obj.data)
    return 1


def _remove_modifiers_max(obj, p: int) -> int:
    _enter_object_mode()
    mods = list(getattr(obj, "modifiers", []) or [])
    if not mods:
        return 0

    if int(p) >= 24:
        max_remove = 10_000
    else:
        max_remove = 1 + int((float(p) / 31.0) * 2.0)  # 1..3

    removed = 0
    for m in list(obj.modifiers):
        if removed >= max_remove:
            break
        try:
            obj.modifiers.remove(m)
            removed += 1
        except Exception:
            continue
    return int(removed)


def _simplify_mesh_geometry(obj, p: int) -> bool:
    _enter_edit_mode(obj)
    try:
        bpy.ops.mesh.select_all(action="SELECT")
    except Exception:
        pass

    before_v = len(obj.data.vertices)

    iters = 1 + int((float(p) / 31.0) * 3.0)  # 1..4
    try:
        bpy.ops.mesh.unsubdivide(iterations=int(iters))
    except Exception:
        pass
    if len(obj.data.vertices) < before_v:
        _enter_object_mode()
        return True

    try:
        deg = 5.0 + (float(p) / 31.0) * 30.0
        bpy.ops.mesh.dissolve_limited(angle_limit=float(math.radians(deg)))
    except Exception:
        pass
    if len(obj.data.vertices) < before_v:
        _enter_object_mode()
        return True

    try:
        dist = 1e-5 + (float(p) / 31.0) * 2e-3
        bpy.ops.mesh.merge_by_distance(distance=float(dist))
    except Exception:
        pass
    if len(obj.data.vertices) < before_v:
        _enter_object_mode()
        return True

    try:
        bpy.ops.mesh.delete_loose()
    except Exception:
        pass

    _enter_object_mode()
    return len(obj.data.vertices) < before_v


def _delete_largest_face(obj) -> bool:
    _enter_edit_mode(obj)
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    if not bm.faces:
        _enter_object_mode()
        return False

    for f in bm.faces:
        f.select = False
    face = max(bm.faces, key=lambda f: float(f.calc_area()))
    face.select = True
    bmesh.update_edit_mesh(obj.data)
    try:
        bpy.ops.mesh.delete(type="FACE")
    except Exception:
        _enter_object_mode()
        return False
    _enter_object_mode()
    return True


def _apply_modifier(obj, name: str):
    _enter_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=name)


def _export_obj(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Export only the active object; plans operate on a single active mesh.
    try:
        _enter_object_mode()
        bpy.ops.object.select_all(action="DESELECT")
        obj = bpy.context.view_layer.objects.active
        if obj is not None:
            obj.select_set(True)
    except Exception:
        pass
    # Blender 4.x has wm.obj_export; fall back for older builds.
    try:
        bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=True)
    except Exception:
        bpy.ops.export_scene.obj(filepath=str(path), use_selection=True)


def _mesh_stats(obj):
    mesh = obj.data
    v_count = len(mesh.vertices)
    e_count = len(mesh.edges)
    f_count = len(mesh.polygons)

    # bounding box extents in local space
    corners = [Vector(c) for c in obj.bound_box]
    min_v = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    max_v = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    ext = max_v - min_v

    # avg edge length
    total_len = 0.0
    for e in mesh.edges:
        v0 = mesh.vertices[e.vertices[0]].co
        v1 = mesh.vertices[e.vertices[1]].co
        total_len += (v1 - v0).length
    avg_edge = float(total_len / max(1, e_count))

    # manifold-ish flag (1 if all edges manifold)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    non_man = 0
    for e in bm.edges:
        if not e.is_manifold:
            non_man += 1
    bm.free()
    manifold_flag = 1.0 if non_man == 0 else 0.0

    # crude symmetry score from bbox extents (0..1)
    denom = max(1e-6, max(ext.x, ext.y))
    symmetry = 1.0 - float(abs(ext.x - ext.y) / denom)
    symmetry = max(0.0, min(1.0, symmetry))

    # selected faces count
    selected_faces = 0
    _enter_edit_mode(obj)
    bm2 = bmesh.from_edit_mesh(mesh)
    selected_faces = sum(1 for f in bm2.faces if f.select)
    _enter_object_mode()

    smooth_faces = 0
    tri_faces = 0
    for poly in mesh.polygons:
        if bool(getattr(poly, "use_smooth", False)):
            smooth_faces += 1
        if int(getattr(poly, "loop_total", 0)) == 3:
            tri_faces += 1
    face_denom = max(1, int(f_count))

    decimate_ratio = 1.0
    has_decimate = False
    has_triangulate = False
    has_displace = False
    for mod in getattr(obj, "modifiers", []):
        t = str(getattr(mod, "type", ""))
        if t == "DECIMATE":
            has_decimate = True
            try:
                decimate_ratio = float(getattr(mod, "ratio", decimate_ratio))
            except Exception:
                pass
        elif t == "TRIANGULATE":
            has_triangulate = True
        elif t == "DISPLACE":
            has_displace = True

    return {
        "vertex_count": int(v_count),
        "edge_count": int(e_count),
        "face_count": int(f_count),
        "bounding_box": {"x": float(ext.x), "y": float(ext.y), "z": float(ext.z)},
        "avg_edge_length": float(avg_edge),
        "manifold_flag": float(manifold_flag),
        "symmetry_score": float(symmetry),
        "selected_face_count": int(selected_faces),
        "shade_smooth_fraction": float(smooth_faces / face_denom),
        "triangulated_ratio": float(tri_faces / face_denom),
        "has_decimate_modifier": bool(has_decimate),
        "decimate_ratio": float(decimate_ratio),
        "has_triangulate_modifier": bool(has_triangulate),
        "has_displace_modifier": bool(has_displace),
    }


def execute_step(obj, step, rng, apply_modifiers: bool):
    a_type = int(step.get("action_type", 0))
    p = int(step.get("param", 0))

    # action types match policy/actions.py ActionType enum
    if a_type == 0:  # ADD_CUBE
        _enter_object_mode()
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        obj = bpy.context.view_layer.objects.active
    elif a_type == 1:  # ADD_CYLINDER
        _enter_object_mode()
        verts = 12 + int((p / 31.0) * 20)
        bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=0.5, depth=1.0, location=(0.0, 0.0, 0.0))
        obj = bpy.context.view_layer.objects.active
    elif a_type == 2:  # EXTRUDE
        _enter_edit_mode(obj)
        if _selected_face_count(obj) <= 0:
            _enter_object_mode()
            return obj
        dist = 0.05 + (p / 31.0) * 0.45
        bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0.0, 0.0, dist)})
        _enter_object_mode()
    elif a_type == 3:  # INSET
        _enter_edit_mode(obj)
        if _selected_face_count(obj) <= 0:
            _enter_object_mode()
            return obj
        amount = 0.01 + (p / 31.0) * 0.15
        bpy.ops.mesh.inset(thickness=amount, depth=0.0)
        _enter_object_mode()
    elif a_type == 4:  # BEVEL
        _enter_edit_mode(obj)
        if _selected_face_count(obj) <= 0:
            _enter_object_mode()
            return obj
        offset = 0.005 + (p / 31.0) * 0.12
        seg = 1 + int((p / 31.0) * 2)  # 1..3
        try:
            bpy.ops.mesh.bevel(offset=float(offset), segments=int(seg))
        except Exception:
            pass
        _enter_object_mode()
    elif a_type == 5:  # SCALE
        _enter_object_mode()
        s = 0.6 + (p / 31.0) * 1.0
        obj.scale = (obj.scale.x * s, obj.scale.y * s, obj.scale.z * s)
        try:
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        except Exception:
            pass
    elif a_type == 6:  # SUBDIVIDE
        _enter_edit_mode(obj)
        if _selected_face_count(obj) <= 0:
            _enter_object_mode()
            return obj
        cuts = 1 + int((p / 31.0) * 3)
        try:
            bpy.ops.mesh.subdivide(number_cuts=int(cuts))
        except Exception:
            pass
        _enter_object_mode()
    elif a_type == 7:  # DELETE_FACE
        _enter_edit_mode(obj)
        if _selected_face_count(obj) <= 0:
            _enter_object_mode()
            return obj
        try:
            bpy.ops.mesh.delete(type="FACE")
        except Exception:
            pass
        _enter_object_mode()
    elif a_type == 8:  # SELECT_RANDOM_FACE
        _select_random_face(obj, rng)
        _enter_object_mode()
    elif a_type == 9:  # MIRROR
        _enter_object_mode()
        mod = obj.modifiers.new(name="Mirror", type="MIRROR")
        mod.use_axis = (p % 3 == 0, p % 3 == 1, p % 3 == 2)
        mod.use_bisect_axis = (True, True, True)
        if apply_modifiers:
            _apply_modifier(obj, mod.name)
    elif a_type == 10:  # APPLY_MODIFIER
        _enter_object_mode()
        # deterministic choice from param
        choices = ["SUBSURF", "BEVEL", "SOLIDIFY"]
        kind = choices[p % len(choices)]
        if kind == "SUBSURF":
            mod = obj.modifiers.new(name="Subsurf", type="SUBSURF")
            mod.levels = 1 + (p % 3)
        elif kind == "BEVEL":
            mod = obj.modifiers.new(name="Bevel", type="BEVEL")
            mod.width = 0.01 + (p / 31.0) * 0.1
            mod.segments = 2
        else:
            mod = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
            mod.thickness = 0.02 + (p / 31.0) * 0.2
        if apply_modifiers:
            _apply_modifier(obj, mod.name)

    elif a_type == 11:  # NOOP
        # Intentionally do nothing.
        pass

    return obj


def main():
    args = _parse_args()
    plan_path = Path(args.plan)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = json.loads(plan_path.read_text())
    seed = int(plan.get("seed", 0))
    steps = plan.get("steps", [])

    clear_scene()
    obj = _ensure_active_mesh_object(args.obj_name)

    rng = __import__("random").Random(seed)

    for step in steps:
        obj = execute_step(obj, step, rng, apply_modifiers=bool(args.apply_modifiers))

    # Save outputs
    blend_path = out_dir / "scene.blend"
    obj_path = out_dir / "mesh.obj"
    stats_path = out_dir / "stats.json"

    _enter_object_mode()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    _export_obj(obj_path)

    stats = _mesh_stats(obj)
    stats_path.write_text(json.dumps(stats, indent=2))

    print("OK: executed plan")
    print(json.dumps({"out_dir": str(out_dir), "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
