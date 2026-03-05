"""Long-running Blender worker: apply actions step-by-step and emit stats.

This keeps Blender alive (fast) and avoids importing torch in Blender.
A controller process writes action_i.json; this worker writes state_i.json.

Run:
  Blender --background --python processing/blender_policy_worker.py -- \
    --work-dir data/eval/rollouts/closed_loop_run \
    --steps 64

Files:
  work-dir/ready.json
  work-dir/action_0000.json  -> worker applies
  work-dir/state_0000.json   <- worker writes after step
  ...
  work-dir/scene.blend, mesh.obj, stats_final.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy  # type: ignore
import bmesh  # type: ignore
from mathutils import Vector  # type: ignore
import math


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Atomic JSON write to avoid readers seeing partial files."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Blender policy worker")
    p.add_argument("--work-dir", type=str, required=True)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--obj-name", type=str, default="PolicyObject")
    p.add_argument("--apply-modifiers", action="store_true")
    p.add_argument("--poll-ms", type=int, default=25)
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.add_argument("--max-verts", type=int, default=250_000)
    p.add_argument("--max-faces", type=int, default=250_000)
    p.add_argument(
        "--unsafe-no-geometry-guards",
        action="store_true",
        help="Disable geometry-safety guards (may freeze/crash Blender).",
    )
    return p.parse_args(argv)


def _too_big(stats: dict, *, max_verts: int, max_faces: int) -> bool:
    try:
        v = int(stats.get("vertex_count", 0))
        f = int(stats.get("face_count", 0))
    except Exception:
        return False
    return (max_verts > 0 and v >= int(max_verts)) or (max_faces > 0 and f >= int(max_faces))


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            try:
                block.remove(item)
            except Exception:
                pass


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


def _replace_scene_with_primitive():
    """Delete ALL existing mesh objects so the new primitive becomes the sole mesh.

    The sim env treats ADD_CUBE / ADD_CYLINDER as *replacing* the active mesh
    (it resets vertex/face counts to the new primitive), not as *appending* an
    extra object.  Without this helper the worker would stack N cylinders on
    top of each other, which the policy was trained never to expect.
    """
    _enter_object_mode()
    meshes_to_remove = [
        obj for obj in bpy.data.objects if obj.type == "MESH"
    ]
    for obj in meshes_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)


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


def _select_all_faces():
    bpy.ops.mesh.select_all(action="SELECT")


def _selected_face_count(obj) -> int:
    """Return selected face count in edit-mode.

    Blender's mesh edit operators act on the current selection.
    We avoid any implicit auto-selection here so action semantics
    remain Blender-faithful.
    """
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        return int(sum(1 for f in bm.faces if f.select))
    except Exception:
        return 0


def _selected_face_count_any_mode(obj) -> int:
    """Return selected face count regardless of current mode.

    Many call sites log selection while in Object Mode, but Blender only exposes
    face selection via bmesh in Edit Mode. We temporarily enter Edit Mode to
    read selection, then restore Object Mode.
    """
    try:
        # If we're already in edit mode for this object, avoid mode toggles.
        if getattr(obj, "mode", None) == "EDIT":
            return int(_selected_face_count(obj))
    except Exception:
        pass

    try:
        _enter_edit_mode(obj)
        return int(_selected_face_count(obj))
    except Exception:
        return 0
    finally:
        try:
            _enter_object_mode()
        except Exception:
            pass


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
    """Remove non-applied modifiers from the stack (object-mode).

    This is a high-leverage simplification step and is reversible by
    re-adding stored modifier settings during dataset generation.
    """
    _enter_object_mode()
    mods = list(getattr(obj, "modifiers", []) or [])
    if not mods:
        return 0

    # Param controls aggressiveness: low p removes 1, high p removes all.
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
    """Try to remove lots of geometry deterministically in one step."""
    _enter_edit_mode(obj)
    try:
        bpy.ops.mesh.select_all(action="SELECT")
    except Exception:
        pass

    before_v = len(obj.data.vertices)

    # 1) Unsubdivide (best loop-cut reversal analogue)
    iters = 1 + int((float(p) / 31.0) * 3.0)  # 1..4
    try:
        bpy.ops.mesh.unsubdivide(iterations=int(iters))
    except Exception:
        pass
    if len(obj.data.vertices) < before_v:
        _enter_object_mode()
        return True

    # 2) Limited dissolve (remove coplanar-ish detail)
    try:
        # Angle limit: 5..35 degrees
        deg = 5.0 + (float(p) / 31.0) * 30.0
        bpy.ops.mesh.dissolve_limited(angle_limit=float(math.radians(deg)))
    except Exception:
        pass
    if len(obj.data.vertices) < before_v:
        _enter_object_mode()
        return True

    # 3) Merge by distance (collapse near-duplicate verts)
    try:
        dist = 1e-5 + (float(p) / 31.0) * 2e-3
        bpy.ops.mesh.merge_by_distance(distance=float(dist))
    except Exception:
        pass
    if len(obj.data.vertices) < before_v:
        _enter_object_mode()
        return True

    # 4) Delete loose (orphaned edges/verts)
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
    # Delete the face with largest area (deterministic, high impact)
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
    # Export only the active object; the policy controls a single active mesh.
    # This avoids accidentally exporting multiple primitives if actions like
    # ADD_CYLINDER were used.
    try:
        _enter_object_mode()
        bpy.ops.object.select_all(action="DESELECT")
        obj = bpy.context.view_layer.objects.active
        if obj is not None:
            obj.select_set(True)
    except Exception:
        pass
    try:
        bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=True)
    except Exception:
        bpy.ops.export_scene.obj(filepath=str(path), use_selection=True)


def _mesh_stats(obj):
    mesh = obj.data
    v_count = len(mesh.vertices)
    e_count = len(mesh.edges)
    f_count = len(mesh.polygons)

    corners = [Vector(c) for c in obj.bound_box]
    min_v = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    max_v = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    ext = max_v - min_v

    total_len = 0.0
    for e in mesh.edges:
        v0 = mesh.vertices[e.vertices[0]].co
        v1 = mesh.vertices[e.vertices[1]].co
        total_len += (v1 - v0).length
    avg_edge = float(total_len / max(1, e_count))

    bm = bmesh.new()
    bm.from_mesh(mesh)
    non_man = 0
    for e in bm.edges:
        if not e.is_manifold:
            non_man += 1
    bm.free()
    manifold_flag = 1.0 if non_man == 0 else 0.0

    denom = max(1e-6, max(ext.x, ext.y))
    symmetry = 1.0 - float(abs(ext.x - ext.y) / denom)
    symmetry = max(0.0, min(1.0, symmetry))

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
    displace_strength = 0.0
    displace_midlevel = 0.5
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
            try:
                displace_strength = float(getattr(mod, "strength", displace_strength))
            except Exception:
                pass
            try:
                displace_midlevel = float(getattr(mod, "mid_level", displace_midlevel))
            except Exception:
                pass

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
        "displace_strength": float(displace_strength),
        "displace_midlevel": float(displace_midlevel),
    }


def _modifier_snapshot(obj) -> list[dict]:
    """Return a compact, JSON-safe snapshot of the modifier stack."""
    out: list[dict] = []
    try:
        mods = list(getattr(obj, "modifiers", []))
    except Exception:
        mods = []

    for m in mods:
        try:
            entry = {
                "name": str(getattr(m, "name", "")),
                "type": str(getattr(m, "type", "")),
                "show_viewport": bool(getattr(m, "show_viewport", True)),
                "show_render": bool(getattr(m, "show_render", True)),
            }
        except Exception:
            continue

        t = entry.get("type")
        try:
            if t == "SUBSURF":
                entry.update({
                    "levels": int(getattr(m, "levels", 0)),
                    "render_levels": int(getattr(m, "render_levels", 0)),
                })
            elif t == "BEVEL":
                entry.update({
                    "width": float(getattr(m, "width", 0.0)),
                    "segments": int(getattr(m, "segments", 0)),
                    "limit_method": str(getattr(m, "limit_method", "")),
                })
            elif t == "SOLIDIFY":
                entry.update({
                    "thickness": float(getattr(m, "thickness", 0.0)),
                    "offset": float(getattr(m, "offset", 0.0)),
                })
            elif t == "MIRROR":
                entry.update({
                    "use_axis": list(getattr(m, "use_axis", (False, False, False))),
                    "use_bisect_axis": list(getattr(m, "use_bisect_axis", (False, False, False))),
                })
            elif t == "DISPLACE":
                entry.update({
                    "strength": float(getattr(m, "strength", 0.0)),
                    "mid_level": float(getattr(m, "mid_level", 0.5)),
                    "texture": str(getattr(getattr(m, "texture", None), "name", "")),
                })
            elif t == "DECIMATE":
                entry.update({
                    "ratio": float(getattr(m, "ratio", 1.0)),
                    "decimate_type": str(getattr(m, "decimate_type", "")),
                })
            elif t == "TRIANGULATE":
                entry.update({
                    "quad_method": str(getattr(m, "quad_method", "")),
                    "ngon_method": str(getattr(m, "ngon_method", "")),
                })
        except Exception:
            # Snapshot should never break execution.
            pass

        out.append(entry)
    return out


def _action_name(a_type: int) -> str:
    names = {
        0: "ADD_CUBE",
        1: "ADD_CYLINDER",
        2: "EXTRUDE",
        3: "INSET",
        4: "BEVEL",
        5: "SCALE",
        6: "SUBDIVIDE",
        7: "DELETE_FACE",
        8: "SELECT_RANDOM_FACE",
        9: "MIRROR",
        10: "APPLY_MODIFIER",
        11: "NOOP",
    }
    return names.get(int(a_type), f"UNKNOWN_{int(a_type)}")


def _execute_step(obj, action, rng, apply_modifiers: bool, *, geometry_guards: bool):
    a_type = int(action.get("action_type", 0))
    p = int(action.get("param", 0))

    trace = {
        "action": {"action_type": int(a_type), "action_name": _action_name(a_type), "param": int(p)},
        "selection": {"before": None, "after": None},
        "did_apply": True,
        "noop_reason": None,
        "derived_params": {},
        "modifiers": {"before": [], "after": []},
    }

    try:
        trace["modifiers"]["before"] = _modifier_snapshot(obj)
    except Exception:
        trace["modifiers"]["before"] = []

    # Selection before (needs edit mode to read correctly).
    try:
        trace["selection"]["before"] = int(_selected_face_count_any_mode(obj))
    except Exception:
        trace["selection"]["before"] = None

    if a_type == 0:  # ADD_CUBE
        # Replace the scene with a fresh cube (matches sim env: primitive-add
        # is a *replace*, not an *append*).
        _replace_scene_with_primitive()
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        obj = bpy.context.view_layer.objects.active
    elif a_type == 1:  # ADD_CYLINDER
        # Replace the scene with a fresh cylinder (matches sim env semantics).
        _replace_scene_with_primitive()
        verts = 12 + int((p / 31.0) * 20)
        trace["derived_params"].update({"vertices": int(verts), "radius": 0.5, "depth": 1.0})
        bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=0.5, depth=1.0, location=(0.0, 0.0, 0.0))
        obj = bpy.context.view_layer.objects.active
    elif a_type == 2:  # EXTRUDE
        _enter_edit_mode(obj)
        if _selected_face_count(obj) <= 0:
            _enter_object_mode()
            trace["did_apply"] = False
            trace["noop_reason"] = "no_selection"
            try:
                trace["selection"]["after"] = int(_selected_face_count_any_mode(obj))
            except Exception:
                pass
            try:
                trace["modifiers"]["after"] = _modifier_snapshot(obj)
            except Exception:
                pass
            return obj, trace
        dist = 0.05 + (p / 31.0) * 0.45
        trace["derived_params"].update({"distance": float(dist)})
        bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0.0, 0.0, dist)})
        _enter_object_mode()
    elif a_type == 3:  # INSET
        _enter_edit_mode(obj)
        if _selected_face_count(obj) <= 0:
            _enter_object_mode()
            trace["did_apply"] = False
            trace["noop_reason"] = "no_selection"
            try:
                trace["selection"]["after"] = int(_selected_face_count_any_mode(obj))
            except Exception:
                pass
            try:
                trace["modifiers"]["after"] = _modifier_snapshot(obj)
            except Exception:
                pass
            return obj, trace
        amount = 0.01 + (p / 31.0) * 0.15
        trace["derived_params"].update({"thickness": float(amount), "depth": 0.0})
        bpy.ops.mesh.inset(thickness=amount, depth=0.0)
        _enter_object_mode()
    elif a_type == 4:  # BEVEL
        _enter_edit_mode(obj)
        if _selected_face_count(obj) <= 0:
            _enter_object_mode()
            trace["did_apply"] = False
            trace["noop_reason"] = "no_selection"
            try:
                trace["selection"]["after"] = int(_selected_face_count_any_mode(obj))
            except Exception:
                pass
            try:
                trace["modifiers"]["after"] = _modifier_snapshot(obj)
            except Exception:
                pass
            return obj, trace

        # Geometry guards should never change operator semantics (e.g. by selecting
        # additional geometry). If the scene is too dense, treat as a no-op.
        if geometry_guards:
            v = len(obj.data.vertices)
            f = len(obj.data.polygons)
            if v >= 80_000 or f >= 80_000:
                _enter_object_mode()
                trace["did_apply"] = False
                trace["noop_reason"] = "geometry_guard_dense"
                try:
                    trace["selection"]["after"] = int(_selected_face_count_any_mode(obj))
                except Exception:
                    pass
                try:
                    trace["modifiers"]["after"] = _modifier_snapshot(obj)
                except Exception:
                    pass
                return obj, trace

        offset = 0.005 + (p / 31.0) * 0.12
        seg = 1 + int((p / 31.0) * 2)  # 1..3
        trace["derived_params"].update({"offset": float(offset), "segments": int(seg)})
        try:
            bpy.ops.mesh.bevel(offset=float(offset), segments=int(seg))
        except Exception:
            pass
        _enter_object_mode()
    elif a_type == 5:  # SCALE
        _enter_object_mode()
        s = 0.6 + (p / 31.0) * 1.0
        trace["derived_params"].update({"scale_factor": float(s)})
        obj.scale = (obj.scale.x * s, obj.scale.y * s, obj.scale.z * s)
        try:
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        except Exception:
            pass
    elif a_type == 6:  # SUBDIVIDE
        _enter_edit_mode(obj)
        if _selected_face_count(obj) <= 0:
            _enter_object_mode()
            trace["did_apply"] = False
            trace["noop_reason"] = "no_selection"
            try:
                trace["selection"]["after"] = int(_selected_face_count_any_mode(obj))
            except Exception:
                pass
            try:
                trace["modifiers"]["after"] = _modifier_snapshot(obj)
            except Exception:
                pass
            return obj, trace

        # Guard against runaway growth.
        if geometry_guards and (len(obj.data.vertices) >= 80_000):
            _enter_object_mode()
            trace["did_apply"] = False
            trace["noop_reason"] = "geometry_guard_dense"
            try:
                trace["selection"]["after"] = int(_selected_face_count_any_mode(obj))
            except Exception:
                pass
            try:
                trace["modifiers"]["after"] = _modifier_snapshot(obj)
            except Exception:
                pass
            return obj, trace

        cuts = 1 + int((p / 31.0) * 3)
        trace["derived_params"].update({"number_cuts": int(cuts)})
        try:
            bpy.ops.mesh.subdivide(number_cuts=int(cuts))
        except Exception:
            pass
        _enter_object_mode()
    elif a_type == 7:  # DELETE_FACE
        _enter_edit_mode(obj)
        if _selected_face_count(obj) <= 0:
            _enter_object_mode()
            trace["did_apply"] = False
            trace["noop_reason"] = "no_selection"
            try:
                trace["selection"]["after"] = int(_selected_face_count_any_mode(obj))
            except Exception:
                pass
            try:
                trace["modifiers"]["after"] = _modifier_snapshot(obj)
            except Exception:
                pass
            return obj, trace
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
        trace["derived_params"].update({"use_axis": list(mod.use_axis), "use_bisect_axis": list(mod.use_bisect_axis)})
        if apply_modifiers:
            _apply_modifier(obj, mod.name)
    elif a_type == 10:  # APPLY_MODIFIER
        _enter_object_mode()
        choices = ["SUBSURF", "BEVEL", "SOLIDIFY"]
        kind = choices[p % len(choices)]
        trace["derived_params"].update({"kind": str(kind)})
        if kind == "SUBSURF":
            # Subsurf can explode topology extremely quickly (and repeatedly applying
            # it can become catastrophic). Keep it heavily constrained.
            if geometry_guards:
                v = len(obj.data.vertices)
                f = len(obj.data.polygons)
                if v >= 5_000 or f >= 5_000:
                    trace["noop"] = True
                    trace["noop_reason"] = "geometry_guard: subsurf skipped (v=%d, f=%d >= 5000)" % (v, f)
                    return obj, trace
            mod = obj.modifiers.new(name="Subsurf", type="SUBSURF")
            mod.levels = 1 if geometry_guards else (1 + int((p / 31.0) * 2))
            trace["derived_params"].update({"levels": int(mod.levels)})
        elif kind == "BEVEL":
            # Modifier bevel on dense meshes can also explode topology.
            if geometry_guards:
                v = len(obj.data.vertices)
                f = len(obj.data.polygons)
                if v >= 20_000 or f >= 20_000:
                    trace["noop"] = True
                    trace["noop_reason"] = "geometry_guard: bevel skipped (v=%d, f=%d >= 20000)" % (v, f)
                    return obj, trace
            mod = obj.modifiers.new(name="Bevel", type="BEVEL")
            if geometry_guards:
                mod.width = 0.002 + (p / 31.0) * 0.03
                mod.segments = 1
            else:
                mod.width = 0.01 + (p / 31.0) * 0.1
                mod.segments = 2
            trace["derived_params"].update({"width": float(mod.width), "segments": int(mod.segments)})
        else:
            if geometry_guards:
                v = len(obj.data.vertices)
                f = len(obj.data.polygons)
                if v >= 50_000 or f >= 50_000:
                    trace["noop"] = True
                    trace["noop_reason"] = "geometry_guard: solidify skipped (v=%d, f=%d >= 50000)" % (v, f)
                    return obj, trace
            mod = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
            mod.thickness = 0.02 + (p / 31.0) * 0.2
            trace["derived_params"].update({"thickness": float(mod.thickness)})
        if apply_modifiers:
            _apply_modifier(obj, mod.name)

    elif a_type == 11:  # NOOP
        # Intentionally do nothing.
        pass

    try:
        trace["selection"]["after"] = int(_selected_face_count_any_mode(obj))
    except Exception:
        trace["selection"]["after"] = None
    try:
        trace["modifiers"]["after"] = _modifier_snapshot(obj)
    except Exception:
        trace["modifiers"]["after"] = []

    return obj, trace


def _wait_for(path: Path, poll_ms: int, timeout_s: int) -> dict:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(poll_ms / 1000.0)
    raise TimeoutError(f"Timeout waiting for {path}")


def main():
    args = _parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # wipe prior step files
    for p in work_dir.glob("action_*.json"):
        try:
            p.unlink()
        except Exception:
            pass
    for p in work_dir.glob("state_*.json"):
        try:
            p.unlink()
        except Exception:
            pass

    clear_scene()
    obj = _ensure_active_mesh_object(args.obj_name)

    rng = __import__("random").Random(int(args.seed))

    # Deselect all faces so the first step starts from a neutral selection state.
    try:
        _enter_edit_mode(obj)
        bpy.ops.mesh.select_all(action="DESELECT")
        _enter_object_mode()
    except Exception:
        pass

    ready = {
        "blender": bpy.app.version_string,
        "seed": int(args.seed),
        "stats": _mesh_stats(obj),
    }
    _write_json_atomic(work_dir / "ready.json", ready)

    # Early abort if scene starts in a pathological state.
    if _too_big(ready["stats"], max_verts=int(args.max_verts), max_faces=int(args.max_faces)):
        _write_json_atomic(
            work_dir / "early_stop.json",
            {"reason": "mesh_too_large_at_start", "stats": ready["stats"]},
        )
        bpy.ops.wm.save_as_mainfile(filepath=str(work_dir / "scene.blend"))
        _export_obj(work_dir / "mesh.obj")
        _write_json_atomic(work_dir / "stats_final.json", {"stats": ready["stats"]})
        print("OK: worker early-stopped (mesh too large at start)")
        return

    for i in range(int(args.steps)):
        action_path = work_dir / f"action_{i:04d}.json"
        action = _wait_for(action_path, int(args.poll_ms), int(args.timeout_seconds))
        t_start = time.time()
        obj, trace = _execute_step(
            obj,
            action,
            rng,
            bool(args.apply_modifiers),
            geometry_guards=(not bool(args.unsafe_no_geometry_guards)),
        )
        trace["step"] = int(i)
        trace["time_s"] = float(max(0.0, time.time() - t_start))
        stats = _mesh_stats(obj)
        _write_json_atomic(work_dir / f"state_{i:04d}.json", {"stats": stats})
        _write_json_atomic(work_dir / f"trace_{i:04d}.json", trace)

        # Safety cap: stop before Blender becomes unusable.
        if _too_big(stats, max_verts=int(args.max_verts), max_faces=int(args.max_faces)):
            _write_json_atomic(
                work_dir / "early_stop.json",
                {"reason": "mesh_too_large", "step": int(i), "stats": stats, "max_verts": int(args.max_verts), "max_faces": int(args.max_faces)},
            )
            break

    _enter_object_mode()
    bpy.ops.wm.save_as_mainfile(filepath=str(work_dir / "scene.blend"))
    _export_obj(work_dir / "mesh.obj")
    _write_json_atomic(work_dir / "stats_final.json", {"stats": _mesh_stats(obj)})

    if (work_dir / "early_stop.json").exists():
        print("OK: worker finished (early stop)")
    else:
        print("OK: worker finished")


if __name__ == "__main__":
    main()
