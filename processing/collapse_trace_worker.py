"""Generate a collapse/simplification trace inside Blender (headless).

Goal: produce step-by-step supervision for a reversal curriculum:
  - Remove non-applied modifiers first (high leverage).
  - Then reduce geometry aggressively using unsubdivide (loop-cut reversal analog),
    followed by limited dissolve / merge-by-distance / delete loose.

This is NOT a perfect progressive-mesh implementation (Hoppe), but it creates
compact, deterministic traces that are useful for training a policy on
"simplify" flows without random edge/face choices.

Run:
  /Applications/Blender.app/Contents/MacOS/Blender --background --python processing/collapse_trace_worker.py -- \
    --in-blend /path/to/input.blend \
    --object-name MyObj \
    --out-dir /tmp/collapse_trace \
    --max-steps 64 \
    --target-verts 8

Outputs:
  out_dir/trace.jsonl    (one JSON per step)
  out_dir/final.blend
  out_dir/final.obj

Each step contains pre/post mesh stats plus the operation that was applied.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore
import bmesh  # type: ignore
from mathutils import Vector  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.blender_render import clear_scene, create_mesh_from_data


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Blender collapse trace worker")
    p.add_argument("--in-blend", type=str, default="", help="Optional .blend to open")
    p.add_argument("--object-name", type=str, default="", help="Mesh object to collapse (defaults to active)")
    p.add_argument("--mesh-json", type=str, default="", help="Optional mesh JSON (vertices/faces) to build scene from")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--target-verts", type=int, default=8)
    p.add_argument("--snapshot-every", type=int, default=0, help="If >0, write obj snapshots every N steps")
    return p.parse_args(argv)


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

    denom = max(1e-6, max(ext.x, ext.y))
    symmetry = 1.0 - float(abs(ext.x - ext.y) / denom)
    symmetry = max(0.0, min(1.0, symmetry))

    return {
        "vertex_count": int(v_count),
        "edge_count": int(e_count),
        "face_count": int(f_count),
        "bounding_box": {"x": float(ext.x), "y": float(ext.y), "z": float(ext.z)},
        "avg_edge_length": float(avg_edge),
        "manifold_flag": float(1.0 if non_man == 0 else 0.0),
        "symmetry_score": float(symmetry),
        "modifier_count": int(len(getattr(obj, "modifiers", []) or [])),
    }


def _export_obj(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=False)
    except Exception:
        bpy.ops.export_scene.obj(filepath=str(path), use_selection=False)


def _remove_one_modifier(obj) -> dict | None:
    _enter_object_mode()
    mods = list(getattr(obj, "modifiers", []) or [])
    if not mods:
        return None

    m = mods[-1]  # deterministic: pop from end
    record = {"name": str(m.name), "type": str(m.type)}

    # Store a small subset of common settings when present (best-effort).
    for k in ("levels", "render_levels", "width", "segments", "thickness", "ratio"):
        if not hasattr(m, k):
            continue
        try:
            v = getattr(m, k)
            # Convert tensors/complex types to a plain float where possible.
            if isinstance(v, (int, float)):
                record[k] = float(v)
            else:
                # Fallback: string representation (keeps JSON-serialisable).
                record[k] = str(v)
        except Exception:
            continue

    try:
        obj.modifiers.remove(m)
    except Exception:
        return None
    return record


def _simplify_geometry_once(obj, aggressiveness: float) -> str | None:
    aggressiveness = float(max(0.0, min(1.0, aggressiveness)))

    _enter_edit_mode(obj)
    try:
        bpy.ops.mesh.select_all(action="SELECT")
    except Exception:
        pass

    before_v = len(obj.data.vertices)

    def _sync_edit_mesh() -> None:
        """Sync edit-mode bmesh edits back to obj.data for accurate counts."""
        try:
            bpy.context.view_layer.objects.active = obj
        except Exception:
            pass
        try:
            # Updates mesh datablock from edit-mode representation.
            obj.update_from_editmode()
        except Exception:
            pass
        try:
            obj.data.update()
        except Exception:
            pass

    # 1) Unsubdivide: strongest loop-cut reversal analogue.
    iters = 1 + int(round(aggressiveness * 3.0))  # 1..4
    try:
        bpy.ops.mesh.unsubdivide(iterations=int(iters))
    except Exception:
        pass
    _sync_edit_mesh()
    if len(obj.data.vertices) < before_v:
        _enter_object_mode()
        return "unsubdivide"

    # 2) Limited dissolve
    try:
        deg = 5.0 + aggressiveness * 30.0
        bpy.ops.mesh.dissolve_limited(angle_limit=float(math.radians(deg)))
    except Exception:
        pass
    _sync_edit_mesh()
    if len(obj.data.vertices) < before_v:
        _enter_object_mode()
        return "dissolve_limited"

    # 3) Merge by distance
    try:
        dist = 1e-5 + aggressiveness * 2e-3
        bpy.ops.mesh.merge_by_distance(distance=float(dist))
    except Exception:
        pass
    _sync_edit_mesh()
    if len(obj.data.vertices) < before_v:
        _enter_object_mode()
        return "merge_by_distance"

    # 4) Delete loose
    try:
        bpy.ops.mesh.delete_loose()
    except Exception:
        pass

    _sync_edit_mesh()

    _enter_object_mode()
    if len(obj.data.vertices) < before_v:
        return "delete_loose"
    return None


def _get_target_object(name: str) -> object:
    if name:
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != "MESH":
            raise RuntimeError(f"Object not found or not MESH: {name!r}")
        return obj

    obj = bpy.context.view_layer.objects.active
    if obj is None or obj.type != "MESH":
        raise RuntimeError("No active mesh object")
    return obj


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = None
    if args.mesh_json:
        # Build mesh from JSON (same format as blender_render expects).
        mesh_json_path = Path(args.mesh_json)
        with mesh_json_path.open("r", encoding="utf-8") as jf:
            data = json.load(jf)

        # Flat format {"vertices": [], "faces": []} is enough here.
        vertices = data.get("vertices", [])
        faces = data.get("faces", [])
        if not vertices or not faces:
            raise RuntimeError("mesh-json missing vertices/faces")

        clear_scene()
        obj = create_mesh_from_data(vertices, faces, name=(data.get("mesh_id") or "CollapseMesh"))
    else:
        if args.in_blend:
            bpy.ops.wm.open_mainfile(filepath=str(Path(args.in_blend)))
        obj = _get_target_object(args.object_name)

    trace_path = out_dir / "trace.jsonl"
    f = trace_path.open("w", encoding="utf-8")

    target_verts = max(1, int(args.target_verts))

    for step in range(int(args.max_steps)):
        pre = _mesh_stats(obj)
        if int(pre["vertex_count"]) <= target_verts and int(pre["modifier_count"]) == 0:
            break

        op = None
        op_data = None

        # First: remove modifiers (highest leverage)
        mod_rec = _remove_one_modifier(obj)
        if mod_rec is not None:
            op = "remove_modifier"
            op_data = mod_rec
        else:
            # Then: simplify geometry deterministically
            op = _simplify_geometry_once(obj, aggressiveness=0.85)
            if op is None:
                break

        post = _mesh_stats(obj)
        rec = {
            "step": int(step),
            "op": op,
            "op_data": op_data,
            "pre": pre,
            "post": post,
        }
        f.write(json.dumps(rec) + "\n")
        f.flush()

        if int(args.snapshot_every) > 0 and (step + 1) % int(args.snapshot_every) == 0:
            _export_obj(out_dir / f"snap_{step+1:04d}.obj")

    f.close()

    _enter_object_mode()
    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "final.blend"))
    _export_obj(out_dir / "final.obj")

    (out_dir / "final_stats.json").write_text(json.dumps(_mesh_stats(obj), indent=2))
    print(f"OK: wrote collapse trace to {trace_path}")


if __name__ == "__main__":
    main()
