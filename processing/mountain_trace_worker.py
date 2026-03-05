"""Generate deterministic mountain modeling traces inside Blender.

This worker records forward, artist-like scripted traces:
RESET_SCENE -> ADD_PLANE -> SUBDIVIDE -> DISPLACE(PROC_HEIGHT_BASIC)
-> DECIMATE -> TRIANGULATE(optional) -> SHADE(mode)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import bmesh  # type: ignore
import bpy  # type: ignore
from mathutils import Vector  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from processing.procedural_displacement import (  # noqa: E402
    apply_proc_height_displacement,
    params_from_buckets,
)


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Generate deterministic mountain traces")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--style", type=str, default="stylized", choices=["stylized", "retro"])
    p.add_argument("--name", type=str, default="Mountain")
    p.add_argument("--plane-size", type=float, default=2.0)
    p.add_argument("--max-subdiv-cuts", type=int, default=5)
    p.add_argument("--triangulate", action="store_true")
    return p.parse_args(argv)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.textures):
        for item in list(block):
            try:
                block.remove(item)
            except Exception:
                pass


def _enter_object_mode() -> None:
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass


def _select_only(obj) -> None:
    _enter_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _mesh_stats(obj) -> dict:
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
    non_man = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()

    denom = max(1e-6, max(ext.x, ext.y))
    symmetry = 1.0 - float(abs(ext.x - ext.y) / denom)
    symmetry = max(0.0, min(1.0, symmetry))

    smooth_faces = 0
    tri_faces = 0
    for poly in mesh.polygons:
        if bool(getattr(poly, "use_smooth", False)):
            smooth_faces += 1
        if int(getattr(poly, "loop_total", 0)) == 3:
            tri_faces += 1
    face_denom = max(1, int(f_count))

    has_decimate = False
    decimate_ratio = 1.0
    has_displace = False
    displace_strength = 0.0
    displace_midlevel = 0.5
    has_triangulate = False
    for mod in getattr(obj, "modifiers", []):
        t = str(getattr(mod, "type", ""))
        if t == "DECIMATE":
            has_decimate = True
            try:
                decimate_ratio = float(getattr(mod, "ratio", 1.0))
            except Exception:
                pass
        elif t == "DISPLACE":
            has_displace = True
            try:
                displace_strength = float(getattr(mod, "strength", 0.0))
                displace_midlevel = float(getattr(mod, "mid_level", 0.5))
            except Exception:
                pass
        elif t == "TRIANGULATE":
            has_triangulate = True

    return {
        "vertex_count": int(v_count),
        "edge_count": int(e_count),
        "face_count": int(f_count),
        "bounding_box": {"x": float(ext.x), "y": float(ext.y), "z": float(ext.z)},
        "avg_edge_length": float(avg_edge),
        "manifold_flag": 1.0 if non_man == 0 else 0.0,
        "symmetry_score": float(symmetry),
        "shade_smooth_fraction": float(smooth_faces / face_denom),
        "triangulated_ratio": float(tri_faces / face_denom),
        "has_decimate_modifier": bool(has_decimate),
        "decimate_ratio": float(decimate_ratio),
        "has_displace_modifier": bool(has_displace),
        "displace_strength": float(displace_strength),
        "displace_midlevel": float(displace_midlevel),
        "has_triangulate_modifier": bool(has_triangulate),
    }


def _record_step(trace_path: Path, step: int, action: str, params: dict, pre: dict, post: dict) -> None:
    _append_jsonl(
        trace_path,
        {
            "step": int(step),
            "action": str(action),
            "params": params,
            "pre": pre,
            "post": post,
        },
    )


def _add_modifier(obj, mod_type: str, name: str):
    return obj.modifiers.new(name=name, type=mod_type)


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    rng = random.Random(int(args.seed))

    _clear_scene()

    step = 0
    pre = {}
    post = {}
    _record_step(trace_path, step, "RESET_SCENE", {"seed": int(args.seed)}, pre, post)

    bpy.ops.mesh.primitive_plane_add(size=float(args.plane_size), location=(0.0, 0.0, 0.0))
    obj = bpy.context.view_layer.objects.active
    if obj is None:
        raise RuntimeError("Failed to create plane")
    obj.name = str(args.name)
    obj.data.name = f"{args.name}_mesh"

    step += 1
    pre = {}
    post = _mesh_stats(obj)
    _record_step(trace_path, step, "ADD_PLANE", {"size": float(args.plane_size)}, pre, post)

    cuts = int(rng.randint(2, max(2, int(args.max_subdiv_cuts))))
    _select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.subdivide(number_cuts=int(cuts))
    bpy.ops.object.mode_set(mode="OBJECT")

    step += 1
    pre = post
    post = _mesh_stats(obj)
    _record_step(trace_path, step, "SUBDIVIDE", {"number_cuts": int(cuts)}, pre, post)

    scale_bucket = rng.randint(5, 22)
    detail_bucket = rng.randint(6, 20)
    roughness_bucket = rng.randint(10, 26)
    distortion_bucket = rng.randint(0, 16)
    strength_bucket = rng.randint(6, 20)
    midlevel_bucket = rng.randint(12, 20)
    params = params_from_buckets(
        seed=int(args.seed),
        scale_bucket=scale_bucket,
        detail_bucket=detail_bucket,
        roughness_bucket=roughness_bucket,
        distortion_bucket=distortion_bucket,
        strength_bucket=strength_bucket,
        midlevel_bucket=midlevel_bucket,
    )
    disp_info = apply_proc_height_displacement(obj, params)

    step += 1
    pre = post
    post = _mesh_stats(obj)
    _record_step(trace_path, step, "ADD_PROC_DISPLACE", disp_info, pre, post)

    ratio = 0.25 if args.style == "stylized" else 0.45
    ratio = float(max(0.08, min(0.95, ratio + rng.uniform(-0.05, 0.05))))
    dec = _add_modifier(obj, "DECIMATE", "Decimate")
    dec.decimate_type = "COLLAPSE"
    dec.ratio = float(ratio)

    step += 1
    pre = post
    post = _mesh_stats(obj)
    _record_step(trace_path, step, "ADD_DECIMATE", {"ratio": float(ratio)}, pre, post)

    do_triangulate = bool(args.triangulate) or bool(args.style == "stylized")
    if do_triangulate:
        _add_modifier(obj, "TRIANGULATE", "Triangulate")
        step += 1
        pre = post
        post = _mesh_stats(obj)
        _record_step(trace_path, step, "ADD_TRIANGULATE", {}, pre, post)

    _select_only(obj)
    if args.style == "stylized":
        try:
            bpy.ops.object.shade_flat()
        except Exception:
            try:
                bpy.ops.object.shade_smooth_by_angle(angle=0.0)
            except Exception:
                pass
        action_name = "SHADE_FLAT"
    else:
        try:
            bpy.ops.object.shade_smooth()
        except Exception:
            try:
                bpy.ops.object.shade_smooth_by_angle(angle=0.78539816339)
            except Exception:
                pass
        action_name = "SHADE_SMOOTH"

    step += 1
    pre = post
    post = _mesh_stats(obj)
    _record_step(trace_path, step, action_name, {}, pre, post)

    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "scene.blend"))
    try:
        bpy.ops.wm.obj_export(filepath=str(out_dir / "mesh.obj"), export_selected_objects=False)
    except Exception:
        bpy.ops.export_scene.obj(filepath=str(out_dir / "mesh.obj"), use_selection=False)

    _write_json(
        out_dir / "stats_final.json",
        {
            "style": str(args.style),
            "seed": int(args.seed),
            "steps": int(step),
            "stats": post,
        },
    )
    _write_json(
        out_dir / "metadata.json",
        {
            "prompt": (
                "stylized low poly mountain terrain" if args.style == "stylized" else "retro ps1 low poly mountain terrain"
            ),
            "style": str(args.style),
            "seed": int(args.seed),
            "trace_path": str(trace_path),
        },
    )

    print(f"OK: wrote mountain trace to {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
