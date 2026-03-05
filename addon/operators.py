"""Operators for the Blender Copilot.

Operator catalogue
──────────────────
  aihouse.send_prompt       — send prompt to AI, execute response
  aihouse.stop_generation   — halt the current generation
  aihouse.execute_code      — re-run the last generated code
  aihouse.clear_scene       — wipe all objects
  aihouse.clear_chat        — reset conversation history
"""

import json as _json
import hashlib
import os
import os as _os
import random
import threading
import traceback
import time
from datetime import datetime as _datetime
from pathlib import Path
import subprocess

import bpy  # type: ignore
from bpy.types import Operator  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# Code Execution Helper
# ═══════════════════════════════════════════════════════════════════════════

def _execute_code(code_string):
    """Execute AI-generated code with all blender_tools pre-imported.

    Returns ``(True, "")`` on success or ``(False, error_message)`` on failure.
    """
    import math
    import bmesh  # type: ignore
    from mathutils import Vector, Matrix, Euler  # type: ignore
    from . import blender_tools

    namespace = {
        "__builtins__": __builtins__,
        "bpy": bpy,
        "bmesh": bmesh,
        "math": math,
        "Vector": Vector,
        "Matrix": Matrix,
        "Euler": Euler,
    }
    for attr in dir(blender_tools):
        if not attr.startswith("_"):
            namespace[attr] = getattr(blender_tools, attr)

    try:
        bpy.context.view_layer.update()
    except Exception:
        pass

    try:
        exec(code_string, namespace)
        return True, ""
    except Exception as e:
        tb = traceback.format_exc()
        line_info = ""
        for tb_line in tb.split('\n'):
            if 'File "<string>"' in tb_line:
                line_info = tb_line.strip()
                break
        print("\n[Blender Copilot] ══ CODE EXECUTION ERROR ══")
        print(tb)
        print("══ Generated code was ══")
        for i, line in enumerate(code_string.split('\n'), 1):
            print("  %3d | %s" % (i, line))
        print("═══════════════════════════════════════\n")
        err_msg = "%s: %s" % (type(e).__name__, e)
        if line_info:
            err_msg = "%s [%s]" % (err_msg, line_info)
        return False, err_msg


def _force_viewport_update():
    """Force Blender to redraw viewports."""
    try:
        bpy.context.view_layer.update()
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        bpy.context.evaluated_depsgraph_get()
    except Exception:
        pass


def _is_policy_project_root(p: Path) -> bool:
    try:
        return (p / "scripts" / "rollout_policy_closed_loop.py").exists()
    except Exception:
        return False


def _guess_policy_project_root(fallback: Path) -> Path | None:
    env = os.environ.get("BLENDER_COPILOT_PROJECT_ROOT")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())

    home = Path.home()
    candidates.extend(
        [
            home / "blenderPlugins" / "blender-copilot",
            home / "blender-copilot",
            home / "projects" / "blender-copilot",
            home / "src" / "blender-copilot",
        ]
    )

    candidates.append(fallback)

    for c in candidates:
        if _is_policy_project_root(c):
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Validator helpers
# ═══════════════════════════════════════════════════════════════════════════

_VALIDATOR_QUEUE: list[dict] = []
_VALIDATOR_QUEUE_DIR: Path | None = None
_VALIDATOR_REVIEWED: set[str] = set()
_VALIDATOR_CACHE_DIR: Path | None = None
_VALIDATOR_WORK_DIR: Path | None = None


def _validator_queue_dir(props) -> Path | None:
    p = (props.validator_queue_dir or "").strip()
    if not p:
        return None
    try:
        return Path(p).expanduser().resolve()
    except Exception:
        return None


def _validator_index_path(queue_dir: Path) -> Path:
    return queue_dir / "index.jsonl"


def _validator_reviews_path(queue_dir: Path) -> Path:
    return queue_dir / "reviews.jsonl"


def _validator_guess_project_root(p: Path) -> Path | None:
    cur = p.resolve()
    if cur.is_file():
        cur = cur.parent
    for _ in range(6):
        if (cur / "config.yaml").exists() and (cur / "run.py").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _validator_guess_cache_dir(selected: Path) -> tuple[Path | None, Path | None]:
    """Return (project_root, cache_dir) if we can infer them.

    Resolution order when user selects repo root:
      1. data/training_cache/default/ (pre-built training cache, preferred)
      2. data/processed/.mesh_cache   (legacy inline cache, fallback)
    """
    sel = selected.resolve()

    # If user selected repo root
    if (sel / "config.yaml").exists() and (sel / "run.py").exists():
        # Prefer pre-built training cache (from scripts/build_training_cache.py)
        training_cache = sel / "data" / "training_cache" / "default"
        if training_cache.exists() and list(training_cache.glob("batch_*.pt")):
            return sel, training_cache
        # Fall back to legacy inline cache
        cache = sel / "data" / "processed" / ".mesh_cache"
        return sel, cache if cache.exists() else None

    # If user selected the cache dir directly (any dir with .pt files)
    if sel.is_dir() and list(sel.glob("*.pt")):
        project_root = _validator_guess_project_root(sel)
        return project_root, sel

    # If user selected an exported queue dir (legacy)
    project_root = _validator_guess_project_root(sel)
    if project_root:
        # Same priority: training cache > legacy cache
        training_cache = project_root / "data" / "training_cache" / "default"
        if training_cache.exists() and list(training_cache.glob("batch_*.pt")):
            return project_root, training_cache
        cache = project_root / "data" / "processed" / ".mesh_cache"
        if cache.exists():
            return project_root, cache
    return project_root, None


def _validator_default_work_dir(project_root: Path | None, selected: Path) -> Path:
    if project_root is not None:
        return (project_root / "data" / "validation_queue_live").resolve()
    return (selected / "validation_queue_live").resolve()


def _validator_run_fetch_item(*,
                             context,
                             cache_dir: Path,
                             work_dir: Path,
                             after_cache_pt: str = "",
                             after_item_index: int = -1,
                             cache_pt: str = "",
                             item_index: int = -1,
                             fresh_only: bool = False,
                             fresh_hours: float = 0.0) -> tuple[bool, dict, str]:
    """Call external venv python to fetch/decode the next cache item."""
    prefs = context.preferences.addons[__package__].preferences

    project_root = _guess_policy_project_root(Path(prefs.policy_project_root))
    if project_root is None:
        # Fall back to our own best guess if user didn't set prefs yet.
        project_root = _validator_guess_project_root(cache_dir) or cache_dir

    py = Path(prefs.policy_python) if prefs.policy_python else None
    if not py or not py.exists():
        py = _default_policy_python(project_root)
    if py is None or not py.exists():
        return False, {}, "Policy Python (venv) not found"

    script = project_root / "scripts" / "validator_fetch_item.py"
    if not script.exists():
        return False, {}, f"Missing script: {script}"

    cmd = [
        str(py),
        str(script),
        "--cache-dir", str(cache_dir),
        "--work-dir", str(work_dir),
    ]
    if fresh_only:
        cmd += ["--fresh-only"]
        if float(fresh_hours) > 0.0:
            cmd += ["--fresh-hours", str(float(fresh_hours))]
    if cache_pt and item_index >= 0:
        cmd += ["--cache-pt", str(cache_pt), "--item-index", str(int(item_index))]
    else:
        if after_cache_pt:
            cmd += ["--after-cache-pt", str(after_cache_pt)]
        if after_item_index >= 0:
            cmd += ["--after-item-index", str(int(after_item_index))]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
    except Exception as e:
        return False, {}, str(e)

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, {}, err[-600:] if err else f"validator_fetch_item failed ({proc.returncode})"

    out_txt = (proc.stdout or "").strip()
    try:
        obj = _json.loads(out_txt)
    except Exception:
        return False, {}, f"validator_fetch_item returned non-JSON: {out_txt[:200]}"

    return True, obj, ""


def _validator_load_reviews(queue_dir: Path) -> set[str]:
    reviewed: set[str] = set()
    p = _validator_reviews_path(queue_dir)
    if not p.exists():
        return reviewed
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except Exception:
                continue
            item_id = obj.get("item_id")
            if isinstance(item_id, str) and item_id:
                reviewed.add(item_id)
    except Exception:
        pass
    return reviewed


def _validator_find_next_index(start: int = 0) -> int | None:
    for i in range(max(0, int(start)), len(_VALIDATOR_QUEUE)):
        item_id = _VALIDATOR_QUEUE[i].get("item_id")
        if isinstance(item_id, str) and item_id and item_id not in _VALIDATOR_REVIEWED:
            return i
    return None


def _validator_get_item(i: int) -> dict | None:
    if 0 <= int(i) < len(_VALIDATOR_QUEUE):
        return _VALIDATOR_QUEUE[int(i)]
    return None


def _validator_ensure_collection(name: str = "AI_VALIDATION"):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def _validator_clear_previous():
    col = bpy.data.collections.get("AI_VALIDATION")
    if not col:
        return
    for obj in list(col.objects):
        try:
            mesh_data = getattr(obj, "data", None)
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh_data is not None and getattr(mesh_data, "users", 0) == 0:
                try:
                    bpy.data.meshes.remove(mesh_data)
                except Exception:
                    pass
        except Exception:
            pass


def _fallback_material_rgba(obj_data: dict | None) -> tuple[float, float, float, float]:
    """Pick a deterministic fallback color from label/source metadata."""
    if not isinstance(obj_data, dict):
        return (0.62, 0.62, 0.62, 1.0)

    key = "|".join(
        [
            str(obj_data.get("label", "")),
            str(obj_data.get("item_id", "")),
            str(obj_data.get("data_source", "")),
        ]
    )
    if not key.strip():
        return (0.62, 0.62, 0.62, 1.0)

    digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).digest()
    # Keep values in a readable mid-range so assets don't look washed out or black.
    r = 0.35 + (digest[0] / 255.0) * 0.45
    g = 0.35 + (digest[1] / 255.0) * 0.45
    b = 0.35 + (digest[2] / 255.0) * 0.45
    return (r, g, b, 1.0)


def _apply_default_material(mesh_obj, obj_data=None):
    """Create a deterministic fallback material for material-less cache items."""
    try:
        rgba = _fallback_material_rgba(obj_data)
        mat = bpy.data.materials.new(name="Fallback")
        mat.use_nodes = True
        mat.diffuse_color = rgba
        tree = mat.node_tree
        for nd in tree.nodes:
            if nd.bl_idname == "ShaderNodeBsdfPrincipled":
                inp = nd.inputs.get("Base Color")
                if inp:
                    inp.default_value = rgba
                rough = nd.inputs.get("Roughness")
                if rough:
                    rough.default_value = 0.65
                spec = nd.inputs.get("Specular IOR Level") or nd.inputs.get("Specular")
                if spec:
                    spec.default_value = 0.25
                break
        mesh_obj.data.materials.append(mat)
    except Exception:
        pass


def _apply_json_materials(mesh_obj, obj_data):
    """Reconstruct Blender materials from scene_context JSON data.

    Handles full node tree reconstruction including node properties
    (operation, blend_type, data_type, color_ramp elements, etc.),
    socket default values, and inter-node links.

    Falls back to a simple Principled BSDF if reconstruction fails.
    When no scene_context/materials exist, creates a neutral default material
    so the mesh is visible in Material Preview mode.
    """
    sc = obj_data.get("scene_context")
    if not isinstance(sc, dict) or not sc:
        # No scene_context at all — create a neutral default material
        _apply_default_material(mesh_obj, obj_data)
        return

    # Known node properties that should be set as attributes (not socket inputs)
    _NODE_PROPS = {
        "operation", "blend_type", "data_type", "clamp_type",
        "interpolation", "projection", "color_space", "use_clamp",
        "invert", "subsurface_method",
        # ShaderNodeMix (Blender 4.x)
        "factor_mode", "clamp_factor", "clamp_result",
    }

    # Node type compatibility mapping for deprecated/renamed nodes.
    # Only map nodes with compatible socket layouts.
    # ShaderNodeEeveeSpecular (removed in Blender 4.0) is intentionally
    # NOT mapped — its sockets don't match Principled BSDF, so the
    # fallback (clean Principled BSDF) gives better results.
    _NODE_COMPAT = {
        "ShaderNodeMixRGB": "ShaderNodeMix",
    }

    mat_datas = sc.get("materials", [])
    if not isinstance(mat_datas, list) or not mat_datas:
        # scene_context exists but has no materials — create default
        _apply_default_material(mesh_obj, obj_data)
        return

    for m_data in mat_datas:
        name = m_data.get("name", "Material")
        mat = bpy.data.materials.new(name=name)
        mesh_obj.data.materials.append(mat)

        bc = m_data.get("base_color")
        if isinstance(bc, list) and len(bc) >= 3:
            mat.diffuse_color = (bc[0], bc[1], bc[2], bc[3] if len(bc) > 3 else 1.0)

        if not m_data.get("use_nodes", True):
            continue

        mat.use_nodes = True
        tree = mat.node_tree
        if not tree:
            continue

        # Save default Principled BSDF + Material Output as fallback
        # (Blender creates these automatically when use_nodes = True)
        default_output = None
        default_bsdf = None
        for dn in tree.nodes:
            if dn.bl_idname == "ShaderNodeOutputMaterial":
                default_output = dn
            elif dn.bl_idname == "ShaderNodeBsdfPrincipled":
                default_bsdf = dn

        nodes_data = m_data.get("nodes", [])
        links_data = m_data.get("links", [])

        # Skip reconstruction if no node data — keep Blender defaults
        if not nodes_data:
            if default_bsdf and bc and isinstance(bc, list) and len(bc) >= 3:
                inp = default_bsdf.inputs.get("Base Color")
                if inp:
                    inp.default_value = (bc[0], bc[1], bc[2], bc[3] if len(bc) > 3 else 1.0)
            continue

        # Clear default nodes and reconstruct from JSON
        tree.nodes.clear()
        nodes_by_name = {}
        created_output = False

        for n_data in nodes_data:
            bl_idname = n_data.get("bl_idname")
            if not bl_idname:
                continue
            try:
                n = tree.nodes.new(bl_idname)
            except Exception:
                # Node type may not exist — try compatibility mapping
                compat = _NODE_COMPAT.get(bl_idname)
                if compat:
                    try:
                        n = tree.nodes.new(compat)
                    except Exception:
                        continue
                else:
                    continue
            n.name = n_data.get("name", "")
            nodes_by_name[n.name] = n

            if bl_idname == "ShaderNodeOutputMaterial":
                created_output = True

            loc = n_data.get("location")
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                n.location = (loc[0], loc[1])

            # Set node-level properties (operation, blend_type, etc.)
            for prop in _NODE_PROPS:
                val = n_data.get(prop)
                if val is not None:
                    try:
                        setattr(n, prop, val)
                    except Exception:
                        pass

            # Restore ColorRamp elements
            cr_data = n_data.get("color_ramp")
            if cr_data and hasattr(n, "color_ramp"):
                try:
                    ramp = n.color_ramp
                    interp = cr_data.get("interpolation")
                    if interp:
                        ramp.interpolation = interp
                    elements = cr_data.get("elements", [])
                    # Ramp starts with 2 default elements
                    while len(ramp.elements) < len(elements):
                        ramp.elements.new(0.5)
                    while len(ramp.elements) > len(elements) and len(ramp.elements) > 1:
                        ramp.elements.remove(ramp.elements[-1])
                    for ei, el_data in enumerate(elements):
                        if ei < len(ramp.elements):
                            ramp.elements[ei].position = float(el_data.get("position", 0.0))
                            c = el_data.get("color", [0, 0, 0, 1])
                            if isinstance(c, list) and len(c) >= 4:
                                ramp.elements[ei].color = tuple(c[:4])
                except Exception:
                    pass

            # Set socket default values
            for in_name, in_val in n_data.get("inputs", {}).items():
                if in_val == "LINKED":
                    continue
                if in_name not in n.inputs:
                    continue
                try:
                    sock = n.inputs[in_name]
                    sock_type = type(sock).__name__
                    if sock_type == "NodeSocketColor" and isinstance(in_val, list):
                        if len(in_val) == 3:
                            in_val = list(in_val) + [1.0]
                        sock.default_value = tuple(in_val[:4])
                    elif sock_type == "NodeSocketVector" and isinstance(in_val, list):
                        sock.default_value = tuple(in_val[:3])
                    elif isinstance(in_val, list):
                        sock.default_value = type(sock.default_value)(in_val)
                    else:
                        sock.default_value = in_val
                except Exception:
                    pass

        # Create links — prefer socket INDEX over name for disambiguation
        # (e.g. ShaderNodeMix has multiple "Result" outputs)
        for l_data in links_data:
            from_n = nodes_by_name.get(l_data.get("from_node"))
            to_n = nodes_by_name.get(l_data.get("to_node"))
            if not from_n or not to_n:
                continue
            # Try index-based lookup first (more reliable)
            from_sock = None
            to_sock = None
            fi = l_data.get("from_socket_index")
            ti = l_data.get("to_socket_index")
            if fi is not None and fi < len(from_n.outputs):
                from_sock = from_n.outputs[fi]
            if to_sock is None and ti is not None and ti < len(to_n.inputs):
                to_sock = to_n.inputs[ti]
            # Fall back to name-based lookup
            if from_sock is None:
                from_sock = from_n.outputs.get(l_data.get("from_socket"))
            if to_sock is None:
                to_sock = to_n.inputs.get(l_data.get("to_socket"))
            if from_sock and to_sock:
                try:
                    tree.links.new(from_sock, to_sock)
                except Exception:
                    pass

        # Fallback: if reconstruction didn't produce a Material Output,
        # OR if the Material Output's Surface input is unconnected
        # (e.g. because a critical node like ShaderNodeEeveeSpecular failed),
        # create a minimal Principled BSDF so the mesh has visible shading.
        needs_fallback = not created_output
        if not needs_fallback:
            # Check if Material Output's Surface socket is actually connected
            for nd in tree.nodes:
                if nd.bl_idname == "ShaderNodeOutputMaterial":
                    surf = nd.inputs.get("Surface")
                    if surf and not surf.is_linked:
                        needs_fallback = True
                    break

        if needs_fallback:
            try:
                # Find or create Material Output
                out_node = None
                for nd in tree.nodes:
                    if nd.bl_idname == "ShaderNodeOutputMaterial":
                        out_node = nd
                        break
                if not out_node:
                    out_node = tree.nodes.new("ShaderNodeOutputMaterial")
                    out_node.location = (300, 0)
                bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
                bsdf.location = (0, 0)
                if bc and isinstance(bc, list) and len(bc) >= 3:
                    bsdf.inputs["Base Color"].default_value = (
                        bc[0], bc[1], bc[2], bc[3] if len(bc) > 3 else 1.0
                    )
                tree.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])
            except Exception:
                pass

    # Assign per-face material indices
    face_mat_idx = sc.get("face_material_indices", [])
    n_polys = len(mesh_obj.data.polygons)
    n_mats = len(mesh_obj.data.materials)
    if isinstance(face_mat_idx, list) and n_mats > 0 and n_polys > 0:
        if len(face_mat_idx) == n_polys:
            # Exact match — assign directly
            for i, poly in enumerate(mesh_obj.data.polygons):
                idx = int(face_mat_idx[i])
                if 0 <= idx < n_mats:
                    poly.material_index = idx
        elif len(face_mat_idx) > n_polys:
            # FMI longer than decoded faces (truncated mesh) — proportional mapping
            ratio = len(face_mat_idx) / n_polys
            for i, poly in enumerate(mesh_obj.data.polygons):
                src_i = min(int(i * ratio), len(face_mat_idx) - 1)
                idx = int(face_mat_idx[src_i])
                if 0 <= idx < n_mats:
                    poly.material_index = idx
        # else: FMI shorter — leave all faces on material 0

    # Assign per-face smooth shading
    face_smooth = sc.get("face_smooth", [])
    if isinstance(face_smooth, list) and len(face_smooth) == len(mesh_obj.data.polygons):
        for i, poly in enumerate(mesh_obj.data.polygons):
            poly.use_smooth = bool(face_smooth[i])


def _set_viewport_material_preview():
    """Switch 3D viewport to Material Preview mode so materials are visible."""
    try:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        if space.shading.type in ('WIREFRAME', 'SOLID'):
                            space.shading.type = 'MATERIAL'
                        break
                break
    except Exception:
        pass


def _validator_import_item_mesh(item_json_path: Path) -> tuple[bool, str]:
    try:
        obj = _json.loads(item_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to read item JSON: {e}"

    verts = obj.get("vertices")
    faces = obj.get("faces")
    if not isinstance(verts, list) or not isinstance(faces, list) or not verts or not faces:
        return False, "Item JSON missing vertices/faces"

    try:
        from . import blender_tools
        _validator_clear_previous()
        col = _validator_ensure_collection("AI_VALIDATION")

        v = [tuple(map(float, p)) for p in verts]
        f = [tuple(map(int, tri)) for tri in faces]
        name = f"VAL_{obj.get('item_id', 'item')}"
        mesh_obj = blender_tools.create_mesh(name=name, verts=v, faces=f)

        # Move into validation collection only
        try:
            for c in list(mesh_obj.users_collection):
                c.objects.unlink(mesh_obj)
        except Exception:
            pass
        col.objects.link(mesh_obj)

        bpy.context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)
        try:
            mesh_obj["val_item_id"] = str(obj.get("item_id", ""))
            mesh_obj["val_cache_pt"] = str(obj.get("cache_pt", ""))
            mesh_obj["val_item_index"] = int(obj.get("item_index", -1))
            mesh_obj["val_label"] = str(obj.get("label", ""))
            mats = obj.get("material_names", [])
            if isinstance(mats, list):
                mesh_obj["val_material_names"] = ", ".join(str(m) for m in mats if str(m).strip())
            sc = obj.get("scene_context", {})
            if isinstance(sc, dict):
                mesh_obj["val_scene_keys"] = ", ".join(sorted(str(k) for k in sc.keys()))
                
            _apply_json_materials(mesh_obj, obj)
        except Exception:
            pass

        try:
            item_id = str(obj.get("item_id", "item"))
            info_name = f"AI_VALIDATION_INFO_{item_id[:16]}"
            txt = bpy.data.texts.get(info_name)
            if txt is None:
                txt = bpy.data.texts.new(info_name)
            else:
                txt.clear()
            info_payload = {k: v for k, v in obj.items() if k not in {"vertices", "faces"}}
            info_payload["vertex_count"] = int(len(verts))
            info_payload["face_count"] = int(len(faces))
            txt.write(_json.dumps(info_payload, indent=2))
        except Exception:
            pass

        _force_viewport_update()
        _set_viewport_material_preview()
        return True, ""
    except Exception as e:
        return False, f"Failed to create mesh in Blender: {e}"


def _validator_run_apply_review(*, context, queue_dir: Path, item_id: str, verdict: str, label: str, tags: str) -> tuple[bool, str]:
    prefs = context.preferences.addons[__package__].preferences

    project_root = _guess_policy_project_root(Path(prefs.policy_project_root))
    if project_root is None:
        return False, "Policy Project Root not set or invalid"

    py = Path(prefs.policy_python) if prefs.policy_python else None
    if not py or not py.exists():
        py = _default_policy_python(project_root)
    if py is None or not py.exists():
        return False, "Policy Python (venv) not found"

    script = project_root / "scripts" / "validator_apply_review.py"
    if not script.exists():
        return False, f"Missing script: {script}"

    cmd = [
        str(py),
        str(script),
        "--queue-dir", str(queue_dir),
        "--item-id", str(item_id),
        "--verdict", str(verdict),
        "--label", str(label or ""),
        "--tags", str(tags or ""),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
    except Exception as e:
        return False, str(e)

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err[-600:] if err else f"validator_apply_review failed ({proc.returncode})"

    return True, ""


def _default_policy_python(project_root: Path) -> Path | None:
    candidates = [
        project_root / ".venv" / "bin" / "python",
        project_root / "venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / "venv" / "Scripts" / "python.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _default_policy_checkpoint(project_root: Path) -> Path | None:
    candidates = [
        project_root
        / "checkpoints"
        / "policy_core_prompts_v2hash_balanced_causal"
        / "latest.pt",
        project_root / "checkpoints" / "policy_goal" / "latest.pt",
        project_root / "checkpoints" / "policy_goal" / "latest_self_improve.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Send Prompt — simple modal: generate → execute → auto-fix
# ═══════════════════════════════════════════════════════════════════════════

class AIHOUSE_OT_send_prompt(Operator):
    """Send a prompt to the AI Copilot"""

    bl_idname = "aihouse.send_prompt"
    bl_label = "Send to AI"

    # ── Class-level shared state (modal ↔ thread) ─────────────────────
    _timer = None
    _done: bool = False
    _result = None
    _error: str = ""
    _state: str = "IDLE"       # TOOL_LOOP
    _api_key: str = ""
    _model: str = ""
    _temperature: float = 0.7
    _server_url: str = ""
    _llm_url: str = ""
    _llm_model: str = ""
    _timeout: int = 180
    _stop_requested: bool = False
    _original_prompt: str = ""
    _session_gen: int = 0      # snapshot of ai_engine session generation

    def execute(self, context):
        props = context.scene.ai_copilot
        prefs = context.preferences.addons[__package__].preferences

        # Guard against double-send (e.g. rapid double-click)
        if props.is_generating:
            self.report({"WARNING"}, "Already generating — please wait")
            return {"CANCELLED"}



        prompt = props.prompt_text.strip()
        if not prompt:
            self.report({"ERROR"}, "Type a prompt first")
            return {"CANCELLED"}

        cls = AIHOUSE_OT_send_prompt
        cls._api_key = ""  # Not needed for local server
        cls._model = prefs.llm_model or "qwen2.5vl:32b"
        cls._temperature = prefs.temperature
        cls._server_url = prefs.local_server_url.rstrip("/")
        from .preferences import LLM_URL
        cls._llm_url = LLM_URL.rstrip("/")
        cls._llm_model = prefs.llm_model or "qwen2.5vl:32b"
        cls._timeout = prefs.generation_timeout
        cls._done = False
        cls._result = None
        cls._error = ""
        cls._stop_requested = False
        cls._original_prompt = prompt
        cls._state = "TOOL_LOOP"

        # Snapshot session generation — if clear_chat fires while we're
        # running, the background thread will notice and discard results.
        from . import ai_engine
        cls._session_gen = ai_engine.get_session_generation()
        print("[Blender Copilot] send_prompt: session_gen snapshot = %d" % cls._session_gen)

        props.is_generating = True
        props.last_response = ""
        props.last_code = ""
        props.status = "🤔 Thinking…"

        def worker():
            try:
                from . import ai_engine
                ai_engine.clear_streaming_text()
                sg = cls._session_gen
                summary, is_complete = ai_engine.generate_with_tools(
                    cls._llm_url, cls._llm_model, cls._temperature, prompt,
                    on_status=ai_engine._update_streaming,
                    session_gen=sg, timeout=cls._timeout,
                    mesh_server_url=cls._server_url)
                cls._result = (summary, is_complete)
            except Exception as exc:
                print("[Blender Copilot] Error:\n%s" % traceback.format_exc())
                cls._error = "%s: %s" % (type(exc).__name__, exc)
            cls._done = True

        threading.Thread(target=worker, daemon=True).start()

        cls._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        cls = AIHOUSE_OT_send_prompt
        props = context.scene.ai_copilot

        try:
            return self._modal_inner(context, props, cls)
        except Exception as exc:
            tb = traceback.format_exc()
            print("\n[Blender Copilot] ══ UNHANDLED ERROR ══\n%s\n" % tb)
            return self._finish(context,
                "❌ Internal error: %s" % str(exc)[:60], is_error=True)

    def _modal_inner(self, context, props, cls):
        from . import ai_engine
        ai_engine.process_main_thread_queue()

        # If the session was cleared while we were running, abort silently
        if ai_engine.get_session_generation() != cls._session_gen:
            print("[Blender Copilot] modal: session mismatch! "
                  "expected=%d actual=%d — aborting"
                  % (cls._session_gen, ai_engine.get_session_generation()))
            return self._finish(context, "💬 New chat — ask me anything")

        if cls._stop_requested:
            return self._finish(context, "⏹ Stopped by user")

        if not cls._done:
            # Animate status with streaming text (shows current tool call)
            streaming = ai_engine.get_streaming_text()
            if streaming:
                display = streaming[:80]
                props.status = display
            else:
                import time
                dots = "." * (int(time.time() * 2) % 4)
                base = props.status.rstrip(". ")
                for suffix in ("...", "..", "."):
                    if base.endswith(suffix):
                        base = base[:-len(suffix)].rstrip()
                        break
                props.status = base + dots
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            return {"PASS_THROUGH"}

        # Handle API errors
        if cls._error:
            return self._finish(context, "❌ %s" % cls._error[:80], is_error=True)

        # ── TOOL_LOOP completed ───────────────────────────────────
        if cls._state == "TOOL_LOOP":
            summary, is_complete = cls._result
            props.last_response = summary or "Done"
            props.last_code = ""  # no separate code in tool-calling mode

            _force_viewport_update()

            if is_complete:
                return self._finish(context,
                    "✅ %s" % (summary[:60] if summary else "Done"))
            else:
                return self._finish(context,
                    "📝 %s" % (summary[:60] if summary else "Done"))

        return {"PASS_THROUGH"}

    def _finish(self, context, status_msg, is_error=False):
        # Always remove the timer first
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None

        props = context.scene.ai_copilot
        props.is_generating = False
        props.status = status_msg

        # Clear pending image attachments (consumed by this message)
        try:
            props.reference_images.clear()
            props.active_ref_index = 0
        except Exception:
            pass

        try:
            from . import ai_engine
            ai_engine.clear_streaming_text()
            ai_engine.finalize_iteration()  # archives iteration history & saves
        except Exception:
            pass

        # Reset class state so nothing leaks into the next invocation
        cls = AIHOUSE_OT_send_prompt
        cls._state = "IDLE"
        cls._done = False
        cls._result = None
        cls._error = ""

        if is_error:
            self.report({"ERROR"}, status_msg[:200])
            return {"CANCELLED"}
        return {"FINISHED"}

    def cancel(self, context):
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        props = context.scene.ai_copilot
        props.is_generating = False
        try:
            from . import ai_engine
            ai_engine.clear_streaming_text()
        except Exception:
            pass
        # Reset class state
        cls = AIHOUSE_OT_send_prompt
        cls._state = "IDLE"
        cls._done = False
        cls._result = None
        cls._error = ""


class AIHOUSE_OT_generate_policy(Operator):
    """Generate a mesh via the architecture-compliant policy rollout.

    This runs outside Blender's Python (venv python with torch) and then
    imports the resulting OBJ into the current scene.
    """

    bl_idname = "aihouse.generate_policy"
    bl_label = "Generate (Policy)"

    _timer = None
    _done: bool = False
    _error: str = ""
    _stdout: str = ""
    _out_dir: str = ""
    _log_path: str = ""
    _session_gen: int = 0
    _stop_requested: bool = False
    _was_stopped: bool = False
    _proc = None

    def execute(self, context):
        props = context.scene.ai_copilot
        prefs = context.preferences.addons[__package__].preferences

        if props.is_generating:
            self.report({"WARNING"}, "Already generating — please wait")
            return {"CANCELLED"}

        prompt = props.prompt_text.strip()
        if not prompt:
            self.report({"ERROR"}, "Type a prompt first")
            return {"CANCELLED"}

        project_root_str = (getattr(prefs, "policy_project_root", "") or "").strip()
        project_root = Path(project_root_str).expanduser() if project_root_str else Path(__file__).resolve().parent.parent
        guessed_root = _guess_policy_project_root(project_root)
        if guessed_root is None:
            self.report(
                {"ERROR"},
                "Set 'Policy Project Root' in Preferences (or env BLENDER_COPILOT_PROJECT_ROOT)",
            )
            return {"CANCELLED"}
        project_root = guessed_root
        try:
            prefs.policy_project_root = str(project_root)
        except Exception:
            pass

        policy_py = (prefs.policy_python or "").strip()

        # Blender's FILE_PATH subtype sometimes resolves to the directory when the
        # user picks a venv folder instead of the binary — detect and fix that.
        if policy_py and Path(policy_py).is_dir():
            venv_root = Path(policy_py)
            for cand in (
                venv_root / "bin" / "python3",
                venv_root / "bin" / "python",
                venv_root / "Scripts" / "python.exe",
            ):
                if cand.exists():
                    policy_py = str(cand)
                    try:
                        prefs.policy_python = policy_py
                    except Exception:
                        pass
                    break
            else:
                policy_py = ""  # dir but no python found inside — fall through

        if not policy_py or not Path(policy_py).exists():
            auto_py = _default_policy_python(project_root)
            if auto_py is not None:
                policy_py = str(auto_py)
                try:
                    prefs.policy_python = policy_py
                except Exception:
                    pass
            else:
                self.report(
                    {"ERROR"},
                    "Set a valid 'Policy Python (venv)' in Preferences (e.g. .venv/bin/python)",
                )
                return {"CANCELLED"}

        # Final sanity: make sure we're pointing at a file, not a directory.
        if Path(policy_py).is_dir():
            self.report(
                {"ERROR"},
                f"Policy Python path is a directory, not a binary: {policy_py}",
            )
            return {"CANCELLED"}

        policy_ckpt = (prefs.policy_checkpoint or "").strip()
        if not policy_ckpt or not Path(policy_ckpt).exists():
            auto_ckpt = _default_policy_checkpoint(project_root)
            if auto_ckpt is not None:
                policy_ckpt = str(auto_ckpt)
                try:
                    prefs.policy_checkpoint = policy_ckpt
                except Exception:
                    pass
            else:
                self.report(
                    {"ERROR"},
                    "Set a valid 'Policy Checkpoint' in Preferences (e.g. checkpoints/policy_goal/latest.pt)",
                )
                return {"CANCELLED"}

        # Snapshot session generation — if clear_chat fires while we're running,
        # abort silently on completion.
        from . import ai_engine
        cls = AIHOUSE_OT_generate_policy
        cls._session_gen = ai_engine.get_session_generation()

        # Output dir in Blender temp
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = Path(getattr(bpy.app, "tempdir", "") or "/tmp")
        out_dir = base / f"blender_copilot_policy_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        cls._done = False
        cls._error = ""
        cls._stdout = ""
        cls._out_dir = str(out_dir)
        cls._log_path = str(out_dir / "rollout.log")
        cls._stop_requested = False
        cls._was_stopped = False
        cls._proc = None

        props.is_generating = True
        props.status = "🧠 Policy rollout…"

        def worker():
            try:
                repo_root = project_root
                rollout = repo_root / "scripts" / "rollout_policy_closed_loop.py"
                if not rollout.exists():
                    raise RuntimeError("rollout_policy_closed_loop.py not found under Policy Project Root")

                # Blender sets PYTHONHOME/PYTHONPATH for its embedded Python.
                # If we inherit those into the external venv Python, it can break
                # imports (torch, site-packages) or even interpreter startup.
                safe_env = os.environ.copy()
                for k in (
                    "PYTHONHOME",
                    "PYTHONPATH",
                    "PYTHONEXECUTABLE",
                    "PYTHONUSERBASE",
                ):
                    safe_env.pop(k, None)
                safe_env.setdefault("PYTHONUNBUFFERED", "1")

                cmd = [
                    str(policy_py),
                    str(rollout),
                    "--ckpt",
                    str(policy_ckpt),
                    "--out-dir",
                    str(out_dir),
                    "--steps",
                    str(int(prefs.policy_steps)),
                    "--device",
                    "auto",
                    "--seed",
                    str(int(prefs.policy_seed)),
                    "--prompt",
                    prompt,
                    "--blender",
                    str(bpy.app.binary_path),
                    "--max-verts",
                    "250000",
                    "--max-faces",
                    "250000",
                    "--temperature",
                    "1.0",
                    "--top-k",
                    "0",
                ]
                if bool(prefs.policy_apply_modifiers):
                    cmd.append("--apply-modifiers")

                import subprocess
                import signal

                def _terminate(p):
                    # On macOS/Linux, kill the whole process group so the child
                    # headless Blender doesn't linger after stopping.
                    try:
                        os.killpg(p.pid, signal.SIGTERM)
                    except Exception:
                        pass
                    try:
                        p.terminate()
                    except Exception:
                        pass
                    try:
                        p.wait(timeout=2.0)
                        return
                    except Exception:
                        pass
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    try:
                        p.kill()
                    except Exception:
                        pass

                timeout_s = int(getattr(prefs, "generation_timeout", 0) or 0)
                start_t = time.time()

                log_path = out_dir / "rollout.log"

                # Pre-write a small header so we still have breadcrumbs even if
                # subprocess spawning fails (e.g., PermissionError).
                try:
                    log_path.write_text(
                        "[Blender Copilot] rollout subprocess\n"
                        + f"cwd: {repo_root}\n"
                        + f"python: {policy_py}\n"
                        + f"ckpt: {policy_ckpt}\n"
                        + f"blender: {bpy.app.binary_path}\n"
                        + "cmd:\n  "
                        + " ".join(cmd)
                        + "\n\n",
                        encoding="utf-8",
                        errors="replace",
                    )
                except Exception:
                    pass
                cls._log_path = str(log_path)

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=safe_env,
                    start_new_session=True,
                )
                cls._proc = proc

                out = ""
                # Poll communicate in short time slices so Stop can work
                # even while the rollout is running.
                while True:
                    if cls._stop_requested:
                        _terminate(proc)
                        cls._was_stopped = True
                        try:
                            proc.communicate(timeout=0.2)
                        except Exception:
                            pass
                        return

                    if timeout_s > 0 and (time.time() - start_t) > timeout_s:
                        _terminate(proc)
                        raise TimeoutError(f"Policy rollout timed out after {timeout_s}s")

                    try:
                        out, _ = proc.communicate(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired as te:
                        partial = getattr(te, "output", None) or getattr(te, "stdout", None) or ""
                        if partial:
                            out = partial

                cls._stdout = (out or "")[-8000:]

                try:
                    # Append stdout/stderr to the header we wrote above.
                    with log_path.open("a", encoding="utf-8", errors="replace") as f:
                        f.write(out or "")
                except Exception:
                    pass

                if proc.returncode != 0:
                    if cls._stop_requested:
                        cls._was_stopped = True
                        return
                    tail = (cls._stdout or "").strip()
                    if tail:
                        print("[Blender Copilot] Policy rollout output (tail):\n%s" % tail)
                    extra = f" (see {log_path})" if log_path.exists() else ""
                    raise RuntimeError(f"Policy rollout failed (code {proc.returncode}){extra}")

                obj_path = out_dir / "mesh.obj"
                if not obj_path.exists():
                    extra = f" (see {log_path})" if log_path.exists() else ""
                    raise RuntimeError("Policy rollout finished but mesh.obj was not found" + extra)

            except Exception as exc:
                # Keep the message short for the UI but include a strong breadcrumb.
                msg = f"{type(exc).__name__}: {exc}"
                tail = (getattr(cls, "_stdout", "") or "").strip()
                if tail and len(msg) < 220:
                    msg = (msg + " | tail: " + tail.replace("\n", " "))[:260]
                cls._error = msg

                # Also persist the full traceback into the out_dir for debugging.
                try:
                    import traceback
                    (out_dir / "rollout_exception.txt").write_text(
                        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:],
                        encoding="utf-8",
                        errors="replace",
                    )
                except Exception:
                    pass
            finally:
                cls._done = True

        threading.Thread(target=worker, daemon=True).start()

        cls._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        props = context.scene.ai_copilot
        cls = AIHOUSE_OT_generate_policy

        # If the session was cleared while we were running, abort silently.
        from . import ai_engine
        if ai_engine.get_session_generation() != cls._session_gen:
            return self._finish(context, "💬 New chat — ask me anything")

        if not cls._done:
            # Keep status alive (light animation)
            dots = "." * (int(time.time() * 2) % 4)
            if cls._stop_requested:
                # Best-effort: terminate external rollout quickly
                proc = getattr(cls, "_proc", None)
                try:
                    if proc is not None and getattr(proc, "poll", None) and proc.poll() is None:
                        proc.terminate()
                except Exception:
                    pass
                props.status = "⏹ Stopping" + dots
            else:
                props.status = "🧠 Policy rollout" + dots
            return {"PASS_THROUGH"}

        if cls._was_stopped:
            return self._finish(context, "⏹ Stopped by user")

        if cls._error:
            # Surface the underlying traceback/output inside Blender.
            # Users can open the Text Editor and inspect "BlenderCopilot_rollout.log".
            log_path_str = ""
            try:
                log_path = Path(getattr(cls, "_log_path", "") or "")
                if not log_path and getattr(cls, "_out_dir", ""):
                    log_path = Path(cls._out_dir) / "rollout.log"
                if log_path and log_path.exists():
                    log_path_str = str(log_path)
                    txt = bpy.data.texts.get("BlenderCopilot_rollout.log")
                    if txt is None:
                        txt = bpy.data.texts.new("BlenderCopilot_rollout.log")
                    try:
                        txt.clear()
                    except Exception:
                        pass
                    try:
                        txt.write(log_path.read_text(encoding="utf-8", errors="replace"))
                    except Exception:
                        # Fallback to the captured tail if reading fails.
                        try:
                            txt.write(getattr(cls, "_stdout", "") or "")
                        except Exception:
                            pass
            except Exception:
                pass

            # Show the actual error (truncated) and point to where the full
            # log/traceback was written.
            err = (cls._error or "").strip()
            msg = "Policy rollout failed"
            if err:
                msg += ": " + err
            if log_path_str:
                msg += f" — log: {log_path_str}"
                exc_path = str(Path(log_path_str).with_name("rollout_exception.txt"))
                msg += f" — exc: {exc_path}"
            msg += " — Text Editor: BlenderCopilot_rollout.log"
            return self._finish(context, "❌ " + msg[:200], is_error=True)

        # Import OBJ into current scene
        obj_path = Path(cls._out_dir) / "mesh.obj"
        if not obj_path.exists():
            return self._finish(context,
                "❌ mesh.obj not found — rollout may have failed silently",
                is_error=True)

        before = set(bpy.data.objects.keys())
        import_err: str = ""
        try:
            bpy.ops.wm.obj_import(filepath=str(obj_path))
        except Exception as e1:
            import_err = str(e1)
            try:
                bpy.ops.import_scene.obj(filepath=str(obj_path))
                import_err = ""
            except Exception as e2:
                import_err = f"{e1} / {e2}"

        if import_err:
            return self._finish(context,
                "❌ OBJ import failed: %s" % import_err[:80],
                is_error=True)

        added = set(bpy.data.objects.keys()) - before
        if not added:
            return self._finish(context,
                "❌ OBJ import ran but added no objects — file may be empty",
                is_error=True)

        # Deselect all, then select and focus the new mesh.
        bpy.ops.object.select_all(action="DESELECT")
        for name in added:
            obj = bpy.data.objects.get(name)
            if obj:
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
        _force_viewport_update()

        return self._finish(context, "✅ Policy mesh imported (%d object(s))" % len(added))

    def _finish(self, context, status_msg, is_error: bool = False):
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None

        props = context.scene.ai_copilot
        props.is_generating = False
        props.status = status_msg
        if is_error:
            self.report({"ERROR"}, status_msg[:200])
            return {"CANCELLED"}
        self.report({"INFO"}, status_msg[:200])
        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Direct Mesh Generate (bypass LLM, hit mesh server immediately)
# ═══════════════════════════════════════════════════════════════════════════

class AIHOUSE_OT_generate_direct(Operator):
    """Generate a mesh directly from the prompt — skips the LLM and posts the
    prompt straight to the mesh server.  Useful when ollama is slow or missing."""

    bl_idname = "aihouse.generate_direct"
    bl_label = "Generate Direct"

    _timer = None
    _done: bool = False
    _result: str = ""
    _error: str = ""

    def execute(self, context):
        props = context.scene.ai_copilot
        prefs = context.preferences.addons[__package__].preferences

        if props.is_generating:
            self.report({"WARNING"}, "Already generating — please wait")
            return {"CANCELLED"}

        prompt = props.prompt_text.strip()
        if not prompt:
            self.report({"ERROR"}, "Type a prompt first")
            return {"CANCELLED"}

        cls = AIHOUSE_OT_generate_direct
        cls._done = False
        cls._result = ""
        cls._error = ""

        server_url = prefs.local_server_url.rstrip("/")
        temperature = prefs.temperature

        props.is_generating = True
        props.status = "⚡ Generating mesh directly…"

        def worker():
            try:
                from . import ai_engine
                mesh_data = ai_engine._generate_mesh_local(
                    server_url, prompt,
                    temperature=temperature,
                    timeout=120,
                    max_faces=400,
                )
                print("[Direct Generate] server response keys:", list(mesh_data.keys()))
                objs = mesh_data.get("objects", [])
                print("[Direct Generate] objects returned:", len(objs))
                if objs:
                    print("[Direct Generate] first obj verts:", len(objs[0].get("mesh", {}).get("vertices", [])))
                if not objs or not objs[0].get("mesh", {}).get("vertices"):
                    raise RuntimeError("Server returned 0 vertices. Response: %s" % str(mesh_data)[:300])

                # Persist feedback context for rating/compare tools.
                try:
                    gen_tokens = mesh_data.get("tokens", [])
                    if gen_tokens:
                        props.last_generation_tokens = json.dumps(gen_tokens)
                        props.compare_prompt = prompt
                except Exception:
                    pass

                cls._result = ai_engine._mesh_to_code(mesh_data)
            except Exception as exc:
                cls._error = "%s: %s" % (type(exc).__name__, exc)
                print("[Direct Generate] Error:\n%s" % traceback.format_exc())
            cls._done = True

        threading.Thread(target=worker, daemon=True).start()

        cls._timer = context.window_manager.event_timer_add(0.25, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        cls = AIHOUSE_OT_generate_direct
        props = context.scene.ai_copilot

        if not cls._done:
            return {"PASS_THROUGH"}

        context.window_manager.event_timer_remove(cls._timer)
        cls._timer = None
        props.is_generating = False

        if cls._error:
            props.status = "Direct generate error: %s" % cls._error[:100]
            self.report({"ERROR"}, cls._error[:200])
            return {"CANCELLED"}

        code = cls._result or ""
        if not code or code.strip() == "# No mesh generated":
            props.status = "Server returned empty mesh — server running? Check port 8420"
            self.report({"WARNING"}, "Direct generate: mesh server returned no geometry. Is 'python run.py serve' running?")
            return {"CANCELLED"}

        props.last_code = code
        ok, err = _execute_code(code)
        if ok:
            props.status = "✅ Mesh generated (direct)"
            self.report({"INFO"}, "Direct mesh generated successfully")
        else:
            props.status = "Code error: %s" % err[:120]
            self.report({"ERROR"}, "Mesh code error: %s" % err[:200])
            return {"CANCELLED"}

        return {"FINISHED"}

    def cancel(self, context):
        cls = AIHOUSE_OT_generate_direct
        if cls._timer:
            context.window_manager.event_timer_remove(cls._timer)
            cls._timer = None
        context.scene.ai_copilot.is_generating = False


# ═══════════════════════════════════════════════════════════════════════════
# Execute Code (manual)
# ═══════════════════════════════════════════════════════════════════════════

class AIHOUSE_OT_execute_code(Operator):
    """Run the last generated Python code in Blender"""

    bl_idname = "aihouse.execute_code"
    bl_label = "Execute Code"

    def execute(self, context):
        props = context.scene.ai_copilot
        code = props.last_code
        if not code:
            self.report({"WARNING"}, "No code to execute")
            return {"CANCELLED"}

        bpy.ops.ed.undo_push(message="Blender Copilot")
        success, err_msg = _execute_code(code)
        if success:
            props.status = "✅ Code executed"
            self.report({"INFO"}, "Code executed successfully")
        else:
            props.status = "❌ " + err_msg[:70]
            self.report({"ERROR"}, err_msg[:200])
        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Stop Generation
# ═══════════════════════════════════════════════════════════════════════════

class AIHOUSE_OT_stop_generation(Operator):
    """Stop the current AI generation"""

    bl_idname = "aihouse.stop_generation"
    bl_label = "Stop"

    def execute(self, context):
        AIHOUSE_OT_send_prompt._stop_requested = True
        AIHOUSE_OT_generate_policy._stop_requested = True
        proc = getattr(AIHOUSE_OT_generate_policy, "_proc", None)
        try:
            if proc is not None and getattr(proc, "poll", None) and proc.poll() is None:
                proc.terminate()
        except Exception:
            pass
        context.scene.ai_copilot.status = "⏹ Stopping…"
        self.report({"INFO"}, "Stop requested")
        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

class AIHOUSE_OT_clear_scene(Operator):
    """Remove all objects from the scene"""

    bl_idname = "aihouse.clear_scene"
    bl_label = "Clear Scene"

    def execute(self, context):
        bpy.ops.ed.undo_push(message="Clear Scene")
        from . import blender_tools
        blender_tools.clear_scene()
        context.scene.ai_copilot.status = "🗑️ Scene cleared"
        self.report({"INFO"}, "Scene cleared")
        return {"FINISHED"}


class AIHOUSE_OT_clear_chat(Operator):
    """Clear the AI conversation history"""

    bl_idname = "aihouse.clear_chat"
    bl_label = "New Chat"

    def execute(self, context):
        # 1) Force-stop any ongoing generation
        AIHOUSE_OT_send_prompt._stop_requested = True
        AIHOUSE_OT_send_prompt._done = True
        AIHOUSE_OT_send_prompt._state = "IDLE"

        AIHOUSE_OT_generate_policy._stop_requested = True
        proc = getattr(AIHOUSE_OT_generate_policy, "_proc", None)
        try:
            if proc is not None and getattr(proc, "poll", None) and proc.poll() is None:
                proc.terminate()
        except Exception:
            pass

        # 2) Clear AI conversation (this also bumps the session
        #    generation counter, so any in-flight background threads
        #    will have their writes silently discarded).
        from . import ai_engine
        ai_engine.clear_history()        # bumps _session_generation
        ai_engine.clear_streaming_text()

        # 3) Reset all scene-level UI state
        #    Scene objects are NOT deleted — the AI will re-scan the
        #    scene with get_scene_context() on the next prompt.
        props = context.scene.ai_copilot
        props.is_generating = False
        props.last_response = ""
        props.last_code = ""
        props.reference_images.clear()
        props.active_ref_index = 0
        props.status = "💬 New chat — ask me anything"
        self.report({"INFO"}, "Chat cleared")
        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Server Testing Operator
# ═══════════════════════════════════════════════════════════════════════════

class AIHOUSE_OT_test_local_server(Operator):
    """Test connection to the AI inference server"""

    bl_idname = "aihouse.test_local_server"
    bl_label = "Test Server Connection"

    def execute(self, context):
        import json
        import urllib.request
        prefs = context.preferences.addons[__package__].preferences
        url = prefs.local_server_url.rstrip("/")
        try:
            req = urllib.request.Request(f"{url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    params = data.get("model_params", 0)
                    device = data.get("device", "?")
                    self.report({"INFO"},
                                f"Connected! Model: {params:,} params on {device}")
                else:
                    self.report({"WARNING"}, f"Unexpected response: {data}")
        except Exception as exc:
            self.report({"ERROR"},
                        f"Cannot connect to {url}: {str(exc)[:120]}")
        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Reference Image Operators
# ═══════════════════════════════════════════════════════════════════════════

class AIHOUSE_OT_add_reference_image(Operator):
    """Browse for a reference image to include in the AI context"""

    bl_idname = "aihouse.add_ref_image"
    bl_label = "Add Reference Image"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')  # type: ignore
    filter_glob: bpy.props.StringProperty(  # type: ignore
        default="*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.gif",
        options={'HIDDEN'},
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        props = context.scene.ai_copilot
        if not self.filepath:
            self.report({"WARNING"}, "No file selected")
            return {"CANCELLED"}

        import os
        if not os.path.exists(self.filepath):
            self.report({"ERROR"}, "File not found: %s" % self.filepath)
            return {"CANCELLED"}

        # Check for duplicate
        for ref in props.reference_images:
            if ref.filepath == self.filepath:
                self.report({"INFO"}, "Image already added")
                return {"CANCELLED"}

        item = props.reference_images.add()
        item.filepath = self.filepath
        props.active_ref_index = len(props.reference_images) - 1
        self.report({"INFO"}, "Reference image added: %s" % os.path.basename(self.filepath))
        return {"FINISHED"}


class AIHOUSE_OT_remove_reference_image(Operator):
    """Remove the selected reference image"""

    bl_idname = "aihouse.remove_ref_image"
    bl_label = "Remove Reference Image"

    index: bpy.props.IntProperty(default=-1)  # type: ignore

    def execute(self, context):
        props = context.scene.ai_copilot
        idx = self.index if self.index >= 0 else props.active_ref_index
        if 0 <= idx < len(props.reference_images):
            props.reference_images.remove(idx)
            if props.active_ref_index >= len(props.reference_images):
                props.active_ref_index = max(0, len(props.reference_images) - 1)
            self.report({"INFO"}, "Reference image removed")
        return {"FINISHED"}


class AIHOUSE_OT_clear_reference_images(Operator):
    """Remove all reference images"""

    bl_idname = "aihouse.clear_ref_images"
    bl_label = "Clear All References"

    def execute(self, context):
        props = context.scene.ai_copilot
        props.reference_images.clear()
        props.active_ref_index = 0
        self.report({"INFO"}, "All reference images cleared")
        return {"FINISHED"}


class AIHOUSE_OT_search_reference_images(Operator):
    """Search the web for reference images based on the current prompt"""

    bl_idname = "aihouse.search_ref_images"
    bl_label = "Search References"

    query: bpy.props.StringProperty(default="")  # type: ignore

    def execute(self, context):
        import threading

        props = context.scene.ai_copilot
        query = self.query or props.prompt_text.strip()
        if not query:
            self.report({"WARNING"}, "Type a prompt first for image search")
            return {"CANCELLED"}

        props.status = "🔍 Searching for reference images…"

        def _search():
            try:
                from . import ai_engine
                paths = ai_engine.search_reference_images(query, max_results=3)
                # Schedule adding to UI on main thread
                def _add_refs():
                    for path in paths:
                        # Avoid duplicates
                        exists = False
                        for ref in props.reference_images:
                            if ref.filepath == path:
                                exists = True
                                break
                        if not exists:
                            item = props.reference_images.add()
                            item.filepath = path
                    props.active_ref_index = max(0, len(props.reference_images) - 1)
                    props.status = "✅ Found %d reference images" % len(paths)
                    # Redraw
                    for area in bpy.context.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
                ai_engine._main_thread_queue.put(_add_refs)
            except Exception as exc:
                err_msg = str(exc)[:50]
                def _report_err(msg=err_msg):
                    props.status = "⚠️ Image search failed: %s" % msg
                from . import ai_engine
                ai_engine._main_thread_queue.put(_report_err)

        threading.Thread(target=_search, daemon=True).start()
        return {"FINISHED"}


class AIHOUSE_OT_drop_reference_image(Operator):
    """Handle a dropped image file as a reference image"""

    bl_idname = "aihouse.drop_ref_image"
    bl_label = "Drop Reference Image"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')  # type: ignore

    def execute(self, context):
        props = context.scene.ai_copilot
        if not self.filepath:
            return {"CANCELLED"}

        import os
        if not os.path.exists(self.filepath):
            self.report({"WARNING"}, "File not found")
            return {"CANCELLED"}

        # Check for duplicate
        for ref in props.reference_images:
            if ref.filepath == self.filepath:
                self.report({"INFO"}, "Image already added")
                return {"FINISHED"}

        item = props.reference_images.add()
        item.filepath = self.filepath
        props.active_ref_index = len(props.reference_images) - 1
        self.report({"INFO"}, "Reference image added: %s" % os.path.basename(self.filepath))
        return {"FINISHED"}


class AIHOUSE_OT_open_ref_image(Operator):
    """Open a reference image with the system viewer"""

    bl_idname = "aihouse.open_ref_image"
    bl_label = "Open Reference Image"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')  # type: ignore

    def execute(self, context):
        import os
        import subprocess
        import sys

        if not self.filepath or not os.path.exists(self.filepath):
            self.report({"WARNING"}, "File not found")
            return {"CANCELLED"}

        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", self.filepath])
            elif sys.platform == "win32":
                os.startfile(self.filepath)  # type: ignore
            else:
                subprocess.Popen(["xdg-open", self.filepath])
        except Exception as exc:
            self.report({"WARNING"}, "Could not open image: %s" % str(exc)[:80])
            return {"CANCELLED"}

        return {"FINISHED"}


# Blender 4.0+ FileHandler for drag-and-drop image support
class AIHOUSE_FH_drop_image(bpy.types.FileHandler):
    """Accept image files dropped onto the 3D viewport as reference images"""

    bl_idname = "AIHOUSE_FH_drop_image"
    bl_label = "AI Copilot Reference Image"
    bl_import_operator = "aihouse.drop_ref_image"
    bl_file_extensions = ".png;.jpg;.jpeg;.webp;.bmp;.gif;.tiff"

    @classmethod
    def poll_drop(cls, context):
        return context.area and context.area.type == 'VIEW_3D'


# ═══════════════════════════════════════════════════════════════════════════
# RLHF Feedback Operators — A/B Comparison & Rating
# ═══════════════════════════════════════════════════════════════════════════

def _send_feedback_to_server(server_url: str, endpoint: str, payload: dict) -> dict:
    """Send feedback data to the inference server."""
    import json
    import urllib.request
    import ssl

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}


class AIHOUSE_OT_start_comparison(Operator):
    """Generate two options and compare them side by side"""

    bl_idname = "aihouse.start_comparison"
    bl_label = "Compare Outputs"

    _timer = None
    _done = False
    _result = None
    _error = ""

    def execute(self, context):
        props = context.scene.ai_copilot
        prefs = context.preferences.addons[__package__].preferences

        prompt = props.prompt_text.strip()
        if not prompt:
            self.report({"ERROR"}, "Type a prompt first")
            return {"CANCELLED"}

        if not prefs.local_server_url:
            self.report({"ERROR"}, "Set server URL in preferences")
            return {"CANCELLED"}

        props.is_comparing = True
        props.compare_prompt = prompt
        props.compare_choice = "NONE"
        props.feedback_status = "Generating comparison options..."

        cls = AIHOUSE_OT_start_comparison
        cls._done = False
        cls._result = None
        cls._error = ""

        server_url = prefs.local_server_url.rstrip("/")

        def worker():
            try:
                result = _send_feedback_to_server(
                    server_url, "feedback/compare",
                    {"prompt": prompt, "n_candidates": 4})
                cls._result = result
            except Exception as e:
                cls._error = str(e)
            cls._done = True

        import threading
        threading.Thread(target=worker, daemon=True).start()

        self._timer = context.window_manager.event_timer_add(
            0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        cls = AIHOUSE_OT_start_comparison
        props = context.scene.ai_copilot

        if not cls._done:
            return {"PASS_THROUGH"}

        # Remove timer
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None

        if cls._error:
            props.is_comparing = False
            props.feedback_status = "Comparison failed: %s" % cls._error[:60]
            return {"CANCELLED"}

        result = cls._result
        if not result or "error" in result:
            props.is_comparing = False
            props.feedback_status = "Failed: %s" % (result or {}).get("error", "unknown")[:60]
            return {"CANCELLED"}

        # Store the comparison data globally for the submit operator
        AIHOUSE_OT_start_comparison._comparison_data = result

        # Create mesh objects for both options
        try:
            from . import blender_tools
            bpy.ops.ed.undo_push(message="AI Comparison")

            option_a = result.get("option_a", {})
            option_b = result.get("option_b", {})

            # Create Option A mesh (left)
            verts_a = option_a.get("vertices", [])
            faces_a = option_a.get("faces", [])
            if verts_a and faces_a:
                blender_tools.create_mesh(
                    name="Compare_A", verts=verts_a,
                    faces=faces_a, location=(-1.5, 0, 0))

            # Create Option B mesh (right)
            verts_b = option_b.get("vertices", [])
            faces_b = option_b.get("faces", [])
            if verts_b and faces_b:
                blender_tools.create_mesh(
                    name="Compare_B", verts=verts_b,
                    faces=faces_b, location=(1.5, 0, 0))

            _force_viewport_update()
            props.feedback_status = "Choose: A (left) or B (right)"

        except Exception as e:
            props.feedback_status = "Display error: %s" % str(e)[:40]

        return {"FINISHED"}


class AIHOUSE_OT_submit_comparison(Operator):
    """Submit your A vs B preference choice"""

    bl_idname = "aihouse.submit_comparison"
    bl_label = "Submit Preference"

    choice: bpy.props.StringProperty(default="A")  # type: ignore

    def execute(self, context):
        props = context.scene.ai_copilot
        prefs = context.preferences.addons[__package__].preferences

        if not props.is_comparing:
            self.report({"WARNING"}, "No active comparison")
            return {"CANCELLED"}

        comparison_data = getattr(
            AIHOUSE_OT_start_comparison, "_comparison_data", None)
        if not comparison_data:
            self.report({"ERROR"}, "No comparison data")
            return {"CANCELLED"}

        option_a = comparison_data.get("option_a", {})
        option_b = comparison_data.get("option_b", {})
        prompt = props.compare_prompt

        server_url = prefs.local_server_url.rstrip("/")

        if self.choice == "A":
            chosen = option_a.get("tokens", [])
            rejected = option_b.get("tokens", [])
        elif self.choice == "B":
            chosen = option_b.get("tokens", [])
            rejected = option_a.get("tokens", [])
        else:
            # Tie — submit both as accept
            _send_feedback_to_server(server_url, "feedback/accept",
                                     {"prompt": prompt, "tokens": option_a.get("tokens", [])})
            _send_feedback_to_server(server_url, "feedback/accept",
                                     {"prompt": prompt, "tokens": option_b.get("tokens", [])})
            props.is_comparing = False
            props.feedback_count += 1
            props.feedback_status = "Tie recorded (both accepted)"
            self._cleanup_comparison(context)
            return {"FINISHED"}

        # Submit pairwise preference
        result = _send_feedback_to_server(server_url, "feedback/pairwise", {
            "prompt": prompt,
            "chosen_tokens": chosen,
            "rejected_tokens": rejected,
            "metadata": {"choice": self.choice},
        })

        if not result or result.get("status") != "ok":
            props.is_comparing = False
            props.feedback_status = "Feedback failed: %s" % (result or {}).get("error", "unknown")[:60]
            self._cleanup_comparison(context)
            self.report({"WARNING"}, props.feedback_status)
            return {"CANCELLED"}

        props.is_comparing = False
        props.feedback_count += 1

        stats = result.get("feedback_stats", {})
        total = stats.get("total_feedback_ever", 0)
        props.feedback_status = "Preference recorded (%d total)" % total

        self._cleanup_comparison(context)
        self.report({"INFO"}, "Preference submitted: Option %s" % self.choice)
        return {"FINISHED"}

    def _cleanup_comparison(self, context):
        """Remove comparison objects from scene."""
        try:
            for name in ("Compare_A", "Compare_B"):
                obj = bpy.data.objects.get(name)
                if obj:
                    bpy.data.objects.remove(obj, do_unlink=True)
            _force_viewport_update()
        except Exception:
            pass


class AIHOUSE_OT_regenerate_comparison(Operator):
    """Neither option is good — regenerate both A and B"""

    bl_idname = "aihouse.regenerate_comparison"
    bl_label = "Regenerate Comparison"

    def execute(self, context):
        props = context.scene.ai_copilot

        if not props.is_comparing:
            self.report({"WARNING"}, "No active comparison")
            return {"CANCELLED"}

        # Submit both as rejected so the reward model learns
        comparison_data = getattr(
            AIHOUSE_OT_start_comparison, "_comparison_data", None)
        if comparison_data:
            prefs = context.preferences.addons[__package__].preferences
            server_url = prefs.local_server_url.rstrip("/")
            prompt = props.compare_prompt
            for key in ("option_a", "option_b"):
                tokens = comparison_data.get(key, {}).get("tokens", [])
                if tokens:
                    _send_feedback_to_server(
                        server_url, "feedback/reject",
                        {"prompt": prompt, "tokens": tokens,
                         "metadata": {"reason": "regenerated"}})
            props.feedback_count += 1

        # Clean up old meshes
        try:
            for name in ("Compare_A", "Compare_B"):
                obj = bpy.data.objects.get(name)
                if obj:
                    bpy.data.objects.remove(obj, do_unlink=True)
            _force_viewport_update()
        except Exception:
            pass

        # Reset state and re-trigger comparison
        props.is_comparing = False
        props.feedback_status = "Regenerating..."

        # Re-invoke the start_comparison operator
        bpy.ops.aihouse.start_comparison()
        return {"FINISHED"}


class AIHOUSE_OT_cancel_comparison(Operator):
    """Cancel the current A/B comparison without submitting"""

    bl_idname = "aihouse.cancel_comparison"
    bl_label = "Cancel Comparison"

    def execute(self, context):
        props = context.scene.ai_copilot
        props.is_comparing = False
        props.feedback_status = "Comparison cancelled"

        # Remove comparison objects
        try:
            for name in ("Compare_A", "Compare_B"):
                obj = bpy.data.objects.get(name)
                if obj:
                    bpy.data.objects.remove(obj, do_unlink=True)
            _force_viewport_update()
        except Exception:
            pass

        return {"FINISHED"}


class AIHOUSE_OT_accept_output(Operator):
    """Accept the current output as good (positive feedback)"""

    bl_idname = "aihouse.accept_output"
    bl_label = "Accept"

    def execute(self, context):
        props = context.scene.ai_copilot
        prefs = context.preferences.addons[__package__].preferences

        if not prefs.local_server_url:
            self.report({"WARNING"}, "Server URL not set")
            return {"CANCELLED"}

        # Get the last generation tokens
        import json
        tokens = []
        if props.last_generation_tokens:
            try:
                tokens = json.loads(props.last_generation_tokens)
            except Exception:
                pass

        if not tokens:
            self.report({"WARNING"}, "No generation to rate")
            return {"CANCELLED"}

        prompt = props.compare_prompt or props.prompt_text.strip()
        server_url = prefs.local_server_url.rstrip("/")

        result = _send_feedback_to_server(server_url, "feedback/accept", {
            "prompt": prompt, "tokens": tokens,
        })

        if not result or result.get("status") != "ok":
            props.feedback_status = "Accept failed: %s" % (result or {}).get("error", "unknown")[:60]
            self.report({"WARNING"}, props.feedback_status)
            return {"CANCELLED"}

        props.feedback_count += 1
        props.last_generation_tokens = ""
        total = result.get("feedback_stats", {}).get("total_feedback_ever", 0)
        props.feedback_status = "Accepted (%d total)" % total
        self.report({"INFO"}, "Positive feedback submitted")
        return {"FINISHED"}


class AIHOUSE_OT_reject_output(Operator):
    """Reject the current output as bad (negative feedback)"""

    bl_idname = "aihouse.reject_output"
    bl_label = "Reject"

    def execute(self, context):
        props = context.scene.ai_copilot
        prefs = context.preferences.addons[__package__].preferences

        if not prefs.local_server_url:
            self.report({"WARNING"}, "Server URL not set")
            return {"CANCELLED"}

        import json
        tokens = []
        if props.last_generation_tokens:
            try:
                tokens = json.loads(props.last_generation_tokens)
            except Exception:
                pass

        if not tokens:
            self.report({"WARNING"}, "No generation to rate")
            return {"CANCELLED"}

        prompt = props.compare_prompt or props.prompt_text.strip()
        server_url = prefs.local_server_url.rstrip("/")

        result = _send_feedback_to_server(server_url, "feedback/reject", {
            "prompt": prompt, "tokens": tokens,
        })

        if not result or result.get("status") != "ok":
            props.feedback_status = "Reject failed: %s" % (result or {}).get("error", "unknown")[:60]
            self.report({"WARNING"}, props.feedback_status)
            return {"CANCELLED"}

        props.feedback_count += 1
        props.last_generation_tokens = ""
        total = result.get("feedback_stats", {}).get("total_feedback_ever", 0)
        props.feedback_status = "Rejected (%d total)" % total
        self.report({"INFO"}, "Negative feedback submitted")
        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Training Data Loop — random prompt → generate → approve/reject → repeat
# ═══════════════════════════════════════════════════════════════════════════

# Diverse prompt pool — ~200 items across many categories
_TRAINING_PROMPTS = [
    # ── Furniture ─────────────────────────────────────────────────
    "Create a simple wooden table",
    "Create a dining table with 6 legs",
    "Create a round coffee table",
    "Create a glass coffee table",
    "Create a wooden chair",
    "Create a modern office chair",
    "Create a bar stool",
    "Create a rocking chair",
    "Create a park bench",
    "Create a garden bench",
    "Create a wooden bookshelf",
    "Create a tall narrow bookshelf",
    "Create a TV stand",
    "Create a bedside nightstand",
    "Create a single bed",
    "Create a bunk bed",
    "Create a queen size bed with headboard",
    "Create a wardrobe",
    "Create a chest of drawers",
    "Create a shoe rack",
    "Create a coat rack",
    "Create a desk with drawers",
    "Create a standing desk",
    "Create a computer desk with monitor shelf",
    "Create a kitchen island",
    "Create a kitchen cabinet",
    "Create a dining room hutch",
    "Create a wine rack",
    "Create a floating shelf",
    "Create a corner shelf unit",
    "Create a toy chest",
    "Create a hope chest",
    "Create a piano bench",
    "Create a vanity table with mirror",
    "Create a console table",
    # ── Seating ───────────────────────────────────────────────────
    "Create a bean bag chair",
    "Create an armchair",
    "Create a loveseat sofa",
    "Create a 3-seat sofa",
    "Create an L-shaped sectional sofa",
    "Create a throne chair",
    "Create a hammock",
    "Create a swing seat",
    "Create a patio chair",
    "Create a folding chair",
    # ── Lighting ──────────────────────────────────────────────────
    "Create a desk lamp",
    "Create a floor lamp",
    "Create a table lamp with shade",
    "Create a pendant ceiling light",
    "Create a chandelier",
    "Create a wall sconce",
    "Create a lantern",
    "Create a candelabra",
    "Create a lava lamp",
    "Create a spotlight",
    "Create a street lamp",
    "Create a neon sign",
    # ── Kitchen / Dining ──────────────────────────────────────────
    "Create a coffee mug",
    "Create a teapot",
    "Create a wine glass",
    "Create a beer bottle",
    "Create a water pitcher",
    "Create a dinner plate",
    "Create a bowl",
    "Create a cutting board",
    "Create a rolling pin",
    "Create a frying pan",
    "Create a cooking pot with lid",
    "Create a blender appliance",
    "Create a toaster",
    "Create a salt and pepper shaker set",
    "Create a cake stand",
    "Create a fruit bowl",
    "Create a knife block",
    # ── Architecture ──────────────────────────────────────────────
    "Create a simple house",
    "Create a two-story house",
    "Create a log cabin",
    "Create a church with steeple",
    "Create a lighthouse",
    "Create a medieval tower",
    "Create a castle turret",
    "Create a skyscraper",
    "Create a barn",
    "Create a gazebo",
    "Create a bridge",
    "Create an arch gateway",
    "Create a well",
    "Create a windmill",
    "Create a water tower",
    "Create a dog house",
    "Create a bird house",
    "Create a treehouse",
    "Create a pyramid",
    "Create a dome building",
    "Create stairs with railing",
    "Create a fence section",
    "Create a brick wall",
    "Create a window frame",
    "Create a door frame with door",
    "Create a chimney",
    "Create a balcony with railing",
    "Create a fire escape",
    # ── Nature / Organic ──────────────────────────────────────────
    "Create a tree",
    "Create a pine tree",
    "Create a palm tree",
    "Create a cactus",
    "Create a mushroom",
    "Create a flower in a pot",
    "Create a sunflower",
    "Create a bush",
    "Create a rock",
    "Create a boulder pile",
    "Create a mountain",
    "Create a snowman",
    "Create a pumpkin",
    "Create an apple",
    "Create a banana",
    "Create a watermelon slice",
    "Create a log",
    "Create a tree stump",
    "Create a leaf",
    # ── Vehicles ──────────────────────────────────────────────────
    "Create a simple car",
    "Create a pickup truck",
    "Create a school bus",
    "Create a train locomotive",
    "Create a sailboat",
    "Create a rowboat",
    "Create a canoe",
    "Create a hot air balloon",
    "Create a helicopter",
    "Create a rocket ship",
    "Create a bicycle",
    "Create a skateboard",
    "Create a shopping cart",
    "Create a wheelbarrow",
    "Create a wagon",
    # ── Tools / Objects ───────────────────────────────────────────
    "Create a hammer",
    "Create a screwdriver",
    "Create a wrench",
    "Create a saw",
    "Create an axe",
    "Create a shovel",
    "Create a broom",
    "Create a bucket",
    "Create a watering can",
    "Create a ladder",
    "Create a toolbox",
    "Create a paint roller",
    # ── Electronics / Tech ────────────────────────────────────────
    "Create a computer monitor",
    "Create a laptop",
    "Create a smartphone",
    "Create a keyboard",
    "Create a computer mouse",
    "Create a speaker",
    "Create headphones",
    "Create a camera",
    "Create a microphone",
    "Create a game controller",
    "Create a TV with stand",
    "Create a radio",
    # ── Sports / Recreation ───────────────────────────────────────
    "Create a soccer ball",
    "Create a basketball hoop",
    "Create a baseball bat",
    "Create a tennis racket",
    "Create a bowling pin",
    "Create a dumbbell",
    "Create a barbell with weights",
    "Create a trophy",
    "Create a chess piece king",
    "Create a chess piece pawn",
    "Create a dart board",
    "Create a pool table",
    # ── Decorative / Art ──────────────────────────────────────────
    "Create a picture frame",
    "Create a mirror with frame",
    "Create a clock",
    "Create a grandfather clock",
    "Create a vase",
    "Create a candle",
    "Create a candle holder",
    "Create a sculpture pedestal",
    "Create a globe on a stand",
    "Create a snow globe",
    "Create a hourglass",
    "Create a treasure chest",
    "Create a jewelry box",
    "Create a photo album stand",
    "Create a wind chime",
    # ── Miscellaneous ─────────────────────────────────────────────
    "Create a mailbox",
    "Create a fire hydrant",
    "Create a trash can",
    "Create a recycling bin",
    "Create a traffic cone",
    "Create a stop sign",
    "Create a flag pole with flag",
    "Create a tent",
    "Create an umbrella",
    "Create a suitcase",
    "Create a backpack",
    "Create a gift box with bow",
    "Create a crown",
    "Create a top hat",
    "Create a wizard hat",
    "Create a sword",
    "Create a shield",
    "Create a key",
    "Create a padlock",
    "Create a bell",
    "Create a megaphone",
    "Create a telescope",
    "Create a binoculars",
    "Create an hourglass",
    "Create a compass",
    "Create a life preserver ring",
    "Create an anchor",
    "Create a barrel",
    "Create a crate",
    "Create a wooden sign post",
]

# Track which prompts have been used so we don't repeat before exhausting pool
_training_used_prompts: list = []
_training_log_path = ""


def _pick_random_prompt() -> str:
    """Pick a random prompt from the pool, avoiding repeats until all used."""
    global _training_used_prompts
    available = [p for p in _TRAINING_PROMPTS if p not in _training_used_prompts]
    if not available:
        _training_used_prompts.clear()
        available = list(_TRAINING_PROMPTS)
    choice = random.choice(available)
    _training_used_prompts.append(choice)
    return choice


def _save_training_record(prompt: str, feedback: str, scene_info: str):
    """Save a training feedback record to the JSONL log."""
    global _training_log_path
    if not _training_log_path:
        log_dir = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "data", "training_feedback")
        _os.makedirs(log_dir, exist_ok=True)
        stamp = _datetime.now().strftime("%Y%m%d_%H%M%S")
        _training_log_path = _os.path.join(
            log_dir, "feedback_%s.jsonl" % stamp)

    record = {
        "timestamp": _datetime.now().isoformat(),
        "prompt": prompt,
        "feedback": feedback,
        "scene_info": scene_info,
    }
    try:
        with open(_training_log_path, "a") as f:
            f.write(_json.dumps(record) + "\n")
    except Exception as e:
        print("[Training] Failed to save record: %s" % e)


def _get_scene_summary() -> str:
    """Get a quick summary of current scene objects for the training record."""
    try:
        objs = []
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                mats = [s.material.name for s in obj.material_slots
                        if s.material] if obj.material_slots else []
                objs.append({
                    "name": obj.name,
                    "verts": len(obj.data.vertices),
                    "faces": len(obj.data.polygons),
                    "location": [round(v, 3) for v in obj.location],
                    "dimensions": [round(v, 3) for v in obj.dimensions],
                    "materials": mats,
                })
        return _json.dumps(objs)
    except Exception:
        return "[]"


class AIHOUSE_OT_training_start(Operator):
    """Start the training data collection loop"""

    bl_idname = "aihouse.training_start"
    bl_label = "Start Training Loop"

    _timer = None

    def execute(self, context):
        props = context.scene.ai_copilot

        if props.training_active:
            self.report({"WARNING"}, "Training loop already active")
            return {"CANCELLED"}

        if props.is_generating:
            self.report({"WARNING"}, "Wait for current generation to finish")
            return {"CANCELLED"}

        global _training_log_path
        _training_log_path = ""

        props.training_active = True
        props.training_awaiting = False
        props.training_approved = 0
        props.training_rejected = 0
        props.training_skipped = 0
        props.training_total = 0

        self._timer = context.window_manager.event_timer_add(
            0.5, window=context.window)
        context.window_manager.modal_handler_add(self)

        self._trigger_next_prompt(context)

        self.report({"INFO"}, "Training loop started")
        return {"RUNNING_MODAL"}

    def _trigger_next_prompt(self, context):
        """Clear scene, pick a new prompt, send it."""
        props = context.scene.ai_copilot

        # Guard: don't send if already generating (race prevention)
        if props.is_generating:
            print("[Training] _trigger_next_prompt: skipping — already generating")
            return

        from . import blender_tools, ai_engine
        blender_tools.clear_scene()
        ai_engine.clear_history()
        ai_engine.clear_streaming_text()

        prompt = _pick_random_prompt()
        props.training_prompt = prompt
        props.training_awaiting = False
        props.prompt_text = prompt

        print("[Training] _trigger_next_prompt: sending '%s'" % prompt[:60])
        bpy.ops.aihouse.send_prompt()

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        props = context.scene.ai_copilot

        if not props.training_active:
            return self._finish(context)

        if props.training_awaiting:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            return {"PASS_THROUGH"}

        if props.is_generating:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            return {"PASS_THROUGH"}

        if not props.training_awaiting and not props.is_generating:
            if props.training_prompt:
                props.training_awaiting = True
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()

        return {"PASS_THROUGH"}

    def _finish(self, context):
        print("[Training] training_start._finish called")
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None

        props = context.scene.ai_copilot

        # Stop any running generation FIRST — this prevents clear_history
        # from racing with an active background thread.
        if props.is_generating:
            AIHOUSE_OT_send_prompt._stop_requested = True
            print("[Training] _finish: stopped active generation")

        props.training_active = False
        props.training_awaiting = False
        # Clean up stale state
        from . import ai_engine
        ai_engine.flush_main_thread_queue()
        ai_engine.clear_history()
        ai_engine.clear_streaming_text()
        return {"FINISHED"}

    def cancel(self, context):
        print("[Training] training_start.cancel called by Blender")
        if self._timer:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        props = context.scene.ai_copilot

        # Stop any running generation FIRST
        if props.is_generating:
            AIHOUSE_OT_send_prompt._stop_requested = True
            print("[Training] cancel: stopped active generation")

        props.training_active = False
        props.training_awaiting = False
        # Clean up stale state so it doesn't leak into next prompt
        from . import ai_engine
        ai_engine.flush_main_thread_queue()
        ai_engine.clear_history()
        ai_engine.clear_streaming_text()


class AIHOUSE_OT_training_stop(Operator):
    """Stop the training data collection loop"""

    bl_idname = "aihouse.training_stop"
    bl_label = "Stop Training Loop"

    def execute(self, context):
        props = context.scene.ai_copilot

        AIHOUSE_OT_send_prompt._stop_requested = True

        props.training_active = False
        props.training_awaiting = False

        # Clean up: clear conversation history and flush any pending
        # operations from the stopped generation so they don't leak
        # into the user's next manual prompt.
        from . import ai_engine, blender_tools
        ai_engine.flush_main_thread_queue()
        ai_engine.clear_history()
        ai_engine.clear_streaming_text()
        blender_tools.clear_scene()

        total = props.training_total
        approved = props.training_approved
        rejected = props.training_rejected
        self.report({"INFO"},
                    "Training stopped: %d approved, %d rejected out of %d"
                    % (approved, rejected, total))
        return {"FINISHED"}


class AIHOUSE_OT_training_approve(Operator):
    """Approve the current output as good training data"""

    bl_idname = "aihouse.training_approve"
    bl_label = "Approve"

    def execute(self, context):
        props = context.scene.ai_copilot

        if not props.training_active or not props.training_awaiting:
            return {"CANCELLED"}

        prompt = props.training_prompt
        scene_info = _get_scene_summary()

        _save_training_record(prompt, "approved", scene_info)

        prefs = context.preferences.addons[__package__].preferences
        server_url = prefs.local_server_url.rstrip("/")
        tokens = []
        if props.last_generation_tokens:
            try:
                tokens = _json.loads(props.last_generation_tokens)
            except Exception:
                pass
        if tokens:
            _send_feedback_to_server(server_url, "feedback/accept", {
                "prompt": prompt, "tokens": tokens,
            })

        props.training_approved += 1
        props.training_total += 1
        props.training_awaiting = False
        props.feedback_count += 1

        self._trigger_next(context)
        return {"FINISHED"}

    def _trigger_next(self, context):
        props = context.scene.ai_copilot
        if props.training_active:
            if props.is_generating:
                print("[Training] approve._trigger_next: skipping — still generating")
                return
            from . import blender_tools, ai_engine
            blender_tools.clear_scene()
            ai_engine.clear_history()
            ai_engine.clear_streaming_text()

            prompt = _pick_random_prompt()
            props.training_prompt = prompt
            props.prompt_text = prompt
            bpy.ops.aihouse.send_prompt()


class AIHOUSE_OT_training_reject(Operator):
    """Reject the current output as bad training data"""

    bl_idname = "aihouse.training_reject"
    bl_label = "Reject"

    def execute(self, context):
        props = context.scene.ai_copilot

        if not props.training_active or not props.training_awaiting:
            return {"CANCELLED"}

        prompt = props.training_prompt
        scene_info = _get_scene_summary()

        _save_training_record(prompt, "rejected", scene_info)

        prefs = context.preferences.addons[__package__].preferences
        server_url = prefs.local_server_url.rstrip("/")
        tokens = []
        if props.last_generation_tokens:
            try:
                tokens = _json.loads(props.last_generation_tokens)
            except Exception:
                pass
        if tokens:
            _send_feedback_to_server(server_url, "feedback/reject", {
                "prompt": prompt, "tokens": tokens,
            })

        props.training_rejected += 1
        props.training_total += 1
        props.training_awaiting = False
        props.feedback_count += 1

        self._trigger_next(context)
        return {"FINISHED"}

    def _trigger_next(self, context):
        props = context.scene.ai_copilot
        if props.training_active:
            if props.is_generating:
                print("[Training] reject._trigger_next: skipping — still generating")
                return
            from . import blender_tools, ai_engine
            blender_tools.clear_scene()
            ai_engine.clear_history()
            ai_engine.clear_streaming_text()

            prompt = _pick_random_prompt()
            props.training_prompt = prompt
            props.prompt_text = prompt
            bpy.ops.aihouse.send_prompt()


class AIHOUSE_OT_training_skip(Operator):
    """Skip this output without recording any feedback"""

    bl_idname = "aihouse.training_skip"
    bl_label = "Skip"

    def execute(self, context):
        props = context.scene.ai_copilot

        if not props.training_active or not props.training_awaiting:
            return {"CANCELLED"}

        props.training_skipped += 1
        props.training_total += 1
        props.training_awaiting = False

        if props.training_active:
            if props.is_generating:
                print("[Training] skip: skipping next — still generating")
                return {"FINISHED"}
            from . import blender_tools, ai_engine
            blender_tools.clear_scene()
            ai_engine.clear_history()
            ai_engine.clear_streaming_text()

            prompt = _pick_random_prompt()
            props.training_prompt = prompt
            props.prompt_text = prompt
            bpy.ops.aihouse.send_prompt()

        return {"FINISHED"}


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Validator operators
# ═══════════════════════════════════════════════════════════════════════════


class AIHOUSE_OT_validator_load_queue(Operator):
    """Load a validation queue exported from the mesh cache."""

    bl_idname = "aihouse.validator_load_queue"
    bl_label = "Load Validation Queue"

    def execute(self, context):
        props = context.scene.ai_copilot
        queue_dir = _validator_queue_dir(props)
        if queue_dir is None:
            self.report({"ERROR"}, "Set Validation Queue directory first")
            return {"CANCELLED"}

        global _VALIDATOR_QUEUE_DIR, _VALIDATOR_REVIEWED, _VALIDATOR_CACHE_DIR, _VALIDATOR_WORK_DIR

        # Live-cache mode: user can point at repo root OR the .mesh_cache dir.
        project_root, cache_dir = _validator_guess_cache_dir(queue_dir)
        if cache_dir is None or not cache_dir.exists():
            # Legacy exported queue support (index.jsonl)
            idx_path = _validator_index_path(queue_dir)
            if not idx_path.exists():
                self.report({"ERROR"}, "Select repo root, data/training_cache/default, or data/processed/.mesh_cache")
                return {"CANCELLED"}
            # Keep legacy behavior
            _VALIDATOR_QUEUE_DIR = queue_dir
            _VALIDATOR_WORK_DIR = queue_dir
            _VALIDATOR_CACHE_DIR = None
            _VALIDATOR_REVIEWED = _validator_load_reviews(queue_dir)
            global _VALIDATOR_QUEUE
            _VALIDATOR_QUEUE = []
            try:
                for line in idx_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        _VALIDATOR_QUEUE.append(_json.loads(line))
                    except Exception:
                        continue
            except Exception as e:
                self.report({"ERROR"}, f"Failed to read index: {e}")
                return {"CANCELLED"}
            props.validator_loaded = True
            props.validator_status = "Legacy queue loaded"
            bpy.ops.aihouse.validator_load_current()
            return {"FINISHED"}

        work_dir = _validator_default_work_dir(project_root, queue_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        _VALIDATOR_QUEUE_DIR = work_dir
        _VALIDATOR_WORK_DIR = work_dir
        _VALIDATOR_CACHE_DIR = cache_dir
        _VALIDATOR_REVIEWED = _validator_load_reviews(work_dir)

        # Read total item count from config.json if available
        total_items = 0
        try:
            cfg_path = cache_dir / "config.json"
            if cfg_path.exists():
                import json as _j
                cfg_data = _j.loads(cfg_path.read_text(encoding="utf-8"))
                total_items = int(cfg_data.get("total_items", 0))
            if total_items <= 0:
                # Estimate from batch file count (50 items/batch default)
                n_batches = len(list(cache_dir.glob("batch_*.pt")))
                if n_batches > 0:
                    batch_size = int(cfg_data.get("batch_size", 50)) if cfg_path.exists() else 50
                    total_items = n_batches * batch_size
        except Exception:
            pass

        props.validator_loaded = True
        props.validator_total = total_items
        props.validator_index = 0
        props.validator_approved = 0
        props.validator_rejected = 0
        props.validator_skipped = 0
        props.validator_quality_weight = 0.0
        props.validator_human_verdict = ""
        props.validator_flags = ""
        props.validator_status = f"Live cache mode: {cache_dir.name}"
        if getattr(props, "validator_fresh_only", False):
            props.validator_status += f" (fresh-only, {float(getattr(props, 'validator_fresh_hours', 0.0)):.1f}h)"

        # Immediately fetch first unreviewed item
        bpy.ops.aihouse.validator_load_current()
        return {"FINISHED"}


class AIHOUSE_OT_validator_load_current(Operator):
    """Load the current item into Blender for inspection."""

    bl_idname = "aihouse.validator_load_current"
    bl_label = "Load Current Item"

    def execute(self, context):
        props = context.scene.ai_copilot

        if not props.validator_loaded:
            self.report({"ERROR"}, "Load first")
            return {"CANCELLED"}

        # Live-cache mode
        if _VALIDATOR_CACHE_DIR is not None and _VALIDATOR_WORK_DIR is not None:
            after_cache = props.validator_cache_pt if props.validator_cache_pt else ""
            after_idx = props.validator_item_index if props.validator_item_index >= 0 else -1

            ok, out, err = _validator_run_fetch_item(
                context=context,
                cache_dir=_VALIDATOR_CACHE_DIR,
                work_dir=_VALIDATOR_WORK_DIR,
                after_cache_pt=after_cache,
                after_item_index=after_idx,
                fresh_only=bool(getattr(props, "validator_fresh_only", False)),
                fresh_hours=float(getattr(props, "validator_fresh_hours", 0.0) or 0.0),
            )
            if not ok:
                self.report({"ERROR"}, err)
                props.validator_status = f"Fetch failed: {err[:60]}"
                return {"CANCELLED"}
            if not out.get("ok") and out.get("done"):
                props.validator_current_item_id = ""
                props.validator_status = "Done — no more unreviewed items"
                _validator_clear_previous()
                return {"FINISHED"}

            item_id = str(out.get("item_id", ""))
            item_json_path = Path(str(out.get("item_json", "")))
            if not item_id or not item_json_path.exists():
                self.report({"ERROR"}, "Fetch returned no item")
                return {"CANCELLED"}

            ok2, err2 = _validator_import_item_mesh(item_json_path)
            if not ok2:
                self.report({"ERROR"}, err2)
                props.validator_status = f"Load failed: {err2[:60]}"
                return {"CANCELLED"}

            props.validator_current_item_id = item_id
            props.validator_current_item_path = str(item_json_path)
            props.validator_source = str(out.get("data_source", ""))
            props.validator_sample_type = str(out.get("sample_type", ""))
            props.validator_label = str(out.get("label", ""))
            tags = out.get("tags", [])
            if isinstance(tags, list):
                props.validator_tags = ", ".join(str(t) for t in tags if str(t).strip())
            try:
                props.validator_quality_weight = float(out.get("quality_weight", 0.0) or 0.0)
            except Exception:
                props.validator_quality_weight = 0.0
            props.validator_human_verdict = str(out.get("human_verdict", ""))
            flags = out.get("flags", [])
            if isinstance(flags, list):
                props.validator_flags = ", ".join(str(f) for f in flags if str(f).strip())
            else:
                props.validator_flags = ""
            mats = out.get("material_names", [])
            if isinstance(mats, list):
                props.validator_materials = ", ".join(str(m) for m in mats if str(m).strip())
            else:
                props.validator_materials = ""
            scene_keys = out.get("scene_context_keys", [])
            if isinstance(scene_keys, list):
                props.validator_scene_keys = ", ".join(str(k) for k in scene_keys if str(k).strip())
            else:
                props.validator_scene_keys = ""
            props.validator_scene_json = str(out.get("scene_json", ""))
            props.validator_cache_pt = str(out.get("cache_pt", ""))
            try:
                props.validator_item_index = int(out.get("item_index", -1))
            except Exception:
                props.validator_item_index = -1

            props.validator_index += 1
            if props.validator_flags:
                props.validator_status = f"Loaded {item_id} — ⚠ {props.validator_flags}"
            else:
                props.validator_status = f"Loaded {item_id}"
            return {"FINISHED"}

        # Legacy queue mode (exported index.jsonl + items)
        if not _VALIDATOR_QUEUE:
            self.report({"ERROR"}, "Load a queue first")
            return {"CANCELLED"}

        item = _validator_get_item(props.validator_index)
        if not item:
            self.report({"ERROR"}, "Invalid queue index")
            return {"CANCELLED"}

        item_id = item.get("item_id", "")
        item_json = item.get("item_json", "")
        if not isinstance(item_id, str) or not item_id:
            self.report({"ERROR"}, "Queue entry missing item_id")
            return {"CANCELLED"}
        if not isinstance(item_json, str) or not item_json:
            self.report({"ERROR"}, "Queue entry missing item_json")
            return {"CANCELLED"}

        item_json_path = Path(item_json)
        if not item_json_path.is_absolute() and _VALIDATOR_QUEUE_DIR is not None:
            item_json_path = (_VALIDATOR_QUEUE_DIR / item_json_path).resolve()
        ok, err = _validator_import_item_mesh(item_json_path)
        if not ok:
            self.report({"ERROR"}, err)
            props.validator_status = f"Load failed: {err[:60]}"
            return {"CANCELLED"}

        props.validator_current_item_id = item_id
        props.validator_current_item_path = str(item_json_path)
        props.validator_source = str(item.get("data_source", ""))
        props.validator_sample_type = str(item.get("sample_type", ""))
        props.validator_label = str(item.get("label", ""))
        props.validator_tags = ""
        props.validator_quality_weight = 0.0
        props.validator_human_verdict = ""
        props.validator_flags = ""
        props.validator_materials = ""
        props.validator_scene_keys = ""
        props.validator_scene_json = ""
        props.validator_cache_pt = ""
        props.validator_item_index = -1
        # tags are stored only in item json payload, but may also be present in index
        try:
            payload = _json.loads(item_json_path.read_text(encoding="utf-8"))
            tags = payload.get("tags", [])
            if isinstance(tags, list):
                props.validator_tags = ", ".join(str(t) for t in tags if str(t).strip())
            cache_pt = payload.get("cache_pt")
            if isinstance(cache_pt, str):
                props.validator_cache_pt = cache_pt
            item_index = payload.get("item_index")
            try:
                props.validator_item_index = int(item_index)
            except Exception:
                props.validator_item_index = -1
            try:
                props.validator_quality_weight = float(payload.get("quality_weight", 0.0) or 0.0)
            except Exception:
                props.validator_quality_weight = 0.0
            human_verdict = payload.get("human_verdict")
            if isinstance(human_verdict, str):
                props.validator_human_verdict = human_verdict
            flags = payload.get("flags", [])
            if isinstance(flags, list):
                props.validator_flags = ", ".join(str(f) for f in flags if str(f).strip())
            elif isinstance(flags, str):
                props.validator_flags = flags
            sample_type = payload.get("sample_type")
            if isinstance(sample_type, str):
                props.validator_sample_type = sample_type
            mats = payload.get("material_names", [])
            if isinstance(mats, list):
                props.validator_materials = ", ".join(str(m) for m in mats if str(m).strip())
            sc = payload.get("scene_context", {})
            if isinstance(sc, dict):
                props.validator_scene_keys = ", ".join(str(k) for k in sorted(sc.keys()) if str(k).strip())
            scene_json = payload.get("scene_json")
            if isinstance(scene_json, str):
                props.validator_scene_json = scene_json
        except Exception:
            pass

        if props.validator_flags:
            props.validator_status = f"Loaded {item_id} — ⚠ {props.validator_flags}"
        else:
            props.validator_status = f"Loaded {item_id}"
        return {"FINISHED"}


def _validator_advance(props) -> None:
    # In live-cache mode, just fetch the next item.
    bpy.ops.aihouse.validator_load_current()


class AIHOUSE_OT_validator_approve_next(Operator):
    bl_idname = "aihouse.validator_approve_next"
    bl_label = "Approve + Next"

    def execute(self, context):
        props = context.scene.ai_copilot
        if _VALIDATOR_QUEUE_DIR is None:
            self.report({"ERROR"}, "Load a queue first")
            return {"CANCELLED"}
        item_id = (props.validator_current_item_id or "").strip()
        if not item_id:
            self.report({"ERROR"}, "No current item loaded")
            return {"CANCELLED"}

        ok, err = _validator_run_apply_review(
            context=context,
            queue_dir=_VALIDATOR_QUEUE_DIR,
            item_id=item_id,
            verdict="approve",
            label=props.validator_label,
            tags=props.validator_tags,
        )
        if not ok:
            self.report({"ERROR"}, err)
            props.validator_status = f"Approve failed: {err[:60]}"
            return {"CANCELLED"}

        _VALIDATOR_REVIEWED.add(item_id)
        props.validator_approved += 1
        props.validator_status = f"Approved {item_id}"
        _validator_advance(props)
        return {"FINISHED"}


class AIHOUSE_OT_validator_reject_next(Operator):
    bl_idname = "aihouse.validator_reject_next"
    bl_label = "Reject + Next"

    def execute(self, context):
        props = context.scene.ai_copilot
        if _VALIDATOR_QUEUE_DIR is None:
            self.report({"ERROR"}, "Load a queue first")
            return {"CANCELLED"}
        item_id = (props.validator_current_item_id or "").strip()
        if not item_id:
            self.report({"ERROR"}, "No current item loaded")
            return {"CANCELLED"}

        ok, err = _validator_run_apply_review(
            context=context,
            queue_dir=_VALIDATOR_QUEUE_DIR,
            item_id=item_id,
            verdict="reject",
            label=props.validator_label,
            tags=props.validator_tags,
        )
        if not ok:
            self.report({"ERROR"}, err)
            props.validator_status = f"Reject failed: {err[:60]}"
            return {"CANCELLED"}

        _VALIDATOR_REVIEWED.add(item_id)
        props.validator_rejected += 1
        props.validator_status = f"Rejected {item_id} (quality_weight=0)"
        _validator_advance(props)
        return {"FINISHED"}


class AIHOUSE_OT_validator_skip_next(Operator):
    bl_idname = "aihouse.validator_skip_next"
    bl_label = "Skip + Next"

    def execute(self, context):
        props = context.scene.ai_copilot
        if _VALIDATOR_QUEUE_DIR is None:
            self.report({"ERROR"}, "Load a queue first")
            return {"CANCELLED"}
        item_id = (props.validator_current_item_id or "").strip()
        if not item_id:
            self.report({"ERROR"}, "No current item loaded")
            return {"CANCELLED"}

        ok, err = _validator_run_apply_review(
            context=context,
            queue_dir=_VALIDATOR_QUEUE_DIR,
            item_id=item_id,
            verdict="skip",
            label=props.validator_label,
            tags=props.validator_tags,
        )
        if not ok:
            self.report({"ERROR"}, err)
            props.validator_status = f"Skip failed: {err[:60]}"
            return {"CANCELLED"}

        _VALIDATOR_REVIEWED.add(item_id)
        props.validator_skipped += 1
        props.validator_status = f"Skipped {item_id}"
        _validator_advance(props)
        return {"FINISHED"}


class AIHOUSE_OT_validator_reconstruct_scene(Operator):
    """Reconstruct the full scene (all sibling objects from the same source file)."""

    bl_idname = "aihouse.validator_reconstruct_scene"
    bl_label = "Reconstruct Full Scene"

    def execute(self, context):
        props = context.scene.ai_copilot

        cache_pt = getattr(props, "validator_cache_pt", "")
        item_index = getattr(props, "validator_item_index", -1)
        if not cache_pt or int(item_index) < 0:
            self.report({"ERROR"}, "No item loaded (need cache_pt + item_index)")
            return {"CANCELLED"}

        prefs = context.preferences.addons[__package__].preferences
        project_root = _guess_policy_project_root(Path(prefs.policy_project_root))
        if project_root is None and _VALIDATOR_CACHE_DIR is not None:
            project_root = _validator_guess_project_root(_VALIDATOR_CACHE_DIR)
        if project_root is None:
            project_root = Path(cache_pt).parent.parent

        py = Path(prefs.policy_python) if prefs.policy_python else None
        if not py or not py.exists():
            py = _default_policy_python(project_root)
        if py is None or not py.exists():
            self.report({"ERROR"}, "Policy Python (venv) not found")
            return {"CANCELLED"}

        script = project_root / "scripts" / "validator_fetch_item.py"
        if not script.exists():
            self.report({"ERROR"}, f"Missing script: {script}")
            return {"CANCELLED"}

        work_dir = _VALIDATOR_WORK_DIR
        if work_dir is None:
            work_dir = _validator_default_work_dir(project_root, project_root)

        cmd = [
            str(py), str(script),
            "--cache-dir", str(_VALIDATOR_CACHE_DIR or Path(cache_pt).parent),
            "--work-dir", str(work_dir),
            "--reconstruct-scene",
            "--cache-pt", str(cache_pt),
            "--item-index", str(int(item_index)),
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=str(project_root), timeout=120)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to run scene reconstruction: {e}")
            return {"CANCELLED"}

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            self.report({"ERROR"}, f"Reconstruction failed: {err[-300:]}")
            return {"CANCELLED"}

        try:
            result = _json.loads((proc.stdout or "").strip())
        except Exception:
            self.report({"ERROR"}, "Reconstruction returned non-JSON")
            return {"CANCELLED"}

        if not result.get("ok"):
            self.report({"ERROR"}, result.get("error", "Unknown error"))
            return {"CANCELLED"}

        source_blend_path = Path(str(result.get("source_blend_path", "")).strip())
        if source_blend_path.exists() and source_blend_path.suffix.lower() == ".blend":
            try:
                _validator_clear_previous()
                col = _validator_ensure_collection("AI_VALIDATION")

                with bpy.data.libraries.load(str(source_blend_path), link=False) as (data_from, data_to):
                    data_to.collections = list(data_from.collections)
                    data_to.worlds = list(data_from.worlds)

                linked_collections = 0
                for appended_col in (data_to.collections or []):
                    if appended_col is None:
                        continue
                    try:
                        if appended_col.name not in bpy.context.scene.collection.children.keys():
                            bpy.context.scene.collection.children.link(appended_col)
                        linked_collections += 1
                    except Exception:
                        pass

                try:
                    worlds = [w for w in (data_to.worlds or []) if w is not None]
                    if worlds:
                        bpy.context.scene.world = worlds[0]
                except Exception:
                    pass

                # Re-apply source visibility (object + collection + layer collection).
                # Appending collections into a different scene does not always preserve
                # viewport hide/exclude flags from the original view layer.
                try:
                    source_scene_payload = {}
                    source_scene_exact_json = Path(str(result.get("source_scene_exact_json", "")).strip())
                    if source_scene_exact_json.exists():
                        source_scene_payload = _json.loads(source_scene_exact_json.read_text(encoding="utf-8"))
                    else:
                        scene_json_fallback = Path(str(result.get("scene_json", "")).strip())
                        if scene_json_fallback.exists():
                            scene_data_fallback = _json.loads(scene_json_fallback.read_text(encoding="utf-8"))
                            if isinstance(scene_data_fallback, dict):
                                source_scene_payload = scene_data_fallback.get("source_scene", {})

                    if isinstance(source_scene_payload, dict):
                        obj_vis_map = {}
                        for sobj in source_scene_payload.get("objects", []) if isinstance(source_scene_payload.get("objects"), list) else []:
                            if not isinstance(sobj, dict):
                                continue
                            sname = str(sobj.get("name", "")).strip()
                            if not sname:
                                continue
                            hide_viewport = bool(sobj.get("hide_viewport", False))
                            if "visible" in sobj:
                                hide_viewport = hide_viewport or (not bool(sobj.get("visible", True)))
                            obj_vis_map[sname] = {
                                "hide_viewport": hide_viewport,
                                "hide_render": bool(sobj.get("hide_render", False)),
                                "hide_select": bool(sobj.get("hide_select", False)),
                            }

                        col_vis_map = {}
                        for scd in source_scene_payload.get("collections", []) if isinstance(source_scene_payload.get("collections"), list) else []:
                            if not isinstance(scd, dict):
                                continue
                            cname = str(scd.get("name", "")).strip()
                            if not cname:
                                continue
                            col_vis_map[cname] = {
                                "hide_viewport": bool(scd.get("hide_viewport", False)),
                                "hide_render": bool(scd.get("hide_render", False)),
                                "hide_select": bool(scd.get("hide_select", False)),
                            }

                        # Object-level visibility
                        for obj_name, vis in obj_vis_map.items():
                            target = bpy.data.objects.get(obj_name)
                            if target is None and "." in obj_name:
                                base, suffix = obj_name.rsplit(".", 1)
                                if suffix.isdigit():
                                    target = bpy.data.objects.get(base)
                            if target is None:
                                continue
                            try:
                                target.hide_viewport = bool(vis.get("hide_viewport", False))
                            except Exception:
                                pass
                            try:
                                target.hide_render = bool(vis.get("hide_render", False))
                            except Exception:
                                pass
                            try:
                                target.hide_select = bool(vis.get("hide_select", False))
                            except Exception:
                                pass

                        # Collection-level datablock visibility
                        for col_name, vis in col_vis_map.items():
                            target_col = bpy.data.collections.get(col_name)
                            if target_col is None:
                                continue
                            try:
                                target_col.hide_viewport = bool(vis.get("hide_viewport", False))
                            except Exception:
                                pass
                            try:
                                target_col.hide_render = bool(vis.get("hide_render", False))
                            except Exception:
                                pass
                            try:
                                target_col.hide_select = bool(vis.get("hide_select", False))
                            except Exception:
                                pass

                        # Layer-collection viewport visibility for active view layer.
                        def _apply_layer_visibility(layer_coll):
                            if layer_coll is None:
                                return
                            cname = str(getattr(layer_coll.collection, "name", ""))
                            vis = col_vis_map.get(cname)
                            if vis is not None:
                                try:
                                    layer_coll.hide_viewport = bool(vis.get("hide_viewport", False))
                                except Exception:
                                    pass
                                try:
                                    layer_coll.exclude = False
                                except Exception:
                                    pass
                            for child in getattr(layer_coll, "children", []):
                                _apply_layer_visibility(child)

                        try:
                            _apply_layer_visibility(bpy.context.view_layer.layer_collection)
                        except Exception:
                            pass
                except Exception:
                    pass

                props.validator_scene_json = str(source_blend_path)
                props.validator_status = (
                    f"Reconstructed from BLEND ({linked_collections} collections): "
                    f"{source_blend_path.name}"
                )
                self.report({"INFO"}, f"Reconstructed exact scene from: {source_blend_path.name}")
                return {"FINISHED"}
            except Exception as e:
                self.report({"WARNING"}, f"BLEND append failed, falling back to JSON: {e}")

        scene_json_path = Path(str(result.get("scene_json", "")))
        if not scene_json_path.exists():
            self.report({"ERROR"}, "Scene JSON file not found")
            return {"CANCELLED"}
        props.validator_scene_json = str(scene_json_path)

        # Load full scene JSON with geometry
        try:
            scene_data = _json.loads(scene_json_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read scene JSON: {e}")
            return {"CANCELLED"}

        scene_objects = scene_data.get("objects", [])
        scene_cameras = scene_data.get("cameras", [])
        scene_lights = scene_data.get("lights", [])
        world_data = scene_data.get("world", {})
        images_data = scene_data.get("images", {})
        scene_node_groups = scene_data.get("node_groups", [])
        scene_collections = scene_data.get("collections", [])

        hidden_viewport_collections = set()
        hidden_render_collections = set()
        if isinstance(scene_collections, list):
            for _cd in scene_collections:
                if not isinstance(_cd, dict):
                    continue
                _cn = str(_cd.get("name", "")).strip()
                if not _cn:
                    continue
                if bool(_cd.get("hide_viewport", False)):
                    hidden_viewport_collections.add(_cn)
                if bool(_cd.get("hide_render", False)):
                    hidden_render_collections.add(_cn)

        if not scene_objects and not scene_cameras and not scene_lights:
            self.report({"ERROR"}, "No objects in scene reconstruction")
            return {"CANCELLED"}

        # Import all objects into Blender
        try:
            from . import blender_tools

            _validator_clear_previous()
            col = _validator_ensure_collection("AI_VALIDATION")

            current_obj_index = int(result.get("current_object_index", -1))
            file_label = str(result.get("file_label", ""))
            source_file = str(result.get("source_file", ""))
            total = int(len(scene_objects))

            imported_count = 0
            active_mesh_obj = None
            rigid_body_queue: list[tuple] = []  # (mesh_obj, rb_info) pairs
            created_by_source_name: dict[str, bpy.types.Object] = {}
            total_instances_imported = 0

            # ── Reconstruct standalone node groups (Geometry/Shader groups) ──
            def _restore_node_groups(_groups_data):
                if not isinstance(_groups_data, list) or not _groups_data:
                    return 0

                _created = 0
                _tree_map = {
                    "GEOMETRY": "GeometryNodeTree",
                    "SHADER": "ShaderNodeTree",
                    "COMPOSITING": "CompositorNodeTree",
                    "TEXTURE": "TextureNodeTree",
                }

                for _ngd in _groups_data:
                    if not isinstance(_ngd, dict):
                        continue
                    _ng_name = str(_ngd.get("name", "")).strip()
                    if not _ng_name:
                        continue

                    _ng_type_raw = str(_ngd.get("type", "GEOMETRY")).upper()
                    _ng_tree_type = _tree_map.get(_ng_type_raw, _ngd.get("type", "GeometryNodeTree"))
                    if _ng_name in bpy.data.node_groups:
                        _ng = bpy.data.node_groups[_ng_name]
                    else:
                        try:
                            _ng = bpy.data.node_groups.new(name=_ng_name, type=_ng_tree_type)
                        except Exception:
                            continue
                        _created += 1

                    try:
                        if hasattr(_ng, "interface") and _ng.interface:
                            try:
                                _items = list(getattr(_ng.interface, "items_tree", []))
                                for _it in _items:
                                    try:
                                        _ng.interface.remove(_it)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                            for _inp in (_ngd.get("inputs") or []):
                                if not isinstance(_inp, dict):
                                    continue
                                try:
                                    _ng.interface.new_socket(
                                        name=str(_inp.get("name", "Input")),
                                        in_out="INPUT",
                                        socket_type=str(_inp.get("socket_type", "NodeSocketFloat")),
                                    )
                                except Exception:
                                    pass

                            for _outp in (_ngd.get("outputs") or []):
                                if not isinstance(_outp, dict):
                                    continue
                                try:
                                    _ng.interface.new_socket(
                                        name=str(_outp.get("name", "Output")),
                                        in_out="OUTPUT",
                                        socket_type=str(_outp.get("socket_type", "NodeSocketFloat")),
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    _nodes_data = _ngd.get("nodes", [])
                    _links_data = _ngd.get("links", [])
                    if not isinstance(_nodes_data, list) or not _nodes_data:
                        continue

                    try:
                        _ng.nodes.clear()
                    except Exception:
                        pass

                    _nodes_by_name = {}
                    for _nd in _nodes_data:
                        if not isinstance(_nd, dict):
                            continue
                        _bl = _nd.get("bl_idname")
                        if not _bl:
                            continue
                        try:
                            _n = _ng.nodes.new(str(_bl))
                        except Exception:
                            continue
                        _n.name = str(_nd.get("name", _n.name))
                        _nodes_by_name[_n.name] = _n

                        _loc = _nd.get("location")
                        if isinstance(_loc, (list, tuple)) and len(_loc) >= 2:
                            try:
                                _n.location = (float(_loc[0]), float(_loc[1]))
                            except Exception:
                                pass

                        for _in_name, _in_val in (_nd.get("inputs") or {}).items():
                            if _in_val == "LINKED" or _in_name not in _n.inputs:
                                continue
                            try:
                                _sock = _n.inputs[_in_name]
                                if isinstance(_in_val, list):
                                    if hasattr(_sock, "default_value"):
                                        _cur = _sock.default_value
                                        if hasattr(_cur, "__len__"):
                                            _sock.default_value = tuple(_in_val[:len(_cur)])
                                else:
                                    _sock.default_value = _in_val
                            except Exception:
                                pass

                    for _lk in (_links_data or []):
                        if not isinstance(_lk, dict):
                            continue
                        _fn = _nodes_by_name.get(_lk.get("from_node"))
                        _tn = _nodes_by_name.get(_lk.get("to_node"))
                        if not _fn or not _tn:
                            continue
                        try:
                            _fs = _fn.outputs.get(_lk.get("from_socket"))
                            _ts = _tn.inputs.get(_lk.get("to_socket"))
                            if _fs and _ts:
                                _ng.links.new(_fs, _ts)
                        except Exception:
                            pass

                return _created

            restored_node_groups = _restore_node_groups(scene_node_groups)

            # Grid layout for single-object datasets (all verts near origin):
            # detect whether the scene has meaningful world-space spread.
            import math as _math
            _grid_centers = []
            for _so in scene_objects:
                _dc = _so.get("denorm_offset", [0, 0, 0])
                try:
                    _grid_centers.append([float(x) for x in _dc])
                except Exception:
                    _grid_centers.append([0.0, 0.0, 0.0])
            _spread = 0.0
            if len(_grid_centers) > 1:
                _cx = sum(c[0] for c in _grid_centers) / len(_grid_centers)
                _cy = sum(c[1] for c in _grid_centers) / len(_grid_centers)
                _cz = sum(c[2] for c in _grid_centers) / len(_grid_centers)
                _spread = max(
                    _math.sqrt((c[0]-_cx)**2 + (c[1]-_cy)**2 + (c[2]-_cz)**2)
                    for c in _grid_centers
                )
            use_grid_layout = _spread < 0.01  # all within 1 cm → side-by-side grid
            grid_spacing = 2.5
            grid_cols = max(1, _math.ceil(_math.sqrt(len(scene_objects))))

            for si, sobj in enumerate(scene_objects):
                use_raw_local_mesh = bool(
                    sobj.get("geometry_space") == "RAW_LOCAL"
                    and isinstance(sobj.get("raw_vertices"), list)
                    and isinstance(sobj.get("raw_faces"), list)
                    and sobj.get("raw_vertices")
                    and sobj.get("raw_faces")
                )
                if use_raw_local_mesh:
                    verts = sobj.get("raw_vertices", [])
                    faces = sobj.get("raw_faces", [])
                else:
                    verts = sobj.get("vertices", [])
                    faces = sobj.get("faces", [])
                if not verts or not faces:
                    # Instancing-only objects (Geometry Nodes / collection instances)
                    # are handled in a second pass.
                    continue

                obj_name = str(sobj.get("name", f"Object_{sobj.get('object_index', 0)}"))
                is_current = bool(sobj.get("is_current_item", False))

                # Prefix with SCENE_ so user knows this is a reconstructed scene
                mesh_name = f"SCENE_{obj_name}"
                if is_current:
                    mesh_name = f"CURRENT_{obj_name}"

                # Denormalize vertices back to world space.
                # The extractor stored verts as: v_norm = (v_world - center) / scale
                # Reversing: v_world = v_norm * scale + center
                # This ensures all objects share the same world coordinate frame.
                _dc = sobj.get("denorm_offset", [0, 0, 0])
                _ds = float(sobj.get("denorm_scale", 1.0))
                try:
                    _ox, _oy, _oz = float(_dc[0]), float(_dc[1]), float(_dc[2])
                except Exception:
                    _ox, _oy, _oz = 0.0, 0.0, 0.0
                if _ds < 1e-9:
                    _ds = 1.0
                ws_verts = [
                    (float(p[0]) * _ds + _ox,
                     float(p[1]) * _ds + _oy,
                     float(p[2]) * _ds + _oz)
                    for p in verts
                ]
                f = [tuple(map(int, tri)) for tri in faces]
                mesh_obj = blender_tools.create_mesh(name=mesh_name, verts=ws_verts, faces=f)

                if use_raw_local_mesh:
                    # Raw mesh is local-space; restore source object transform.
                    _tr = sobj.get("transforms", {}) if isinstance(sobj.get("transforms"), dict) else {}
                    try:
                        _loc = _tr.get("location", [0, 0, 0])
                        mesh_obj.location = tuple(float(x) for x in list(_loc)[:3])
                    except Exception:
                        mesh_obj.location = (0.0, 0.0, 0.0)
                    try:
                        _rot = _tr.get("rotation_euler", [0, 0, 0])
                        mesh_obj.rotation_euler = tuple(float(x) for x in list(_rot)[:3])
                    except Exception:
                        pass
                    try:
                        _scl = _tr.get("scale", [1, 1, 1])
                        mesh_obj.scale = tuple(float(x) for x in list(_scl)[:3])
                    except Exception:
                        mesh_obj.scale = (1.0, 1.0, 1.0)
                elif use_grid_layout:
                    # Grid mode: spread objects side-by-side for easy comparison
                    row = si // grid_cols
                    col_idx = si % grid_cols
                    mesh_obj.location = (
                        col_idx * grid_spacing,
                        -row * grid_spacing,
                        0.0,
                    )
                else:
                    # Scene mode: vertices are already in world space → object at origin
                    mesh_obj.location = (0.0, 0.0, 0.0)
                mesh_obj.scale = (1.0, 1.0, 1.0)

                # Move into validation collection
                try:
                    for c in list(mesh_obj.users_collection):
                        c.objects.unlink(mesh_obj)
                except Exception:
                    pass
                col.objects.link(mesh_obj)

                # Store metadata as custom properties
                mesh_obj["scene_object_index"] = int(sobj.get("object_index", 0))
                mesh_obj["scene_object_name"] = obj_name
                mesh_obj["scene_object_label"] = str(sobj.get("label", ""))
                mesh_obj["scene_is_current_item"] = is_current
                mesh_obj["scene_file_label"] = file_label
                mesh_obj["scene_source_file"] = source_file

                # Restore object visibility flags from source scene.
                try:
                    hide_viewport = bool(sobj.get("hide_viewport", False))
                    if "visible" in sobj:
                        hide_viewport = hide_viewport or (not bool(sobj.get("visible", True)))
                    sobj_cols = sobj.get("collections", [])
                    if isinstance(sobj_cols, list):
                        if any(str(c) in hidden_viewport_collections for c in sobj_cols):
                            hide_viewport = True
                    mesh_obj.hide_viewport = hide_viewport
                except Exception:
                    pass
                try:
                    hide_render = bool(sobj.get("hide_render", False))
                    sobj_cols = sobj.get("collections", [])
                    if isinstance(sobj_cols, list):
                        if any(str(c) in hidden_render_collections for c in sobj_cols):
                            hide_render = True
                    mesh_obj.hide_render = hide_render
                except Exception:
                    pass
                try:
                    mesh_obj.hide_select = bool(sobj.get("hide_select", False))
                except Exception:
                    pass

                # Map original source object name → created Blender object
                created_by_source_name[obj_name] = mesh_obj

                # Apply materials from scene data
                obj_data = {
                    "scene_context": {
                        "materials": sobj.get("materials", []),
                        "face_material_indices": sobj.get("face_material_indices"),
                        "face_smooth": sobj.get("face_smooth"),
                    },
                    "label": str(sobj.get("label", "")),
                    "item_id": f"scene_{sobj.get('object_index', 0)}",
                    "data_source": str(result.get("data_source", "")),
                }
                _apply_json_materials(mesh_obj, obj_data)

                # Apply face smooth
                face_smooth = sobj.get("face_smooth")
                if isinstance(face_smooth, list) and len(face_smooth) == len(mesh_obj.data.polygons):
                    for fi, poly in enumerate(mesh_obj.data.polygons):
                        poly.use_smooth = bool(face_smooth[fi])

                # ── Modifier stack ────────────────────────────────────────────
                # Re-apply stored modifier stack when available.
                # In RAW_LOCAL mode this restores the original construction.
                # In EVALUATED_WORLD mode modifiers are typically skipped at source.
                obj_modifiers = sobj.get("modifiers", []) if isinstance(sobj.get("modifiers", []), list) else []
                for mod_info in obj_modifiers:
                    mod_type = str(mod_info.get("type", "")).upper()
                    if not mod_type:
                        continue
                    mod_name = str(mod_info.get("name", mod_type))
                    try:
                        mod = mesh_obj.modifiers.new(name=mod_name, type=mod_type)
                        if not mod_info.get("show_viewport", True):
                            mod.show_viewport = False
                        if mod_type == "SUBSURF":
                            mod.levels = int(mod_info.get("levels", 1))
                            mod.render_levels = int(mod_info.get("render_levels", 2))
                            try:
                                mod.subdivision_type = str(mod_info.get("subdivision_type", "CATMULL_CLARK"))
                            except Exception:
                                pass
                        elif mod_type == "MIRROR":
                            use_axis = mod_info.get("use_axis", [True, False, False])
                            for _ax_i, _ax_v in enumerate(use_axis[:3]):
                                mod.use_axis[_ax_i] = bool(_ax_v)
                            mod.use_clip = bool(mod_info.get("use_clip", False))
                        elif mod_type == "SOLIDIFY":
                            mod.thickness = float(mod_info.get("thickness", 0.01))
                            mod.offset = float(mod_info.get("offset", -1.0))
                        elif mod_type == "BEVEL":
                            mod.width = float(mod_info.get("width", 0.1))
                            mod.segments = int(mod_info.get("segments", 1))
                            if mod_info.get("limit_method"):
                                try:
                                    mod.limit_method = str(mod_info["limit_method"])
                                except Exception:
                                    pass
                        elif mod_type == "ARRAY":
                            mod.count = int(mod_info.get("count", 2))
                            mod.use_relative_offset = bool(mod_info.get("use_relative_offset", True))
                            if mod_info.get("relative_offset_displace"):
                                mod.relative_offset_displace = tuple(
                                    float(v) for v in mod_info["relative_offset_displace"][:3]
                                )
                        elif mod_type == "SCREW":
                            mod.angle = float(mod_info.get("angle", 6.2832))
                            mod.steps = int(mod_info.get("steps", 16))
                            mod.render_steps = int(mod_info.get("render_steps", 16))
                            if mod_info.get("axis"):
                                try:
                                    mod.axis = str(mod_info["axis"])
                                except Exception:
                                    pass
                        elif mod_type == "DECIMATE":
                            if mod_info.get("decimate_type"):
                                try:
                                    mod.decimate_type = str(mod_info["decimate_type"])
                                except Exception:
                                    pass
                            mod.ratio = float(mod_info.get("ratio", 0.5))
                        elif mod_type == "WIREFRAME":
                            mod.thickness = float(mod_info.get("thickness", 0.02))
                        elif mod_type == "SIMPLE_DEFORM":
                            if mod_info.get("deform_method"):
                                try:
                                    mod.deform_method = str(mod_info["deform_method"])
                                except Exception:
                                    pass
                            mod.angle = float(mod_info.get("angle", 0.7854))
                            if mod_info.get("deform_axis"):
                                try:
                                    mod.deform_axis = str(mod_info["deform_axis"])
                                except Exception:
                                    pass
                        elif mod_type == "NODES":
                            # Geometry Nodes — link the stored node group by name
                            ng_name = (mod_info.get("node_group_name")
                                       or mod_info.get("node_group"))
                            if ng_name and ng_name in bpy.data.node_groups:
                                try:
                                    mod.node_group = bpy.data.node_groups[ng_name]
                                except Exception:
                                    pass
                        elif mod_type == "ARMATURE":
                            arm_name = mod_info.get("armature_object")
                            if arm_name and arm_name in bpy.data.objects:
                                try:
                                    mod.object = bpy.data.objects[arm_name]
                                except Exception:
                                    pass
                            mod.use_vertex_groups = bool(
                                mod_info.get("use_vertex_groups", True))
                    except Exception:
                        pass  # Skip unsupported modifier types gracefully

                # Collect rigid body data for post-import processing
                rb_info = sobj.get("rigid_body")
                if rb_info and isinstance(rb_info, dict):
                    rigid_body_queue.append((mesh_obj, rb_info))

                if is_current:
                    active_mesh_obj = mesh_obj

                imported_count += 1

            # ── Second pass: materialize instance transforms ─────────────────
            # Instances reference prototype objects by original Blender name.
            # We create linked duplicates that share mesh data for efficiency.
            try:
                from mathutils import Matrix as _Matrix
            except Exception:
                _Matrix = None

            for sobj in (scene_objects or []):
                inst_list = sobj.get("instances")
                if not inst_list or not isinstance(inst_list, list):
                    continue
                parent_name = str(sobj.get("name", ""))
                for ii, inst in enumerate(inst_list):
                    if not isinstance(inst, dict) or inst.get("_truncated"):
                        continue
                    proto_name = str(inst.get("source_object", ""))
                    mat = inst.get("matrix_world")
                    if not proto_name or not mat or not isinstance(mat, list) or len(mat) != 16:
                        continue
                    proto_obj = created_by_source_name.get(proto_name)
                    if proto_obj is None or not getattr(proto_obj, "data", None):
                        continue

                    inst_obj_name = f"SCENE_INST_{parent_name}_{proto_name}_{ii:04d}"
                    try:
                        inst_obj = bpy.data.objects.new(name=inst_obj_name, object_data=proto_obj.data)
                    except Exception:
                        continue

                    # Place in world space
                    try:
                        if _Matrix is not None:
                            inst_obj.matrix_world = _Matrix([
                                [float(mat[0]), float(mat[1]), float(mat[2]), float(mat[3])],
                                [float(mat[4]), float(mat[5]), float(mat[6]), float(mat[7])],
                                [float(mat[8]), float(mat[9]), float(mat[10]), float(mat[11])],
                                [float(mat[12]), float(mat[13]), float(mat[14]), float(mat[15])],
                            ])
                        else:
                            inst_obj.location = (float(mat[3]), float(mat[7]), float(mat[11]))
                    except Exception:
                        pass

                    try:
                        for c in list(inst_obj.users_collection):
                            c.objects.unlink(inst_obj)
                    except Exception:
                        pass
                    try:
                        col.objects.link(inst_obj)
                    except Exception:
                        pass

                    inst_obj["scene_instance_source"] = proto_name
                    inst_obj["scene_instance_parent"] = parent_name
                    inst_obj["scene_file_label"] = file_label
                    inst_obj["scene_source_file"] = source_file

                    total_instances_imported += 1

            # ── Apply rigid body physics ──────────────────────────────────────
            if rigid_body_queue:
                try:
                    # Ensure the scene has a rigidbody world
                    if not bpy.context.scene.rigidbody_world:
                        bpy.ops.rigidbody.world_add()
                    for rb_obj, rb_info in rigid_body_queue:
                        try:
                            bpy.context.view_layer.objects.active = rb_obj
                            bpy.ops.object.select_all(action="DESELECT")
                            rb_obj.select_set(True)
                            bpy.ops.rigidbody.object_add()
                            if rb_obj.rigid_body:
                                rbd = rb_obj.rigid_body
                                rb_type = str(rb_info.get("type", "ACTIVE")).upper()
                                rbd.type = rb_type if rb_type in ("ACTIVE", "PASSIVE") else "ACTIVE"
                                rbd.mass = float(rb_info.get("mass", 1.0))
                                rbd.friction = float(rb_info.get("friction", 0.5))
                                rbd.restitution = float(rb_info.get("restitution", 0.0))
                                rbd.linear_damping = float(rb_info.get("linear_damping", 0.04))
                                rbd.angular_damping = float(rb_info.get("angular_damping", 0.1))
                                rbd.kinematic = bool(rb_info.get("kinematic", False))
                                rbd.enabled = bool(rb_info.get("enabled", True))
                                cs = str(rb_info.get("collision_shape", "CONVEX_HULL")).upper()
                                try:
                                    rbd.collision_shape = cs
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass

            # ── Reconstruct keyframe animation ────────────────────────────────
            for _kf_sobj in scene_objects:
                _kf_data = _kf_sobj.get("keyframes")
                if not _kf_data:
                    continue
                _kf_prefix = "CURRENT_" if _kf_sobj.get("is_current_item") else "SCENE_"
                _kf_target = bpy.data.objects.get(
                    _kf_prefix + str(_kf_sobj.get("name", "")))
                if not _kf_target:
                    continue
                try:
                    _act_name = str(_kf_sobj.get(
                        "action_name", f"{_kf_prefix}{_kf_sobj.get('name', '')}Action"))
                    _action = bpy.data.actions.new(name=_act_name)
                    _kf_target.animation_data_create()
                    _kf_target.animation_data.action = _action
                    for _fcd in _kf_data:
                        if not isinstance(_fcd, dict) or "_truncated" in _fcd:
                            continue
                        _dp = str(_fcd.get("data_path", ""))
                        _ai = int(_fcd.get("array_index", 0))
                        if not _dp:
                            continue
                        try:
                            _fc = _action.fcurves.new(data_path=_dp, index=_ai)
                            for _kp in (_fcd.get("keyframe_points") or []):
                                _co = _kp.get("co")
                                if not _co or len(_co) < 2:
                                    continue
                                try:
                                    _kpn = _fc.keyframe_points.insert(
                                        float(_co[0]), float(_co[1]))
                                    _interp = _kp.get("interpolation")
                                    if _interp:
                                        try:
                                            _kpn.interpolation = str(_interp)
                                        except Exception:
                                            pass
                                    _hl = _kp.get("handle_left")
                                    _hr = _kp.get("handle_right")
                                    if _hl and len(_hl) >= 2:
                                        _kpn.handle_left.x = float(_hl[0])
                                        _kpn.handle_left.y = float(_hl[1])
                                    if _hr and len(_hr) >= 2:
                                        _kpn.handle_right.x = float(_hr[0])
                                        _kpn.handle_right.y = float(_hr[1])
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass

            # ── Set scene frame range from source file metadata ───────────────
            try:
                bpy.context.scene.frame_start = int(scene_data.get("frame_start", 1))
                bpy.context.scene.frame_end = int(scene_data.get("frame_end", 250))
                bpy.context.scene.render.fps = int(scene_data.get("fps", 24))
                bpy.context.scene.frame_current = int(
                    scene_data.get("frame_current", scene_data.get("frame_start", 1))
                )
            except Exception:
                pass

            # ── Load image textures (full-res PNG preferred, JPEG fallback) ───
            imported_images = 0
            if images_data and isinstance(images_data, dict):
                import base64, tempfile, os as _os
                for img_name, img_info in images_data.items():
                    if not isinstance(img_info, dict):
                        continue
                    if img_name in bpy.data.images:
                        imported_images += 1
                        continue
                    # Prefer full-res PNG; fall back to small JPEG thumbnail
                    img_b64 = img_info.get("image_data") or img_info.get("thumbnail")
                    if not img_b64:
                        continue
                    fmt = "png" if img_info.get("image_data") else img_info.get("thumbnail_format", "jpeg").lower()
                    try:
                        img_bytes = base64.b64decode(img_b64)
                        ext = ".png" if fmt == "png" else ".jpg"
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                            tf.write(img_bytes)
                            tmp_path = tf.name
                        blender_img = bpy.data.images.load(tmp_path, check_existing=False)
                        blender_img.name = img_name
                        blender_img.pack()
                        _os.unlink(tmp_path)
                        cs = img_info.get("colorspace", "sRGB")
                        try:
                            blender_img.colorspace_settings.name = cs
                        except Exception:
                            pass
                        imported_images += 1
                    except Exception:
                        pass

            # ── Create light objects ──────────────────────────────────────────
            imported_lights = 0
            for lobj in (scene_lights or []):
                try:
                    ldata_d = lobj.get("light", {})
                    ltype = str(ldata_d.get("type", "POINT")).upper()
                    if ltype not in ("SUN", "POINT", "SPOT", "AREA"):
                        ltype = "POINT"
                    light_name = f"SCENE_{lobj.get('name', 'Light')}"
                    bl_light = bpy.data.lights.new(name=light_name, type=ltype)
                    color = ldata_d.get("color", [1.0, 1.0, 1.0])
                    try:
                        bl_light.color = tuple(min(1.0, max(0.0, float(c))) for c in list(color)[:3])
                    except Exception:
                        pass
                    try:
                        bl_light.energy = float(ldata_d.get("energy", 10.0))
                    except Exception:
                        pass
                    if ltype == "SUN":
                        angle = ldata_d.get("angle")
                        if angle is not None:
                            try:
                                bl_light.angle = float(angle)
                            except Exception:
                                pass
                    elif ltype == "SPOT":
                        try:
                            bl_light.spot_size = float(ldata_d.get("spot_size", 1.0471))
                            bl_light.spot_blend = float(ldata_d.get("spot_blend", 0.15))
                        except Exception:
                            pass
                    elif ltype == "AREA":
                        area_size = ldata_d.get("size")
                        if area_size is not None:
                            try:
                                bl_light.size = float(area_size)
                            except Exception:
                                pass
                    bl_light_obj = bpy.data.objects.new(name=light_name, object_data=bl_light)
                    _tr = lobj.get("transforms", {})
                    try:
                        _loc = _tr.get("location", [0, 0, 0])
                        bl_light_obj.location = tuple(float(x) for x in list(_loc)[:3])
                    except Exception:
                        pass
                    try:
                        _rot = _tr.get("rotation_euler", [0, 0, 0])
                        bl_light_obj.rotation_euler = tuple(float(x) for x in list(_rot)[:3])
                    except Exception:
                        pass
                    for c in list(bl_light_obj.users_collection):
                        c.objects.unlink(bl_light_obj)
                    col.objects.link(bl_light_obj)
                    bl_light_obj["scene_object_name"] = lobj.get("name", "")
                    bl_light_obj["scene_file_label"] = file_label
                    imported_lights += 1
                except Exception:
                    pass

            # ── Create camera objects ─────────────────────────────────────────
            imported_cameras = 0
            active_cam_obj = None
            for cobj in (scene_cameras or []):
                try:
                    cdata_d = cobj.get("camera", {})
                    cam_name = f"SCENE_{cobj.get('name', 'Camera')}"
                    bl_cam = bpy.data.cameras.new(name=cam_name)
                    try:
                        bl_cam.lens = float(cdata_d.get("lens", 50.0))
                    except Exception:
                        pass
                    try:
                        bl_cam.clip_start = float(cdata_d.get("clip_start", 0.1))
                        bl_cam.clip_end   = float(cdata_d.get("clip_end", 1000.0))
                    except Exception:
                        pass
                    try:
                        bl_cam.sensor_width  = float(cdata_d.get("sensor_width", 36.0))
                        bl_cam.sensor_height = float(cdata_d.get("sensor_height", 24.0))
                    except Exception:
                        pass
                    cam_type = str(cdata_d.get("type", "PERSP")).upper()
                    if cam_type in ("PERSP", "ORTHO", "PANO"):
                        bl_cam.type = cam_type
                    dof_dist = cdata_d.get("dof_distance")
                    if dof_dist and cdata_d.get("dof_use"):
                        try:
                            bl_cam.dof.use_dof = True
                            bl_cam.dof.focus_distance = float(dof_dist)
                        except Exception:
                            pass
                    bl_cam_obj = bpy.data.objects.new(name=cam_name, object_data=bl_cam)
                    _tr = cobj.get("transforms", {})
                    try:
                        _loc = _tr.get("location", [0, 0, 0])
                        bl_cam_obj.location = tuple(float(x) for x in list(_loc)[:3])
                    except Exception:
                        pass
                    try:
                        _rot = _tr.get("rotation_euler", [0, 0, 0])
                        bl_cam_obj.rotation_euler = tuple(float(x) for x in list(_rot)[:3])
                    except Exception:
                        pass
                    for c in list(bl_cam_obj.users_collection):
                        c.objects.unlink(bl_cam_obj)
                    col.objects.link(bl_cam_obj)
                    bl_cam_obj["scene_object_name"] = cobj.get("name", "")
                    bl_cam_obj["scene_file_label"] = file_label
                    bl_cam_obj["scene_is_active_camera"] = bool(cobj.get("is_active", False))
                    if cobj.get("is_active"):
                        active_cam_obj = bl_cam_obj
                    imported_cameras += 1
                except Exception:
                    pass

            if active_cam_obj is not None:
                try:
                    bpy.context.scene.camera = active_cam_obj
                except Exception:
                    pass

            # ── Apply world shader (full node tree + HDRI) ───────────────────
            if world_data and isinstance(world_data, dict):
                try:
                    _world = bpy.context.scene.world
                    if _world is None:
                        _world = bpy.data.worlds.new("World")
                        bpy.context.scene.world = _world
                    _world.use_nodes = True
                    _wnt = _world.node_tree
                    _w_nodes_data = world_data.get("nodes", [])
                    _w_links_data = world_data.get("links", [])

                    if _w_nodes_data:
                        # Full world node tree reconstruction
                        _wnt.nodes.clear()
                        _w_nodes_by_name: dict = {}
                        for _wnd in _w_nodes_data:
                            _wbl = _wnd.get("bl_idname")
                            if not _wbl:
                                continue
                            try:
                                _wn = _wnt.nodes.new(_wbl)
                            except Exception:
                                continue
                            _wn.name = _wnd.get("name", "")
                            _w_nodes_by_name[_wn.name] = _wn
                            # Assign image for TEX_ENVIRONMENT / TEX_IMAGE
                            if _wnd.get("type") in ("TEX_ENVIRONMENT", "TEX_IMAGE"):
                                _img_name = _wnd.get("image_name")
                                if _img_name:
                                    # Look in already-loaded images first
                                    _img = bpy.data.images.get(_img_name)
                                    if _img is None and _img_name in (images_data or {}):
                                        # Load from base64 data
                                        _ientry = images_data[_img_name]
                                        _ib64 = _ientry.get("image_data") or ""
                                        if _ib64:
                                            import base64 as _b64
                                            import tempfile as _tf
                                            import os as _os
                                            try:
                                                # Use stored format (e.g. "png") not filename extension
                                                # (e.g. "forest.exr" may be stored as PNG)
                                                _fmt = _ientry.get("image_data_format", "").lower()
                                                if _fmt in ("png", "jpg", "jpeg", "tga", "bmp", "tiff", "exr", "hdr"):
                                                    ext = "." + _fmt
                                                else:
                                                    ext = _os.path.splitext(_img_name)[1] or ".png"
                                                with _tf.NamedTemporaryFile(
                                                        suffix=ext, delete=False) as _tmp:
                                                    _tmp.write(_b64.b64decode(_ib64))
                                                    _tmp_path = _tmp.name
                                                _img = bpy.data.images.load(_tmp_path)
                                                _img.name = _img_name
                                                _img.pack()
                                                _os.unlink(_tmp_path)
                                            except Exception:
                                                _img = None
                                    if _img and hasattr(_wn, "image"):
                                        try:
                                            _wn.image = _img
                                        except Exception:
                                            pass
                            # Set socket default values
                            for _wk, _wv in _wnd.get("inputs", {}).items():
                                if _wv == "LINKED" or _wk not in _wnt.nodes[_wn.name].inputs:
                                    continue
                                try:
                                    _wsock = _wnt.nodes[_wn.name].inputs[_wk]
                                    _wstype = type(_wsock).__name__
                                    if _wstype == "NodeSocketColor" and isinstance(_wv, list):
                                        if len(_wv) == 3:
                                            _wv = list(_wv) + [1.0]
                                        _wsock.default_value = tuple(_wv[:4])
                                    elif _wstype == "NodeSocketVector" and isinstance(_wv, list):
                                        _wsock.default_value = tuple(_wv[:3])
                                    elif isinstance(_wv, list):
                                        pass  # skip ambiguous list types
                                    else:
                                        _wsock.default_value = _wv
                                except Exception:
                                    pass
                        # Recreate world links
                        for _wl in _w_links_data:
                            _wfn = _w_nodes_by_name.get(_wl.get("from_node"))
                            _wtn = _w_nodes_by_name.get(_wl.get("to_node"))
                            if not _wfn or not _wtn:
                                continue
                            try:
                                _wfsock = _wfn.outputs.get(_wl.get("from_socket"))
                                _wtsock = _wtn.inputs.get(_wl.get("to_socket"))
                                if _wfsock and _wtsock:
                                    _wnt.links.new(_wfsock, _wtsock)
                            except Exception:
                                pass
                    else:
                        # Fallback: plain background color
                        _bg = _wnt.nodes.get("Background") or _wnt.nodes.new("ShaderNodeBackground")
                        _out = _wnt.nodes.get("World Output") or _wnt.nodes.new("ShaderNodeOutputWorld")
                        if not any(_l.to_node == _out for _l in _wnt.links):
                            _wnt.links.new(_bg.outputs["Background"], _out.inputs["Surface"])
                except Exception:
                    pass

            # Select and focus on the current item's object
            if active_mesh_obj is not None:
                bpy.context.view_layer.objects.active = active_mesh_obj
                active_mesh_obj.select_set(True)

            _force_viewport_update()
            _set_viewport_material_preview()

            # Switch 3D viewport to look through the reconstructed camera
            if active_cam_obj is not None:
                try:
                    for _area in bpy.context.screen.areas:
                        if _area.type == 'VIEW_3D':
                            for _space in _area.spaces:
                                if _space.type == 'VIEW_3D':
                                    _space.use_local_camera = False
                                    # Directly set camera perspective (avoids toggle issue)
                                    _space.region_3d.view_perspective = 'CAMERA'
                            break
                except Exception:
                    pass
            elif not use_grid_layout:
                # No camera — frame all objects so user can see the scene
                try:
                    for _area in bpy.context.screen.areas:
                        if _area.type == 'VIEW_3D':
                            with bpy.context.temp_override(area=_area):
                                bpy.ops.view3d.view_all(center=False)
                            break
                except Exception:
                    pass

            extras = []
            if imported_cameras:
                extras.append(f"{imported_cameras} cam{'s' if imported_cameras != 1 else ''}")
            if imported_lights:
                extras.append(f"{imported_lights} light{'s' if imported_lights != 1 else ''}")
            if imported_images:
                extras.append(f"{imported_images} tex")
            if total_instances_imported:
                extras.append(f"{total_instances_imported} inst")
            if restored_node_groups:
                extras.append(f"{restored_node_groups} nodegrp")
            extra_str = f" + {', '.join(extras)}" if extras else ""
            layout_mode = "grid (origin-stacked scene)" if use_grid_layout else "world-space"
            props.validator_status = (
                f"Scene reconstructed [{layout_mode}]: {imported_count}/{total} meshes{extra_str} "
                f"from {source_file or 'unknown'}"
            )
            self.report({"INFO"},
                        f"Loaded {imported_count} meshes{extra_str} ({layout_mode})")
            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Scene import failed: {e}")
            return {"CANCELLED"}


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

classes = (
    AIHOUSE_OT_send_prompt,
    AIHOUSE_OT_generate_policy,
    AIHOUSE_OT_generate_direct,
    AIHOUSE_OT_stop_generation,
    AIHOUSE_OT_execute_code,
    AIHOUSE_OT_clear_scene,
    AIHOUSE_OT_clear_chat,
    AIHOUSE_OT_test_local_server,
    AIHOUSE_OT_add_reference_image,
    AIHOUSE_OT_remove_reference_image,
    AIHOUSE_OT_clear_reference_images,
    AIHOUSE_OT_search_reference_images,
    AIHOUSE_OT_drop_reference_image,
    AIHOUSE_OT_open_ref_image,
    AIHOUSE_FH_drop_image,
    AIHOUSE_OT_start_comparison,
    AIHOUSE_OT_submit_comparison,
    AIHOUSE_OT_regenerate_comparison,
    AIHOUSE_OT_cancel_comparison,
    AIHOUSE_OT_accept_output,
    AIHOUSE_OT_reject_output,
    AIHOUSE_OT_training_start,
    AIHOUSE_OT_training_stop,
    AIHOUSE_OT_training_approve,
    AIHOUSE_OT_training_reject,
    AIHOUSE_OT_training_skip,
    AIHOUSE_OT_validator_load_queue,
    AIHOUSE_OT_validator_load_current,
    AIHOUSE_OT_validator_approve_next,
    AIHOUSE_OT_validator_reject_next,
    AIHOUSE_OT_validator_skip_next,
    AIHOUSE_OT_validator_reconstruct_scene,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
