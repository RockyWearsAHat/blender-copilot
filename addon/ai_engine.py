"""
AI Engine — Blender Copilot chat engine.

Architecture (v4 — lean single-pass copilot):
  1. User sends a prompt (text)
  2. Engine builds rich scene context (selected objects, viewport, full scene)
  3. AI generates an explanation + executable Python code
  4. Code is executed in Blender with full blender_tools namespace
  5. If code errors, AI auto-fixes (up to 2 retries)
  6. Conversation history is maintained for multi-turn editing
"""

import json
import logging
import re
import os
import base64
import urllib.request
import urllib.error
import urllib.parse
import ssl
import tempfile
import threading
import queue
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Local Mesh Server Detection
# ═══════════════════════════════════════════════════════════════════════════

def _is_local_mesh_server(server_url: str) -> bool:
    """Check if server is our local mesh generation server."""
    return "127.0.0.1" in server_url or "localhost" in server_url


def _generate_mesh_local(server_url: str, prompt: str, temperature: float = 0.8,
                         timeout: int = 60, max_faces: int = 512,
                         top_k: int = 96, top_p: float = 0.97,
                         cfg_scale: float = 1.5) -> dict:
    """Call local mesh generation server directly.
    
    Parameters:
    - prompt: text description of what to generate
    - temperature: sampling temperature (higher = more creative/random)
    - timeout: max seconds to wait for generation
    - max_faces: maximum faces to generate (capped server-side to model capacity)
    """
    def _post(payload_dict: dict) -> dict:
        data = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            f"{server_url}/generate/mesh",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    payload = {
        "prompt": prompt,
        "temperature": float(temperature),
        "max_faces": int(max_faces),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "cfg_scale": float(cfg_scale),
    }
    result = _post(payload)

    err = str(result.get("error", "")).lower() if isinstance(result, dict) else ""
    if "trivial primitive" in err or "collapsed" in err:
        retry_payload = dict(payload)
        retry_payload["temperature"] = max(float(temperature), 0.9)
        retry_payload["top_k"] = max(int(top_k), 96)
        retry_payload["top_p"] = max(float(top_p), 0.97)
        retry_payload["cfg_scale"] = min(float(cfg_scale), 1.5)
        retry_payload["max_faces"] = min(int(max_faces), 192)
        result = _post(retry_payload)

    return result


def _mesh_to_code(mesh_data: dict) -> str:
    """Convert mesh generation result to executable Blender code."""
    if not mesh_data.get("objects"):
        return "# No mesh generated"
    
    obj_data = mesh_data["objects"][0]
    mesh_info = obj_data["mesh"]
    name = obj_data["name"]
    vertices = mesh_info["vertices"]
    faces = mesh_info["faces"]
    
    # Generate Python code to create the mesh
    code_lines = [
        "import bpy",
        "import bmesh",
        "from mathutils import Vector",
        "",
        f"# Generated mesh: {name}",
        f"# {len(vertices)} vertices, {len(faces)} faces",
        "",
        "# Create mesh and object",
        "mesh = bpy.data.meshes.new(name='{}')" .format(name),
        "obj = bpy.data.objects.new(name='{}', object_data=mesh)".format(name),
        "bpy.context.collection.objects.link(obj)",
        "",
        "# Create mesh data",
        "bm = bmesh.new()",
        "",
        "# Add vertices",
        "verts = []",
    ]
    
    # Add vertices in batches to avoid huge single lines
    for i in range(0, len(vertices), 10):
        batch = vertices[i:i+10]
        for v in batch:
            code_lines.append(f"verts.append(bm.verts.new(({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})))")
    
    code_lines.extend([
        "",
        "bm.verts.ensure_lookup_table()",
        "",
        "# Add faces",
    ])
    
    # Add faces (skip duplicates)
    code_lines.append("_seen_faces = set()")
    for face in faces:
        v_refs = ", ".join(f"verts[{vi}]" for vi in face)
        face_key = tuple(sorted(face))
        code_lines.append(f"_fk = {face_key}")
        code_lines.append("if _fk not in _seen_faces:")
        code_lines.append("    _seen_faces.add(_fk)")
        code_lines.append("    try:")
        code_lines.append(f"        bm.faces.new([{v_refs}])")
        code_lines.append("    except ValueError:")
        code_lines.append("        pass")
    
    code_lines.extend([
        "",
        "# Update mesh and clean up",
        "bm.to_mesh(mesh)",
        "bm.free()",
        "mesh.update()",
        "",
        "# Center geometry at origin",
        "bpy.ops.object.select_all(action='DESELECT')",
        "obj.select_set(True)",
        "bpy.context.view_layer.objects.active = obj",
        "bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')",
        "obj.location = (0, 0, 0)",
        "",
        "# Shade smooth",
        "for face in mesh.polygons:",
        "    face.use_smooth = True",
    ])
    
    return "\n".join(code_lines)


# ═══════════════════════════════════════════════════════════════════════════
# Conversation History  (module-level, survives across calls in a session)
# ═══════════════════════════════════════════════════════════════════════════

_conversation = []   # list of {"role": "user"/"assistant", "content": str}
MAX_HISTORY = 30     # keep last N messages to limit token usage

# ── Chat logging ──────────────────────────────────────────────────────
_CHAT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_logs")
_chat_session_id = None  # set on first message, reset on clear

# ── Session generation counter — bumped on every clear_history() ──────
# Background threads capture this at start; if it changes before they
# finish, their writes to _conversation are silently discarded.
_session_generation = 0
_state_lock = threading.Lock()   # guards _conversation, _iteration_history, _session_generation

# ── Streaming state (updated by background thread, read by UI) ────────
_streaming_text = ""   # partial response text during streaming

# ── Iteration history (separate from main conversation) ───────────────
# Assessment and fix messages go here, NOT into _conversation.
# This keeps the chat display clean (only real user/AI messages)
# and avoids polluting the context with huge diagnostic prompts.
_iteration_history = []      # list of {"role": ..., "content": ..., "_seq": int}
MAX_ITERATION_HISTORY = 20   # keep recent iteration context only

# Finalized iterations — permanently accumulated so they're never lost
# from the chat log file. finalize_iteration() moves messages here.
_finalized_iterations = []   # list of {"role": "iteration_user/assistant", "content": ..., "_seq": int}

# Global sequence counter — ensures all messages can be sorted
# into the correct chronological order in the chat log.
_message_seq = 0


# ═══════════════════════════════════════════════════════════════════════════
# Main-Thread Dispatch  (background threads request bpy.context work here)
# ═══════════════════════════════════════════════════════════════════════════

_main_thread_queue = queue.Queue()


def _run_on_main_thread(fn):
    """Execute *fn* on the main thread and return its result."""
    if threading.current_thread() is threading.main_thread():
        return fn()

    result_holder: list = [None, None]
    event = threading.Event()

    def _wrapper():
        try:
            result_holder[0] = fn()
        except Exception as exc:
            result_holder[1] = exc
        event.set()

    _main_thread_queue.put(_wrapper)
    event.wait(timeout=30)

    if result_holder[1] is not None:
        raise result_holder[1]
    return result_holder[0]


def process_main_thread_queue():
    """Drain the dispatch queue.  **Must** be called from the main thread."""
    while True:
        try:
            fn = _main_thread_queue.get_nowait()
            fn()
        except queue.Empty:
            break
        except Exception:
            pass


def flush_main_thread_queue():
    """Discard all pending items in the main-thread queue.

    Call this when stopping a generation to prevent stale code from
    a previous session executing after a new prompt starts.
    Each discarded wrapper's event is set so blocked background threads
    unblock instead of hanging for 30 seconds.
    """
    discarded = 0
    while True:
        try:
            fn = _main_thread_queue.get_nowait()
            # Try to set the event so any blocked background thread unblocks
            try:
                if hasattr(fn, '__closure__') and fn.__closure__:
                    for cell in fn.__closure__:
                        obj = cell.cell_contents
                        if isinstance(obj, threading.Event):
                            obj.set()
                            break
            except (ValueError, AttributeError):
                pass
            discarded += 1
        except queue.Empty:
            break
    if discarded:
        print(f"[Blender Copilot] Flushed {discarded} stale queue items")


def get_session_generation():
    """Return the current session generation counter.

    Callers snapshot this BEFORE spawning a background thread, then
    compare again when the thread finishes. If it changed, the session
    was cleared/reset and results should be discarded.
    """
    return _session_generation


def get_streaming_text():
    return _streaming_text


def clear_streaming_text():
    global _streaming_text
    _streaming_text = ""


def _update_streaming(text):
    global _streaming_text
    _streaming_text = text


# ── Iteration history helpers ─────────────────────────────────────────

def _add_iteration_message(role, content, session_gen=None):
    """Store a message in iteration history (not the main conversation).

    These messages are used for API calls during assessment/fix loops
    but do NOT appear in the user's chat display.
    They ARE saved to the chat log file for debugging.
    """
    global _message_seq
    with _state_lock:
        if session_gen is not None and session_gen != _session_generation:
            return  # stale session — discard
        _message_seq += 1
        _iteration_history.append({"role": role, "content": content, "_seq": _message_seq})
        while len(_iteration_history) > MAX_ITERATION_HISTORY:
            _iteration_history.pop(0)
    # Auto-save so iteration logs are never lost
    _save_chat_log()


def clear_iteration_history():
    """Reset iteration history — called when starting a new prompt."""
    _iteration_history.clear()


def finalize_iteration():
    """Clean up after an iteration loop completes (success or failure).

    Moves iteration messages into _finalized_iterations so they persist
    in every future save of the chat log.  Then clears the active
    iteration history (used for API context window).
    """
    with _state_lock:
        for msg in _iteration_history:
            _finalized_iterations.append({
                "role": "iteration_%s" % msg["role"],
                "content": msg["content"],
                "_seq": msg.get("_seq", 0),
            })
        _iteration_history.clear()
    _save_chat_log()


def add_message(role, content, session_gen=None):
    """Append a message to the conversation.

    If *session_gen* is provided and doesn't match the current
    ``_session_generation``, the write is silently dropped (the session
    was cleared while a background thread was running).
    """
    global _chat_session_id, _message_seq
    with _state_lock:
        if session_gen is not None and session_gen != _session_generation:
            return  # stale session — discard
        if not _chat_session_id:
            _chat_session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _message_seq += 1
        _conversation.append({"role": role, "content": content, "_seq": _message_seq})
        while len(_conversation) > MAX_HISTORY:
            _conversation.pop(0)


def get_history():
    with _state_lock:
        return list(_conversation)


def _save_chat_log():
    """Write the current conversation to chat_logs/ as a JSON file."""
    global _chat_session_id
    if not _conversation:
        return  # nothing to save
    try:
        os.makedirs(_CHAT_LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sid = _chat_session_id or ts
        filename = "chat_%s.json" % sid
        filepath = os.path.join(_CHAT_LOG_DIR, filename)
        # Build combined message list sorted chronologically by _seq
        combined = []
        for msg in _conversation:
            combined.append({
                "role": msg["role"],
                "content": msg["content"],
                "_seq": msg.get("_seq", 0),
            })
        for msg in _finalized_iterations:
            combined.append({
                "role": msg["role"],
                "content": msg["content"],
                "_seq": msg.get("_seq", 0),
            })
        for msg in _iteration_history:
            combined.append({
                "role": "iteration_%s" % msg["role"],
                "content": msg["content"],
                "_seq": msg.get("_seq", 0),
            })
        combined.sort(key=lambda m: m["_seq"])
        # Strip _seq from output (internal bookkeeping only)
        all_messages = [{"role": m["role"], "content": m["content"]}
                        for m in combined]
        total_iterations = len(_finalized_iterations) + len(_iteration_history)
        payload = {
            "session_id": sid,
            "started": sid,
            "saved_at": ts,
            "message_count": len(_conversation),
            "iteration_count": total_iterations,
            "messages": all_messages,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("[Blender Copilot] Chat saved → %s" % filepath)
    except Exception as exc:
        print("[Blender Copilot] Failed to save chat log: %s" % exc)


def save_current_chat():
    """Public API — save the current chat (called on unregister / shutdown)."""
    _save_chat_log()


def clear_history():
    global _chat_session_id, _message_seq, _session_generation
    _save_chat_log()   # persist before clearing
    with _state_lock:
        old_gen = _session_generation
        _session_generation += 1   # invalidate all in-flight background threads
        import traceback as _tb
        caller = _tb.extract_stack(limit=3)
        caller_info = " → ".join(
            "%s:%d:%s" % (f.filename.split("/")[-1], f.lineno, f.name)
            for f in caller[:-1])
        print("[Blender Copilot] clear_history: session_gen %d → %d  caller: %s"
              % (old_gen, _session_generation, caller_info))
        _conversation.clear()
        _iteration_history.clear()
        _finalized_iterations.clear()
        _chat_session_id = None
        _message_seq = 0
    clear_streaming_text()


def get_display_history():
    result = []
    for msg in _conversation:
        role = msg["role"]
        text = msg["content"]
        display = re.sub(r'```[\s\S]*?```', '[code]', text)
        if len(display) > 120:
            display = display[:117] + "..."
        result.append((role, display))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Scene Context — tells the AI what currently exists in Blender
# ═══════════════════════════════════════════════════════════════════════════

def get_scene_context():
    return _run_on_main_thread(_get_scene_context_impl)


def _get_mesh_profile_data(obj, num_slices=8):
    """Generate compact mesh profile data along BOTH X and Y axes.

    Auto-detects which horizontal axis is the 'length' axis (longer span)
    and labels accordingly. Z is always up.

    Returns a string like:
    dim=[4.5, 1.9, 0.6] LENGTH_AXIS=X
    PROFILE along X (length): x=-2.0 w=1.2 h=0.5 | ...
    PROFILE along Y (width):  y=-0.9 w=4.0 h=0.5 | ...
    """
    try:
        from . import blender_tools
        dims = obj.dimensions
        x_span = dims.x
        y_span = dims.y
        z_span = dims.z

        # Determine which horizontal axis is "length" (longer)
        if x_span >= y_span:
            length_axis, width_axis = 'X', 'Y'
        else:
            length_axis, width_axis = 'Y', 'X'

        lines = []
        lines.append("dim=[%.2f, %.2f, %.2f] LENGTH_AXIS=%s" % (
            x_span, y_span, z_span, length_axis))

        # Profile along length axis
        profile_len = blender_tools.get_mesh_profile(obj, length_axis, num_slices)
        if profile_len:
            parts = []
            for s in profile_len:
                parts.append("%s=%.2f w=%.2f h=%.2f" % (
                    length_axis.lower(), s["pos"], s["width"], s["height"]))
            lines.append("PROFILE along %s (length): %s" % (
                length_axis, " | ".join(parts)))

        # Profile along width axis
        profile_wid = blender_tools.get_mesh_profile(obj, width_axis, 5)
        if profile_wid:
            parts = []
            for s in profile_wid:
                parts.append("%s=%.2f w=%.2f h=%.2f" % (
                    width_axis.lower(), s["pos"], s["width"], s["height"]))
            lines.append("PROFILE along %s (width):  %s" % (
                width_axis, " | ".join(parts)))

        return "\n    ".join(lines)
    except Exception:
        return ""


def _obj_to_dict(obj, include_profile=False):
    """Serialize a single Blender object to a compact dict.

    If *include_profile* is True, also includes cross-section profile data
    for mesh objects — gives the AI detailed shape information for editing.
    """
    import math as _math

    o = {"name": obj.name, "type": obj.type}

    loc = obj.location
    o["pos"] = [round(loc.x, 2), round(loc.y, 2), round(loc.z, 2)]

    dims = obj.dimensions
    if dims.length > 0.001:
        o["dim"] = [round(dims.x, 2), round(dims.y, 2), round(dims.z, 2)]

    rot = obj.rotation_euler
    if abs(rot.x) + abs(rot.y) + abs(rot.z) > 0.01:
        o["rot"] = [round(_math.degrees(rot.x)),
                    round(_math.degrees(rot.y)),
                    round(_math.degrees(rot.z))]

    if obj.parent:
        o["parent"] = obj.parent.name

    obj_cols = [c.name for c in obj.users_collection
                if c.name != "Scene Collection"]
    if obj_cols:
        o["col"] = obj_cols

    if obj.type == 'MESH' and obj.data:
        mesh = obj.data
        o["verts"] = len(mesh.vertices)
        o["faces"] = len(mesh.polygons)
        mats = [mm.name for mm in mesh.materials if mm]
        if mats:
            o["mat"] = mats
        if not mesh.uv_layers:
            o["uv"] = False
        if obj.modifiers:
            mods = []
            for mod in obj.modifiers:
                md = {"type": mod.type}
                if mod.type == 'BEVEL':
                    md["width"] = round(mod.width, 3)
                    md["seg"] = mod.segments
                elif mod.type == 'SUBSURF':
                    md["levels"] = mod.levels
                elif mod.type == 'SOLIDIFY':
                    md["thick"] = round(mod.thickness, 3)
                elif mod.type == 'MIRROR':
                    md["axes"] = [a for a, on in enumerate(mod.use_axis) if on]
                elif mod.type == 'ARRAY':
                    md["count"] = mod.count
                elif mod.type == 'BOOLEAN':
                    md["op"] = mod.operation
                mods.append(md)
            o["mods"] = mods

        # Include cross-section profile for detailed shape analysis
        if include_profile and len(mesh.vertices) >= 4:
            profile = _get_mesh_profile_data(obj)
            if profile:
                o["profile"] = profile

    elif obj.type == 'LIGHT' and obj.data:
        o["light"] = obj.data.type
        o["energy"] = round(obj.data.energy, 1)
        o["color"] = [round(c, 2) for c in obj.data.color]

    elif obj.type == 'CAMERA' and obj.data:
        o["lens"] = round(obj.data.lens, 1)

    elif obj.type == 'CURVE' and obj.data:
        o["splines"] = len(obj.data.splines)
        if obj.data.bevel_depth > 0:
            o["bevel"] = round(obj.data.bevel_depth, 3)

    return o


def _material_list():
    import bpy  # type: ignore
    result = []
    for mat in list(bpy.data.materials)[:30]:
        m = {"name": mat.name}
        if mat.use_nodes and mat.node_tree:
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bc = node.inputs.get("Base Color")
                    if bc and hasattr(bc, 'default_value'):
                        c = bc.default_value
                        m["col"] = [round(c[0], 2), round(c[1], 2), round(c[2], 2)]
                    r = node.inputs.get("Roughness")
                    if r and hasattr(r, 'default_value'):
                        m["rough"] = round(r.default_value, 2)
                    mt = node.inputs.get("Metallic")
                    if mt and hasattr(mt, 'default_value'):
                        m["metal"] = round(mt.default_value, 2)
                    a = node.inputs.get("Alpha")
                    if a and hasattr(a, 'default_value') and a.default_value < 0.99:
                        m["alpha"] = round(a.default_value, 2)
                    norm = node.inputs.get("Normal")
                    if norm and norm.is_linked:
                        m["normal_map"] = True
                    if any(n.type.startswith('TEX_') for n in mat.node_tree.nodes):
                        m["has_textures"] = True
                    break
                elif node.type == 'BSDF_GLASS':
                    m["shader"] = "glass"
                    break
                elif node.type == 'EMISSION':
                    m["shader"] = "emission"
                    break
        users = [o.name for o in bpy.data.objects
                 if o.data and hasattr(o.data, 'materials') and
                 mat.name in [mm.name for mm in o.data.materials if mm]]
        if users:
            m["used_by"] = users[:6]
        result.append(m)
    return result


def _get_viewport_center_and_radius():
    import bpy  # type: ignore
    for area in bpy.context.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for space in area.spaces:
            if space.type != 'VIEW_3D':
                continue
            r3d = space.region_3d
            if r3d is None:
                continue
            center = r3d.view_location.copy()
            radius = r3d.view_distance * 1.5
            return center, radius
    return None


def _get_scene_context_impl():
    """Build scene context with priority pipeline."""
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    obj_count = len(bpy.data.objects)
    if obj_count == 0:
        return "SCENE: empty — no objects exist yet."

    all_objs = list(bpy.data.objects)
    lines = []

    # Phase 1: Selected
    selected_set = set(o.name for o in bpy.context.selected_objects)
    selected_objs = [o for o in all_objs if o.name in selected_set]

    # Phase 2: Viewport
    viewport_info = _get_viewport_center_and_radius()
    viewport_set = set()
    if viewport_info:
        center, radius = viewport_info
        for obj in all_objs:
            if obj.name in selected_set:
                continue
            dist = (obj.location - center).length
            max_dim = max(obj.dimensions) if obj.dimensions.length > 0 else 0
            if dist - max_dim / 2 < radius:
                viewport_set.add(obj.name)
    viewport_objs = [o for o in all_objs if o.name in viewport_set]

    # Phase 3: Camera
    camera_set = set()
    cam = bpy.context.scene.camera
    if cam:
        cam_loc = cam.location
        cam_forward = cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))
        for obj in all_objs:
            if obj.name in selected_set or obj.name in viewport_set:
                continue
            to_obj = obj.location - cam_loc
            dist = to_obj.length
            if dist < 50.0 and dist > 0:
                if to_obj.normalized().dot(cam_forward) > 0.1:
                    camera_set.add(obj.name)
    camera_objs = [o for o in all_objs if o.name in camera_set]

    # Phase 4: Remaining
    remaining_objs = [o for o in all_objs
                      if o.name not in selected_set
                      and o.name not in viewport_set
                      and o.name not in camera_set]

    lines.append("SCENE: %d objects" % obj_count)

    # Collections
    cols = {}
    for c in bpy.data.collections:
        members = [o.name for o in c.objects]
        if not members:
            continue
        hidden = False
        try:
            vl = bpy.context.view_layer
            lc = vl.layer_collection.children.get(c.name)
            if (lc and lc.exclude) or c.hide_viewport:
                hidden = True
        except Exception:
            pass
        key = c.name + (" [HIDDEN]" if hidden else "")
        cols[key] = members[:20]
        if len(members) > 20:
            cols[key].append("...+%d" % (len(members) - 20))
    if cols:
        lines.append("COLLECTIONS: " + json.dumps(cols, separators=(',', ':')))

    # Materials
    mats = _material_list()
    if mats:
        lines.append("MATERIALS (reuse with find_material(\"Name\")):")
        for m in mats:
            lines.append(json.dumps(m, separators=(',', ':')))

    # Selected — include mesh profile data for detailed editing
    if selected_objs:
        lines.append("SELECTED (%d):" % len(selected_objs))
        for obj in selected_objs:
            lines.append(json.dumps(_obj_to_dict(obj, include_profile=True), separators=(',', ':')))

    # Viewport
    if viewport_objs:
        lines.append("VIEWPORT_NEARBY (%d):" % len(viewport_objs))
        for obj in viewport_objs:
            lines.append(json.dumps(_obj_to_dict(obj), separators=(',', ':')))

    # Camera
    if camera_objs:
        lines.append("CAMERA_VISIBLE (%d):" % len(camera_objs))
        for obj in camera_objs:
            lines.append(json.dumps(_obj_to_dict(obj), separators=(',', ':')))

    # Remaining
    if remaining_objs:
        if len(remaining_objs) > 80:
            remaining_objs.sort(
                key=lambda o: max(o.dimensions) if o.dimensions.length > 0 else 0,
                reverse=True)
            keep = remaining_objs[:60]
            dropped = len(remaining_objs) - 60
            lines.append("OTHER (%d, %d small objects omitted):" % (len(keep), dropped))
            for obj in keep:
                lines.append(json.dumps(_obj_to_dict(obj), separators=(',', ':')))
        else:
            lines.append("OTHER (%d):" % len(remaining_objs))
            for obj in remaining_objs:
                lines.append(json.dumps(_obj_to_dict(obj), separators=(',', ':')))

    # World
    world = bpy.context.scene.world
    if world and world.use_nodes and world.node_tree:
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                ci = node.inputs.get("Color")
                si = node.inputs.get("Strength")
                if ci and hasattr(ci, 'default_value'):
                    c = ci.default_value
                    w = {"bg": [round(c[0], 2), round(c[1], 2), round(c[2], 2)]}
                    if si:
                        w["strength"] = round(si.default_value, 2)
                    lines.append("WORLD: " + json.dumps(w, separators=(',', ':')))
                break
            elif node.type == 'TEX_SKY':
                lines.append("WORLD: sky_texture")
                break

    # Render settings
    render = bpy.context.scene.render
    lines.append("RENDER: engine=%s, res=%dx%d" % (
        render.engine, render.resolution_x, render.resolution_y))

    # Warnings
    mesh_objs = [o for o in all_objs if o.type == 'MESH']
    no_mat = [o.name for o in mesh_objs
              if not o.data.materials or all(m is None for m in o.data.materials)]
    if no_mat:
        lines.append("WARNING: no material on: " + ", ".join(no_mat[:15]))

    return "\n".join(lines)


def get_mesh_context_for_iteration():
    """Get detailed mesh profile data for ALL mesh objects.

    Used during assessment/iteration to give the AI fine-grained
    shape information it can use for in-place edits.
    """
    return _run_on_main_thread(_get_mesh_context_impl)


def _get_mesh_context_impl():
    """Build mesh analysis context for all key objects.

    Profiles both X and Y axes. Auto-detects which is length.
    Z is always vertical.
    """
    import bpy  # type: ignore
    lines = ["MESH PROFILES (Z=up, LENGTH_AXIS=whichever of X/Y is longer):"]
    for obj in list(bpy.data.objects)[:20]:  # cap to avoid huge context
        if obj.type != 'MESH' or not obj.data or len(obj.data.vertices) < 4:
            continue
        profile = _get_mesh_profile_data(obj, num_slices=8)
        if profile:
            lines.append("  %s:\n    %s" % (obj.name, profile))
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def get_selection_context():
    return _run_on_main_thread(_get_selection_context_impl)


def _get_selection_context_impl():
    import bpy  # type: ignore
    selected = list(bpy.context.selected_objects)
    active = bpy.context.active_object
    if not selected:
        return "No objects selected."
    lines = ["Selected %d object(s):" % len(selected)]
    for obj in selected[:20]:
        d = _obj_to_dict(obj)
        if obj == active:
            d["active"] = True
        lines.append(json.dumps(d, separators=(',', ':')))
    if len(selected) > 20:
        lines.append("... and %d more" % (len(selected) - 20))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Tool Reference — auto-generated from blender_tools.py
# ═══════════════════════════════════════════════════════════════════════════

def _get_tools_reference():
    import ast as _ast
    tools_path = os.path.join(os.path.dirname(__file__), "blender_tools.py")
    try:
        with open(tools_path, "r") as f:
            tree = _ast.parse(f.read())
    except Exception:
        return "# blender_tools.py not found"

    _HIDDEN_TOOLS = {
        'frame_all', 'set_viewport_shading', 'capture_viewport',
        # Hide coordinate-heavy functions that LLMs can't use well
        'shape_from_profiles', 'mesh_from_outlines', 'revolve_profile',
        'extrude_shape', 'loft_sections', 'profile_ring',
        'create_bezier_curve', 'create_nurbs_curve', 'create_profile_from_points',
        'sweep_profile_along_curve', 'create_mesh',
    }

    lines = ["# blender_tools — pre-imported helpers (call directly)"]
    for node in _ast.iter_child_nodes(tree):
        if not isinstance(node, _ast.FunctionDef):
            continue
        if node.name in _HIDDEN_TOOLS or node.name.startswith('_'):
            continue
        args = node.args
        params = []
        defaults_offset = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            name = arg.arg
            didx = i - defaults_offset
            if didx >= 0:
                params.append(name + "=" + _ast.unparse(args.defaults[didx]))
            else:
                params.append(name)
        if args.vararg:
            params.append("*" + args.vararg.arg)
        if args.kwonlyargs:
            if not args.vararg:
                params.append("*")
            for j, kw in enumerate(args.kwonlyargs):
                _default = args.kw_defaults[j]
                kwd = _ast.unparse(_default) if _default is not None else ""
                params.append(kw.arg + ("=" + kwd if kwd else ""))
        if args.kwarg:
            params.append("**" + args.kwarg.arg)
        sig = node.name + "(" + ", ".join(params) + ")"
        doc = _ast.get_docstring(node) or ""
        first = doc.split("\n")[0].strip() if doc else ""
        if first:
            lines.append(sig + "  # " + first)
        else:
            lines.append(sig)

    # Also include procedural materials from materials.py
    mat_path = os.path.join(os.path.dirname(__file__), "materials.py")
    try:
        with open(mat_path, "r") as f:
            mat_tree = _ast.parse(f.read())
        lines.append("")
        lines.append("# Procedural PBR materials (also pre-imported):")
        for node in _ast.iter_child_nodes(mat_tree):
            if not isinstance(node, _ast.FunctionDef):
                continue
            if node.name.startswith('_'):
                continue
            args = node.args
            params = []
            defaults_offset = len(args.args) - len(args.defaults)
            for i, arg in enumerate(args.args):
                name = arg.arg
                didx = i - defaults_offset
                if didx >= 0:
                    params.append(name + "=" + _ast.unparse(args.defaults[didx]))
                else:
                    params.append(name)
            sig = node.name + "(" + ", ".join(params) + ")"
            doc = _ast.get_docstring(node) or ""
            first = doc.split("\n")[0].strip() if doc else ""
            if first:
                lines.append(sig + "  # " + first)
            else:
                lines.append(sig)
    except Exception:
        pass

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# System Prompt — General-purpose Blender Copilot
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are **Blender Copilot** — an expert 3D artist and Blender Python developer.
You can create, modify, and manage anything in Blender: models, materials,
lighting, cameras, animations, simulations, render settings, and more.

═══ RESPONSE FORMAT ═══
1. Brief explanation of your approach (1-3 sentences).
2. Python code in a single ```python ... ``` block.
   If the user asks a question that doesn't need code, just answer
   without a code block.

═══ CODE ENVIRONMENT ═══
Pre-imported: bpy, bmesh, math, Vector, Matrix, Euler.
All functions from blender_tools are pre-imported (see reference below).
When modifying existing objects: use get("Name") and check for None.
get_bounds(obj) returns Bounds with .min_x .max_z .center_x .width etc.

═══ SCENE DATA FORMAT ═══
Scene data is provided as structured JSON-lines by priority:
  SELECTED — objects the user has selected (highest priority)
  VIEWPORT_NEARBY — objects near the user's current view
  CAMERA_VISIBLE — objects the scene camera can see
  OTHER — everything else
Each object: {{"name":"X","type":"MESH","pos":[x,y,z],"dim":[w,d,h],
"verts":N,"faces":N,"mat":["MatName"],"col":["CollName"],...}}
Selected objects include "profile" data — cross-section measurements
along BOTH X and Y axes, with LENGTH_AXIS auto-detected.

═══ AXIS CONVENTIONS ═══
  • Z is ALWAYS up/down (vertical).
  • X and Y are BOTH horizontal — they are interchangeable.
  • There is NO fixed convention for which horizontal axis is "forward".
  • When you create an object, YOU choose which axis is length vs width.
  • create_box(width=W, depth=D, height=H) maps width→X, depth→Y, height→Z.
  • When editing existing objects, ALWAYS check the actual dimensions first:
    - dims = obj.dimensions → (X_size, Y_size, Z_size)
    - The LONGER horizontal dimension is the length axis.
    - The SHORTER horizontal dimension is the width axis.
    - MESH PROFILES show LENGTH_AXIS=X or LENGTH_AXIS=Y to help you.
  • Pass the CORRECT axis to taper(), scale_section(), etc. based on
    the actual geometry, NOT an assumption.

═══ CORE PRINCIPLES ═══

1. SPATIAL AWARENESS
   EVERY new object must be positioned RELATIVE to existing geometry.
   NEVER hardcode coordinates when objects already exist.
   Pattern: ref = get("Name"); b = get_bounds(ref); use b.max_z etc.

2. PROFESSIONAL MODELING
   • Build from sub-parts, not single primitives. A table = legs + top.
   • Define dimensions as named variables at top of script (meters).
   • Everything has thickness — no paper-thin geometry.
   • Use BMesh ops (extrude_faces, inset_faces, bevel_edges) for detail.
   • Real-world scale: human ~1.75m, door ~0.9×2.1m, table ~0.75m tall.

3. CLEAN WORKFLOW
   • Reuse existing materials: find_material("Name") or quick_material().
   • Different surfaces get different materials — no grey defaults.
   • For multi-part objects: use assign_material() on each separate object.
   • For single-mesh objects with multiple surfaces: use assign_material_to_faces()
     or assign_material_by_normal() to apply different materials to different faces.
   • Procedural PBR materials available: make_wood(), make_brick(), make_concrete(),
     make_stone(), make_metal(), make_stucco(), make_grass(), make_glass().
   • quick_material() accepts hex colors: quick_material("Wood", color="#8B4513").
   • Organize into semantic collections with new_collection() + link_to_collection().
   • merge_by_distance() after booleans. recalc_normals() for clean mesh.
   • apply_transforms() before booleans.

4. RESPECT THE SCENE
   • When modifying: NEVER call clear_scene() unless explicitly asked.
   • Only change what the user asked for. Preserve everything else.
   • Reference existing objects with get("Name"), always check for None.

5. NEVER control the viewport: no set_viewport_shading(), frame_all(),
   or camera manipulation unless the user specifically asks.

═══ HOW YOU WORK ═══

**STEP 1 — ANALYZE THE REQUEST**
Before writing ANY code, decompose the user's prompt into components:
  • SUBJECT — What specific thing? ("lamborghini svj" → specific car model)
  • STYLE — How should it look? ("low poly" → faceted, limited geometry)
  • MATERIAL/COLOR — What finish? ("matte black" → dark, matte surface)
  • CONTEXT — Scene requirements? ("on a road" → needs ground plane)
State each component explicitly in your response so the user sees your
understanding before you start building.

**STEP 2 — BUILD A MENTAL BLUEPRINT**
Use your training knowledge to describe the subject's key specs:
  • Real-world dimensions (length, width, height in meters)
  • Key proportions and ratios that make it recognizable
  • Distinctive visual features (what makes THIS thing look like itself?)
  • Front / side / top silhouette characteristics
  • For the SIDE PROFILE: describe the height at MANY positions
    along the length — this directly becomes your top_curve values.
    Focus on WHERE the profile changes sharply vs stays flat.
  • For the TOP VIEW: describe where the body is WIDE (fenders),
    where it NARROWS (cabin/doors), and how the nose/tail taper.
Write this blueprint out — it becomes your modeling target.
YOUR CURVE VALUES MUST MATCH THESE DIMENSIONS. If the subject is
1.1m tall, your top_curve peak must reach ~1.1. If the nose is
0.35m, your top_curve must start near 0.35. VERIFY before coding.

CRITICAL — what makes a shape RECOGNIZABLE vs GENERIC:
  • It's the TRANSITIONS, not the overall outline. A Lamborghini has
    a very flat hood that SUDDENLY rises at the windshield. A Porsche
    has a gradual rise from nose to roof. These differences are what
    makes each car look like ITSELF.
  • Every real object has flat areas and sharp transitions between them.
    Your curves must capture BOTH — flat stretches AND abrupt changes.
  • Think about the cross-section shape at different stations:
    - Front: narrow, low, wedge-like
    - At cabin: wider body, taller, flat floor
    - At rear fenders: WIDEST (for sports cars with fender flares)
    - Tail: narrowing quickly

Example for a Lamborghini Aventador SVJ:
  Length ≈ 4.80 m, Width ≈ 2.03 m, Height ≈ 1.14 m, Wheelbase ≈ 2.70 m
  ⚠️ Height is 1.14m for BOTH coupe AND Spyder/Roadster versions.
  Convertibles are the SAME height — only the removable roof panel
  differs (~2-3cm). Your top_curve peak MUST reach ~1.12-1.14m.
  Extremely low wedge — hood height only ~0.40m for over 1.5m of length!
  Then STEEP windshield rise (0.40→0.90m in just ~0.3m)
  Very low roofline (it's mid-engine — cabin is FORWARD)
  Engine cover behind cabin is slightly lower than roof
  DRAMATIC rear fender flares — rear is WIDER than front
  ⚠️ Angular design → use sharpness=8.0, flat_top=0.7
  flat_top creates the FLAT HOOD and FLAT ROOF that define angular cars.
  Without flat_top, the body will look like a rounded blob.
  Triangular headlights, hexagonal intakes
  Massive rear diffuser and high-mounted wing on SVJ variant

**STEP 3 — CREATE A MODELING PLAN**
Write a numbered checklist of every part you'll build:
  1. Body shell — MOST IMPORTANT. Build with correct resolution
     (num_sections=20, ring_points=16), set flat_top for angular
     vehicles, then flat_shade for low-poly look.
     ⚠️ Do NOT use limited_dissolve on vehicle bodies.
  2. Wheels × 4 (tires + rims) at correct wheelbase and track width
  3. Windshield (glass panel, angled, matching body width)
  4. Headlights & taillights (emissive boxes at corners)
  5. Spoiler / wing
  6. Air intakes, mirrors, diffuser, grille
  7. Materials & colors
  ...etc.
⚠️ BUILD ALL PARTS IN YOUR INITIAL CODE BLOCK. Do NOT plan to
add details one-at-a-time in later iterations. Build body + wheels +
windshield + lights + wing ALL in iteration 0.

**STEP 4 — BUILD IT**
Execute your plan in one comprehensive code block. Key rules:
  • Name every object clearly — you'll need to find them later.
  • For the main body, choose the right tool for the shape:

    ★ shape_from_profiles() — Best for elongated shapes (vehicles, boats,
      aircraft, furniture, bottles). Define 3 intuitive curves:
        top_curve=[(pos, z), ...]      — roofline / top edge in side view
        bottom_curve=[(pos, z), ...]   — underside in side view
        width_curve=[(pos, hw), ...]   — half-width from top view
      Use sharpness=2 for organic, 3 for rounded vehicles,
      8.0 for angular/aggressive vehicles (supercars, sports cars), 10+ for boxy.
      Use bottom_flat=0 for round, 0.5 for cars, 1.0 for flat-bottomed.
      Use flat_top=0.7 for angular vehicles (flat hood/roof),
      flat_top=0.0 for organic shapes (bottles, fish).
      ⚠️ flat_top is CRITICAL for angular vehicles! Without it, the
      body will have a domed top that looks blobby, not angular.

      ⚠️ VEHICLE BODIES — DO NOT USE limited_dissolve:
      limited_dissolve destroys center-seam vertices, creating a
      tent/groove artifact visible from the front. Instead:
        body = shape_from_profiles(..., num_sections=20, ring_points=16)
        flat_shade(body)  # 20×16 with flat_shade IS low-poly
      The 320-face mesh with flat shading looks properly faceted.
      Only use limited_dissolve on non-vehicle shapes (buildings, etc).

      ⚠️ CRITICAL BOUNDS AXIS MAPPING:
      b = get_bounds(body)  # for a car aligned along X axis:
        b.width  = X extent = CAR LENGTH (NOT the lateral width!)
        b.depth  = Y extent = CAR LATERAL WIDTH
        b.height = Z extent = CAR HEIGHT
      For placing parts on left/right sides of the car:
        USE:  y = side * (b.depth * 0.5 - inset)
        NOT:  y = side * (b.width * 0.5 - inset)  ← WRONG! Uses length!

      ⚠️ CURVE QUALITY IS EVERYTHING — the 3D shape is only as good as
      the curves that define it. Each curve point should correspond to
      a real feature on the subject (nose tip, hood line, windshield
      base, roof peak, engine deck, tail, etc.).

      KEY PRINCIPLE: Use enough geometry for an accurate silhouette.
        • Vehicles: 20-30 control points per curve, num_sections=20,
          ring_points=16. This gives ~320 faces — perfect for low-poly.
        • Other subjects: 12-20 control points, num_sections=24+.
        • NEVER build with num_sections < 16 or ring_points < 12 —
          that just creates unrecognizable blobs.
      Derive Z-values and half-widths from your STEP 2 mental blueprint.
      The top_curve Z-values should span from ground-clearance height
      at nose to full vehicle height at the roof peak.

      ⚠️ CURVE VALUES MUST HAVE SHARP TRANSITIONS, NOT SMOOTH RAMPS.
      Real vehicles have ABRUPT changes in profile:
        - Hood is FLAT for a stretch, then JUMPS steeply at windshield
        - Roof is relatively FLAT, then DROPS sharply at rear window
        - Width HOLDS CONSTANT through middle, PINCHES at nose/tail
      Use CLUSTERS of close points at transition zones (windshield base,
      beltline, fender edges) and FEWER points in flat stretches.
      BAD: (1.0, 0.40), (1.5, 0.70), (2.0, 1.00)  ← smooth ramp
      GOOD: (1.0, 0.42), (1.3, 0.43), (1.5, 0.44), (1.6, 0.70),
            (1.7, 0.95), (1.8, 1.05), (2.0, 1.10)  ← flat then JUMP

      Vehicle example (mid-engine sports car, ~4.5m long, ~1.1m tall):
        top_curve = [  # 20 points — note SHARP transitions
            (0.0, 0.38),   # nose tip
            (0.3, 0.35),   # front splitter — FLAT and LOW
            (0.6, 0.36),   # hood stays flat
            (0.9, 0.38),   # still flat hood
            (1.2, 0.40),   # hood still low — this is a sports car!
            (1.4, 0.42),   # approaching windshield
            (1.5, 0.55),   # SHARP RISE — windshield base
            (1.6, 0.80),   # steep windshield climb
            (1.7, 1.00),   # near top of windshield
            (1.8, 1.08),   # roof leading edge
            (2.0, 1.12),   # roof peak
            (2.2, 1.10),   # roof still high
            (2.5, 1.05),   # rear window / engine cover start
            (2.8, 0.95),   # engine cover descent
            (3.2, 0.82),   # engine bay
            (3.6, 0.72),   # rear haunches
            (4.0, 0.62),   # approaching tail
            (4.2, 0.55),   # rear deck
            (4.4, 0.50),   # tail
            (4.5, 0.45)]   # tail tip
        bottom_curve = [  # 12 points
            (0.0, 0.12), (0.3, 0.10), (0.7, 0.08), (1.2, 0.06),
            (1.8, 0.06), (2.2, 0.06), (2.8, 0.06), (3.2, 0.08),
            (3.6, 0.10), (4.0, 0.14), (4.3, 0.18), (4.5, 0.22)]
        width_curve = [  # 14 points — captures fender flares
            (0.0, 0.40),   # narrow nose
            (0.3, 0.55),   # widening
            (0.6, 0.72),   # front fenders start
            (0.9, 0.88),   # front fender peak
            (1.2, 0.95),   # behind front wheels
            (1.5, 0.85),   # door pinch — NARROWER than fenders
            (2.0, 0.82),   # cabin is narrow
            (2.5, 0.88),   # widening to rear
            (3.0, 1.00),   # rear fender flare — WIDEST
            (3.3, 1.02),   # rear haunch peak
            (3.6, 0.98),   # past rear wheels
            (4.0, 0.80),   # narrowing to tail
            (4.3, 0.55),   # tail narrows
            (4.5, 0.38)]   # tail tip
        body = shape_from_profiles("Body", top_curve, bottom_curve,
            width_curve, sharpness=8.0, bottom_flat=0.6, flat_top=0.7,
            num_sections=20, ring_points=16)
        flat_shade(body)  # angular faceted low-poly look
        # ⚠️ Do NOT use limited_dissolve on vehicle bodies —
        # it destroys center-seam vertices → tent/groove artifact.
        # 20×16 with flat_shade IS low-poly.
      Note how:
        - top_curve HOLDS FLAT at 0.35-0.42 for the hood, then JUMPS
          0.42→0.80 in just 0.2m at the windshield — that's how real
          cars work. Smooth ramps = generic blob.
        - width_curve has a PINCH at the cabin (0.82-0.85) between
          wider fenders (0.95 front, 1.02 rear) — this creates the
          muscular look of a sports car.
        - Curves are DENSE near transitions, SPARSE in flat areas.
        - sharpness=8.0 gives angular cross-sections for supercars.
          flat_top=0.7 flattens the hood/roof for that angular look.
          Use sharpness=3.0 + flat_top=0.0 for rounded vehicles.

    ★ WHEEL PLACEMENT — correct pattern for L/R mirroring:
        # Place wheels at correct wheelbase + track positions
        front_axle = 0.92   # front overhang from nose
        rear_axle  = front_axle + 2.70  # + wheelbase
        half_track = 1.00   # half of track width
        # Create EACH wheel independently at its position:
        for axle_x, tag in [(front_axle, 'F'), (rear_axle, 'R')]:
            for side_y, side in [(half_track, 'R'), (-half_track, 'L')]:
                tire = create_cylinder(name='Tire_%s%s' % (tag, side),
                    radius=0.34, depth=0.28, vertices=16,
                    location=(axle_x, side_y, 0.34))
                rotate_deg(tire, x=90)  # align to Y axis
                assign_material(tire, find_material('TireMat'))
                flat_shade(tire)
      ⚠️ Do NOT duplicate+offset for L/R mirroring — it causes placement
      bugs. Create EACH part at its EXACT final position.

    ★ mesh_from_outlines() — Best when tracing from reference images.
      Trace the CLOSED silhouette from side view and top view:
        side_outline=[(pos, z), ...]  — closed polygon around side silhouette
        top_outline=[(pos, y), ...]   — closed polygon around top silhouette
      The function intersects both outlines to carve the 3D shape.

    ★ revolve_profile() — Best for rotationally symmetric shapes
      (bottles, vases, columns, wheels, chess pieces, lamp shades).
        profile=[(radius, height), ...]  — the profile curve

    ★ extrude_shape() — Best for uniform cross-section shapes
      (brackets, logos, panels, floor plans).
        outline=[(a, b), ...]  — 2D cross-section polygon

  • Use create_mesh(verts, faces) for flat panels (glass, trim).
  • Use create_cylinder/create_box for simple sub-parts.
  • Apply materials to everything — no grey defaults.
  • Position parts using real dimensions from your blueprint.

  ★ "LOW POLY" STYLE — THE CORRECT WORKFLOW:
    The WRONG way: build with sparse geometry from the start.
      → This creates unrecognizable blobs. NEVER do this.
    The RIGHT way:
      FOR VEHICLES (cars, trucks, etc.):
        shape_from_profiles(..., num_sections=20, ring_points=16,
            sharpness=8.0, flat_top=0.7)
        flat_shade(body)
        ⚠️ Do NOT use limited_dissolve on vehicle bodies!
        It destroys center-seam vertices → tent/groove artifact.
        20×16 with flat_shade already gives clean low-poly (~320 faces).

      FOR NON-VEHICLES (buildings, furniture, etc.):
        shape_from_profiles(..., num_sections=32, ring_points=24)
        limited_dissolve(obj, angle_limit=5.0)
        flat_shade(obj)

    ⚠️ SHARPNESS + FLAT_TOP for angular vehicles:
    sharpness=3.0, flat_top=0.0  → rounded cross-sections (sedans, SUVs)
    sharpness=5.0, flat_top=0.3  → semi-angular (muscle cars)
    sharpness=8.0, flat_top=0.7  → angular with flat hood/roof (supercars)
    sharpness=10.0, flat_top=0.9 → very boxy (trucks, buildings)
    For Lamborghini/Ferrari/McLaren: sharpness=8.0, flat_top=0.7.
    flat_top flattens the TOP surface → flat hood and roof panels.
    Without flat_top, even high sharpness produces a domed top.

    ⚠️ Do NOT use decimate(COLLAPSE) for low-poly — it randomly removes
    vertices and DESTROYS your carefully designed curves (e.g. hood at
    0.38m becomes 0.22m after decimate).

After your code executes, the system will take a viewport screenshot
and show it to you. You then assess and refine.

**STEP 5 — ASSESS & REFINE**
You will receive viewport screenshots AND mesh profile data after each pass.
CRITICAL: You are a 3D modeler, not just a code writer. Your goal is to
make the model RECOGNIZABLE as the specific subject.

The #1 priority across ALL iterations is the BODY SHAPE. If the body
doesn't look like the subject, no amount of accessories will save it.

For each iteration:
  1. STUDY the screenshot + mesh profile data against your mental blueprint
  2. DESCRIBE what's wrong (be specific: "hood should be FLAT at 0.40m
     but rises gradually to 0.70m — needs sharper windshield transition")
  3. Write code to fix the MOST IMPORTANT problem:

  ★ Body shape is wrong → REBUILD with better curves.
    Don't waste iterations with scale_section/sculpt_move on a shape
    that's fundamentally not right. A rebuild with good curves takes
    1 iteration. Trying to nudge bad geometry into shape takes 5+
    iterations and still looks bad. When rebuilding:
    - Use 20-30 control points with SHARP transitions
    - Build at num_sections=20, ring_points=16
    - flat_shade(body) — do NOT use limited_dissolve on vehicles

  ★ Body shape is close but needs tweaks → in-place tools:
    • taper(obj, axis, start_scale, end_scale)
    • sculpt_move(obj, center, radius, offset)
    • scale_section(obj, axis, position, tol, scale_y=, scale_z=)
    • pinch / bulge / bend / move_section
    • find_verts_near() + move_verts() for precise edits

  ★ Body shape is good → add detail parts:
    • Windshield, lights, intakes, spoiler, trim, etc.
    • Each as a separate object with its own material.

  4. Focus on ONE specific improvement per iteration.
  5. When the result genuinely matches your blueprint → COMPLETE: <summary>

You decide how many refinement passes are needed. Simple objects might
need 1-2. Complex subjects might need 5+. But each pass MUST make
measurable progress or you should declare COMPLETE.

═══ TOOLS BY SHAPE TYPE ═══
  ★ Elongated bodies (vehicles, boats, furniture, bottles, aircraft):
    → shape_from_profiles(name, top_curve, bottom_curve, width_curve,
        sharpness=3.0, bottom_flat=0.5, flat_top=0.0)
    Define 3 curves: top edge, bottom edge, half-width.
    For angular vehicles: sharpness=8.0, flat_top=0.7
    This is the PRIMARY body-building tool.

  ★ Traced from reference images:
    → mesh_from_outlines(name, side_outline, top_outline,
        sharpness=3.0, bottom_flat=0.5, flat_top=0.0)
    Trace closed silhouettes from side + top views.

  ★ Rotationally symmetric (bottles, vases, columns, wheels):
    → revolve_profile(name, profile=[(radius, height), ...], segments=32)

  ★ Uniform cross-section (brackets, logos, panels):
    → extrude_shape(name, outline=[(a, b), ...], depth=0.1)

  • Flat panels → create_mesh(name, verts=[...], faces=[...])
  • Round parts → create_cylinder(name, radius, depth, vertices=16)
  • Angular parts → create_box() + loop_cut() + move_verts()
  • Organic → create_ico_sphere() + subsurf + proportional_translate()

═══ QUALITY STANDARDS ═══
  What makes a model look PROFESSIONAL:
  • Accurate silhouette — matches the real subject from every angle.
  • Correct proportions — length:width:height ratios are right.
  • Recognizable as the SPECIFIC subject, not just the category.
    A Lamborghini should look like a Lamborghini, not "some car."
  • Multiple distinct parts — body, glass, wheels, lights, trim are
    ALL separate objects with different materials.

  ⚠️ RECOGNITION REALITY CHECK:
  The body shell from shape_from_profiles will always look somewhat
  generic on its own — it's a superellipse loft, not a precise CAD model.
  What makes a car RECOGNIZABLE is the COMBINATION of:
    body shell (proportions + flat_top for angular look) + wheels +
    windshield + headlights + intakes + spoiler + correct materials.
  If the body dimensions are within ~15% of target, it IS recognizable —
  move on to adding detail parts. Do NOT keep rebuilding the body
  expecting it to magically look like a perfect rendering.

  ⚠️ DETAIL PARTS MUST BE VISIBLE:
  Detail parts that are too small are invisible and waste iterations.
  On a ~4.5m vehicle, minimum dimensions for visible detail parts:
    • Headlights:  width >= 0.30m, height >= 0.10m
    • Taillights:  width >= 0.25m, height >= 0.08m
    • Windshield:  width should match body width, depth >= 0.05m
    • Side mirrors: >= 0.12m in each dimension
    • Air intakes:  width >= 0.30m
  Parts smaller than 0.10m in any dimension will be INVISIBLE.
  Scale detail parts to 5-8% of body length minimum.

  • Proper shading:
    - "low poly" vehicles = shape_from_profiles(num_sections=20,
      ring_points=16) + flat_shade(). No limited_dissolve on vehicles.
    - "low poly" non-vehicles = shape_from_profiles(32, 24) +
      limited_dissolve() + flat_shade().
    - Detailed/realistic = smooth_shade() + shade_auto_smooth(angle=30).
  • Material variety — different surfaces get different materials.
    For generated meshes (single object): use assign_material_to_faces()
    or assign_material_by_normal() to apply different materials to
    different regions (top, sides, bottom).
    Example workflow for a generated fruit bowl:
      bowl = get("FruitBowl")
      clay = make_stucco("Clay", color=(0.85, 0.7, 0.5))
      glaze = quick_material("Glaze", color=(0.9, 0.85, 0.75), roughness=0.2)
      assign_material(bowl, clay)  # base material for all faces
      assign_material_by_normal(bowl, glaze, axis='Z', direction='UP', threshold=0.6)
  • No floating geometry — everything connects or sits properly.
  • Clean topology — merge_by_distance(), recalc_normals().

  What makes a model look AMATEUR (avoid these):
  ✗ Using limited_dissolve on vehicle bodies — it destroys center-seam
    vertices and creates a tent/groove artifact visible from the front.
  ✗ Using decimate(COLLAPSE) for low-poly — it randomly removes
    vertices and DESTROYS your designed shape.
  ✗ Confusing b.width (X=length) with b.depth (Y=lateral width)
    when positioning parts. CHECK THE AXIS MAPPING.
  ✗ Smooth, gradual curves where there should be SHARP transitions.
    Real objects have flat stretches and abrupt angle changes.
  ✗ Wrong proportions (too tall, too narrow, too short).
  ✗ Giving up on the body shape and adding accessories instead.
    A spoiler on a blob is still a blob with a spoiler.
  ✗ Using scale_section/sculpt_move to fix a fundamentally wrong shape.
    If the base shape doesn't look like the subject, REBUILD it with
    better curves — don't try to push vertices around.
  ✗ Single material on everything.
  ✗ Declaring "done" when it clearly doesn't look like the subject.

═══ IN-PLACE REFINEMENT WORKFLOW ═══
When iterating on an existing model, follow this workflow:
  1. FIRST: Check the MESH PROFILES for each object — they show:
     dim=[x, y, z] LENGTH_AXIS=X or =Y
     PROFILE along X/Y: positions, widths, heights at each cross-section
  2. Compare actual dimensions against your blueprint.
  3. STRONGLY PREFER in-place editing tools over delete-and-rebuild:
     • taper(obj, axis, start_scale, end_scale) — narrow/widen along axis
     • sculpt_move(obj, center, radius, offset) — push/pull regions
     • scale_section(obj, axis, pos, tol, scale_y=, scale_z=) — resize
     • move_section(obj, axis, pos, tol, offset=) — shift cross-section
     • pinch(obj, axis, pos, radius, strength) — narrow locally
     • bulge(obj, axis, pos, radius, strength) — widen locally
     • add_detail_cuts(obj, axis, num) — more vertices for finer edits
     • find_verts_near() + move_verts() — precise vertex adjustments
  4. Add MISSING PARTS as separate objects. Each significant feature
     should be its own object with its own material.
  5. Check shading: use flat_shade() for low-poly, smooth+auto for detailed.
  6. Only declare COMPLETE when it genuinely looks like the subject.

  ⚠️ ITERATION DISCIPLINE:
  • Each iteration must focus on ONE specific, measurable improvement.
    State what you're changing and by how much BEFORE writing code.
  • BODY SHAPE comes FIRST. Do NOT add accessories (spoiler, lights,
    intakes) until the body passes the recognition test (items 1-3).
    A spoiler on a blob is still a blob.
  • If the body shape is fundamentally wrong (doesn't look like the
    subject, just looks like a generic wedge/blob), REBUILD IT with
    better curves. This is the fastest path to a good result.
    Rebuilding with BETTER curves is GOOD.
    Rebuilding with the SAME quality curves is WASTEFUL.
  • You may rebuild the body AT MOST 1 time total across all iterations.
    The initial build = attempt #1. ONE rebuild = attempt #2. That's it.
    After 1 rebuild, you MUST use in-place tools (sculpt_move,
    scale_section, bulge, pinch, taper) or declare COMPLETE.
    Rebuilding with similar curves wastes iterations — use in-place edits.
  • If you cannot make the model significantly better with the available
    tools, declare COMPLETE with an honest summary of what was achieved.
  • When you rebuild, ALWAYS use the correct workflow:
    shape_from_profiles(..., num_sections=20, ring_points=16,
        sharpness=8.0, flat_top=0.7)  # for angular vehicles
    flat_shade(body)  # do NOT use limited_dissolve on vehicles

Example — in-place body refinement (PREFERRED over rebuild):
```python
# The body is too narrow in the rear — widen it incrementally
body = get("CarBody")
# Widen the rear section
scale_section(body, axis='X', position=1.8, tol=0.5, scale_y=1.3)
# Raise the roof slightly at center
bulge(body, axis='X', position=0.0, radius=1.0, strength=0.1)
# Add detail cuts for finer control
add_detail_cuts(body, axis='X', num_cuts=3)
flat_shade(body)
```

Example — adding missing detail parts (good iteration progress):
```python
# Add headlights that were missing
mat_hl = emission_material("HeadlightMat", color=(1,0.95,0.8), strength=5)
for side in [1, -1]:
    hl = create_box(name="Headlight_%s" % ("L" if side==1 else "R"),
                    width=0.15, depth=0.03, height=0.06)
    b = get_bounds(get("CarBody"))
    move_to(hl, x=b.min_x+0.1, y=0.5*side, z=0.25)
    assign_material(hl, mat_hl)
    flat_shade(hl)
```

═══ REFERENCE IMAGES & TRACING ═══
When reference images are attached, use them to TRACE accurate geometry:

  **TRACING WORKFLOW (with reference images):**
  1. Study the reference image carefully. Note the silhouette from
     side view and top view.
  2. TRACE the side-view outline as (position, height) coordinate pairs.
     Start at the front top, go along the top edge to the rear,
     then along the bottom edge back to the front (closed loop).
  3. TRACE the top-view outline as (position, half-width) coordinate pairs.
     Start at the front center, go along one side to the rear,
     then the other side back to the front (closed loop).
  4. Use mesh_from_outlines(side_outline=..., top_outline=...)
     to generate the accurate 3D body shape.
  5. Alternatively, extract top_curve, bottom_curve, width_curve
     from the reference and use shape_from_profiles().

  **WITHOUT REFERENCE IMAGES:**
  Rely on your training knowledge to build a mental blueprint.
  Use shape_from_profiles() with curves derived from known dimensions.
  Provide 8-15+ control points per curve for accuracy.

  **KEY PRINCIPLE:** The 2D curves/outlines you define are the
  MOST IMPORTANT part. Spend effort getting the curves right —
  the 3D mesh will only be as good as the curves that define it.
  Compare your viewport output against references during refinement.

═══ HELPER FUNCTIONS ═══
{tools_reference}
"""


def _build_system_prompt():
    tools = _get_tools_reference()
    return _SYSTEM_PROMPT.format(tools_reference=tools)


# ═══════════════════════════════════════════════════════════════════════════
# HTTP / OpenAI helpers
# ═══════════════════════════════════════════════════════════════════════════

def _http_post(url, headers, payload, timeout=None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        # HTTPError is a subclass of URLError — must catch it FIRST
        err = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError("API error %d: %s" % (e.code, err[:500])) from e
    except urllib.error.URLError as e:
        if "Connection refused" in str(e) or "No connection" in str(e):
            raise RuntimeError(
                "Cannot connect to LLM server. "
                "Make sure ollama is running: ollama serve"
            ) from e
        raise RuntimeError("Connection error: %s" % str(e)[:300]) from e
    return json.loads(body)


def _chat(server_url, model, messages, temperature=0.7, max_tokens=16384, on_chunk=None):
    if on_chunk is not None:
        return _chat_stream(server_url, model, messages, temperature,
                            max_tokens, on_chunk)

    _model_lower = model.lower()
    _is_reasoning = _model_lower.startswith(("o1", "o3", "o4"))

    payload = {
        "model": model,
        "messages": messages,
    }
    if _is_reasoning:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["temperature"] = temperature
        payload["max_completion_tokens"] = max_tokens

    result = _http_post(
        f"{server_url}/v1/chat/completions",
        {
            "Content-Type": "application/json",
        },
        payload,
    )
    return result["choices"][0]["message"]["content"]


def _chat_stream(server_url, model, messages, temperature=0.7,
                 max_tokens=16384, on_chunk=None):
    _model_lower = model.lower()
    _is_reasoning = _model_lower.startswith(("o1", "o3", "o4"))

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if _is_reasoning:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["temperature"] = temperature
        payload["max_completion_tokens"] = max_tokens

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )
    ctx = ssl.create_default_context()

    full_text = ""
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        if on_chunk:
                            on_chunk(full_text)
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError("API error %d: %s" % (e.code, err[:500])) from e

    return full_text


# ═══════════════════════════════════════════════════════════════════════════
# Tool-Calling API — structured tool use (VS Code Copilot-style)
# ═══════════════════════════════════════════════════════════════════════════

def _strip_special_tokens(text):
    """Remove LLM chat template tokens from text output."""
    if not text:
        return text
    # Qwen / ollama chat template markers
    text = re.sub(r'<\|im_start\|>.*?\n?', '', text)
    text = re.sub(r'<\|im_end\|>', '', text)
    text = re.sub(r'<\|endoftext\|>', '', text)
    text = re.sub(r'<\|im_sep\|>', '', text)
    # Generic special tokens
    text = re.sub(r'<\|[a-z_]+\|>', '', text)
    return text.strip()


# Models that have been detected as NOT supporting the structured
# "tools" API parameter.  Cached per-session so we only pay the
# cost of one failed request per model.
_MODELS_WITHOUT_TOOLS_API: set = set()


def _chat_with_tools(server_url, model, messages, tools, temperature=0.7,
                     max_tokens=16384, on_status=None):
    """Call chat API with tool definitions.

    Returns the full message dict (may contain tool_calls or content).
    Handles both structured tool_calls (OpenAI/ollama) and text-based
    tool calls (common with vision models or older local LLMs).

    If the model doesn't support the ``tools`` API parameter (e.g.
    qwen2.5vl) the function automatically retries WITHOUT ``tools``
    and relies on ``_parse_tool_calls_from_text`` to extract calls
    from the model's plain-text output.  The result is cached so
    subsequent rounds skip the failed attempt.
    """
    use_tools_api = model not in _MODELS_WITHOUT_TOOLS_API

    # --- First attempt: with tools (if model supports it) ----------------
    if use_tools_api:
        payload = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "stream": False,
        }
        try:
            result = _http_post(
                f"{server_url}/v1/chat/completions",
                {"Content-Type": "application/json"},
                payload,
            )
            msg = result["choices"][0]["message"]

            if msg.get("content"):
                msg["content"] = _strip_special_tokens(msg["content"])

            if msg.get("tool_calls"):
                return msg

            # Model responded but without structured tool_calls — parse text
            content = msg.get("content", "")
            parsed_calls = _parse_tool_calls_from_text(content)
            if parsed_calls:
                msg["tool_calls"] = parsed_calls
                msg["_text_parsed"] = True
            return msg

        except RuntimeError as e:
            err_str = str(e)
            if "does not support tools" in err_str or \
               ("400" in err_str and "tool" in err_str.lower()):
                # Model doesn't support tools API — remember and fall through
                _MODELS_WITHOUT_TOOLS_API.add(model)
                logger.info("Model %s does not support tools API — "
                            "falling back to text-based tool calling", model)
            else:
                raise  # Re-raise unrelated errors

    # --- Fallback: without tools, rely on text parsing -------------------
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    result = _http_post(
        f"{server_url}/v1/chat/completions",
        {"Content-Type": "application/json"},
        payload,
    )
    msg = result["choices"][0]["message"]

    if msg.get("content"):
        msg["content"] = _strip_special_tokens(msg["content"])

    content = msg.get("content", "")
    parsed_calls = _parse_tool_calls_from_text(content)
    if parsed_calls:
        msg["tool_calls"] = parsed_calls
        msg["_text_parsed"] = True

    return msg


def _parse_tool_calls_from_text(text):
    """Parse tool calls from text output when models don't use structured format.

    Handles patterns like:
    - {"name": "execute_code", "arguments": {"code": "..."}}
    - ```json\n{"name": ...}\n```
    - <tool_call>{"name": ...}</tool_call>
    """
    import re
    import uuid

    if not text:
        return None

    tool_calls = []

    # Pattern 1: <tool_call> tags (Qwen often uses these)
    tag_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    tag_matches = re.findall(tag_pattern, text, re.DOTALL)

    # Pattern 2: JSON blocks with "name" and "arguments"
    json_pattern = r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}'
    json_matches = re.findall(json_pattern, text, re.DOTALL)

    # Pattern 3: code blocks with ```python or ```json containing tool calls
    code_block_pattern = r'```(?:json|python)?\s*\n(\{.*?\})\s*\n```'
    code_matches = re.findall(code_block_pattern, text, re.DOTALL)

    # Try tag matches first
    for match in tag_matches:
        try:
            data = json.loads(match)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if name and name in ("generate_mesh", "execute_code", "inspect_scene", "inspect_object",
                                 "get_object_bounds", "inspect_timeline", "inspect_animation",
                                 "capture_viewport", "declare_complete"):
                tool_calls.append({
                    "id": "call_%s" % uuid.uuid4().hex[:8],
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments,
                    },
                })
        except (json.JSONDecodeError, KeyError):
            continue

    if tool_calls:
        return tool_calls

    # Try direct JSON matches
    for name, args_str in json_matches:
        if name in ("generate_mesh", "execute_code", "inspect_scene", "inspect_object",
                     "get_object_bounds", "inspect_timeline", "inspect_animation",
                     "capture_viewport", "declare_complete"):
            try:
                args = json.loads(args_str)
                tool_calls.append({
                    "id": "call_%s" % uuid.uuid4().hex[:8],
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                })
            except json.JSONDecodeError:
                continue

    if tool_calls:
        return tool_calls

    # Try code block matches
    for match in code_matches:
        try:
            data = json.loads(match)
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if name:
                tool_calls.append({
                    "id": "call_%s" % uuid.uuid4().hex[:8],
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments,
                    },
                })
        except (json.JSONDecodeError, KeyError):
            continue

    if tool_calls:
        return tool_calls

    # Last resort: if the content contains Python code that looks like
    # Blender code (import bpy, etc), wrap it in an execute_code call
    if ("import bpy" in text or "bpy.ops." in text or "bpy.data." in text or
            "bpy.context." in text):
        # Extract code from markdown code blocks if present
        code_extract = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
        if code_extract:
            code = "\n".join(code_extract)
        else:
            # Try to find the code portion (skip any explanation text)
            lines = text.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                if any(kw in line for kw in ("import bpy", "import bmesh",
                       "bpy.ops.", "bpy.data.", "bpy.context.",
                       "create_box", "create_cylinder", "create_sphere",
                       "quick_material", "pbr_material", "subsurf",
                       "mirror(", "add_detail_cuts", "shape_body",
                       "scale_section", "shade_auto_smooth")):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            code = "\n".join(code_lines) if code_lines else ""

        if code.strip():
            tool_calls.append({
                "id": "call_%s" % uuid.uuid4().hex[:8],
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": json.dumps({"code": code}),
                },
            })
            return tool_calls

    return None


def generate_with_tools(server_url, model, temperature, user_prompt,
                        on_status=None, session_gen=None, timeout=180,
                        mesh_server_url=None):
    """Tool-calling loop: local LLM reasons and calls tools to build in Blender.

    Args:
        server_url: LLM API endpoint (e.g. ollama at localhost:11434)
        model: LLM model name (e.g. qwen2.5vl:32b)
        temperature: Sampling temperature
        user_prompt: User's text request
        on_status: Callback for UI progress updates
        session_gen: Session generation counter
        timeout: Max seconds for mesh generation
        mesh_server_url: Optional URL for the trained mesh generation server
    """
    # Always use the chat tools path with local LLM for reasoning
    return _generate_with_chat_tools(server_url, model, temperature, user_prompt,
                                     on_status, session_gen,
                                     mesh_server_url=mesh_server_url)


def _generate_with_local_mesh(server_url, prompt, temperature, on_status, session_gen, timeout):
    """Generate mesh using local server and return as code."""
    try:
        if on_status:
            on_status(f"🎨 Generating 3D mesh (timeout: {timeout}s)...")
        
        # Call local mesh server
        mesh_data = _generate_mesh_local(server_url, prompt, temperature, timeout)
        
        # Check for errors in response
        if "error" in mesh_data:
            error_msg = f"❌ Server error: {mesh_data['error']}"
            add_message("user", prompt, session_gen=session_gen)
            add_message("assistant", error_msg, session_gen=session_gen)
            return error_msg, False
        
        if not mesh_data.get("objects") or len(mesh_data["objects"]) == 0:
            error_msg = "❌ No mesh generated - server returned empty result"
            add_message("user", prompt, session_gen=session_gen)
            add_message("assistant", error_msg, session_gen=session_gen)
            return error_msg, False
        
        if on_status:
            on_status("📝 Converting to Blender code...")
        
        # Convert mesh to executable code
        code = _mesh_to_code(mesh_data)
        
        # Execute the code on the MAIN THREAD (bpy is not thread-safe)
        success, error = _run_on_main_thread(lambda: _execute_mesh_code(code))  # type: ignore[misc]
        
        if success:
            summary = f"✅ Created mesh: {mesh_data['objects'][0]['name']}\n"
            summary += f"Vertices: {mesh_data['objects'][0]['mesh']['num_vertices']}, "
            summary += f"Faces: {mesh_data['objects'][0]['mesh']['num_faces']}\n"
            summary += f"Generation time: {mesh_data.get('generation_time', 0)}s"
            
            # Store tokens + prompt for feedback (must run on main thread)
            gen_tokens = mesh_data.get("tokens", [])
            gen_prompt = prompt
            if gen_tokens:
                def _store_feedback_data():
                    import bpy as _bpy  # type: ignore
                    _props = _bpy.context.scene.ai_copilot
                    _props.last_generation_tokens = json.dumps(gen_tokens)
                    _props.compare_prompt = gen_prompt
                _run_on_main_thread(_store_feedback_data)
            
            # Add to conversation history
            add_message("user", prompt, session_gen=session_gen)
            add_message("assistant", f"{summary}\n\n```python\n{code}\n```", session_gen=session_gen)
            
            return summary, True
        else:
            error_msg = f"❌ Failed to execute mesh code:\n{error[:500]}"
            add_message("user", prompt, session_gen=session_gen)
            add_message("assistant", error_msg, session_gen=session_gen)
            return error_msg, False
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8') if e.fp else str(e)
        error_msg = f"❌ Server HTTP error {e.code}: {error_body[:200]}"
        add_message("user", prompt, session_gen=session_gen)
        add_message("assistant", error_msg, session_gen=session_gen)
        return error_msg, False
    except urllib.error.URLError as e:
        error_msg = f"❌ Cannot connect to server: {str(e.reason)[:200]}"
        add_message("user", prompt, session_gen=session_gen)
        add_message("assistant", error_msg, session_gen=session_gen)
        return error_msg, False
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        error_msg = f"❌ Mesh generation failed: {str(e)}\n{tb[:500]}"
        add_message("user", prompt, session_gen=session_gen)
        add_message("assistant", error_msg, session_gen=session_gen)
        return error_msg, False


def _execute_mesh_code(code: str) -> tuple[bool, str]:
    """Execute mesh generation code in Blender."""
    import bpy  # type: ignore
    import bmesh  # type: ignore
    from mathutils import Vector  # type: ignore
    
    try:
        namespace = {
            "__builtins__": __builtins__,
            "bpy": bpy,
            "bmesh": bmesh,
            "Vector": Vector,
        }
        exec(code, namespace)
        return True, ""
    except Exception:
        import traceback
        return False, traceback.format_exc()


def _generate_with_chat_tools(server_url, model, temperature, user_prompt,
                              on_status, session_gen, mesh_server_url=None):
    """Tool-calling loop using local LLM (ollama) as the reasoning brain.
    
    The LLM decides what to do, calls tools, sees results, and continues
    until it calls declare_complete or reaches the safety limit.
    
    Args:
        server_url: LLM API endpoint URL (e.g. ollama at localhost:11434)
        model: LLM model name (e.g. qwen2.5vl:32b)
        temperature: Sampling temperature
        user_prompt: User's text request
        on_status: Callback for UI progress updates
        session_gen: Session generation counter
        mesh_server_url: Optional URL for trained mesh generation server
    
    Returns:
        (summary_text, is_complete): Result summary and completion status
    """
    from . import tool_defs

    clear_iteration_history()

    # Wire mesh server URL so generate_mesh tool can reach it
    if mesh_server_url:
        tool_defs.set_mesh_server_url(mesh_server_url)
    else:
        # Default to localhost if not explicitly set
        tool_defs.set_mesh_server_url("http://localhost:8420")

    # 1) Build initial context
    scene_ctx = get_scene_context()
    sel_ctx = get_selection_context()

    parts = []
    parts.append("[Current Blender scene]\n%s" % scene_ctx)
    if sel_ctx and "No objects selected" not in sel_ctx:
        parts.append("[Selected objects]\n%s" % sel_ctx)
    parts.append("[Your request]\n%s" % user_prompt)
    user_content = "\n\n".join(parts)

    # 2) Build messages — include history BEFORE adding current prompt
    system_prompt = _build_tool_system_prompt(model)
    messages = [{"role": "system", "content": system_prompt}]

    # Include recent conversation history for multi-turn context
    recent = _conversation[-MAX_HISTORY:]
    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add current user message (after history, so it's not duplicated)
    messages.append({"role": "user", "content": user_content})
    add_message("user", user_content, session_gen=session_gen)

    # Vision is supported via capture_viewport tool (qwen2.5vl handles images natively)

    # 3) Get tool definitions
    tools = tool_defs.get_tool_definitions()

    # 4) Tool-calling loop
    MAX_ROUNDS = 15  # safety limit (each round can have multiple parallel tool calls)
    MAX_TOTAL_CALLS = 40  # total tool invocations cap
    total_calls = 0
    consecutive_text_rounds = 0  # rounds where LLM returned text instead of tool calls
    repetitive_call_count = 0
    dead_end_rounds = 0
    recent_call_keys = []
    summary = ""
    is_complete = False

    # Push an undo point before we start modifying the scene.
    # Must run on main thread; and bpy.ops.ed.undo_push can still fail
    # depending on context, so we fall back silently.
    def _safe_undo_push():
        try:
            import bpy as _bpy  # type: ignore
            _bpy.ops.ed.undo_push(message="Blender Copilot")
        except Exception:
            pass
    _run_on_main_thread(_safe_undo_push)

    for round_num in range(MAX_ROUNDS):
        # Check for session cancellation
        if session_gen is not None and session_gen != _session_generation:
            print("[Blender Copilot] SESSION CANCELLED: round %d, "
                  "expected gen=%d, current gen=%d"
                  % (round_num, session_gen, _session_generation))
            return "Session cancelled", False

        # Update status
        if on_status:
            _update_streaming("🤔 Thinking (round %d)..." % (round_num + 1))

        # Call the API with tools
        response_msg = _chat_with_tools(
            server_url, model, messages, tools, temperature)

        # If the model was just detected as not supporting tools API,
        # update the system prompt to include text-based tool instructions
        # so subsequent rounds have the right guidance.
        if model in _MODELS_WITHOUT_TOOLS_API and \
                _TEXT_TOOL_CALL_INSTRUCTIONS not in messages[0]["content"]:
            messages[0]["content"] = _build_tool_system_prompt(model)
            logger.info("Updated system prompt with text-based tool instructions")

        # Extract tool calls from the response
        tool_calls = response_msg.get("tool_calls") or []
        text_parsed = response_msg.get("_text_parsed", False)

        # Log what the model returned for debugging
        tc_names = [tc["function"]["name"] for tc in tool_calls] if tool_calls else []
        logger.info("Round %d: %d tool calls %s (text_parsed=%s, total_calls=%d)",
                     round_num + 1, len(tool_calls), tc_names, text_parsed, total_calls)

        # Check if AI returned a text response (no tool calls = done thinking)
        if not tool_calls:
            consecutive_text_rounds += 1
            content = _strip_special_tokens(response_msg.get("content", ""))
            logger.info("Round %d: text-only response (consecutive=%d, total_calls=%d)",
                        round_num + 1, consecutive_text_rounds, total_calls)

            # Check for COMPLETE marker
            has_complete = False
            if content:
                for line in content.split("\n"):
                    if line.strip().upper().startswith("COMPLETE"):
                        has_complete = True
                        break

            if has_complete:
                messages.append(response_msg)
                summary = content
                is_complete = True
                add_message("assistant", content, session_gen=session_gen)
                break

            # Auto-complete if the LLM keeps returning text instead of
            # tool calls — it's stuck and nudging won't help
            if consecutive_text_rounds >= 3 and total_calls >= 1:
                logger.info("Auto-completing: %d consecutive text rounds, %d total calls",
                            consecutive_text_rounds, total_calls)
                summary = content or "Generation complete"
                is_complete = True
                add_message("assistant", summary, session_gen=session_gen)
                break

            # Nudge it to use tools — but only once or twice
            messages.append(response_msg)
            if total_calls < 2:
                messages.append({"role": "user", "content":
                    "You must call tools to build the scene. Start by calling "
                    "generate_mesh with a prompt describing the object. "
                    "Then use execute_code to add materials."})
            else:
                messages.append({"role": "user", "content":
                    "If you are done, call declare_complete with a summary. "
                    "Otherwise call execute_code to add materials or adjust objects."})
            continue

        # Add assistant message to conversation
        if text_parsed:
            # For text-parsed tool calls: keep original text as assistant message
            # (the model output text, not structured tool_calls)
            asst_content = response_msg.get("content", "")
            messages.append({"role": "assistant", "content": asst_content})
        else:
            # For structured tool calls: keep tool_calls in the message
            asst_msg = {"role": "assistant", "content": response_msg.get("content", "")}
            asst_msg["tool_calls"] = response_msg["tool_calls"]
            messages.append(asst_msg)

        # Got real tool calls — reset text-only counter
        consecutive_text_rounds = 0

        # Execute each tool call and collect results
        tool_results_text = []
        round_had_error = False
        round_repeated_calls = 0
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            call_key = "%s|%s" % (
                func_name,
                json.dumps(args, sort_keys=True, separators=(",", ":")),
            )
            recent_call_keys.append(call_key)
            if len(recent_call_keys) > 12:
                recent_call_keys = recent_call_keys[-12:]
            if len(recent_call_keys) >= 3 and all(k == call_key for k in recent_call_keys[-3:]):
                repetitive_call_count += 1
                round_repeated_calls += 1

            total_calls += 1

            # Update streaming status
            status_text = "🔧 %s(%s)..." % (
                func_name,
                ", ".join("%s=%s" % (k, str(v)[:30])
                          for k, v in list(args.items())[:3])
            )
            _update_streaming(status_text)

            # declare_complete finishes the loop when the model decides the task is done.
            if func_name == "declare_complete":
                summary = args.get("summary", "Model complete")
                is_complete = True
                tool_result = json.dumps({"status": "COMPLETE"})
            # capture_viewport — capture image and send to VL model for
            # visual verification.  qwen2.5vl supports vision natively.
            elif func_name == "capture_viewport":
                # Run the real capture on main thread (renders via bpy)
                capture_result = _run_on_main_thread(
                    lambda: tool_defs.execute_tool("capture_viewport", args))
                # Also grab scene text data for context
                scene_data = _run_on_main_thread(
                    lambda: tool_defs.execute_tool("inspect_scene", {}))
                # If we got a base64 image, inject it as a vision message
                # so the VL model can actually *see* the viewport.
                img_b64 = None
                if isinstance(capture_result, dict):
                    img_data = capture_result.get("image", "")
                    if img_data.startswith("data:image"):
                        # Strip data URI prefix → raw base64
                        img_b64 = img_data.split(",", 1)[-1]
                if img_b64 and "vl" in model.lower():
                    # Ollama OpenAI-compat: multipart content with image_url
                    vision_msg = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": (
                                "Here is the current viewport screenshot. "
                                "Scene data: " + json.dumps(scene_data)
                            )},
                            {"type": "image_url", "image_url": {
                                "url": "data:image/png;base64," + img_b64
                            }},
                        ],
                    }
                    messages.append(vision_msg)
                    tool_result = json.dumps({
                        "status": "viewport image sent for visual inspection",
                        "scene_description": scene_data,
                    })
                else:
                    # Non-VL model or capture failed — text fallback
                    tool_result = json.dumps({
                        "status": "viewport captured",
                        "scene_description": scene_data,
                    })
            elif func_name == "generate_mesh":
                # Special handling: HTTP fetch runs in background thread,
                # only the Blender object creation runs on main thread.
                # This avoids freezing Blender UI during model inference.
                mesh_data = tool_defs.fetch_mesh_from_server(args)
                if "error" in mesh_data:
                    tool_result = json.dumps(mesh_data)
                else:
                    # Create mesh on main thread
                    result = _run_on_main_thread(
                        lambda a=args, md=mesh_data: tool_defs.create_mesh_object(a, md))

                    # Store tokens for RLHF feedback (approve/reject)
                    gen_tokens = mesh_data.get("tokens", [])
                    gen_prompt = args.get("prompt", user_prompt)
                    if gen_tokens:
                        def _store_tokens(t=gen_tokens, p=gen_prompt):
                            import bpy as _bpy
                            _props = _bpy.context.scene.ai_copilot
                            _props.last_generation_tokens = json.dumps(t)
                            _props.compare_prompt = p
                        _run_on_main_thread(_store_tokens)

                    result_str = json.dumps(result)
                    if len(result_str) > 4000:
                        result_str = result_str[:4000] + '..."}'
                    tool_result = result_str
            else:
                # Execute tool on main thread
                result = _run_on_main_thread(
                    lambda fn=func_name, a=args: tool_defs.execute_tool(fn, a))
                # Cap result size for token efficiency
                result_str = json.dumps(result)
                if isinstance(result, dict) and result.get("error"):
                    round_had_error = True
                if len(result_str) > 4000:
                    result_str = result_str[:4000] + '..."}'
                tool_result = result_str

            if text_parsed:
                # Collect results for a combined user message
                tool_results_text.append("[%s result]: %s" % (func_name, tool_result))
            else:
                # Structured tool calls: use proper tool role
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

        # For text-parsed calls: send all results as a single user message
        # so the model can see them (it doesn't understand "tool" role)
        if text_parsed and tool_results_text:
            results_msg = "Your code has been executed. Do NOT repeat it.\n\n" + "\n".join(tool_results_text)
            results_msg += "\n\nIf the task is complete, call declare_complete. Otherwise, continue with the next step."
            messages.append({"role": "user", "content": results_msg})

        if round_repeated_calls > 0 or round_had_error:
            dead_end_rounds += 1
            messages.append({
                "role": "user",
                "content": (
                    "You are in a recovery state. Do not repeat the same tool call with the same args. "
                    "Use PLAN-FIRST execution: (1) one-sentence decomposition, "
                    "(2) inspect_scene or inspect_object, "
                    "(3) one corrective tool call, "
                    "(4) capture_viewport or inspect_scene to verify, then continue or declare_complete."
                ),
            })
        else:
            dead_end_rounds = max(0, dead_end_rounds - 1)

        # After declare_complete, break out
        if is_complete:
            add_message("assistant",
                        "COMPLETE: %s" % summary, session_gen=session_gen)
            break

        # Safety check: too many total calls
        if total_calls >= MAX_TOTAL_CALLS:
            summary = "Reached tool call limit (%d calls)" % MAX_TOTAL_CALLS
            is_complete = True
            break

    # Force viewport update
    try:
        import bpy as _bpy_local  # type: ignore
        _run_on_main_thread(lambda: _bpy_local.context.view_layer.update())
    except Exception:
        pass

    finalize_iteration()
    _save_chat_log()

    if summary:
        summary += "\n\nTool loop stats: total_calls=%d, repetitive_calls=%d, dead_end_rounds=%d" % (
            total_calls,
            repetitive_call_count,
            dead_end_rounds,
        )

    return summary, is_complete


# ═══════════════════════════════════════════════════════════════════════════
# Compact System Prompt — for tool-calling mode
# ═══════════════════════════════════════════════════════════════════════════

_TOOL_SYSTEM_PROMPT = """You are Blender Copilot, an AI that creates 3D scenes through iterative refinement.
You have a trained mesh generation model that produces detailed geometry (up to 8000 faces).

═══ YOUR WORKFLOW: PLAN → GENERATE → REFINE → ITERATE ═══

**STEP 1: COMPOSITIONAL PLANNING** (CRITICAL — DO THIS FIRST)
Before generating anything, mentally decompose the request into parts:

Example: "treasure chest"
→ Think: "A treasure chest has: bottom box, lid, hinges, lock, metal bands, maybe treasure inside"
→ Decision: Generate bottom and lid as separate meshes. Add hinges and details programmatically.

Example: "sports car"  
→ Think: "A car has: body, wheels (4), windows, headlights, grille"
→ Decision: Generate the main body mesh. Create wheels with primitives + array. Add details.

Example: "coffee mug"
→ Think: "Simple object — cylinder body + handle"
→ Decision: Generate the mug as one mesh. The model can handle this complexity.

**CRITICAL RULES FOR PLANNING:**
- Complex multi-part objects (furniture, vehicles, buildings) → generate main components separately
- Simple objects (cups, balls, boxes) → generate as one mesh
- Symmetric objects → generate one half, use mirror modifier
- Repeated elements (wheels, windows, tiles) → generate one, use array modifier
- Fine details (screws, bolts, text) → add via execute_code, not generate_mesh

For complex objects, ALWAYS decompose into semantic parts before generating:
- Vehicle: [body, wheels, windows, interior, engine/details]
- Character: [body, head, hands, feet, clothes, accessories]
- Building: [foundation, walls, roof, windows/doors, details]
- Furniture: [frame, surfaces, supports, cushions/padding]
Generate each part separately, place with location+scale, then merge with
join_objects_with_cleanup() after all parts are correctly placed.

**STEP 2: GENERATE BASE GEOMETRY**
Call generate_mesh for each main component with DETAILED prompts:
- Good: "detailed sports car body with aerodynamic curves, hood, roof, trunk"
- Bad: "car body" (too vague)

The model can generate complex shapes (up to 8000 faces). Be specific about:
- Style (modern, vintage, sci-fi, organic)
- Detail level (simple, detailed, high-poly)
- Key features (curved, angular, smooth, rough)

**STEP 3: INSPECT & REFINE**
After generation, use inspect_object to check:
- Vertex/face count — is it detailed enough?
- Dimensions — correct scale?
- Visual quality — call capture_viewport if unsure

Then refine using vertex-level editing:
```python
# Select problematic vertices
verts = select_vertices_by_location(obj, 'Z', min_val=2.0)
# Move them
move_vertices(obj, verts, offset=(0, 0, 0.5))
# Smooth sharp areas
smooth_vertices(obj, verts, iterations=2, factor=0.7)
```

**STEP 4: ASSIGN MATERIALS TO REGIONS**
Don't just assign one material to the whole object. Use face selection:
```python
# Select faces by orientation
top_faces = select_faces_by_normal(car, (0, 0, 1), 0.8)
paint = quick_material("Paint", color=(0.8, 0.1, 0.1), metallic=0.9)
assign_material_to_faces(car, paint, top_faces)

# Windows
window_faces = select_faces_by_normal(car, (0, 0, 1), 0.95)
glass = glass_material("Glass", color=(0.7, 0.8, 0.9))
assign_material_to_faces(car, glass, window_faces)
```

**STEP 5: ADD FINAL DETAIL**
- Subdivision surface for smoothness: `subsurf(obj, levels=2)`
- Bevel edges: `bevel(obj, width=0.01, segments=2)`
- Mirror for symmetry: `mirror(obj, axis='X')`
- Array for repetition: `array(wheels, count=4, offset=(2, 0, 0))`

**STEP 6: ITERATE**
- Call inspect_scene or capture_viewport to check results
- If something looks wrong: select vertices/faces and refine
- Add more detail if needed
- Adjust materials

**STEP 7: DECLARE COMPLETE**
Only when the scene looks good and all requested elements are present.

═══ TOOLS ═══
1. **generate_mesh**(prompt, name, temperature, max_faces) — Generate detailed 3D geometry
2. **execute_code**(code) — Run Python with all blender_tools pre-imported
3. **inspect_scene**() / **inspect_object**(name) / **get_object_bounds**(name)
4. **inspect_timeline**() — Get fps, frame range, per-object keyframes with timestamps
5. **inspect_animation**(name) — Detailed animation data for one object (FCurves, NLA, drivers)
6. **capture_viewport**() — Screenshot for visual verification
7. **declare_complete**(summary) — Signal completion

Inspection helpers via execute_code:
- inspect_materials_detail(obj) — full material node tree info
- inspect_modifiers(obj) — all modifiers with settings
- inspect_constraints(obj) — all constraints with settings
- inspect_armature(obj) — bone hierarchy, IK chains, pose
- inspect_physics(obj) — rigid body, cloth, particles
- inspect_shape_keys(obj) — shape keys with values/drivers
- inspect_vertex_groups(obj) — vertex groups with counts
- inspect_uv_maps(obj) — UV layers
- inspect_scene_hierarchy() — collection tree + parenting
- inspect_render_settings() — engine, resolution, samples
- inspect_world() — environment/HDRI settings

═══ BLENDER_TOOLS REFERENCE (pre-imported in execute_code) ═══
All 290+ functions below are pre-imported. Call them directly in execute_code.

**Vertex/Face Selection & Editing:**
- select_vertices_by_location(obj, axis, min_val, max_val) → vertex indices
- select_faces_by_normal(obj, direction, threshold) → face indices
- select_faces_by_position(obj, axis, min_val, max_val) → face indices
- move_vertices(obj, vertex_indices, offset=(x,y,z))
- scale_vertices(obj, vertex_indices, scale=(sx,sy,sz), pivot='center')
- smooth_vertices(obj, vertex_indices, iterations=2, factor=0.5)
- get_vertex_positions(obj, vertex_indices) → list of (x,y,z)
- extrude_faces(obj, face_indices, offset=1.0)
- inset_faces(obj, face_indices, thickness=0.1)
- bevel_edges(obj, edge_indices, width=0.05, segments=3)
- bridge_edge_loops(obj, loop1, loop2)
- loop_cut(obj, edge_index, cuts=1)

**Material & Texturing:**
- quick_material(name, color, roughness, metallic) → material
- glass_material(name, color, roughness, ior)
- emission_material(name, color, strength)
- pbr_material(name, base_color, roughness, metallic, normal_strength)
- subsurface_material(name, color, subsurface, roughness) — skin/wax/marble
- image_texture_material(name, image_path, roughness, metallic)
- gradient_material(name, color1, color2, axis)
- noise_texture_material(name, base_color, scale, detail)
- wood_material(name, base_color, grain_scale)
- brick_texture_material(name, brick_color, mortar_color)
- environment_texture(image_path, strength) — HDRI world lighting
- assign_material(obj, mat) / assign_material_to_faces(obj, mat, face_indices)
- assign_material_by_normal(obj, mat, axis, direction, threshold)
- add_normal_map(mat, image_path, strength)
- add_displacement(mat, strength, midlevel, image_path)

**Positioning & Transforms:**
- get(name) → object by name
- move_to(obj, x, y, z) / rotate_deg(obj, x, y, z) / scale_to(obj, x, y, z)
- get_bounds(obj) → bounding box
- place_on_ground(obj) / stack_on(obj, target, gap)
- snap_object_to(obj, target, snap='BOTTOM_TO_TOP')
- align_objects(objects, axis, mode)
- set_parent(child, parent) / apply_transforms(obj)

**Modifiers:**
- subsurf(obj, levels) / bevel(obj, width, segments)
- mirror(obj, axis) / array(obj, count, offset)
- solidify(obj, thickness) / boolean_cut(target, cutter)
- boolean_join(target, other) / decimate(obj, ratio)
- simple_deform(obj, method, angle, axis)
- shrinkwrap(obj, target) / wireframe_modifier(obj, thickness)
- screw_modifier(obj, angle_deg, steps, axis)

**Curves:**
- create_bezier_curve(name, points) / create_nurbs_curve(name, points)
- sweep_profile_along_curve(name, profile, path)
- revolve_profile(name, profile, axis, segments)

**Animation & Keyframing:**
- set_frame_range(start, end) / set_current_frame(frame) / set_fps(fps)
- insert_keyframe(obj, data_path, frame) — keyframe any property
- animate_location(obj, [(frame, (x,y,z)), ...])
- animate_rotation(obj, [(frame, (rx,ry,rz)), ...])
- animate_scale(obj, [(frame, (sx,sy,sz)), ...])
- animate_value(obj, data_path, [(frame, value), ...]) — animate anything
- set_interpolation(obj, 'LINEAR'|'BEZIER'|'CONSTANT')
- set_extrapolation(obj, 'CONSTANT'|'LINEAR'|'MAKE_CYCLIC')
- create_action(name) / set_action(obj, action) / push_to_nla(obj, track)
- add_follow_path(obj, curve) — animate along a curve
- clear_animation(obj) / add_marker(name, frame)

**Armature & Rigging:**
- create_armature(name, location) → armature object
- add_bone(arm, name, head, tail, parent_bone, connected)
- extrude_bone(arm, parent_bone, name, tail_offset)
- create_bone_chain(arm, name, joints) — chain from joint positions
- parent_to_armature(mesh, armature, method='ARMATURE_AUTO') — auto weights
- set_bone_ik(arm, bone, target, chain_length) — inverse kinematics
- add_bone_constraint(arm, bone, type, target, **kwargs)
- pose_bone(arm, bone, rotation_euler=(rx,ry,rz))
- keyframe_bone(arm, bone, data_path, frame)
- get_bones(arm) → list of bone info
- mirror_bones(arm) — mirror .L bones to .R

**Shape Keys:**
- add_shape_key(obj, name) — add a morph target
- edit_shape_key(obj, name, {vert_idx: (dx,dy,dz), ...})
- set_shape_key_value(obj, name, value) — blend 0.0–1.0
- animate_shape_key(obj, name, [(frame, value), ...])
- get_shape_keys(obj) → list of shape key info

**Drivers:**
- add_driver(obj, driven_path, driver_obj, driver_path, expression)
- add_expression_driver(obj, path, expr, variables={name: (obj, path)})
- remove_driver(obj, path)

**Object Constraints:**
- add_constraint(obj, type, target, **kwargs) — any constraint
- add_copy_location(obj, target) / add_copy_rotation(obj, target)
- add_track_to(obj, target) — always look at target
- add_floor_constraint(obj, target) / add_follow_path(obj, curve)
- remove_constraint(obj, name)

**Particle Systems:**
- add_particle_emitter(obj, count, lifetime, velocity, size)
- add_hair_system(obj, count, length, segments)
- set_particle_instance(particle_obj, instance_obj) — scatter objects
- remove_particles(obj)

**Physics:**
- add_rigid_body(obj, 'ACTIVE'|'PASSIVE', mass, bounciness)
- add_cloth(obj, stiffness, damping) / add_collision(obj)
- add_soft_body(obj, mass, goal_strength)
- add_force_field(field_type='WIND'|'TURBULENCE'|'VORTEX', strength)
- set_gravity(x, y, z) / bake_physics(start, end) / free_physics_bake()

**Rendering:**
- set_render_engine('CYCLES'|'BLENDER_EEVEE_NEXT')
- set_render_resolution(x, y) / set_render_samples(samples, denoise)
- render_image(filepath) / render_animation(filepath)
- set_output_path(path, file_format)
- set_transparent_background(True) / set_color_management(view_transform)

**Cameras & Lighting:**
- add_camera(name, location, look_at, lens) / set_active_camera(cam)
- add_depth_of_field(cam, focus_object, fstop)
- setup_sun(energy, rotation_deg) / setup_point_light / setup_area_light
- setup_sky_texture(sun_elevation) / setup_gradient_world(colors)

**Compositor (quick recipes):**
- setup_compositor() / add_glare(type, threshold)
- add_color_correction(brightness, contrast, saturation)
- add_vignette(amount)

**Compositor (generic node editing):**
- add_compositor_node(type, name) — 'glare','blur','bright_contrast','hue_sat', etc.
- get_compositor_node(name) / remove_compositor_node(name)
- connect_compositor_nodes(from_node, from_output, to_node, to_input)
- insert_compositor_node_between(type, before, after) — auto-relinks chain
- auto_arrange_compositor() / clear_compositor()
- add_viewer_node() — for previewing compositor output
- set_node_input(node, input_name, value)
- set_node_property(node, prop_name, value)

**Shader Node Editing (generic — build ANY material):**
- add_shader_node(mat, type, name) — 'noise','voronoi','colorramp','bump', etc.
- get_node(mat, name) / remove_node(mat, name)
- connect_nodes(mat, from_node, from_output, to_node, to_input)
- disconnect_node_input(mat, node, input_name)
- set_node_input(node, input_name, value) / get_node_input(node, input_name)
- set_node_property(node, prop_name, value) — operation, blend_type, etc.
- set_colorramp_stops(node, [(pos, color), ...])
- list_node_inputs(node) / list_node_outputs(node)
- auto_arrange_nodes(mat) — clean horizontal layout
- clear_material_nodes(mat) / duplicate_material(mat, new_name)

**Weight Painting & Vertex Groups:**
- create_vertex_group(obj, name, vert_indices, weight)
- set_vertex_weights(obj, group, verts, weight)
- paint_weight_gradient(obj, group, axis, min_weight, max_weight)
- normalize_weights(obj) / get_vertex_weights(obj, group)

**Scene Utilities:**
- get_scene_info() → comprehensive scene data
- list_actions() / list_materials()
- clear_scene() / scene_objects()

**Inspection Tools (via execute_code):**
- inspect_materials_detail(obj) → full material node trees with colors/values
- inspect_modifiers(obj) → all modifiers with type-specific settings
- inspect_constraints(obj) → all constraints with targets/influence
- inspect_armature(obj) → bone hierarchy, pose bones, IK chains
- inspect_physics(obj) → rigid body, cloth, particles, collision
- inspect_shape_keys(obj) → key blocks with values, ranges, drivers
- inspect_vertex_groups(obj) → groups with vertex counts
- inspect_uv_maps(obj) → UV layers with active status
- inspect_scene_hierarchy() → collection tree + parent relationships
- inspect_render_settings() → engine, resolution, samples, output
- inspect_world() → environment, HDRI, sky settings

═══ WORKFLOW PATTERNS FOR DIFFERENT TASKS ═══

**Animation Workflow:**
1. Set timeline: set_frame_range(1, 120), set_fps(24)
2. Keyframe: animate_location(obj, [(1, start), (60, mid), (120, end)])
3. Adjust curves: set_interpolation(obj, 'BEZIER')
4. For looping: set_extrapolation(obj, 'MAKE_CYCLIC')

**Rigging Workflow:**
1. Create armature: arm = create_armature("Rig")
2. Add bones: add_bone(arm, "spine", head, tail), add_bone(arm, "chest", ...)
3. Parent mesh: parent_to_armature(mesh, arm, 'ARMATURE_AUTO')
4. Add IK: set_bone_ik(arm, "hand.L", target_obj=ik_empty, chain_length=3)
5. Pose & keyframe: pose_bone(arm, "upper_arm.L", rotation_euler=(0,0,-45))
                     keyframe_bone(arm, "upper_arm.L", 'rotation_euler', frame=1)

**Physics Workflow:**
1. Set up: add_rigid_body(ball, 'ACTIVE', mass=2), add_rigid_body(floor, 'PASSIVE')
2. Optional: add_force_field(field_type='WIND', strength=5)
3. Bake: bake_physics(1, 250)

**Material Workflow (simple):**
1. Create: mat = pbr_material("Metal", roughness=0.2, metallic=1.0)
2. Add detail: add_normal_map(mat, "//textures/scratches.jpg")
3. Assign to faces: assign_material_to_faces(obj, mat, top_faces)

**Material Workflow (node editing for complex shaders):**
1. Start: mat = quick_material("Rock"); clear_material_nodes(mat)
2. Add nodes: noise = add_shader_node(mat, 'noise', name='RockNoise')
3. Configure: set_node_input(noise, 'Scale', 15.0)
4. Add color ramp: ramp = add_shader_node(mat, 'colorramp')
   set_colorramp_stops(ramp, [(0, (0.2,0.15,0.1,1)), (1, (0.5,0.4,0.3,1))])
5. Wire up: connect_nodes(mat, noise, 'Fac', ramp, 'Fac')
   bsdf = get_node(mat, 'Principled BSDF')
   connect_nodes(mat, ramp, 'Color', bsdf, 'Base Color')
6. Clean layout: auto_arrange_nodes(mat)

**Compositor Workflow:**
1. Enable: setup_compositor()
2. Add effects: glare = add_compositor_node('glare', name='Bloom')
   set_node_property(glare, 'glare_type', 'FOG_GLOW')
   set_node_input(glare, 'Threshold', 0.8)
3. Or insert inline: insert_compositor_node_between('glare', 'Render Layers', 'Composite')
4. Preview: add_viewer_node()
5. Clean layout: auto_arrange_compositor()
"""


def _build_tool_system_prompt(model=None):
    """Build the tool-calling system prompt.

    If *model* is in the no-tools-API cache, append explicit JSON
    call-format instructions so the model knows how to invoke tools
    via plain text output.
    """
    prompt = _TOOL_SYSTEM_PROMPT
    if model and model in _MODELS_WITHOUT_TOOLS_API:
        prompt += _TEXT_TOOL_CALL_INSTRUCTIONS
    return prompt


_TEXT_TOOL_CALL_INSTRUCTIONS = """

═══ HOW TO CALL TOOLS (IMPORTANT) ═══
Your model does NOT support structured tool calls.
To invoke a tool, output a JSON block wrapped in <tool_call> tags:

<tool_call>
{"name": "generate_mesh", "arguments": {"prompt": "a wooden chair", "max_faces": 2048}}
</tool_call>

<tool_call>
{"name": "execute_code", "arguments": {"code": "mat = quick_material('Wood', color=(0.4, 0.25, 0.1))\nassign_material(get('wooden_chair'), mat)"}}
</tool_call>

<tool_call>
{"name": "inspect_scene", "arguments": {}}
</tool_call>

<tool_call>
{"name": "inspect_timeline", "arguments": {}}
</tool_call>

<tool_call>
{"name": "inspect_animation", "arguments": {"name": "Cube"}}
</tool_call>

<tool_call>
{"name": "declare_complete", "arguments": {"summary": "Created a wooden chair with material"}}
</tool_call>

Rules:
- You may output multiple <tool_call> blocks in a single response.
- Always use valid JSON inside the tags.
- Available tool names: generate_mesh, execute_code, inspect_scene,
  inspect_object, get_object_bounds, inspect_timeline, inspect_animation,
  capture_viewport, declare_complete.
- After each response the system will execute the tools and show you the
  results. Continue calling tools until done, then call declare_complete.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Code Extraction
# ═══════════════════════════════════════════════════════════════════════════

def _sanitize_code(code):
    """Replace common Unicode look-alikes that break Python parsing."""
    replacements = {
        '\u2212': '-',   # − (minus sign) → -
        '\u2013': '-',   # – (en dash) → -
        '\u2014': '-',   # — (em dash) → -
        '\u2018': "'",   # ' (left single quote) → '
        '\u2019': "'",   # ' (right single quote) → '
        '\u201c': '"',   # " (left double quote) → "
        '\u201d': '"',   # " (right double quote) → "
        '\u00d7': '*',   # × (multiplication sign) → *
        '\u2264': '<=',  # ≤ → <=
        '\u2265': '>=',  # ≥ → >=
        '\u2260': '!=',  # ≠ → !=
    }
    for old, new in replacements.items():
        code = code.replace(old, new)
    return code


def _extract_code(text):
    """Extract explanation and Python code from AI response."""
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        code = "\n\n".join(m.strip() for m in matches)
        explanation = text[:text.find('```')].strip()
        return explanation, _sanitize_code(code)

    pattern2 = r'```\s*\n(.*?)```'
    matches2 = re.findall(pattern2, text, re.DOTALL)
    if matches2:
        code = "\n\n".join(m.strip() for m in matches2)
        explanation = text[:text.find('```')].strip()
        return explanation, _sanitize_code(code)

    return text.strip(), ""


# ═══════════════════════════════════════════════════════════════════════════
# Reference Image Search — web search for visual references
# ═══════════════════════════════════════════════════════════════════════════

_REF_IMAGE_CACHE = {}  # query -> [(url, local_path), ...]
_REF_IMAGE_DIR = os.path.join(tempfile.gettempdir(), "ai_copilot_refs")


def search_reference_images(query, max_results=3, api_key=None):
    """Search the web for reference images matching *query*.

    Uses Google Custom Search API (if configured in prefs) or falls back
    to a direct image URL fetch approach.

    Returns a list of local file paths to downloaded images.
    """
    if query in _REF_IMAGE_CACHE:
        return [p for _, p in _REF_IMAGE_CACHE[query] if os.path.exists(p)]

    os.makedirs(_REF_IMAGE_DIR, exist_ok=True)

    urls = _search_image_urls(query, max_results, api_key)
    results = []

    for i, url in enumerate(urls):
        try:
            local_path = os.path.join(
                _REF_IMAGE_DIR,
                "ref_%s_%d.jpg" % (re.sub(r'[^a-zA-Z0-9]', '_', query)[:30], i)
            )
            _download_image(url, local_path)
            if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
                results.append((url, local_path))
        except Exception as exc:
            print("[Blender Copilot] Failed to download ref image: %s" % exc)

    _REF_IMAGE_CACHE[query] = results
    return [p for _, p in results]


def _search_image_urls(query, max_results=3, api_key=None):
    """Fetch image URLs via OpenAI-powered URL generation.

    Since we can't reliably use Google/Bing without API keys,
    we use OpenAI to suggest reference image search URLs and then
    try to find actual image URLs from those results.
    """
    # Use direct image URLs from known free sources
    urls = []
    try:
        # Try Unsplash source (free, no API key)
        encoded_q = urllib.parse.quote(query)
        for i in range(max_results):
            urls.append("https://source.unsplash.com/800x600/?%s&sig=%d" % (encoded_q, i))
    except Exception:
        pass

    return urls[:max_results]


def _download_image(url, local_path, timeout=10):
    """Download an image from a URL to a local file."""
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Blender AI Copilot)',
    })
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        data = resp.read()
    with open(local_path, 'wb') as f:
        f.write(data)


def get_reference_image_paths():
    """Get all pending reference image paths from the current scene."""
    try:
        import bpy as _bpy  # type: ignore
        props = _bpy.context.scene.ai_copilot
        paths = []
        for ref in props.reference_images:
            if ref.filepath and os.path.exists(ref.filepath):
                paths.append(ref.filepath)
        return paths
    except Exception:
        return []


def _get_reference_images_base64():
    """Encode all enabled reference images as base64 data URLs."""
    paths = _run_on_main_thread(get_reference_image_paths)
    results = []
    for path in (paths or []):
        try:
            data_url = _image_to_base64(path)
            results.append(data_url)
        except Exception:
            pass
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Image helpers
# ═══════════════════════════════════════════════════════════════════════════

def _image_to_base64(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    }.get(ext, "image/png")
    with open(filepath, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return "data:%s;base64,%s" % (mime, data)


def _capture_viewport_image():
    """Try to capture a viewport screenshot. Returns base64 data URL or None."""
    try:
        from . import blender_tools
        import tempfile
        img_path = os.path.join(tempfile.gettempdir(), "ai_copilot_viewport.png")
        blender_tools.capture_viewport(filepath=img_path, resolution=(1920, 1080))
        if os.path.exists(img_path) and os.path.getsize(img_path) > 1000:
            return _image_to_base64(img_path)
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main API — generate_response  (single-pass, the ONLY generation path)
# ═══════════════════════════════════════════════════════════════════════════

def generate_response(api_key, model, temperature, user_prompt,
                      on_chunk=None, session_gen=None):
    """Send a prompt to the AI and get back (explanation, code).

    Automatically includes scene context, selection context, and
    conversation history. Works for both fresh builds and iterative edits.

    *session_gen* — snapshot of ``get_session_generation()`` taken before
    the background thread was spawned.  If the session was cleared while
    this call was in flight, message writes are silently dropped.
    """
    # Clear any leftover iteration history from a previous prompt
    clear_iteration_history()

    scene_ctx = get_scene_context()
    sel_ctx = get_selection_context()

    # Build the user message with full context
    parts = []
    parts.append("[Current Blender scene]\n%s" % scene_ctx)
    if sel_ctx and "No objects selected" not in sel_ctx:
        parts.append("[Selected objects]\n%s" % sel_ctx)
    parts.append("[Your request]\n%s" % user_prompt)
    user_content = "\n\n".join(parts)

    add_message("user", user_content, session_gen=session_gen)

    # Build messages with conversation history
    system_prompt = _build_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]

    # Include recent history for multi-turn context
    recent = _conversation[-MAX_HISTORY:]
    for msg in recent:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Optionally include viewport image for vision-capable models
    model_lower = model.lower()
    supports_vision = not model_lower.startswith(("o1", "o3"))
    if supports_vision:
        viewport_image = _run_on_main_thread(_capture_viewport_image)
        ref_images = _get_reference_images_base64()
        if viewport_image or ref_images:
            # Replace last user message with multimodal content
            last_msg = messages[-1]
            content_parts: list[dict] = [{"type": "text", "text": last_msg["content"]}]
            if viewport_image:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": viewport_image, "detail": "high"},
                })
            for ref_b64 in ref_images[:5]:  # limit to 5 reference images
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": ref_b64, "detail": "high"},
                })
            messages[-1] = {"role": "user", "content": content_parts}  # type: ignore[assignment]

    response_text = _chat(api_key, model, messages, temperature, on_chunk=on_chunk)
    explanation, code = _extract_code(response_text)
    add_message("assistant", response_text, session_gen=session_gen)

    return explanation, code


def generate_fix(api_key, model, temperature, error_message, failed_code,
                 session_gen=None):
    """Send an error back to the AI for auto-repair.

    Fix messages are stored in _iteration_history (not _conversation)
    so the user's chat stays clean — they don't need to see error traces.
    """
    fix_prompt = (
        "The code produced this error:\n"
        "```\n%s\n```\n\n"
        "Failed code:\n"
        "```python\n%s\n```\n\n"
        "Fix the code. Return ONLY the corrected ```python``` block."
    ) % (error_message[:500], failed_code[:3000])

    _add_iteration_message("user", fix_prompt, session_gen=session_gen)

    system_prompt = _build_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_conversation[-MAX_HISTORY:])
    messages.extend(_iteration_history)

    response_text = _chat(api_key, model, messages, temperature)
    explanation, code = _extract_code(response_text)
    _add_iteration_message("assistant", response_text, session_gen=session_gen)

    return explanation, code


def get_scene_hash():
    """Return a lightweight hash of the current scene state for stagnation detection.

    Compares object names, vertex counts, dimensions, and positions.
    Returns empty string if scene cannot be read.
    """
    try:
        import hashlib
        ctx = _run_on_main_thread(get_scene_context)
        if ctx:
            return hashlib.md5(ctx.encode()).hexdigest()
    except Exception:
        pass
    return ""


def assess_result(api_key, model, temperature, original_request,
                  iteration, on_chunk=None, session_gen=None):
    """Provide the AI with a viewport screenshot, scene data, AND mesh profiles,
    let it decide whether to iterate or declare COMPLETE.

    Architecture note: This function stores messages in _iteration_history
    (NOT _conversation) so the user's chat stays clean. The AI sees:
      system prompt → real conversation → iteration observations/responses.

    Returns ``(is_complete, explanation, code)``.
    """
    scene_ctx = get_scene_context()
    mesh_ctx = get_mesh_context_for_iteration()
    viewport_image = _run_on_main_thread(_capture_viewport_image)

    # ── Build iteration-aware assessment prompt ──
    observe_prompt = (
        "Refinement pass %d. Original request: \"%s\"\n\n"
        "Scene:\n%s\n\n"
    ) % (iteration, original_request, scene_ctx)

    if mesh_ctx:
        observe_prompt += "%s\n\n" % mesh_ctx

    observe_prompt += (
        "Study the viewport screenshot AND the mesh profile data above. "
        "Compare against your mental blueprint of this subject.\n\n"
    )

    observe_prompt += (
        "QUALITY CHECKLIST (evaluate each, be brutally honest):\n"
        "  □ 1. RECOGNITION — Would someone immediately recognize this as\n"
        "       the specific subject? If not, WHY not? What's wrong?\n"
        "  □ 2. SILHOUETTE — Does the outline match from side/front/top?\n"
        "       Compare the actual profile curves vs your mental blueprint.\n"
        "  □ 3. PROPORTIONS — Read the PROFILE data. State actual vs target:\n"
        "       'Length is 4.5m ✓ but height is 0.9m, should be 1.1m ✗'\n"
        "  □ 4. PARTS COUNT — Are ALL planned parts present as separate\n"
        "       objects? Body, glass, wheels, lights, trim, details?\n"
        "  □ 5. SHADING — Is shading correct for the style?\n"
        "       Low-poly → flat_shade(). Realistic → smooth + auto_smooth.\n"
        "       Smooth shading on sparse meshes = BLOBBY = WRONG.\n"
        "  □ 6. MATERIALS — Does every part have the correct material?\n"
        "  □ 7. DISTINCTIVE FEATURES — What makes this subject unique?\n"
        "       Are those features present and recognizable?\n\n"
        "DIMENSIONAL VALIDATION (REQUIRED):\n"
        "  ACTUAL:  length=___m  width=___m  height=___m  (from profile data)\n"
        "  TARGET:  length=___m  width=___m  height=___m  (from blueprint)\n"
        "  VERDICT: ✓ matches / ✗ off by ___m\n\n"
    )

    # ── Progressive iteration strategy ──
    if iteration <= 1:
        observe_prompt += (
            "ACTION RULES (iteration %d — BODY SHAPE IS EVERYTHING):\n"
            "  The body shape is the FOUNDATION. Nothing else matters until\n"
            "  it looks recognizably like the subject.\n\n"
            "  CHECK THE BODY MESH:\n"
            "  • If the body shape looks wrong, DELETE and REBUILD with:\n"
            "    - 20-30 control points per curve (with SHARP transitions)\n"
            "    - num_sections=20, ring_points=16\n"
            "    - sharpness=8.0, flat_top=0.7 for angular/sporty vehicles\n"
            "    - sharpness=3.0, flat_top=0.0 for rounded vehicles\n"
            "    - flat_top creates FLAT hood/roof — essential for angular cars!\n"
            "    - Then: flat_shade(body)\n"
            "    - ⚠️ Do NOT use limited_dissolve on vehicle bodies!\n"
            "    (Do NOT use decimate — it destroys your designed shape!)\n"
            "  • If the silhouette doesn't look like the specific subject,\n"
            "    DELETE and REBUILD with CORRECTED curves. Your curves must\n"
            "    have SHARP transitions at feature boundaries.\n"
            "  • If dimensions are off by more than 15%% in any axis,\n"
            "    DELETE and REBUILD. Do NOT scale — scaling distorts\n"
            "    cross-section proportions.\n\n"
            "  ⚠️ BOUNDS AXIS WARNING:\n"
            "  b = get_bounds(body)\n"
            "  b.width = X = CAR LENGTH (NOT lateral width!)\n"
            "  b.depth = Y = CAR LATERAL WIDTH\n"
            "  For placing parts on left/right sides: y = side * (b.depth/2)\n\n"
            "  ⚠️ This is your ONE allowed rebuild. After this iteration,\n"
            "  you MUST use in-place tools (sculpt_move, scale_section,\n"
            "  bulge, pinch) — no more body rebuilds.\n\n"
            "  ℹ️ PROFILE NOTE: y=0.00 showing w=0.00 h=0.00 is NORMAL.\n"
            "  The mesh center seam has no cross-section width. This does\n"
            "  NOT mean geometry collapsed — ignore this line.\n\n"
            "  DO NOT add accessories yet. Get the BODY right first.\n"
            "  You may add wheels for proportion reference, nothing else.\n\n"
        ) % iteration
    elif iteration <= 3:
        observe_prompt += (
            "ACTION RULES (iteration %d — ADD ALL DETAIL PARTS NOW):\n"
            "  🚫 You have ALREADY used your ONE allowed body rebuild.\n"
            "  Do NOT delete and rebuild the body again.\n"
            "  If body dimensions are within ~15%% of target, it's FINE.\n\n"
            "  ⚠️ BOUNDS AXIS WARNING:\n"
            "  b = get_bounds(body)\n"
            "  b.width = X = CAR LENGTH (NOT lateral width!)\n"
            "  b.depth = Y = CAR LATERAL WIDTH\n"
            "  For placing parts on left/right sides: y = side * (b.depth/2)\n\n"
            "  YOUR PRIORITY: Add ALL remaining detail parts IN THIS ITERATION:\n"
            "  • Wheels (4x tires + 4x rims) if not already present\n"
            "  • Windshield (glass panel, >= 0.05m thick)\n"
            "  • Headlights (angular boxes at front corners, 0.30m wide)\n"
            "  • Taillights (emission boxes at rear corners)\n"
            "  • Spoiler/wing (flat box behind roof)\n"
            "  • Intakes, mirrors\n"
            "  Add EVERYTHING in ONE code block. Do NOT add 1 part per iteration.\n"
            "  Each part needs its own material.\n\n"
            "  ℹ️ PROFILE NOTE: y=0.00 showing w=0.00 h=0.00 is NORMAL.\n"
            "  The mesh center seam has no cross-section width. Ignore it.\n\n"
        ) % iteration
    elif iteration <= 6:
        observe_prompt += (
            "ACTION RULES (iteration %d — FINISH UP & DECLARE COMPLETE):\n"
            "  🚫 Do NOT rebuild or significantly modify the body shape.\n"
            "  You should have body + wheels + most detail parts by now.\n"
            "  Add any REMAINING missing parts, then declare COMPLETE.\n\n"
            "  ⚠️ BOUNDS AXIS WARNING (for positioning):\n"
            "  b.width = X = CAR LENGTH, b.depth = Y = CAR LATERAL WIDTH\n"
            "  For sides: y = side * (b.depth/2 - inset)\n\n"
            "  If the model has: body + 4 wheels + windshield + some lights\n"
            "  + correct materials → declare COMPLETE.\n"
            "  Perfection is not required. A recognizable model with good\n"
            "  proportions and appropriate details is COMPLETE.\n\n"
        ) % iteration
    else:
        observe_prompt += (
            "ACTION RULES (iteration %d — MUST DECLARE COMPLETE):\n"
            "  You are at a LATE iteration. Unless there is a critical\n"
            "  visible defect, declare COMPLETE now.\n"
            "  Do NOT keep adding tiny details or tweaking.\n\n"
        ) % iteration

    # After several iterations, wrap-up guidance
    if iteration >= 3:
        observe_prompt += (
            "NOTE: This is iteration %d. You should have all major parts.\n"
            "If the model has a recognizable body + wheels + detail parts +\n"
            "correct materials → declare COMPLETE.\n"
            "Perfection is not required. A recognizable model with good\n"
            "proportions and appropriate details is COMPLETE.\n\n"
        ) % iteration

    if iteration >= 4:
        observe_prompt += (
            "⚠️ APPROACHING FINAL ITERATIONS. You MUST either add one\n"
            "targeted detail part or declare COMPLETE. Do NOT modify body.\n"
            "Do NOT keep tweaking the same dimension repeatedly.\n\n"
        )

    observe_prompt += (
        "If ALL quality checklist items pass AND dimensional validation\n"
        "passes AND the model is recognizable as the requested subject\n"
        "→ COMPLETE: <summary>\n"
        "If not → describe the SINGLE most important problem and write\n"
        "targeted fix code (prefer in-place edits over rebuilds).\n"
    )

    _add_iteration_message("user", observe_prompt, session_gen=session_gen)

    # Build messages: system + real conversation + iteration history
    system_prompt = _build_system_prompt()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_conversation[-MAX_HISTORY:])
    messages.extend(_iteration_history)

    # Attach viewport image if vision-capable
    model_lower = model.lower()
    supports_vision = not model_lower.startswith(("o1", "o3"))
    if supports_vision:
        last_msg = messages[-1]
        content_parts: list[dict] = [{"type": "text", "text": last_msg["content"]}]
        if viewport_image:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": viewport_image, "detail": "high"},
            })
        # Include reference images so the AI can compare against them
        ref_images = _get_reference_images_base64()
        for ref_b64 in ref_images[:5]:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": ref_b64, "detail": "high"},
            })
        if len(content_parts) > 1:
            messages[-1] = {"role": "user", "content": content_parts}  # type: ignore[assignment]

    response_text = _chat(api_key, model, messages, temperature, on_chunk=on_chunk)
    _add_iteration_message("assistant", response_text, session_gen=session_gen)

    # Detect COMPLETE marker
    is_complete = False
    stripped = response_text.strip()
    for line in stripped.split("\n"):
        if line.strip().upper().startswith("COMPLETE"):
            is_complete = True
            break

    explanation, code = _extract_code(response_text)
    # If marked complete, discard any accidental code
    if is_complete:
        code = ""

    return is_complete, explanation, code
