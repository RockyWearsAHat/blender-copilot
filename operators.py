"""Operators for the Blender Copilot.

Operator catalogue
──────────────────
  aihouse.send_prompt       — send prompt to AI, execute response
  aihouse.stop_generation   — halt the current generation
  aihouse.execute_code      — re-run the last generated code
  aihouse.clear_scene       — wipe all objects
  aihouse.clear_chat        — reset conversation history
  aihouse.paste_api_key     — paste key from clipboard
  aihouse.open_openai_keys  — open OpenAI key page in browser
  aihouse.test_api_key      — test API connection
  aihouse.refresh_models    — refresh model list from OpenAI
"""

import threading
import traceback

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

        if not prefs.openai_api_key:
            self.report({"ERROR"},
                        "Set your OpenAI API key in Preferences → Add-ons")
            return {"CANCELLED"}

        prompt = props.prompt_text.strip()
        if not prompt:
            self.report({"ERROR"}, "Type a prompt first")
            return {"CANCELLED"}

        cls = AIHOUSE_OT_send_prompt
        cls._api_key = prefs.openai_api_key
        cls._model = prefs.model
        cls._temperature = prefs.temperature
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
                    cls._api_key, cls._model, cls._temperature, prompt,
                    on_status=ai_engine._update_streaming,
                    session_gen=sg)
                cls._result = (summary, is_complete)
            except Exception as exc:
                print("[Blender Copilot] Error:\n%s" % traceback.format_exc())
                cls._error = "%s: %s" % (type(exc).__name__, exc)
            cls._done = True

        threading.Thread(target=worker, daemon=True).start()

        self._timer = context.window_manager.event_timer_add(
            0.1, window=context.window)
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
# API-Key / Preferences Operators
# ═══════════════════════════════════════════════════════════════════════════

class AIHOUSE_OT_paste_api_key(Operator):
    """Paste your OpenAI API key from the system clipboard"""

    bl_idname = "aihouse.paste_api_key"
    bl_label = "Paste API Key"

    def execute(self, context):
        clipboard = context.window_manager.clipboard
        if not clipboard:
            self.report({"WARNING"}, "Clipboard is empty")
            return {"CANCELLED"}
        key = clipboard.strip().replace("\n", "").replace("\r", "").replace(" ", "")
        prefs = context.preferences.addons[__package__].preferences
        prefs.openai_api_key = key
        self.report({"INFO"}, "API key pasted (%d chars)" % len(key))
        return {"FINISHED"}


class AIHOUSE_OT_open_openai_keys(Operator):
    """Open the OpenAI API keys page in your browser"""

    bl_idname = "aihouse.open_openai_keys"
    bl_label = "Open OpenAI Keys Page"

    def execute(self, _context):
        import webbrowser
        webbrowser.open("https://platform.openai.com/api-keys")
        return {"FINISHED"}


class AIHOUSE_OT_test_api_key(Operator):
    """Test the OpenAI API connection"""

    bl_idname = "aihouse.test_api_key"
    bl_label = "Test API Key"

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        if not prefs.openai_api_key:
            self.report({"ERROR"}, "No API key set")
            return {"CANCELLED"}
        try:
            from .preferences import fetch_openai_models
            models = fetch_openai_models(prefs.openai_api_key)
            self.report({"INFO"}, "Connection OK — %d models available" % len(models))
        except Exception as exc:
            self.report({"ERROR"}, "Connection failed: %s" % str(exc)[:150])
        return {"FINISHED"}


class AIHOUSE_OT_refresh_models(Operator):
    """Refresh the list of available OpenAI models"""

    bl_idname = "aihouse.refresh_models"
    bl_label = "Refresh Models"

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences
        if not prefs.openai_api_key:
            self.report({"ERROR"}, "Set your API key first")
            return {"CANCELLED"}
        try:
            from .preferences import fetch_openai_models, _cached_model_items
            items = fetch_openai_models(prefs.openai_api_key)
            _cached_model_items.clear()
            _cached_model_items.extend(items)
            self.report({"INFO"}, "Loaded %d models" % len(items))
        except Exception as exc:
            self.report({"ERROR"}, "Failed: %s" % str(exc)[:150])
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
# Registration
# ═══════════════════════════════════════════════════════════════════════════

classes = (
    AIHOUSE_OT_send_prompt,
    AIHOUSE_OT_stop_generation,
    AIHOUSE_OT_execute_code,
    AIHOUSE_OT_clear_scene,
    AIHOUSE_OT_clear_chat,
    AIHOUSE_OT_paste_api_key,
    AIHOUSE_OT_open_openai_keys,
    AIHOUSE_OT_test_api_key,
    AIHOUSE_OT_refresh_models,
    AIHOUSE_OT_add_reference_image,
    AIHOUSE_OT_remove_reference_image,
    AIHOUSE_OT_clear_reference_images,
    AIHOUSE_OT_search_reference_images,
    AIHOUSE_OT_drop_reference_image,
    AIHOUSE_OT_open_ref_image,
    AIHOUSE_FH_drop_image,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
