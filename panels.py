"""UI Panels for the Blender Copilot.

Clean chat interface with streaming text display.
"""

import re
import os
import bpy  # type: ignore
from bpy.types import Panel  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _wrap_text(layout, text, width=42):
    """Word-wrap *text* into compact label rows."""
    if not text:
        return
    col = layout.column(align=True)
    col.scale_y = 0.8
    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            continue
        words = stripped.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > width:
                col.label(text=line)
                line = word
            else:
                line = (line + " " + word) if line else word
        if line:
            col.label(text=line)


def _clean_for_display(text):
    """Strip code blocks and internal markers from text for display."""
    display = re.sub(r'```[\s\S]*?```', '', text).strip()
    # Strip scene-context block
    if "[Current Blender scene]" in display:
        marker = "[Your request]"
        idx = display.find(marker)
        if idx >= 0:
            display = display[idx + len(marker):].strip()
    # Strip selection context block
    if "[Selected objects]" in display:
        marker = "[Your request]"
        idx = display.find(marker)
        if idx >= 0:
            display = display[idx + len(marker):].strip()
    return display


# ═══════════════════════════════════════════════════════════════════════════
# Main Chat Panel
# ═══════════════════════════════════════════════════════════════════════════

class AIHOUSE_PT_copilot(Panel):
    bl_label = "Blender Copilot"
    bl_idname = "AIHOUSE_PT_copilot"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Copilot"

    def draw(self, context):
        layout = self.layout
        props = context.scene.ai_copilot

        # ── Get data ──────────────────────────────────────────────────
        try:
            from . import ai_engine
            history = ai_engine.get_history()
            streaming = ai_engine.get_streaming_text()
        except Exception:
            history = []
            streaming = ""

        has_content = bool(history or streaming)

        if has_content:
            chat = layout.column(align=False)

            # ── Conversation messages ─────────────────────────────────
            for msg in history[-8:]:
                role = msg["role"]
                text = msg["content"]
                display = _clean_for_display(text)
                if not display:
                    continue

                box = chat.box()
                if role == "user":
                    box.label(text="You", icon="USER")
                else:
                    box.label(text="Copilot", icon="LIGHT")
                _wrap_text(box, display[:800], width=38)
                chat.separator(factor=0.05)

            # ── Live streaming text ───────────────────────────────────
            if props.is_generating and streaming:
                stream_display = _clean_for_display(streaming)
                if stream_display:
                    sbox = chat.box()
                    sbox.label(text="Copilot ✍", icon="LIGHT")
                    _wrap_text(sbox, stream_display[:800], width=38)
                    chat.separator(factor=0.05)

        else:
            # ── Empty state ───────────────────────────────────────────
            box = layout.box()
            col = box.column(align=True)
            col.label(text="Blender Copilot", icon="LIGHT")
            col.separator(factor=0.3)
            col.label(text="Your AI assistant for Blender.")
            col.label(text="Create, modify, or ask anything.")

        # ── Status bar ────────────────────────────────────────────────
        if props.is_generating:
            status_box = layout.box()
            status_box.alert = True
            status_box.label(text=props.status, icon="SORTTIME")
        elif props.status and not props.status.startswith("Ready"):
            layout.label(text=props.status, icon="INFO")

        layout.separator(factor=0.3)

        # ── Prompt bar ────────────────────────────────────────────────
        prompt_box = layout.box()
        prompt_box.prop(props, "prompt_text", text="")

        # ── Attached image thumbnails (Copilot-style) ────────────────
        if props.reference_images:
            img_col = prompt_box.column(align=True)
            for i, ref in enumerate(props.reference_images):
                row = img_col.row(align=True)
                row.scale_y = 0.7
                fname = os.path.basename(ref.filepath) if ref.filepath else "(image)"
                if len(fname) > 30:
                    fname = fname[:27] + "…"
                # Click filename to open/preview the image
                op = row.operator("aihouse.open_ref_image", text=fname,
                                 icon="IMAGE_DATA", emboss=False)
                op.filepath = ref.filepath
                # Remove button
                op = row.operator("aihouse.remove_ref_image", text="",
                                 icon="X", emboss=False)
                op.index = i

        # ── Action row: Send + Attach + Search ────────────────────────
        row = prompt_box.row(align=True)
        row.scale_y = 1.4
        if props.is_generating:
            row.operator("aihouse.stop_generation", text="Stop", icon="CANCEL")
        else:
            row.operator("aihouse.send_prompt", text="Send", icon="PLAY")
            row.operator("aihouse.add_ref_image", text="", icon="IMAGE_DATA")
            row.operator("aihouse.search_ref_images", text="", icon="WORLD")

        # ── Utility row ───────────────────────────────────────────────
        util = layout.row(align=True)
        util.scale_y = 0.75
        util.operator("aihouse.clear_scene", text="Clear Scene", icon="TRASH")
        util.operator("aihouse.clear_chat", text="New Chat", icon="FILE_NEW")

        # ── Settings toggles ─────────────────────────────────────────
        settings = layout.column(align=True)
        settings.scale_y = 0.8
        row = settings.row(align=True)
        row.prop(props, "auto_iterate", text="Iterate", icon="FILE_REFRESH", toggle=True)
        row.prop(props, "auto_fix", text="Auto-fix", icon="TOOL_SETTINGS", toggle=True)


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

classes = (
    AIHOUSE_PT_copilot,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
