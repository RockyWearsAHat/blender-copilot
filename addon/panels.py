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
    # Strip LLM special tokens (Qwen/ollama chat template markers)
    display = re.sub(r'<\|im_start\|>.*?\n?', '', text)
    display = re.sub(r'<\|im_end\|>', '', display)
    display = re.sub(r'<\|endoftext\|>', '', display)
    display = re.sub(r'<\|im_sep\|>', '', display)
    # Strip code blocks
    display = re.sub(r'```[\s\S]*?```', '', display).strip()
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
        if props.is_generating:
            stop_row = prompt_box.row(align=True)
            stop_row.scale_y = 1.4
            stop_row.operator("aihouse.stop_generation", text="Stop", icon="CANCEL")
        else:
            gen_row = prompt_box.row(align=True)
            gen_row.scale_y = 1.4
            gen_row.operator("aihouse.generate_direct", text="Generate", icon="PLAY")
            gen_row.operator("aihouse.generate_policy", text="Agent", icon="OUTLINER_OB_MESH")

        # ── Utility row ───────────────────────────────────────────────
        util = layout.row(align=True)
        util.scale_y = 0.75
        util.operator("aihouse.clear_scene", text="Clear Scene", icon="TRASH")
        util.operator("aihouse.clear_chat", text="New Chat", icon="FILE_NEW")

        # ── Inline feedback after generation ──────────────────────────
        if not props.is_generating and props.last_generation_tokens:
            fb_box = layout.box()
            fb_row = fb_box.row(align=True)
            fb_row.label(text="Rate this output:", icon="SOLO_ON")
            fb_row.operator("aihouse.accept_output", text="",
                            icon="CHECKMARK")
            fb_row.operator("aihouse.reject_output", text="",
                            icon="CANCEL")

        # Assistant-only toggles intentionally hidden in modeling-first UI.


# ═══════════════════════════════════════════════════════════════════════════
# Human Feedback Panel — A/B Comparison & Quality Rating
# ═══════════════════════════════════════════════════════════════════════════

class AIHOUSE_PT_feedback(Panel):
    bl_label = "Feedback"
    bl_idname = "AIHOUSE_PT_feedback"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Copilot"
    bl_options = set()

    def draw(self, context):
        layout = self.layout
        props = context.scene.ai_copilot

        # ── A/B Comparison section ────────────────────────────────────
        if props.is_comparing:
            box = layout.box()
            box.alert = True
            col = box.column(align=True)
            col.label(text="A/B Comparison Active", icon="HIDE_OFF")
            col.separator(factor=0.3)

            # Show the prompt being compared
            if props.compare_prompt:
                _wrap_text(col, props.compare_prompt[:120], width=36)
                col.separator(factor=0.3)

            col.label(text="Which output is better?", icon="QUESTION")

            # Choice buttons
            row = col.row(align=True)
            row.scale_y = 1.4
            op_a = row.operator("aihouse.submit_comparison", text="Option A",
                                icon="TRIA_LEFT")
            op_a.choice = "A"
            op_b = row.operator("aihouse.submit_comparison", text="Option B",
                                icon="TRIA_RIGHT")
            op_b.choice = "B"

            # Tie / Regenerate / Cancel row
            row2 = col.row(align=True)
            row2.scale_y = 1.0
            op_tie = row2.operator("aihouse.submit_comparison",
                                   text="Tie", icon="ARROW_LEFTRIGHT")
            op_tie.choice = "TIE"
            row2.operator("aihouse.regenerate_comparison",
                          text="Redo", icon="FILE_REFRESH")
            row2.operator("aihouse.cancel_comparison", text="",
                          icon="X")

        else:
            # ── Quick feedback on last output ─────────────────────────
            has_tokens = bool(props.last_generation_tokens)

            box = layout.box()
            col = box.column(align=True)
            col.label(text="Rate Last Output", icon="SOLO_ON")

            if has_tokens and props.compare_prompt:
                sub = col.column(align=True)
                sub.scale_y = 0.7
                prompt_display = props.compare_prompt
                if len(prompt_display) > 60:
                    prompt_display = prompt_display[:57] + "..."
                sub.label(text=prompt_display, icon="NONE")

            col.separator(factor=0.2)

            row = col.row(align=True)
            row.scale_y = 1.3
            row.enabled = has_tokens
            row.operator("aihouse.accept_output", text="Good",
                         icon="CHECKMARK")
            row.operator("aihouse.reject_output", text="Bad",
                         icon="CANCEL")

            if not has_tokens:
                col.label(text="Generate something first", icon="INFO")

            col.separator(factor=0.3)

            # ── Start new comparison ──────────────────────────────────
            col.operator("aihouse.start_comparison",
                         text="Compare A/B", icon="MOD_BOOLEAN")

        # ── Status / stats ────────────────────────────────────────────
        if props.feedback_status:
            layout.separator(factor=0.2)
            stat_box = layout.box()
            stat_col = stat_box.column(align=True)
            stat_col.scale_y = 0.7
            stat_col.label(text=props.feedback_status, icon="INFO")
            if props.feedback_count > 0:
                stat_col.label(
                    text="Session feedback: %d" % props.feedback_count)


# ═══════════════════════════════════════════════════════════════════════════
# Training Data Loop Panel
# ═══════════════════════════════════════════════════════════════════════════

class AIHOUSE_PT_training(Panel):
    bl_label = "Training Data Loop"
    bl_idname = "AIHOUSE_PT_training"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Copilot"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.ai_copilot

        if not props.training_active:
            # ── Start button ──────────────────────────────────────
            box = layout.box()
            col = box.column(align=True)
            col.label(text="Rapid Training Mode", icon="SEQUENCE")
            col.separator(factor=0.2)
            col.scale_y = 0.8
            col.label(text="Auto-generates random prompts.")
            col.label(text="Approve or reject each result")
            col.label(text="to build training data fast.")
            col.separator(factor=0.4)
            row = col.row()
            row.scale_y = 1.6
            row.operator("aihouse.training_start",
                         text="Start Training Loop", icon="PLAY")
        else:
            # ── Active training session ───────────────────────────
            # Current prompt
            box = layout.box()
            col = box.column(align=True)
            col.label(text="Current Prompt:", icon="ARMATURE_DATA")
            if props.training_prompt:
                sub = col.column(align=True)
                sub.scale_y = 0.85
                sub.alert = True
                _wrap_text(sub, props.training_prompt, width=36)

            if props.training_awaiting:
                # ── Approve / Reject / Skip ───────────────────────
                col.separator(factor=0.4)
                col.label(text="Rate this output:", icon="QUESTION")
                row = col.row(align=True)
                row.scale_y = 1.8
                row.operator("aihouse.training_approve",
                             text="Approve", icon="CHECKMARK")
                row.operator("aihouse.training_reject",
                             text="Reject", icon="CANCEL")
                skip_row = col.row(align=True)
                skip_row.scale_y = 1.0
                skip_row.operator("aihouse.training_skip",
                                  text="Skip (don't record)", icon="FORWARD")
            elif props.is_generating:
                col.separator(factor=0.3)
                col.label(text=props.status, icon="SORTTIME")
            else:
                col.separator(factor=0.3)
                col.label(text="Preparing next prompt...", icon="TIME")

            # ── Stats ─────────────────────────────────────────────
            stat_box = layout.box()
            stat_col = stat_box.column(align=True)
            stat_col.scale_y = 0.8
            total = props.training_total
            stat_col.label(text="Session Stats:", icon="GRAPH")
            stat_col.label(
                text="  Approved: %d  |  Rejected: %d  |  Skipped: %d"
                % (props.training_approved, props.training_rejected,
                   props.training_skipped))
            stat_col.label(text="  Total: %d" % total)
            if total > 0:
                rate = props.training_approved / total * 100
                stat_col.label(text="  Approval rate: %.0f%%" % rate)

            # ── Stop button ───────────────────────────────────────
            layout.separator(factor=0.3)
            row = layout.row()
            row.scale_y = 1.3
            row.alert = True
            row.operator("aihouse.training_stop",
                         text="Stop Training Loop", icon="PAUSE")


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Validator Panel
# ═══════════════════════════════════════════════════════════════════════════

class AIHOUSE_PT_dataset_validator(Panel):
    bl_label = "Dataset Validator"
    bl_idname = "AIHOUSE_PT_dataset_validator"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AI Copilot"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        try:
            props = context.scene.ai_copilot

            box = layout.box()
            col = box.column(align=True)
            # NOTE: some icon IDs differ across Blender versions; prefer very common icons.
            col.label(text="Validate training data in Blender", icon="INFO")
            col.label(text="Select repo root or a cache directory", icon="DOT")
            col.label(text="(auto-detects training_cache or .mesh_cache)", icon="DOT")
            col.separator(factor=0.2)
            col.prop(props, "validator_queue_dir")
            col.prop(props, "validator_fresh_only")
            if getattr(props, "validator_fresh_only", False):
                col.prop(props, "validator_fresh_hours")
                col.label(text="Only new/regenerated items in this window are queued", icon="TIME")

            row = col.row(align=True)
            row.scale_y = 1.2
            row.operator("aihouse.validator_load_queue", text="Load Queue", icon="FILE_REFRESH")
            row.operator("aihouse.validator_load_current", text="Load Current", icon="IMPORT")

            if props.validator_loaded:
                col.separator(factor=0.2)
                prog = col.row(align=True)
                prog.scale_y = 0.8
                prog.label(text=f"Item {props.validator_index + 1} / {props.validator_total}", icon="INFO")
                if props.validator_source:
                    prog.label(text=props.validator_source, icon="INFO")

            if props.validator_current_item_id:
                col.separator(factor=0.2)
                info = col.box()
                info_col = info.column(align=True)
                info_col.scale_y = 0.8
                info_col.label(text=f"Item ID: {props.validator_current_item_id}", icon="INFO")
                if props.validator_current_item_path:
                    info_col.label(text=f"Item JSON: {os.path.basename(props.validator_current_item_path)}", icon="FILE")
                if getattr(props, "validator_cache_pt", ""):
                    info_col.label(text=f"Cache: {os.path.basename(props.validator_cache_pt)}", icon="FILE")
                if getattr(props, "validator_item_index", -1) >= 0:
                    info_col.label(text=f"Cache index: {props.validator_item_index}", icon="INFO")
                if getattr(props, "validator_sample_type", ""):
                    info_col.label(text=f"Sample type: {props.validator_sample_type}", icon="INFO")
                info_col.label(text=f"Quality weight: {props.validator_quality_weight:.3f}", icon="INFO")
                if props.validator_human_verdict:
                    info_col.label(text=f"Prior verdict: {props.validator_human_verdict}", icon="INFO")
                if props.validator_flags:
                    info_col.label(text=f"Flags: {props.validator_flags}", icon="ERROR")
                if getattr(props, "validator_materials", ""):
                    info_col.label(text=f"Materials: {props.validator_materials}", icon="MATERIAL")
                if getattr(props, "validator_scene_keys", ""):
                    info_col.label(text=f"Scene keys: {props.validator_scene_keys}", icon="OUTLINER")
                if getattr(props, "validator_scene_json", ""):
                    info_col.label(text=f"Scene JSON: {os.path.basename(props.validator_scene_json)}", icon="FILE")
                col.prop(props, "validator_label")
                col.prop(props, "validator_tags")

                col.separator(factor=0.3)
                col.label(text="One-click decision:", icon="QUESTION")
                row2 = col.row(align=True)
                row2.scale_y = 1.6
                row2.operator("aihouse.validator_approve_next", text="Approve + Next", icon="CHECKMARK")
                row2.operator("aihouse.validator_reject_next", text="Reject + Next", icon="CANCEL")
                row3 = col.row(align=True)
                row3.scale_y = 1.0
                row3.operator("aihouse.validator_skip_next", text="Skip + Next", icon="FORWARD")

                # ── Reconstruct Full Scene ────────────────────────
                col.separator(factor=0.3)
                scene_row = col.row(align=True)
                scene_row.scale_y = 1.2
                scene_row.operator("aihouse.validator_reconstruct_scene",
                                   text="Reconstruct Full Scene", icon="SCENE_DATA")

            if props.validator_status:
                layout.separator(factor=0.2)
                stat = layout.box()
                stat_col = stat.column(align=True)
                stat_col.scale_y = 0.8
                stat_col.label(text=props.validator_status, icon="INFO")
                stat_col.label(
                    text="Approved: %d  |  Rejected: %d  |  Skipped: %d" % (
                        props.validator_approved,
                        props.validator_rejected,
                        props.validator_skipped,
                    )
                )
        except Exception as e:
            # Never fail silently: show something actionable in the UI.
            box = layout.box()
            box.alert = True
            box.label(text="Dataset Validator UI error", icon="ERROR")
            box.label(text=str(e)[:120])


# ═══════════════════════════════════════════════════════════════════════════
# Registration
# ═══════════════════════════════════════════════════════════════════════════

classes = (
    AIHOUSE_PT_copilot,
    AIHOUSE_PT_feedback,
    AIHOUSE_PT_training,
    AIHOUSE_PT_dataset_validator,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
