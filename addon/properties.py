"""Scene-level properties for the Blender Copilot.

One PropertyGroup registered on ``bpy.types.Scene.ai_copilot``.
"""

import bpy  # type: ignore
from bpy.props import (StringProperty, BoolProperty, IntProperty,  # type: ignore
                        FloatProperty, CollectionProperty, EnumProperty)
from bpy.types import PropertyGroup  # type: ignore


class AICopilotRefImage(PropertyGroup):
    """A single image attached to the current prompt."""
    filepath: StringProperty(  # type: ignore
        name="File",
        description="Absolute path to the image on disk",
        subtype='FILE_PATH',
        default="",
    )


class AICopilotProperties(PropertyGroup):
    """Properties attached to every Scene via ``scene.ai_copilot``."""

    prompt_text: StringProperty(  # type: ignore
        name="Prompt",
        description="Describe what you want to create or modify",
        default="",
    )

    last_response: StringProperty(  # type: ignore
        name="Response",
        description="Last AI explanation",
        default="",
    )

    last_code: StringProperty(  # type: ignore
        name="Code",
        description="Last generated Python code",
        default="",
    )

    status: StringProperty(  # type: ignore
        name="Status",
        description="Current copilot status",
        default="Ready — type a prompt below",
    )

    is_generating: BoolProperty(  # type: ignore
        name="Generating",
        description="True while waiting for the AI",
        default=False,
    )

    auto_execute: BoolProperty(  # type: ignore
        name="Auto-execute",
        description="Automatically run generated code in Blender",
        default=True,
    )

    auto_fix: BoolProperty(  # type: ignore
        name="Auto-fix errors",
        description="If code errors, send the error back to the AI for repair",
        default=True,
    )

    auto_iterate: BoolProperty(  # type: ignore
        name="Auto-iterate",
        description="After generating, assess the result and keep refining until the AI is satisfied",
        default=True,
    )

    show_code: BoolProperty(  # type: ignore
        name="Show Code",
        description="Display generated code in the panel",
        default=False,
    )

    # ── Pending image attachments (cleared after send) ─────────────
    reference_images: CollectionProperty(  # type: ignore
        type=AICopilotRefImage,
        name="Attached Images",
        description="Images attached to the current prompt",
    )

    active_ref_index: IntProperty(  # type: ignore
        name="Active Attachment",
        default=0,
    )

    # ── RLHF Feedback Properties ──────────────────────────────────
    is_comparing: BoolProperty(  # type: ignore
        name="Comparing",
        description="True while showing A/B comparison",
        default=False,
    )

    compare_prompt: StringProperty(  # type: ignore
        name="Compare Prompt",
        description="The prompt used for current comparison",
        default="",
    )

    compare_choice: EnumProperty(  # type: ignore
        name="Choice",
        description="Which output the user prefers",
        items=[
            ("NONE", "Not chosen", "No preference selected"),
            ("A", "Option A", "Prefer option A"),
            ("B", "Option B", "Prefer option B"),
            ("TIE", "About equal", "Both are roughly equal"),
        ],
        default="NONE",
    )

    feedback_status: StringProperty(  # type: ignore
        name="Feedback Status",
        description="Current feedback / RLHF status",
        default="",
    )

    last_generation_tokens: StringProperty(  # type: ignore
        name="Last Tokens",
        description="JSON-encoded mesh tokens from last generation (for feedback)",
        default="",
    )

    feedback_count: IntProperty(  # type: ignore
        name="Feedback Count",
        description="Total feedback items submitted this session",
        default=0,
    )

    # ── Training Loop Properties ──────────────────────────────────
    training_active: BoolProperty(  # type: ignore
        name="Training Active",
        description="True while the training data loop is running",
        default=False,
    )

    training_prompt: StringProperty(  # type: ignore
        name="Training Prompt",
        description="The current prompt being evaluated in training mode",
        default="",
    )

    training_awaiting: BoolProperty(  # type: ignore
        name="Awaiting Feedback",
        description="True when waiting for user approve/reject",
        default=False,
    )

    training_approved: IntProperty(  # type: ignore
        name="Approved",
        description="Number of approved outputs in this training session",
        default=0,
    )

    training_rejected: IntProperty(  # type: ignore
        name="Rejected",
        description="Number of rejected outputs in this training session",
        default=0,
    )

    training_skipped: IntProperty(  # type: ignore
        name="Skipped",
        description="Number of skipped outputs in this training session",
        default=0,
    )

    training_total: IntProperty(  # type: ignore
        name="Total",
        description="Total prompts processed in this training session",
        default=0,
    )

    # ── Dataset Validation (human-in-the-loop data hygiene) ─────────
    validator_queue_dir: StringProperty(  # type: ignore
        name="Validation Queue",
        description="Select repo root, data/training_cache/default, data/processed/.mesh_cache, or a legacy exported queue",
        subtype='DIR_PATH',
        default="",
    )

    validator_fresh_only: BoolProperty(  # type: ignore
        name="Fresh Scope Only",
        description="Only review newly regenerated/newly pulled/generated items within the time window",
        default=True,
    )

    validator_fresh_hours: FloatProperty(  # type: ignore
        name="Fresh Window (hours)",
        description="How far back to consider items 'new' for visual review",
        default=72.0,
        min=0.0,
        max=24.0 * 365.0,
    )

    validator_loaded: BoolProperty(  # type: ignore
        name="Queue Loaded",
        default=False,
    )

    validator_index: IntProperty(  # type: ignore
        name="Index",
        default=0,
        min=0,
    )

    validator_total: IntProperty(  # type: ignore
        name="Total",
        default=0,
        min=0,
    )

    validator_current_item_id: StringProperty(  # type: ignore
        name="Item ID",
        default="",
    )

    validator_current_item_path: StringProperty(  # type: ignore
        name="Item JSON",
        subtype='FILE_PATH',
        default="",
    )

    validator_cache_pt: StringProperty(  # type: ignore
        name="Cache PT",
        description="Path to the source .pt cache file containing this item",
        subtype='FILE_PATH',
        default="",
    )

    validator_item_index: IntProperty(  # type: ignore
        name="Cache Index",
        description="Index within the source cache file",
        default=-1,
    )

    validator_source: StringProperty(  # type: ignore
        name="Source",
        default="",
    )

    validator_sample_type: StringProperty(  # type: ignore
        name="Sample Type",
        description="Type of cache sample (object or scene_composition)",
        default="",
    )

    validator_label: StringProperty(  # type: ignore
        name="Label",
        description="Edit the training label for this object",
        default="",
    )

    validator_tags: StringProperty(  # type: ignore
        name="Tags",
        description="Comma-separated tags for later filtering/training",
        default="",
    )

    validator_quality_weight: FloatProperty(  # type: ignore
        name="Quality Weight",
        description="Training quality weight from cache",
        default=0.0,
    )

    validator_human_verdict: StringProperty(  # type: ignore
        name="Human Verdict",
        description="Prior human verdict attached to this cache item",
        default="",
    )

    validator_flags: StringProperty(  # type: ignore
        name="Flags",
        description="Data quality flags inferred during cache materialization",
        default="",
    )

    validator_materials: StringProperty(  # type: ignore
        name="Materials",
        description="Comma-separated material names extracted from scene context",
        default="",
    )

    validator_scene_keys: StringProperty(  # type: ignore
        name="Scene Keys",
        description="Scene context keys present on this cache item",
        default="",
    )

    validator_scene_json: StringProperty(  # type: ignore
        name="Scene JSON",
        description="Path to exported scene_context JSON for this item",
        subtype='FILE_PATH',
        default="",
    )

    validator_status: StringProperty(  # type: ignore
        name="Validator Status",
        default="",
    )

    validator_approved: IntProperty(  # type: ignore
        name="Approved",
        default=0,
        min=0,
    )

    validator_rejected: IntProperty(  # type: ignore
        name="Rejected",
        default=0,
        min=0,
    )

    validator_skipped: IntProperty(  # type: ignore
        name="Skipped",
        default=0,
        min=0,
    )


# ── Registration ──────────────────────────────────────────────────────────

classes = (AICopilotRefImage, AICopilotProperties,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ai_copilot = bpy.props.PointerProperty(type=AICopilotProperties)


def unregister():
    if hasattr(bpy.types.Scene, "ai_copilot"):
        del bpy.types.Scene.ai_copilot
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
