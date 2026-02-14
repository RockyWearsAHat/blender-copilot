"""Scene-level properties for the Blender Copilot.

One PropertyGroup registered on ``bpy.types.Scene.ai_copilot``.
"""

import bpy  # type: ignore
from bpy.props import (StringProperty, BoolProperty, IntProperty,  # type: ignore
                        CollectionProperty)
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
