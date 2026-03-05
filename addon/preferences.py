"""Addon preferences — local server configuration.

This addon intentionally keeps heavy ML deps (torch) out of Blender's Python.
Policy rollouts are executed via an external Python (venv) process, and the
resulting OBJ is imported back into the current scene.
"""

import bpy  # type: ignore
from bpy.types import AddonPreferences  # type: ignore
from bpy.props import StringProperty, FloatProperty, IntProperty, BoolProperty  # type: ignore

from pathlib import Path


# Hardcoded LLM server URL (ollama)
LLM_URL = "http://127.0.0.1:11434"


class BlenderCopilotPreferences(AddonPreferences):
    bl_idname = __package__

    # ── LLM Brain (ollama / local reasoning model) ──────────────
    llm_model: StringProperty(  # type: ignore
        name="LLM Model",
        description="Model name for the reasoning LLM (e.g. qwen2.5vl:32b)",
        default="qwen2.5vl:32b",
    )

    # ── Mesh Generation Server (our trained model) ──────────────
    local_server_url: StringProperty(  # type: ignore
        name="Mesh Server URL",
        description="URL of the trained mesh generation server (for RLHF feedback)",
        default="http://127.0.0.1:8420",
    )

    temperature: FloatProperty(  # type: ignore
        name="Temperature",
        description="Creativity of the AI (0 = deterministic, 1 = creative)",
        default=0.4,
        min=0.0,
        max=1.0,
    )

    generation_timeout: IntProperty(  # type: ignore
        name="Generation Timeout (seconds)",
        description="Maximum time to wait for mesh generation (0 = no timeout)",
        default=0,
        min=0,
        max=3600,
    )

    # ── Policy (architecture-compliant) rollout runner ────────────
    policy_project_root: StringProperty(  # type: ignore
        name="Policy Project Root",
        description="Path to the blender-copilot repo (contains scripts/, checkpoints/, .venv/)",
        subtype='DIR_PATH',
        default=str(Path.home() / "blenderPlugins" / "blender-copilot"),
    )

    policy_python: StringProperty(  # type: ignore
        name="Policy Python (venv)",
        description="Path to Python executable with torch installed (outside Blender)",
        subtype='FILE_PATH',
        default="",
    )

    policy_checkpoint: StringProperty(  # type: ignore
        name="Policy Checkpoint",
        description="Path to policy checkpoint (.pt), e.g. checkpoints/policy_goal/latest.pt",
        subtype='FILE_PATH',
        default="",
    )

    policy_steps: IntProperty(  # type: ignore
        name="Policy Steps",
        description="Number of closed-loop steps for policy generation",
        default=32,
        min=1,
        max=512,
    )

    policy_seed: IntProperty(  # type: ignore
        name="Policy Seed",
        description="Seed for deterministic rollouts",
        default=0,
        min=0,
        max=2**31 - 1,
    )

    policy_apply_modifiers: BoolProperty(  # type: ignore
        name="Apply Modifiers",
        description="Apply generated modifiers (mirror/solidify/etc.) during rollout",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Blender Copilot Settings", icon='LIGHT')
        layout.separator()

        # LLM Brain section
        box = layout.box()
        box.label(text="AI Brain (Local LLM)", icon='LIGHT')
        box.prop(self, "llm_model")
        box.separator()
        box.label(text="Runs via ollama on %s" % LLM_URL, icon='INFO')

        layout.separator()

        # Mesh server section
        box2 = layout.box()
        box2.label(text="Mesh Generation Server", icon='MESH_DATA')
        box2.prop(self, "local_server_url")
        box2.label(text="Trained model for direct mesh generation", icon='INFO')

        layout.separator()
        layout.prop(self, "temperature")
        layout.prop(self, "generation_timeout")

        layout.separator()

        box3 = layout.box()
        box3.label(text="Policy Generator (External Python)", icon='OUTLINER_OB_MESH')
        box3.prop(self, "policy_project_root")
        box3.prop(self, "policy_python")
        box3.prop(self, "policy_checkpoint")
        row = box3.row(align=True)
        row.prop(self, "policy_steps")
        row.prop(self, "policy_seed")
        box3.prop(self, "policy_apply_modifiers")
        box3.label(text="Runs headless rollout and imports OBJ into this scene", icon='INFO')


classes = (
    BlenderCopilotPreferences,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
