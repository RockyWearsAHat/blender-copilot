"""Addon preferences — stores API key, model, temperature, and backend."""

import json
import ssl
import urllib.request
import urllib.error

import bpy  # type: ignore
from bpy.types import AddonPreferences  # type: ignore
from bpy.props import (  # type: ignore
    StringProperty, EnumProperty, FloatProperty,
)


# ═══════════════════════════════════════════════════════════════════════════
# Model cache
# ═══════════════════════════════════════════════════════════════════════════

_CHAT_PREFIXES = (
    "gpt-5", "gpt-4", "gpt-3.5",
    "o1", "o3", "o4",
)

_EXCLUDE_SUBSTRINGS = (
    "audio", "realtime", "tts", "transcribe", "search",
    "instruct", "vision", "embed", "moderation", "dall",
    "whisper", "babbage", "davinci", "codex", "image",
    "computer", "chat-latest", "preview",
)

_cached_model_items: list = [
    ("gpt-4o", "GPT-4o", "Default model — click Refresh to load all models"),
]


def fetch_openai_models(api_key: str) -> list:
    """Call GET /v1/models and return sorted list of chat-capable models."""
    api_key = api_key.strip()
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    req = urllib.request.Request(url, headers=headers, method="GET")
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"OpenAI API error {e.code}: {error_body}") from e

    data = json.loads(body)
    models = data.get("data", [])

    result = []
    for m in models:
        mid = m.get("id", "")
        if not any(mid.startswith(p) for p in _CHAT_PREFIXES):
            continue
        if any(ex in mid for ex in _EXCLUDE_SUBSTRINGS):
            continue
        result.append((mid, mid, mid))

    result.sort(key=lambda x: x[0], reverse=True)

    if not result:
        result.append(("gpt-4o", "GPT-4o", "No chat models found — using default"))

    return result


def _get_model_items(self, context):
    return _cached_model_items


def _on_api_key_changed(self, _context):
    raw = self.openai_api_key
    cleaned = raw.replace('\n', '').replace('\r', '').replace(' ', '').replace('\t', '')
    if cleaned != raw:
        self['openai_api_key'] = cleaned
    if cleaned:
        try:
            items = fetch_openai_models(cleaned)
            _cached_model_items.clear()
            _cached_model_items.extend(items)
        except Exception:
            pass


class BlenderCopilotPreferences(AddonPreferences):
    bl_idname = __package__

    ai_backend: EnumProperty(  # type: ignore
        name="AI Backend",
        description="Which AI backend to use for generation",
        items=[
            ("openai", "OpenAI API", "Use OpenAI cloud API (requires API key)"),
            ("local", "Local Model", "Use locally trained model server (http://127.0.0.1:8420)"),
        ],
        default="openai",
    )

    local_server_url: StringProperty(  # type: ignore
        name="Local Server URL",
        description="URL of the local inference server",
        default="http://127.0.0.1:8420",
    )

    openai_api_key: StringProperty(  # type: ignore
        name="OpenAI API Key",
        description="Your OpenAI API key (starts with sk-…)",
        subtype='PASSWORD',
        default="",
        update=_on_api_key_changed,
    )

    model: EnumProperty(  # type: ignore
        name="Model",
        description="OpenAI model to use (click Refresh to load from API)",
        items=_get_model_items,
    )

    temperature: FloatProperty(  # type: ignore
        name="Temperature",
        description="Creativity of the AI (0 = deterministic, 1 = creative)",
        default=0.4,
        min=0.0,
        max=1.0,
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Blender Copilot Settings", icon='LIGHT')
        layout.separator()

        # Backend selection
        box = layout.box()
        box.label(text="AI Backend", icon='SETTINGS')
        box.prop(self, "ai_backend", expand=True)

        if self.ai_backend == "local":
            # Local model settings
            box2 = layout.box()
            box2.label(text="Local Model Server", icon='NETWORK_DRIVE')
            box2.prop(self, "local_server_url")
            box2.operator("aihouse.test_local_server", text="Test Connection", icon='PLUGIN')
            box2.separator()
            box2.label(text="Start server: python cli.py \u2192 option 5", icon='INFO')
        else:
            # OpenAI settings
            box2 = layout.box()
            box2.label(text="OpenAI API Key", icon='KEYTYPE_KEYFRAME_VEC')

            row = box2.row(align=True)
            row.scale_y = 1.3
            row.operator("aihouse.paste_api_key", text="Paste Key from Clipboard", icon='PASTEDOWN')
            row.operator("aihouse.open_openai_keys", text="", icon='URL')

            box2.prop(self, "openai_api_key")

            if self.openai_api_key:
                key = self.openai_api_key.strip()
                box2.label(text=f"\u2713 Key set ({len(key)} chars, starts with {key[:8]}\u2026)", icon='CHECKMARK')
                box2.operator("aihouse.test_api_key", text="Test Connection", icon='PLUGIN')
            else:
                box2.label(text="Copy your key then click 'Paste Key from Clipboard'", icon='INFO')

            layout.separator()
            row = layout.row(align=True)
            row.prop(self, "model")
            row.operator("aihouse.refresh_models", text="", icon='FILE_REFRESH')

        layout.prop(self, "temperature")


classes = (
    BlenderCopilotPreferences,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
