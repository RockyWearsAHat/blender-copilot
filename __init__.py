import importlib  # noqa: E402
import sys  # noqa: E402
import os  # noqa: E402

# bl_info MUST be at module level for Blender's addon scanner.
bl_info = {
    "name": "Blender AI Copilot",
    "author": "AI Copilot Team",
    "version": (4, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > AI Copilot",
    "description": "AI-powered assistant for Blender — create, modify and explore anything from text prompts",
    "warning": "Requires an OpenAI API key",
    "category": "3D View",
}

# Add the addon directory to sys.path so submodules can import each other
addon_dir = os.path.dirname(os.path.realpath(__file__))
if addon_dir not in sys.path:
    sys.path.append(addon_dir)

from . import properties  # noqa: E402
from . import preferences  # noqa: E402
from . import panels  # noqa: E402
from . import operators  # noqa: E402
from . import ai_engine  # noqa: E402
from . import blender_tools  # noqa: E402
from . import materials  # noqa: E402
from . import tool_defs  # noqa: E402

# Modules that register Blender classes (order matters)
_modules = [
    properties,
    preferences,
    panels,
    operators,
]

# Modules that don't register classes but still need reloading
_reload_only = [
    ai_engine,
    blender_tools,
    materials,
    tool_defs,
]


def register():
    # Reload ALL modules first to pick up code changes from reinstall.
    # Order matters: reload base modules before dependent ones.
    for mod in _reload_only + _modules:
        importlib.reload(mod)
    for mod in _modules:
        mod.register()


def unregister():
    # Persist any open chat before shutting down
    try:
        ai_engine.save_current_chat()
    except Exception:
        pass
    for mod in reversed(_modules):
        mod.unregister()


if __name__ == "__main__":
    register()
