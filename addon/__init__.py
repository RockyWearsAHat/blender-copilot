import sys
import importlib

# bl_info MUST be at module level for Blender's addon scanner.
bl_info = {
    "name": "Blender AI Copilot",
    "author": "AI Copilot Team",
    "version": (4, 0, 1),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > AI Copilot",
    "description": "AI-powered assistant for Blender — create, modify and explore anything from text prompts",
    "warning": "Requires AI server (OpenAI-compatible API)",
    "category": "3D View",
}

# Import all modules
if "bpy" in locals():
    # Reload all modules if already loaded (for F8 script reload)
    importlib.reload(properties)
    importlib.reload(preferences)
    importlib.reload(panels)
    importlib.reload(operators)
    importlib.reload(ai_engine)
    importlib.reload(blender_tools)
    importlib.reload(materials)
    importlib.reload(tool_defs)
else:
    from . import properties
    from . import preferences
    from . import panels
    from . import operators
    from . import ai_engine
    from . import blender_tools
    from . import materials
    from . import tool_defs

# Modules that register Blender classes (order matters)
_modules = [
    properties,
    preferences,
    panels,
    operators,
]


def register():
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
