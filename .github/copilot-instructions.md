# Copilot Instructions — Blender AI Copilot Plugin

## Project Overview

**AIHouseGenerator** is a Blender addon (3.6+) that lets users create and modify
3D models from natural-language text prompts. It uses OpenAI's chat API with
function-calling (tool use) so the AI can iteratively build, inspect, and refine
geometry inside a live Blender session.

The addon lives in `/Users/alexwaldmann/blenderPlugins/AIHouseGenerator/` and is
packaged as a ZIP for Blender's addon installer.

---

## Architecture

### Module Map

| File               | Role                                                                         |
| ------------------ | ---------------------------------------------------------------------------- |
| `__init__.py`      | Blender addon registration, module loading & reload ordering                 |
| `properties.py`    | `bpy.types.PropertyGroup` for per-scene state (prompt, status, chat history) |
| `preferences.py`   | Addon preferences (API key, model selector, temperature)                     |
| `panels.py`        | Sidebar UI (`View3D > Sidebar > AI Copilot`)                                 |
| `operators.py`     | Modal operator — async state machine that drives the AI loop                 |
| `ai_engine.py`     | System prompt, OpenAI API calls, tool-calling loop, conversation history     |
| `tool_defs.py`     | 6 OpenAI tool schemas + dispatch + `execute_code` sandbox                    |
| `blender_tools.py` | **All Blender helper functions** exposed to the AI via `execute_code`        |
| `materials.py`     | Procedural material/shader utilities                                         |
| `oauth.py`         | (Unused) OAuth stub                                                          |

### Data Flow

```
User prompt  →  operators.py (modal, background thread)
             →  ai_engine.py (generate_with_tools loop)
             →  OpenAI API  →  tool calls  →  tool_defs.py  →  blender_tools.py
             →  scene updated  →  inspect/capture  →  next AI round
             →  declare_complete  →  summary shown to user
```

### Threading Model

Blender's Python API is **single-threaded** — `bpy` calls must happen on the
main thread. The plugin uses a producer/consumer pattern:

1. `operators.py` starts a **background thread** that calls
   `ai_engine.generate_with_tools()`.
2. Any Blender-touching code is enqueued via `_run_on_main_thread(fn)`.
3. A **modal timer** (0.1 s) in `operators.py` calls
   `process_main_thread_queue()` to drain the queue on the main thread.
4. The background thread blocks on `threading.Event` until the main thread
   finishes execution.

**Rule:** Never call `bpy.*` directly from the background thread. Always wrap
in `_run_on_main_thread()`.

---

## Design Philosophy

### The Fundamental Insight

> **LLMs cannot invent 3D geometry coordinates.**
>
> They can write syntactically correct Python that places vertices, but the
> spatial reasoning required to produce recognizable shapes from raw (x, y, z)
> tuples is beyond current language models. Asking an LLM to output 200 control
> points for a car silhouette will always produce blobby, unrecognizable shapes.

### The Solution: Modifier-Based Box Modeling

Instead of asking the AI to invent coordinates, give it **the same tools a real
Blender artist would use**:

1. **Start from primitives** — `create_box`, `create_cylinder`, `create_sphere`
2. **Add topology** — `loop_cut`, `add_detail_cuts`, `subdivide_mesh`
3. **Shape with transforms** — `scale_section`, `move_section`, `taper`, `bend`
4. **Refine silhouette** — `extrude_faces`, `inset_faces`, `extrude_and_scale`
5. **Add modifiers** — `subsurf`, `mirror`, `bevel`, `solidify`
6. **Combine** — `boolean_cut`, `boolean_join`
7. **Detail** — `bevel_edges`, `set_edge_crease`, `shade_auto_smooth`
8. **Material** — `quick_material`, `pbr_material`, `glass_material`

Every parameter is a **simple number, enum, or fraction** — never a coordinate
array. The AI leverages its knowledge of Blender workflows (from training on
thousands of tutorials) to choose the right sequence of operations.

### What the AI Is Good At

- Knowing **which** Blender operations to use and in **what order**
- Choosing plausible **real-world dimensions** (car = 4.5 m, door = 0.9 × 2.1 m)
- Selecting **materials** with correct color, roughness, metallic values
- **Relative positioning** — placing wheels at the corners of a body
- **Iterative refinement** — inspecting bounds, then adjusting

### What the AI Is Bad At

- Inventing raw vertex positions or Bézier control points
- Spatial reasoning about complex curved surfaces
- Knowing exact edge indices without inspection
- Any operation that requires "artistic eye" judgment on 3D shape

---

## Tool-Calling Architecture

The plugin uses a **hybrid 6-tool approach**:

| Tool                | Purpose                                                                                |
| ------------------- | -------------------------------------------------------------------------------------- |
| `execute_code`      | **Primary** — runs a Python code block with all `blender_tools` functions pre-imported |
| `inspect_scene`     | Returns all objects, materials, counts                                                 |
| `inspect_object`    | Returns one object's dimensions, bounds, material                                      |
| `get_object_bounds` | Returns bounding box with axis-labelled extents                                        |
| `capture_viewport`  | Takes a screenshot for visual verification                                             |
| `declare_complete`  | Signals the model is done                                                              |

The AI writes **bulk code blocks** (10-20 operations) via `execute_code`,
then verifies with inspection tools, and iterates.

### Why Not 60+ Individual Tools?

We tried this. The AI made one tiny change per round (add one vertex, move one
edge) across 13+ rounds. Bulk code via `execute_code` is dramatically faster
and lets the AI think in terms of complete workflows.

---

## Axis Conventions

| Axis | Direction    | Vehicle Mapping   |
| ---- | ------------ | ----------------- |
| X    | Forward/back | Car LENGTH        |
| Y    | Left/right   | Car LATERAL WIDTH |
| Z    | Up/down      | HEIGHT (always)   |

`get_object_bounds` returns:

- `width_x` = X extent
- `depth_y` = Y extent
- `height_z` = Z extent

**Common mistake:** Using `width_x/2` for left/right placement. For lateral
offset, always use `depth_y/2`.

---

## Key Functions in `blender_tools.py`

### Box Modeling / Section-Based Shaping

These are the **preferred** tools for creating organic/complex shapes:

- `scale_section(obj, axis, position, tolerance, scale_x, scale_y, scale_z)` — scale a cross-section ring
- `move_section(obj, axis, position, tolerance, offset)` — translate a cross-section ring
- `taper(obj, axis, start_scale, end_scale)` — linear taper along an axis
- `bend(obj, axis, angle_deg, center, bend_axis)` — bend geometry
- `pinch(obj, axis, position, radius, strength)` — pinch geometry at a position
- `bulge(obj, axis, position, radius, strength)` — bulge geometry at a position
- `add_detail_cuts(obj, axis, num_cuts)` — evenly-spaced loop cuts
- `crease_edge_loop_at(obj, axis, position, tolerance, sharpness)` — crease for SubD
- `set_profile_shape(obj, axis, profile)` — reshape cross-sections via profile curve

### Primitives → Modifiers Workflow

```
create_box → add_detail_cuts → scale_section (multiple) → mirror →
subsurf → bevel_edges → shade_auto_smooth → materials
```

### Functions to AVOID in System Prompts

- `shape_from_profiles()` — requires AI to invent curve control points
- `mesh_from_outlines()` — same problem
- `revolve_profile()` — same problem
- `create_mesh(verts=..., faces=...)` — raw coordinate invention
- Any function taking a `points=[(x,y,z), ...]` parameter as primary input

These exist for advanced users writing their own code, but the AI should NOT
be guided to use them.

---

## Build & Deploy

### Build ZIP

```bash
cd /Users/alexwaldmann/blenderPlugins
rm -f AIHouseGenerator.zip
zip -r AIHouseGenerator.zip AIHouseGenerator/__init__.py \
  AIHouseGenerator/properties.py AIHouseGenerator/preferences.py \
  AIHouseGenerator/panels.py AIHouseGenerator/operators.py \
  AIHouseGenerator/ai_engine.py AIHouseGenerator/blender_tools.py \
  AIHouseGenerator/materials.py AIHouseGenerator/oauth.py \
  AIHouseGenerator/tool_defs.py AIHouseGenerator/README.md \
  -x '*.pyc' '*__pycache__*' '*.DS_Store' '*chat_logs*' '*_backup*'
```

Or use the VS Code task: **Rebuild Plugin ZIP**.

### Install in Blender

1. Edit > Preferences > Add-ons > Install…
2. Select `AIHouseGenerator.zip`
3. Enable "Blender AI Copilot"
4. Set API key in addon preferences

### Testing

- Open Blender, press N → AI Copilot tab
- Type a prompt like "Create a simple house with a red roof"
- Watch the status messages — the AI should iterate 3-8 rounds
- Check System Console (Window > Toggle System Console) for errors

---

## Development Guidelines

### When Adding New `blender_tools` Functions

1. **Parameters must be simple** — numbers, strings, enums, small tuples.
   Never require the AI to pass coordinate arrays as primary shape input.
2. **Return useful info** — return the created/modified object, or a dict with
   results, so the AI can chain operations.
3. **Add docstrings** — `_get_tools_reference()` in `ai_engine.py` uses AST
   parsing to extract function signatures, so docstrings are what the AI sees.
4. **Handle errors gracefully** — wrap BMesh operations in try/finally to ensure
   `bm.free()` is called. Check for None objects.
5. **No raw `bpy.ops` in tools** — use BMesh or direct data API where possible.
   `bpy.ops` has context requirements that break in background execution.

### When Modifying the System Prompt

- The prompt is in `ai_engine.py` → `_TOOL_SYSTEM_PROMPT`
- It has a `{tools_reference}` placeholder filled by `_get_tools_reference()`
- Keep it concise — every token of system prompt is sent on every API call
- Focus on **workflow patterns** (what sequence of tools to use), not on
  exhaustive API documentation
- Test with "Create a sports car" — this is the hardest case

### When Modifying `operators.py`

- The modal state machine has states: `IDLE`, `TOOL_LOOP`
- All Blender API calls must happen on the main thread
- Use `_run_on_main_thread()` for anything touching `bpy`
- The background thread calls `generate_with_tools()` which returns
  `(summary, is_complete)`
- Never call `bpy.ops.ed.undo_push()` from a background thread

### Common Pitfalls

- **`bpy.ops` from background thread** — crashes Blender silently
- **`undo_push` from wrong thread** — poll() fails, wrap in try/except
- **Edge indices change** — after any mesh edit, all indices are invalidated.
  Re-query with `find_verts_near()` or `find_verts_in_range()`.
- **Boolean solver** — `EXACT` solver is more reliable but slower. Falls back
  to `FAST` automatically on failure.
- **`limited_dissolve` on vehicles** — destroys center-seam vertices, avoid on
  anything with mirror symmetry

---

## File Size Reference

| File               | ~Lines | Notes                                  |
| ------------------ | ------ | -------------------------------------- |
| `blender_tools.py` | 4480   | Largest — all Blender helper functions |
| `ai_engine.py`     | 2210   | System prompt + API + tool loop        |
| `operators.py`     | 713    | Modal operator + state machine         |
| `tool_defs.py`     | 229    | Tool schemas + dispatch                |
| `panels.py`        | ~200   | UI layout                              |
| `properties.py`    | 108    | Scene properties                       |
| `preferences.py`   | ~100   | Addon preferences                      |
| `materials.py`     | ~150   | Material helpers                       |
