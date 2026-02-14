# AI House Generator — Blender Plugin

Generate 3D houses in Blender using AI. Provide a **style reference** (image
or text) and a **blueprint / floor plan** (image or text), and the plugin will
procedurally build a complete house with walls, roof, windows, doors, and PBR
materials.

Supports **OpenAI**, **Anthropic Claude**, and **GitHub Copilot** as AI providers.

![Blender 3.6+](https://img.shields.io/badge/Blender-3.6%2B-orange)
![License MIT](https://img.shields.io/badge/License-MIT-green)

---

## Features

| Feature                    | Description                                                       |
| -------------------------- | ----------------------------------------------------------------- |
| **Multiple AI Providers**  | OpenAI (GPT-4o), Anthropic (Claude Sonnet 4), or GitHub Copilot   |
| **Style from Image**       | Upload a photo of a house to match its architectural style        |
| **Style from Text**        | Describe the style in words (e.g. "modern minimalist, flat roof") |
| **Blueprint from Image**   | Upload a floor plan / blueprint image                             |
| **Blueprint from Text**    | Describe the layout (e.g. "3 bed, 2 bath, open kitchen")          |
| **Procedural Geometry**    | Exterior walls, interior partitions, floor slabs, roof            |
| **Window & Door Openings** | Boolean-cut openings with frame geometry                          |
| **Multiple Roof Types**    | Flat, gable, and hip roofs (auto or manual selection)             |
| **PBR Materials**          | Auto-generated Principled BSDF materials with correct colors      |
| **Non-blocking**           | AI call runs in a background thread; Blender stays responsive     |

---

## Supported AI Providers

| Provider           | Models                                                              | Image Support  |
| ------------------ | ------------------------------------------------------------------- | -------------- |
| **OpenAI**         | GPT-4o, GPT-4o Mini, GPT-4 Turbo                                    | ✅ Full vision |
| **Anthropic**      | Claude Sonnet 4, Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus | ✅ Full vision |
| **GitHub Copilot** | GPT-4o, Claude 3.5 Sonnet, o3-mini                                  | ✅ Via GPT-4o  |

---

## Requirements

- **Blender 3.6** or newer
- An API key for at least one provider:
  - **OpenAI:** [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  - **Anthropic:** [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
  - **GitHub Copilot:** A GitHub PAT with `copilot` scope

No external Python packages are required — the plugin uses only Blender's
bundled Python and the standard library.

---

## Installation

### Option A — Install from ZIP

1. Compress the `AIHouseGenerator` folder into a `.zip` file.
2. In Blender go to **Edit → Preferences → Add-ons → Install…**
3. Select the `.zip` file and click **Install Add-on**.
4. Enable **AI House Generator** in the addon list.

### Option B — Symlink / Copy

1. Copy (or symlink) the `AIHouseGenerator` folder into your Blender addons
   directory:
   - **macOS:** `~/Library/Application Support/Blender/<version>/scripts/addons/`
   - **Windows:** `%APPDATA%\Blender Foundation\Blender\<version>\scripts\addons\`
   - **Linux:** `~/.config/blender/<version>/scripts/addons/`
2. In Blender go to **Edit → Preferences → Add-ons**, search for
   **AI House Generator**, and enable it.

---

## Setup

1. After enabling the addon, open the **Preferences** panel for it
   (or click the **API Key Settings** button in the sidebar panel).
2. Choose your **AI Provider** (OpenAI, Anthropic, or GitHub Copilot).
3. Paste the **API Key / Token** for that provider.
4. Choose a **Model** (GPT-4o and Claude Sonnet 4 are recommended).
5. Close preferences.

---

## Usage

1. Open the **3D Viewport** sidebar (`N` key) and switch to the **AI House**
   tab.
2. **Style Reference**
   - Choose **Text Description** and describe the style, _or_
   - Choose **Reference Image** and select an image file.
3. **Blueprint / Floor Plan**
   - Choose **Text Description** and describe the layout, _or_
   - Choose **Blueprint Image** and select a floor plan image.
4. Adjust **Dimensions** (lot size, floors, wall thickness) as needed.
5. Toggle **Features** (interior walls, roof, windows, doors, materials).
6. Click **Generate House**.
7. Wait for the AI to respond (status shown in the panel). The 3D house
   will appear in the viewport inside an `AI_House` collection.

### Tips

- Use **Clear House** to remove the generated geometry before regenerating.
- **Regen Materials** re-applies default materials if you changed something.
- Lower the **Temperature** in preferences for more predictable results;
  raise it for more creative / varied houses.
- If you provide both a style _image_ and a blueprint _image_, the AI will
  attempt to combine the visual style with the floor plan layout.

---

## Testing

### Automated Test Suite (no API key needed)

Run the built-in test that exercises addon registration and the geometry
pipeline with mock data:

```bash
# macOS
/Applications/Blender.app/Contents/MacOS/Blender --background --python test_addon.py

# Linux
blender --background --python test_addon.py

# Windows
"C:\Program Files\Blender Foundation\Blender <version>\blender.exe" --background --python test_addon.py
```

This runs 5 tests:

1. Addon can be enabled
2. All properties register
3. All operators register
4. Geometry generator works with mock data (creates walls, slabs, roof, etc.)
5. AI engine has all provider functions

### Manual Testing in Blender

1. Install and enable the addon (see Installation above).
2. Open the **AI House** sidebar tab.
3. **Quick test (no API key):** Open Blender's Python Console and run:
   ```python
   import bpy
   from AIHouseGenerator import house_generator, materials
   mock = {
       "style": {"name": "Modern", "wall_color": [0.9,0.9,0.85], "trim_color": [1,1,1],
                  "roof_color": [0.3,0.3,0.3], "roof_type": "flat", "roof_pitch": 5,
                  "window_style": "large", "door_style": "modern",
                  "material_exterior": "concrete", "material_roof": "flat_membrane"},
       "floors": [{"level": 0, "height": 2.8, "rooms": [
           {"name": "Living", "x": 0.25, "y": 0.25, "width": 5, "depth": 5,
            "windows": [{"wall": "south", "position": 0.5, "width": 2, "height": 1.5, "sill_height": 0.8}],
            "doors": [{"wall": "south", "position": 0.15, "width": 0.9, "height": 2.1, "is_exterior": True}]}
       ]}],
       "exterior": {"width": 10, "depth": 6, "porch": False, "garage": False, "balcony": False, "chimney": False}
   }
   props = bpy.context.scene.ai_house_gen
   created = house_generator.generate_house(mock, props)
   materials.apply_materials(created, mock)
   ```
4. **Full test (API key required):** Set an API key in preferences, type a
   style and blueprint description, and click **Generate House**.

---

## Project Structure

```
AIHouseGenerator/
├── __init__.py          # Addon entry point & registration
├── properties.py        # Scene-level properties (settings)
├── preferences.py       # Addon preferences (provider, API keys, models)
├── panels.py            # Sidebar UI panels
├── operators.py         # Blender operators (generate, clear, etc.)
├── ai_engine.py         # Multi-provider AI communication (OpenAI / Claude / Copilot)
├── house_generator.py   # Procedural 3D geometry builder
├── materials.py         # PBR material creation & assignment
├── test_addon.py        # Automated test suite
├── pyrightconfig.json   # Suppresses false Pylance errors for bpy imports
└── README.md            # This file
```

---

## How It Works

1. The user's style and blueprint inputs (text and/or images) are sent to
   the selected AI provider as a multimodal chat message.
2. The AI returns a structured **JSON** object describing:
   - Architectural style (colors, materials, roof type, window style)
   - Floor-by-floor room layouts (positions, sizes)
   - Window and door placements per room
3. The **house generator** builds the geometry procedurally:
   - Exterior walls → boolean-cut window/door openings
   - Interior partition walls
   - Floor slabs and ceiling
   - Roof (flat / gable / hip)
   - Window frames and door frames
4. The **materials** module creates Principled BSDF materials using the
   AI-specified colors and surface properties.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
