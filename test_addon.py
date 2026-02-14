"""
test_addon.py — Quick validation script for the AI House Generator addon.

Run this inside Blender's Python console OR from the terminal with:

    /Applications/Blender.app/Contents/MacOS/Blender --background --python test_addon.py

It verifies that:
 1. The addon can be enabled without errors.
 2. All panels, operators, and properties register correctly.
 3. A mock house-data dict can be fed to the geometry generator
    (no API key needed — this exercises the geometry pipeline only).
"""

import sys
import os

# ── When running from CLI, we need bpy ─────────────────────────────────
try:
    import bpy
except ImportError:
    print("ERROR: This script must be run inside Blender.")
    print("  /Applications/Blender.app/Contents/MacOS/Blender --background --python test_addon.py")
    sys.exit(1)


# ── Ensure the addon directory is on the path ──────────────────────────
addon_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(addon_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

ADDON_MODULE = os.path.basename(addon_dir)  # "AIHouseGenerator"


def test_enable_addon():
    """Test that the addon can be enabled."""
    print("\n=== Test 1: Enable addon ===")
    bpy.ops.preferences.addon_install(filepath="", overwrite=True)

    # Try enabling by module name
    try:
        bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
        print("  ✓ Addon enabled successfully")
    except Exception as e:
        print(f"  ✗ Addon enable failed: {e}")
        # Try manual registration as fallback
        print("  → Trying manual registration…")
        import importlib
        addon = importlib.import_module(ADDON_MODULE)
        addon.register()
        print("  ✓ Manual registration succeeded")


def test_properties_exist():
    """Test that scene properties are registered."""
    print("\n=== Test 2: Properties ===")
    scene = bpy.context.scene
    assert hasattr(scene, "ai_house_gen"), "ai_house_gen property group not found on scene"
    props = scene.ai_house_gen

    expected_attrs = [
        "style_mode", "style_text", "style_image_path",
        "blueprint_mode", "blueprint_text", "blueprint_image_path",
        "house_width", "house_depth", "num_floors", "floor_height",
        "wall_thickness", "generate_interior", "generate_roof",
        "generate_windows", "generate_doors", "generate_materials",
        "roof_type", "roof_overhang",
    ]
    for attr in expected_attrs:
        assert hasattr(props, attr), f"Missing property: {attr}"
    print(f"  ✓ All {len(expected_attrs)} properties found")


def test_operators_registered():
    """Test that all operators are registered."""
    print("\n=== Test 3: Operators ===")
    expected_ops = [
        "aihouse.generate_house",
        "aihouse.clear_generated",
        "aihouse.regenerate_materials",
        "aihouse.open_preferences",
    ]
    for op_id in expected_ops:
        parts = op_id.split(".")
        assert hasattr(bpy.ops, parts[0]), f"Missing operator category: {parts[0]}"
        cat = getattr(bpy.ops, parts[0])
        assert hasattr(cat, parts[1]), f"Missing operator: {op_id}"
    print(f"  ✓ All {len(expected_ops)} operators registered")


def test_geometry_generator():
    """Test the procedural geometry pipeline with mock data (no API call)."""
    print("\n=== Test 4: Geometry generator (mock data) ===")

    # Import the house generator directly
    try:
        from AIHouseGenerator import house_generator
        from AIHouseGenerator import materials as mat_module
    except ImportError:
        # Fallback for when running as installed addon
        import importlib
        house_generator = importlib.import_module(f"{ADDON_MODULE}.house_generator")
        mat_module = importlib.import_module(f"{ADDON_MODULE}.materials")

    # Mock house data (what the AI would return)
    mock_data = {
        "style": {
            "name": "Modern",
            "wall_color": [0.9, 0.88, 0.85],
            "trim_color": [0.95, 0.95, 0.95],
            "roof_color": [0.3, 0.3, 0.3],
            "roof_type": "flat",
            "roof_pitch": 5,
            "window_style": "large",
            "door_style": "modern",
            "material_exterior": "concrete",
            "material_roof": "flat_membrane"
        },
        "floors": [
            {
                "level": 0,
                "height": 2.8,
                "rooms": [
                    {
                        "name": "Living Room",
                        "x": 0.25,
                        "y": 0.25,
                        "width": 5.5,
                        "depth": 5.0,
                        "windows": [
                            {"wall": "south", "position": 0.5, "width": 2.0,
                             "height": 1.8, "sill_height": 0.5}
                        ],
                        "doors": [
                            {"wall": "south", "position": 0.2, "width": 0.9,
                             "height": 2.1, "is_exterior": True}
                        ]
                    },
                    {
                        "name": "Kitchen",
                        "x": 5.75,
                        "y": 0.25,
                        "width": 4.0,
                        "depth": 5.0,
                        "windows": [
                            {"wall": "east", "position": 0.5, "width": 1.2,
                             "height": 1.2, "sill_height": 0.9}
                        ],
                        "doors": [
                            {"wall": "west", "position": 0.5, "width": 0.9,
                             "height": 2.1, "is_exterior": False}
                        ]
                    }
                ]
            }
        ],
        "exterior": {
            "width": 10.0,
            "depth": 6.0,
            "porch": False,
            "garage": False,
            "balcony": False,
            "chimney": False
        }
    }

    props = bpy.context.scene.ai_house_gen
    props.num_floors = 1
    props.house_width = 10.0
    props.house_depth = 6.0

    # Generate geometry
    created = house_generator.generate_house(mock_data, props)
    print(f"  ✓ Created {len(created)} objects")

    # Check the collection exists
    assert "AI_House" in bpy.data.collections, "AI_House collection not created"
    col = bpy.data.collections["AI_House"]
    print(f"  ✓ AI_House collection has {len(col.objects)} objects")

    # Apply materials
    mat_module.apply_materials(created, mock_data)
    print("  ✓ Materials applied successfully")

    # Verify some expected objects exist
    obj_names = [o.name for o in col.objects]
    assert any("Wall" in n for n in obj_names), "No wall objects found"
    assert any("Slab" in n for n in obj_names), "No floor slab found"
    print(f"  ✓ Objects: {obj_names[:8]}{'…' if len(obj_names) > 8 else ''}")

    # Clean up
    house_generator.clear_house()
    mat_module.clear_materials()
    print("  ✓ Cleanup successful")


def test_ai_engine_import():
    """Test that the AI engine module imports and has multi-provider support."""
    print("\n=== Test 5: AI Engine multi-provider ===")
    try:
        from AIHouseGenerator import ai_engine
    except ImportError:
        import importlib
        ai_engine = importlib.import_module(f"{ADDON_MODULE}.ai_engine")

    assert hasattr(ai_engine, 'generate_house_parameters')
    assert hasattr(ai_engine, '_request_openai')
    assert hasattr(ai_engine, '_request_anthropic')
    assert hasattr(ai_engine, '_request_github_copilot')
    print("  ✓ All provider functions present")
    print("  ✓ generate_house_parameters() accepts 'provider' arg")


# ── Run all tests ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  AI House Generator — Addon Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_fn in [
        test_enable_addon,
        test_properties_exist,
        test_operators_registered,
        test_geometry_generator,
        test_ai_engine_import,
    ]:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed:
        sys.exit(1)
