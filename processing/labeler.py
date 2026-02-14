"""Auto-labeler for extracted .blend data.

Generates text descriptions for 3D objects and scenes by analyzing:
- Object names and hierarchy
- Dimensions and proportions
- Materials (colors, textures, properties)
- Modifier stacks
- Scene composition (object count, types, arrangements)
- Metadata from scraping (tags, descriptions, categories)

The text labels serve as the conditioning input for training.
Better labels → better text-to-3D generation.

Usage:
    python -m processing.labeler \
        --input data/filtered/ \
        --output data/labeled/ \
        --config config.yaml
"""

import argparse
import json
import logging
import math
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class AutoLabeler:
    """Generate text descriptions for 3D objects and scenes."""

    # Common Blender default names to clean up
    DEFAULT_NAMES = {
        "cube", "sphere", "cylinder", "cone", "torus", "plane",
        "circle", "monkey", "suzanne", "icosphere", "uvsphere",
        "grid", "empty", "light", "camera", "lamp", "sun",
        "spot", "area", "point", "armature", "bone",
    }

    # Material color name mapping
    COLOR_NAMES = {
        (1.0, 0.0, 0.0): "red",
        (0.0, 1.0, 0.0): "green",
        (0.0, 0.0, 1.0): "blue",
        (1.0, 1.0, 0.0): "yellow",
        (1.0, 0.5, 0.0): "orange",
        (0.5, 0.0, 0.5): "purple",
        (0.0, 1.0, 1.0): "cyan",
        (1.0, 0.0, 1.0): "magenta",
        (1.0, 1.0, 1.0): "white",
        (0.0, 0.0, 0.0): "black",
        (0.5, 0.5, 0.5): "gray",
        (0.6, 0.3, 0.1): "brown",
        (0.9, 0.8, 0.6): "beige",
        (0.7, 0.7, 0.7): "silver",
        (0.8, 0.7, 0.2): "gold",
    }

    def __init__(self, config: dict):
        self.config = config

    def closest_color_name(self, r: float, g: float, b: float) -> str:
        """Find the closest named color."""
        best_name = "colored"
        best_dist = float("inf")

        for (cr, cg, cb), name in self.COLOR_NAMES.items():
            dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
            if dist < best_dist:
                best_dist = dist
                best_name = name

        return best_name if best_dist < 0.15 else "colored"

    def clean_object_name(self, name: str) -> str:
        """Clean up a Blender object name into readable text.

        Examples:
            "Car_Body.001" → "car body"
            "SM_Chair_Wood" → "chair wood"
            "low_poly_house_roof" → "low poly house roof"
        """
        # Remove numeric suffixes like .001, .002
        name = re.sub(r'\.\d{3,}$', '', name)

        # Remove common prefixes (SM_, GEO_, OBJ_, etc.)
        name = re.sub(r'^(SM_|GEO_|OBJ_|MESH_|MAT_|MTL_)', '', name, flags=re.IGNORECASE)

        # Replace separators with spaces
        name = name.replace('_', ' ').replace('-', ' ')

        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip().lower()

        return name

    def describe_dimensions(self, bounds: dict) -> str:
        """Describe object dimensions in human terms."""
        w = bounds.get("width_x", 0)
        d = bounds.get("depth_y", 0)
        h = bounds.get("height_z", 0)

        if max(w, d, h) < 0.001:
            return ""

        # Determine scale category
        max_dim = max(w, d, h)
        if max_dim > 10:
            scale = "large"
        elif max_dim > 2:
            scale = "medium-sized"
        elif max_dim > 0.3:
            scale = "small"
        else:
            scale = "tiny"

        # Describe proportions
        dims = sorted([(w, "wide"), (d, "deep"), (h, "tall")])
        dominant = dims[-1][1]

        return f"{scale}, {dominant}"

    def describe_material(self, material: dict) -> str:
        """Describe a material in text."""
        parts = []

        mat_name = material.get("name", "")
        if mat_name:
            cleaned = self.clean_object_name(mat_name)
            if cleaned and cleaned not in self.DEFAULT_NAMES:
                parts.append(cleaned)

        # Color from base color
        nodes = material.get("nodes", [])
        for node in nodes:
            if node.get("type") == "ShaderNodeBsdfPrincipled":
                inputs = node.get("inputs", [])
                for inp in inputs:
                    if inp.get("name") == "Base Color":
                        val = inp.get("default_value", [])
                        if len(val) >= 3:
                            color = self.closest_color_name(val[0], val[1], val[2])
                            if color != "colored":
                                parts.append(color)
                    elif inp.get("name") == "Metallic":
                        val = inp.get("default_value", 0)
                        if isinstance(val, (int, float)) and val > 0.5:
                            parts.append("metallic")
                    elif inp.get("name") == "Roughness":
                        val = inp.get("default_value", 0.5)
                        if isinstance(val, (int, float)):
                            if val < 0.2:
                                parts.append("glossy")
                            elif val > 0.8:
                                parts.append("matte")

        # Check for texture nodes
        has_texture = any(
            n.get("type", "").startswith("ShaderNodeTex") and
            n.get("type") != "ShaderNodeTexCoord"
            for n in nodes
        )
        if has_texture:
            parts.append("textured")

        return " ".join(parts)

    def describe_modifiers(self, modifiers: list) -> str:
        """Describe a modifier stack in text."""
        if not modifiers:
            return ""

        mod_descriptions = []
        for mod in modifiers:
            mod_type = mod.get("type", "")
            if mod_type == "SUBSURF":
                levels = mod.get("levels", 1)
                mod_descriptions.append(f"subdivided (level {levels})")
            elif mod_type == "MIRROR":
                axes = []
                if mod.get("use_axis_x", True):
                    axes.append("X")
                if mod.get("use_axis_y", False):
                    axes.append("Y")
                if mod.get("use_axis_z", False):
                    axes.append("Z")
                mod_descriptions.append(f"mirrored on {','.join(axes)}")
            elif mod_type == "BEVEL":
                mod_descriptions.append("beveled edges")
            elif mod_type == "SOLIDIFY":
                mod_descriptions.append("solidified")
            elif mod_type == "ARRAY":
                count = mod.get("count", 2)
                mod_descriptions.append(f"arrayed ({count}x)")
            elif mod_type == "BOOLEAN":
                mod_descriptions.append("boolean operation")
            elif mod_type == "SMOOTH":
                mod_descriptions.append("smoothed")
            elif mod_type == "DECIMATE":
                mod_descriptions.append("decimated")

        return ", ".join(mod_descriptions)

    def label_object(self, obj_data: dict, metadata: dict = None) -> str:
        """Generate a text description for a single object.

        Combines:
            1. Object name (cleaned)
            2. Scraping metadata (if available)
            3. Dimensions
            4. Material description
            5. Modifier description
            6. Mesh complexity hint

        Returns:
            Text description string.
        """
        parts = []

        # Object name
        name = obj_data.get("name", "object")
        cleaned_name = self.clean_object_name(name)
        if cleaned_name and cleaned_name.lower() not in self.DEFAULT_NAMES:
            parts.append(cleaned_name)

        # Metadata (from scraping)
        if metadata:
            tags = metadata.get("tags", [])
            if tags:
                # Add relevant tags
                tag_str = " ".join(t.lower() for t in tags[:5])
                parts.append(tag_str)

            desc = metadata.get("description", "")
            if desc and len(desc) > 5:
                # Take first sentence
                first_sentence = desc.split(".")[0].strip()[:100]
                if first_sentence:
                    parts.append(first_sentence.lower())

            category = metadata.get("category", "")
            if category:
                parts.append(category.lower())

        # Dimensions
        mesh = obj_data.get("mesh", {})
        if mesh:
            verts = mesh.get("vertices", [])
            if verts:
                import numpy as np
                v_arr = np.array(verts)
                bounds = {
                    "width_x": float(v_arr[:, 0].max() - v_arr[:, 0].min()),
                    "depth_y": float(v_arr[:, 1].max() - v_arr[:, 1].min()),
                    "height_z": float(v_arr[:, 2].max() - v_arr[:, 2].min()),
                }
                dim_desc = self.describe_dimensions(bounds)
                if dim_desc:
                    parts.append(dim_desc)

            # Complexity hint
            num_verts = len(verts)
            num_faces = len(mesh.get("faces", []))
            if num_faces < 50:
                parts.append("low poly")
            elif num_faces < 500:
                parts.append("medium poly")
            elif num_faces > 5000:
                parts.append("high poly")

        # Materials
        materials = obj_data.get("materials", [])
        if materials:
            for mat in materials[:2]:  # Max 2 material descriptions
                mat_desc = self.describe_material(mat)
                if mat_desc:
                    parts.append(mat_desc)

        # Modifiers
        modifiers = obj_data.get("modifiers", [])
        if modifiers:
            mod_desc = self.describe_modifiers(modifiers)
            if mod_desc:
                parts.append(f"with {mod_desc}")

        # Combine and deduplicate words
        label = " ".join(parts)
        # Deduplicate consecutive words
        words = label.split()
        deduped = [words[0]] if words else []
        for w in words[1:]:
            if w != deduped[-1]:
                deduped.append(w)
        label = " ".join(deduped)

        return label.strip() or "3d object"

    def label_scene(self, data: dict) -> str:
        """Generate a scene-level description.

        Useful for multi-object generation.
        """
        objects = data.get("objects", [])
        metadata = data.get("metadata", {})

        if not objects:
            return "empty scene"

        # Use metadata title/description if available
        title = metadata.get("title", "")
        if title:
            return self.clean_object_name(title)

        # Otherwise, describe the scene composition
        object_names = [
            self.clean_object_name(o.get("name", "object"))
            for o in objects
        ]
        # Filter out default names
        meaningful = [n for n in object_names if n not in self.DEFAULT_NAMES]

        if meaningful:
            if len(meaningful) <= 3:
                return "scene with " + ", ".join(meaningful)
            else:
                return f"scene with {len(meaningful)} objects including " + \
                       ", ".join(meaningful[:3])
        else:
            return f"scene with {len(objects)} objects"

    def label_file(self, data: dict) -> dict:
        """Add text labels to all objects in an extracted file.

        Modifies data in-place and returns it.
        """
        metadata = data.get("metadata", {})

        # Label individual objects
        for obj in data.get("objects", []):
            obj["text_label"] = self.label_object(obj, metadata)

        # Add scene-level label
        data["scene_label"] = self.label_scene(data)

        return data

    def label_directory(self, input_dir: str, output_dir: str):
        """Label all extracted JSON files in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        json_files = sorted(input_path.glob("*.json"))
        logger.info(f"Labeling {len(json_files)} files from {input_dir}")

        total_objects = 0
        labeled_objects = 0

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading {json_file.name}: {e}")
                continue

            labeled_data = self.label_file(data)

            for obj in labeled_data.get("objects", []):
                total_objects += 1
                label = obj.get("text_label", "")
                if label and label != "3d object":
                    labeled_objects += 1
                    logger.debug(f"  '{obj.get('name', '?')}' → \"{label}\"")

            out_file = output_path / json_file.name
            with open(out_file, "w") as f:
                json.dump(labeled_data, f)

        pct = labeled_objects / max(1, total_objects) * 100
        logger.info(f"\nLabeled {labeled_objects}/{total_objects} objects "
                     f"({pct:.1f}%) with meaningful descriptions")
        logger.info(f"Output written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Auto-label extracted .blend data")
    parser.add_argument("--input", required=True, help="Input directory with extracted/filtered JSON")
    parser.add_argument("--output", required=True, help="Output directory for labeled JSON")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    labeler = AutoLabeler(config)
    labeler.label_directory(args.input, args.output)


if __name__ == "__main__":
    main()
