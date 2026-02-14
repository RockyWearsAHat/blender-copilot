"""Extract mesh data from standard 3D formats (GLB, GLTF, OBJ, STL, PLY).

Unlike blend_extractor.py (which requires Blender headless), this uses
trimesh to load and process 3D files directly in Python. This is the
primary extractor for Objaverse data (GLB format).

Outputs the same JSON structure as blend_extractor.py so the downstream
pipeline (tokenizer, dataset_builder) works identically.

Usage:
    python -m processing.mesh_extractor \
        --input data/raw/objaverse/models \
        --output data/processed/objaverse \
        --metadata data/raw/objaverse/metadata
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_mesh(filepath: Path) -> "trimesh.Trimesh | trimesh.Scene | None":
    """Load a 3D file and return a trimesh object."""
    import trimesh
    try:
        result = trimesh.load(str(filepath), force="mesh", process=True)
        return result
    except Exception:
        # Some files are scenes, not single meshes
        try:
            result = trimesh.load(str(filepath), process=True)
            return result
        except Exception as e:
            logger.warning(f"Failed to load {filepath.name}: {e}")
            return None


def scene_to_meshes(scene) -> list:
    """Convert a trimesh Scene to a list of individual meshes with names."""
    import trimesh
    meshes = []
    if isinstance(scene, trimesh.Trimesh):
        meshes.append(("Object", scene))
    elif isinstance(scene, trimesh.Scene):
        for name, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
                meshes.append((name, geom))
    return meshes


def extract_mesh_data(name: str, mesh, config: dict) -> dict | None:
    """Extract structured mesh data from a trimesh object.

    Returns the same format as blend_extractor.py for pipeline compatibility.
    """
    import trimesh

    mesh_config = config.get("processing", {}).get("mesh_extraction", {})
    min_verts = mesh_config.get("min_vertices", 8)
    max_verts = mesh_config.get("max_vertices", 100000)
    precision = mesh_config.get("coordinate_precision", 4)
    normalize = mesh_config.get("normalize", True)

    if not isinstance(mesh, trimesh.Trimesh):
        return None

    if len(mesh.vertices) < min_verts or len(mesh.vertices) > max_verts:
        return None

    # Ensure triangulated
    if not mesh.is_watertight:
        pass  # Still usable, just not watertight

    # Get vertices and faces
    vertices = mesh.vertices.tolist()
    faces = mesh.faces.tolist()

    if len(faces) < 4:  # Need at least a tetrahedron
        return None

    # Normalize to unit sphere centered at origin
    if normalize:
        verts_np = np.array(vertices)
        centroid = verts_np.mean(axis=0)
        verts_np -= centroid
        max_dist = np.linalg.norm(verts_np, axis=1).max()
        if max_dist > 1e-8:
            verts_np /= max_dist
        vertices = np.round(verts_np, precision).tolist()

    # Compute normals
    normals = []
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(vertices):
        normals = np.round(mesh.vertex_normals, 3).tolist()

    # Bounding box (post-normalization)
    verts_np = np.array(vertices)
    bbox_min = verts_np.min(axis=0).tolist()
    bbox_max = verts_np.max(axis=0).tolist()
    dimensions = (np.array(bbox_max) - np.array(bbox_min)).tolist()

    # Extract basic material info if available
    materials = []
    if hasattr(mesh, "visual") and mesh.visual is not None:
        vis = mesh.visual
        if hasattr(vis, "material") and vis.material is not None:
            mat = vis.material
            mat_data = {"name": getattr(mat, "name", "Material")}
            if hasattr(mat, "main_color") and mat.main_color is not None:
                color = mat.main_color
                if len(color) >= 3:
                    mat_data["base_color"] = [
                        round(color[0] / 255.0, 3),
                        round(color[1] / 255.0, 3),
                        round(color[2] / 255.0, 3),
                        round(color[3] / 255.0, 3) if len(color) > 3 else 1.0,
                    ]
            if hasattr(mat, "glossiness"):
                mat_data["roughness"] = round(1.0 - (mat.glossiness or 0), 3)
            materials.append(mat_data)

    return {
        "name": name,
        "mesh": {
            "vertices": vertices,
            "faces": faces,
            "normals": normals,
            "num_vertices": len(vertices),
            "num_faces": len(faces),
        },
        "dimensions": dimensions,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "materials": materials,
        "modifiers": [],  # Standard formats don't have modifiers
    }


def process_file(filepath: Path, output_dir: Path, metadata_dir: Path | None,
                  config: dict) -> int:
    """Process a single 3D file and write extracted data to JSON.

    Returns number of objects extracted.
    """
    scene = load_mesh(filepath)
    if scene is None:
        return 0

    meshes = scene_to_meshes(scene)
    if not meshes:
        return 0

    # Load metadata if available
    meta = {}
    if metadata_dir:
        uid = filepath.stem
        # Try both naming conventions
        for suffix in [".meta.json", ".json"]:
            meta_path = metadata_dir / f"{uid}{suffix}"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                break

    # Build scene-level data matching blend_extractor output format
    scene_data = {
        "source_file": filepath.name,
        "source_format": filepath.suffix.lstrip("."),
        "metadata": meta,
        "objects": [],
    }

    count = 0
    for name, mesh in meshes:
        obj_data = extract_mesh_data(name, mesh, config)
        if obj_data is not None:
            scene_data["objects"].append(obj_data)
            count += 1

    if count == 0:
        return 0

    # Write output JSON (same format as blend_extractor)
    out_path = output_dir / f"{filepath.stem}.json"
    with open(out_path, "w") as f:
        json.dump(scene_data, f, indent=2)

    return count


def process_directory(input_dir: Path, output_dir: Path,
                       metadata_dir: Path | None, config: dict):
    """Process all 3D files in a directory."""
    from .utils import ensure_dir
    ensure_dir(output_dir)

    # Supported extensions
    extensions = {".glb", ".gltf", ".obj", ".stl", ".ply", ".off", ".3ds"}

    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*{ext}"))
        files.extend(input_dir.glob(f"*{ext.upper()}"))

    # Also follow symlinks
    if not files:
        files = [f for f in input_dir.iterdir()
                 if f.suffix.lower() in extensions]

    logger.info(f"Found {len(files)} 3D files to process")

    # Check for progress
    progress_file = output_dir / ".progress.txt"
    processed = set()
    if progress_file.exists():
        processed = set(progress_file.read_text().strip().split("\n"))
    logger.info(f"Already processed: {len(processed)} files")

    total_objects = 0
    errors = 0

    for i, filepath in enumerate(files):
        if filepath.name in processed:
            continue

        try:
            count = process_file(filepath, output_dir, metadata_dir, config)
            total_objects += count
            processed.add(filepath.name)

            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(files)} files, "
                             f"{total_objects} objects extracted")
                # Save progress
                with open(progress_file, "w") as f:
                    f.write("\n".join(sorted(processed)))

        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {e}")
            errors += 1
            continue

    # Final progress save
    with open(progress_file, "w") as f:
        f.write("\n".join(sorted(processed)))

    logger.info(f"Processing complete: {total_objects} objects from "
                 f"{len(files)} files ({errors} errors)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract mesh data from GLB/GLTF/OBJ/STL files")
    parser.add_argument("--input", required=True,
                        help="Input directory with 3D files")
    parser.add_argument("--output", required=True,
                        help="Output directory for extracted JSON")
    parser.add_argument("--metadata", default=None,
                        help="Directory with metadata JSON files")
    parser.add_argument("--config", default="config.yaml",
                        help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [mesh_extractor] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import yaml
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    metadata_dir = Path(args.metadata) if args.metadata else None

    process_directory(input_dir, output_dir, metadata_dir, config)


if __name__ == "__main__":
    main()
