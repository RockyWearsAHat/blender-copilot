"""Quality filter for scraped .blend files.

Filters out unusable data before training:
- Corrupt / empty files
- Meshes too simple (< min_vertices) or too complex (> max_vertices)
- Missing geometry (camera-only scenes, empties only)
- Duplicate meshes (dedup by geometry hash)
- License violations (keeps only permissive licenses)

Usage:
    python -m processing.quality_filter \
        --input data/extracted/ \
        --output data/filtered/ \
        --config config.yaml
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class QualityFilter:
    """Filter extracted .blend data by quality criteria."""

    def __init__(self, config: dict):
        proc = config.get("processing", {})
        mesh_cfg = proc.get("mesh_extraction", {})

        self.min_vertices = mesh_cfg.get("min_vertices", 8)
        self.max_vertices = mesh_cfg.get("max_vertices", 50000)
        self.min_faces = mesh_cfg.get("min_faces", 4)
        self.max_faces = mesh_cfg.get("max_faces", 20000)
        self.require_manifold = proc.get("quality_filters", {}).get(
            "require_manifold", False
        )
        self.min_objects = proc.get("quality_filters", {}).get(
            "min_objects_per_scene", 1
        )
        self.allowed_licenses = set(proc.get("quality_filters", {}).get(
            "allowed_licenses",
            ["CC-0", "CC-BY", "CC-BY-SA", "MIT", "Apache-2.0", "GPL", "public_domain"]
        ))

        # Track geometry hashes for dedup
        self.seen_hashes = set()

        # Statistics
        self.stats = {
            "total_files": 0,
            "passed": 0,
            "rejected_empty": 0,
            "rejected_too_simple": 0,
            "rejected_too_complex": 0,
            "rejected_duplicate": 0,
            "rejected_license": 0,
            "rejected_corrupt": 0,
            "total_objects_in": 0,
            "total_objects_out": 0,
        }

    def geometry_hash(self, mesh_data: dict) -> str:
        """Compute a hash of mesh geometry for deduplication.

        Uses quantized vertex positions + face topology.
        """
        verts = mesh_data.get("vertices", [])
        faces = mesh_data.get("faces", [])

        if not verts or not faces:
            return ""

        # Quantize to 3 decimal places for fuzzy matching
        v_arr = np.array(verts)
        v_quantized = np.round(v_arr, 3)

        # Sort vertices for order-invariant hashing
        sorted_indices = np.lexsort(v_quantized.T)

        hasher = hashlib.sha256()
        hasher.update(v_quantized[sorted_indices].tobytes())
        hasher.update(str(sorted(map(tuple, faces))).encode())

        return hasher.hexdigest()[:16]

    def check_mesh(self, mesh_data: dict) -> tuple[bool, str]:
        """Check if a single mesh passes quality filters.

        Returns:
            (passed, reason)
        """
        verts = mesh_data.get("vertices", [])
        faces = mesh_data.get("faces", [])

        num_verts = len(verts)
        num_faces = len(faces)

        if num_verts == 0 or num_faces == 0:
            return False, "empty_mesh"

        if num_verts < self.min_vertices:
            return False, f"too_few_vertices ({num_verts} < {self.min_vertices})"

        if num_verts > self.max_vertices:
            return False, f"too_many_vertices ({num_verts} > {self.max_vertices})"

        if num_faces < self.min_faces:
            return False, f"too_few_faces ({num_faces} < {self.min_faces})"

        if num_faces > self.max_faces:
            return False, f"too_many_faces ({num_faces} > {self.max_faces})"

        # Check for degenerate geometry
        v_arr = np.array(verts)
        bbox_size = v_arr.max(axis=0) - v_arr.min(axis=0)
        if np.any(bbox_size < 1e-6):
            return False, "degenerate_bbox (zero extent on an axis)"

        # Dedup check
        ghash = self.geometry_hash(mesh_data)
        if ghash and ghash in self.seen_hashes:
            return False, "duplicate_geometry"
        if ghash:
            self.seen_hashes.add(ghash)

        return True, "ok"

    def check_file(self, data: dict) -> tuple[bool, dict, str]:
        """Check an extracted .blend file's data.

        Returns:
            (passed, filtered_data, reason)
        """
        self.stats["total_files"] += 1

        # Check license
        metadata = data.get("metadata", {})
        license_val = metadata.get("license", "unknown")
        if license_val not in self.allowed_licenses and license_val != "unknown":
            self.stats["rejected_license"] += 1
            return False, {}, f"license_rejected ({license_val})"

        # Check objects
        objects = data.get("objects", [])
        if not objects:
            self.stats["rejected_empty"] += 1
            return False, {}, "no_objects"

        self.stats["total_objects_in"] += len(objects)

        # Filter individual objects
        kept_objects = []
        for obj in objects:
            mesh = obj.get("mesh", {})
            if not mesh:
                continue

            passed, reason = self.check_mesh(mesh)
            if passed:
                kept_objects.append(obj)
            else:
                logger.debug(f"  Rejected object '{obj.get('name', '?')}': {reason}")
                if "too_few" in reason or "empty" in reason:
                    self.stats["rejected_too_simple"] += 1
                elif "too_many" in reason:
                    self.stats["rejected_too_complex"] += 1
                elif "duplicate" in reason:
                    self.stats["rejected_duplicate"] += 1

        if len(kept_objects) < self.min_objects:
            self.stats["rejected_empty"] += 1
            return False, {}, f"too_few_objects_after_filter ({len(kept_objects)})"

        self.stats["total_objects_out"] += len(kept_objects)
        self.stats["passed"] += 1

        filtered_data = {
            "metadata": metadata,
            "objects": kept_objects,
            "scene_info": data.get("scene_info", {}),
        }

        return True, filtered_data, "ok"

    def filter_directory(self, input_dir: str, output_dir: str):
        """Filter all extracted JSON files in a directory.

        Args:
            input_dir: Directory with extracted .json files
            output_dir: Directory to write filtered .json files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        json_files = sorted(input_path.glob("*.json"))
        logger.info(f"Filtering {len(json_files)} files from {input_dir}")

        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Corrupt file {json_file.name}: {e}")
                self.stats["rejected_corrupt"] += 1
                continue

            passed, filtered_data, reason = self.check_file(data)

            if passed:
                out_file = output_path / json_file.name
                with open(out_file, "w") as f:
                    json.dump(filtered_data, f)
                logger.debug(f"  ✓ {json_file.name}")
            else:
                logger.debug(f"  ✗ {json_file.name}: {reason}")

        self.print_stats()

    def print_stats(self):
        """Print filtering statistics."""
        s = self.stats
        total = s["total_files"]
        if total == 0:
            logger.info("No files processed.")
            return

        pct = lambda n: f"{n / total * 100:.1f}%" if total > 0 else "0%"

        logger.info(f"\n{'='*50}")
        logger.info(f"Quality Filter Results:")
        logger.info(f"{'='*50}")
        logger.info(f"Total files:           {total}")
        logger.info(f"Passed:                {s['passed']} ({pct(s['passed'])})")
        logger.info(f"Rejected (empty):      {s['rejected_empty']} ({pct(s['rejected_empty'])})")
        logger.info(f"Rejected (too simple): {s['rejected_too_simple']} ({pct(s['rejected_too_simple'])})")
        logger.info(f"Rejected (too complex):{s['rejected_too_complex']} ({pct(s['rejected_too_complex'])})")
        logger.info(f"Rejected (duplicate):  {s['rejected_duplicate']} ({pct(s['rejected_duplicate'])})")
        logger.info(f"Rejected (license):    {s['rejected_license']} ({pct(s['rejected_license'])})")
        logger.info(f"Rejected (corrupt):    {s['rejected_corrupt']} ({pct(s['rejected_corrupt'])})")
        logger.info(f"Objects in:            {s['total_objects_in']}")
        logger.info(f"Objects out:           {s['total_objects_out']}")
        logger.info(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Quality filter for extracted .blend data")
    parser.add_argument("--input", required=True, help="Input directory with extracted JSON")
    parser.add_argument("--output", required=True, help="Output directory for filtered JSON")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    qf = QualityFilter(config)
    qf.filter_directory(args.input, args.output)


if __name__ == "__main__":
    main()
