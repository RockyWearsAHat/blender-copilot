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
            ["CC-0", "CC-BY", "CC-BY-SA", "CC-BY-NC", "CC-BY-NC-SA",
             "MIT", "Apache-2.0", "GPL", "LGPL", "BSD",
             "public_domain", "unknown"]
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
            "rejected_bad_materials": 0,
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

    def check_materials(self, objects: list) -> tuple[bool, str]:
        """Check if scene has adequate material data.

        Rejects files where ALL objects have placeholder/missing materials
        and broken texture references. A few objects without materials
        is fine (e.g. collision boxes), but a whole scene with nothing
        indicates a broken export or template file.
        """
        if not objects:
            return True, "ok"

        total = 0
        placeholder_count = 0
        all_textures_broken = 0

        for obj in objects:
            mat_q = obj.get("material_quality", {})
            if not mat_q:
                continue
            total += 1
            if mat_q.get("is_placeholder", True):
                placeholder_count += 1
            mats = obj.get("materials", [])
            for m in mats:
                if m.get("texture_missing", False):
                    all_textures_broken += 1

        if total == 0:
            return True, "ok"

        # If every single object is placeholder, likely a template
        if placeholder_count == total and total >= 3:
            return False, f"all_objects_placeholder_materials ({total})"

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

        # Check material quality across the whole scene
        mat_ok, mat_reason = self.check_materials(kept_objects)
        if not mat_ok:
            self.stats["rejected_bad_materials"] += 1
            return False, {}, mat_reason

        self.stats["total_objects_out"] += len(kept_objects)
        self.stats["passed"] += 1

        # Score quality of each kept object
        scorer = MeshQualityScorer()
        for obj in kept_objects:
            mesh = obj.get("mesh", {})
            if mesh:
                try:
                    obj["quality_score_info"] = scorer.score(mesh)
                except Exception:
                    obj["quality_score_info"] = {'quality_score': 0.5}

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
        logger.info(f"Rejected (materials):  {s['rejected_bad_materials']} ({pct(s['rejected_bad_materials'])})")
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


# ────────────────────────────────────────────────────────────────────
# Mesh Quality Scorer
# ────────────────────────────────────────────────────────────────────
# Assigns a 0.0-1.0 quality score to each mesh based on geometric
# properties. Higher scores = better training data.
#
# Good geometry:  primarily quads/tris, no self-intersections, clean
#                 edge flow, detail concentrated where it matters,
#                 manifold/watertight.
# Bad geometry:   clipping, n-gons, uniform tessellation, non-manifold
#                 edges, degenerate faces.
#
# This is used for sample weighting in training — no content is
# filtered out, but higher quality meshes are weighted more heavily.
# ────────────────────────────────────────────────────────────────────


class MeshQualityScorer:
    """Score mesh quality on a 0.0-1.0 scale for training sample weighting.

    All content is included — scores reflect GEOMETRIC quality only,
    not subject matter. R-rated, stylized, abstract — all fine.
    We only care about mesh craftsmanship.
    """

    # Weight for each sub-score in the final quality score
    WEIGHTS = {
        'quad_ratio': 0.20,
        'face_regularity': 0.10,
        'manifold': 0.15,
        'no_degenerate': 0.10,
        'detail_distribution': 0.15,
        'edge_flow': 0.15,
        'no_clipping': 0.15,
    }

    def score(self, mesh_data: dict) -> dict:
        """Compute quality score for a mesh.

        Args:
            mesh_data: dict with 'vertices', 'faces', optionally 'edges'

        Returns:
            dict with 'quality_score' (float 0-1), 'sub_scores' (dict),
            and 'quality_tier' ('excellent'/'good'/'fair'/'poor')
        """
        verts = mesh_data.get("vertices", [])
        faces = mesh_data.get("faces", [])

        if not verts or not faces:
            return {
                'quality_score': 0.0,
                'sub_scores': {},
                'quality_tier': 'poor',
            }

        v_arr = np.array(verts, dtype=np.float64)
        face_list = [list(f) for f in faces]

        sub = {}
        sub['quad_ratio'] = self._score_quad_ratio(face_list)
        sub['face_regularity'] = self._score_face_regularity(v_arr, face_list)
        sub['manifold'] = self._score_manifold(face_list)
        sub['no_degenerate'] = self._score_no_degenerate(v_arr, face_list)
        sub['detail_distribution'] = self._score_detail_distribution(
            v_arr, face_list
        )
        sub['edge_flow'] = self._score_edge_flow(face_list)
        sub['no_clipping'] = self._score_no_clipping(v_arr, face_list)

        # Weighted average
        quality = sum(
            sub[k] * self.WEIGHTS[k] for k in self.WEIGHTS
        )
        quality = float(np.clip(quality, 0.0, 1.0))

        # Tier
        if quality >= 0.80:
            tier = 'excellent'
        elif quality >= 0.60:
            tier = 'good'
        elif quality >= 0.35:
            tier = 'fair'
        else:
            tier = 'poor'

        return {
            'quality_score': round(quality, 4),
            'sub_scores': {k: round(v, 4) for k, v in sub.items()},
            'quality_tier': tier,
        }

    # ── Sub-scores ──────────────────────────────────────────────

    def _score_quad_ratio(self, faces: list[list]) -> float:
        """Higher score for meshes made of quads and tris (not n-gons).

        Professional models use quads for SubD, tris for games.
        N-gons (5+ sides) indicate sloppy modeling.
        """
        if not faces:
            return 0.0
        counts = {'tri': 0, 'quad': 0, 'ngon': 0}
        for f in faces:
            n = len(f)
            if n == 3:
                counts['tri'] += 1
            elif n == 4:
                counts['quad'] += 1
            else:
                counts['ngon'] += 1
        total = len(faces)
        # Quads are best, tris are fine, n-gons are bad
        good = counts['quad'] + counts['tri'] * 0.8
        return min(good / total, 1.0)

    def _score_face_regularity(self, verts: np.ndarray,
                                faces: list[list]) -> float:
        """Score how regular/uniform face sizes are.

        Good meshes have consistent face sizes in each region;
        bad meshes have wild size variation (huge face next to tiny face).
        """
        if len(faces) < 4:
            return 0.5

        areas = []
        for f in faces:
            if len(f) < 3:
                continue
            try:
                idxs = [i for i in f if i < len(verts)]
                if len(idxs) < 3:
                    continue
                # Triangle fan area for polygon
                v0 = verts[idxs[0]]
                area = 0.0
                for i in range(1, len(idxs) - 1):
                    e1 = verts[idxs[i]] - v0
                    e2 = verts[idxs[i + 1]] - v0
                    area += np.linalg.norm(np.cross(e1, e2)) * 0.5
                if area > 1e-12:
                    areas.append(area)
            except (IndexError, ValueError):
                continue

        if len(areas) < 4:
            return 0.5

        areas_arr = np.array(areas)
        # Use coefficient of variation (lower = more regular)
        mean_area = areas_arr.mean()
        if mean_area < 1e-12:
            return 0.0
        cv = areas_arr.std() / mean_area
        # CV < 0.5 is very regular, CV > 3 is very irregular
        return float(np.clip(1.0 - cv / 4.0, 0.0, 1.0))

    def _score_manifold(self, faces: list[list]) -> float:
        """Score manifoldness: each edge should be shared by exactly 2 faces.

        Non-manifold edges (shared by 1 or 3+ faces) indicate holes,
        internal faces, or self-penetration.
        """
        if not faces:
            return 0.0

        edge_counts = {}
        for f in faces:
            n = len(f)
            for i in range(n):
                e = tuple(sorted((f[i], f[(i + 1) % n])))
                edge_counts[e] = edge_counts.get(e, 0) + 1

        if not edge_counts:
            return 0.0

        total_edges = len(edge_counts)
        manifold_edges = sum(
            1 for c in edge_counts.values() if c == 2
        )
        # Boundary edges (count=1) are OK for open meshes, minor penalty
        boundary_edges = sum(
            1 for c in edge_counts.values() if c == 1
        )
        non_manifold = sum(
            1 for c in edge_counts.values() if c > 2
        )

        manifold_ratio = manifold_edges / total_edges
        boundary_penalty = boundary_edges / total_edges * 0.3
        non_manifold_penalty = non_manifold / total_edges * 2.0

        return float(np.clip(
            manifold_ratio - boundary_penalty - non_manifold_penalty,
            0.0, 1.0
        ))

    def _score_no_degenerate(self, verts: np.ndarray,
                              faces: list[list]) -> float:
        """Score absence of degenerate faces (zero area, collapsed edges).

        Degenerate faces cause shading artifacts and are a sign of
        sloppy modeling or broken boolean operations.
        """
        if len(faces) < 2:
            return 0.5

        degenerate_count = 0
        total_checked = 0

        for f in faces:
            idxs = [i for i in f if i < len(verts)]
            if len(idxs) < 3:
                degenerate_count += 1
                total_checked += 1
                continue

            total_checked += 1
            try:
                v0 = verts[idxs[0]]
                # Check for collapsed edges (vertices at same position)
                has_collapsed = False
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        dist = np.linalg.norm(
                            verts[idxs[i]] - verts[idxs[j]]
                        )
                        if dist < 1e-8:
                            has_collapsed = True
                            break
                    if has_collapsed:
                        break

                if has_collapsed:
                    degenerate_count += 1
                    continue

                # Check for zero-area face
                e1 = verts[idxs[1]] - v0
                e2 = verts[idxs[2]] - v0
                area = np.linalg.norm(np.cross(e1, e2))
                if area < 1e-10:
                    degenerate_count += 1
            except (IndexError, ValueError):
                degenerate_count += 1

        if total_checked == 0:
            return 0.5

        clean_ratio = 1.0 - (degenerate_count / total_checked)
        return float(np.clip(clean_ratio, 0.0, 1.0))

    def _score_detail_distribution(self, verts: np.ndarray,
                                    faces: list[list]) -> float:
        """Score how well geometry is distributed (concentrated where needed).

        Good models: detail where it matters (edges, curves, focal points).
        Bad models: uniform tessellation everywhere, or all detail in one spot
        with nothing elsewhere.

        Measured by spatial entropy of face centroids — moderate entropy
        is best (not perfectly uniform, not all clumped).
        """
        if len(faces) < 8:
            return 0.5

        # Compute face centroids
        centroids = []
        for f in faces:
            idxs = [i for i in f if i < len(verts)]
            if len(idxs) < 3:
                continue
            centroid = verts[idxs].mean(axis=0)
            centroids.append(centroid)

        if len(centroids) < 8:
            return 0.5

        centroids_arr = np.array(centroids)

        # Normalize to unit cube
        bbox_min = centroids_arr.min(axis=0)
        bbox_max = centroids_arr.max(axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_size = np.where(bbox_size < 1e-6, 1.0, bbox_size)
        normalized = (centroids_arr - bbox_min) / bbox_size

        # Bin into 5x5x5 grid and compute entropy
        n_bins = 5
        bins = (normalized * (n_bins - 0.01)).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)

        grid_indices = bins[:, 0] * n_bins * n_bins + bins[:, 1] * n_bins + bins[:, 2]
        unique, counts = np.unique(grid_indices, return_counts=True)

        # Shannon entropy
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Max entropy for n_bins^3 = 125 bins
        max_entropy = np.log2(min(len(centroids), n_bins ** 3))
        if max_entropy < 1e-6:
            return 0.5

        # Normalized entropy — want 0.4-0.8 range (not perfectly uniform,
        # not totally clumped)
        norm_entropy = entropy / max_entropy

        if 0.3 <= norm_entropy <= 0.85:
            return 1.0
        elif norm_entropy < 0.3:
            return float(norm_entropy / 0.3)
        else:
            return float(1.0 - (norm_entropy - 0.85) / 0.15)

    def _score_edge_flow(self, faces: list[list]) -> float:
        """Score edge flow quality: vertices shared by 3-6 faces is ideal.

        Good edge flow: most vertices have valence 4 (quads) or 5-6.
        Bad edge flow: vertices with valence 1-2 (loose) or 8+ (star poles).
        """
        if not faces:
            return 0.0

        # Count face adjacency per vertex (valence from faces)
        vert_face_count = {}
        for f in faces:
            for v in f:
                vert_face_count[v] = vert_face_count.get(v, 0) + 1

        if not vert_face_count:
            return 0.0

        total_verts = len(vert_face_count)
        good_valence = 0
        for v, count in vert_face_count.items():
            if 3 <= count <= 6:
                good_valence += 1
            elif count == 2:
                good_valence += 0.5  # boundary vertex, partial credit

        return float(np.clip(good_valence / total_verts, 0.0, 1.0))

    def _score_no_clipping(self, verts: np.ndarray,
                            faces: list[list]) -> float:
        """Approximate clipping/self-intersection detection.

        Uses spatial hashing to find face pairs that might intersect.
        Full triangle-triangle intersection is expensive, so we use a
        fast approximation: check if face bounding boxes overlap for
        non-adjacent faces.

        A high ratio of non-adjacent overlapping bboxes suggests
        clipping / self-intersection.
        """
        if len(faces) < 6:
            return 0.8  # Too few faces to meaningfully check

        # Build adjacency (faces sharing a vertex are adjacent)
        vert_to_faces = {}
        for fi, f in enumerate(faces):
            for v in f:
                vert_to_faces.setdefault(v, set()).add(fi)

        # Sample faces to check (full check is O(n^2))
        n_check = min(len(faces), 200)
        check_indices = np.random.choice(
            len(faces), size=n_check, replace=False
        ) if len(faces) > n_check else range(len(faces))

        # Precompute face bboxes
        face_bboxes = []
        for f in faces:
            idxs = [i for i in f if i < len(verts)]
            if len(idxs) < 3:
                face_bboxes.append(None)
                continue
            fv = verts[idxs]
            face_bboxes.append((fv.min(axis=0), fv.max(axis=0)))

        overlapping_non_adj = 0
        total_pairs_checked = 0

        for fi in check_indices:
            if face_bboxes[fi] is None:
                continue
            fi_min, fi_max = face_bboxes[fi]
            fi_adj = set()
            for v in faces[fi]:
                fi_adj.update(vert_to_faces.get(v, set()))

            # Check against a sample of other faces
            others = np.random.choice(
                len(faces), size=min(20, len(faces)), replace=False
            )
            for fj in others:
                if fj == fi or fj in fi_adj:
                    continue
                if face_bboxes[fj] is None:
                    continue

                fj_min, fj_max = face_bboxes[fj]
                total_pairs_checked += 1

                # AABB overlap test
                if (fi_min[0] <= fj_max[0] and fi_max[0] >= fj_min[0] and
                    fi_min[1] <= fj_max[1] and fi_max[1] >= fj_min[1] and
                    fi_min[2] <= fj_max[2] and fi_max[2] >= fj_min[2]):
                    overlapping_non_adj += 1

        if total_pairs_checked == 0:
            return 0.8

        overlap_ratio = overlapping_non_adj / total_pairs_checked
        # Some overlap is normal for concave objects, heavy overlap is bad
        if overlap_ratio < 0.05:
            return 1.0
        elif overlap_ratio < 0.15:
            return 0.8
        elif overlap_ratio < 0.30:
            return 0.5
        else:
            return max(0.0, 1.0 - overlap_ratio)


def score_mesh_file(filepath: str) -> dict:
    """Score a mesh file's quality. Convenience function.

    Args:
        filepath: path to JSON file with 'vertices' and 'faces'

    Returns:
        dict with quality_score, sub_scores, quality_tier
    """
    with open(filepath) as f:
        data = json.load(f)

    scorer = MeshQualityScorer()

    # If file has 'objects', score each and return weighted average
    objects = data.get('objects', [])
    if objects:
        scores = []
        for obj in objects:
            mesh = obj.get('mesh', {})
            if mesh:
                s = scorer.score(mesh)
                n_faces = len(mesh.get('faces', []))
                scores.append((s, n_faces))

        if not scores:
            return scorer.score({})

        # Weight by face count (larger meshes matter more)
        total_faces = sum(n for _, n in scores)
        if total_faces == 0:
            return scores[0][0]

        weighted_quality = sum(
            s['quality_score'] * n / total_faces
            for s, n in scores
        )
        return {
            'quality_score': round(weighted_quality, 4),
            'quality_tier': (
                'excellent' if weighted_quality >= 0.80 else
                'good' if weighted_quality >= 0.60 else
                'fair' if weighted_quality >= 0.35 else
                'poor'
            ),
            'object_scores': [s for s, _ in scores],
        }

    # Single mesh
    mesh = data.get('mesh', data)
    return scorer.score(mesh)


if __name__ == "__main__":
    main()
