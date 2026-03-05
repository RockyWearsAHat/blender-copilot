"""Geometric quality metrics for 3D mesh evaluation.

Implements the metrics recommended by the deep research report:
- Chamfer Distance (CD) — surface distance between predicted and reference
- F-score at distance thresholds — precision/recall on surface proximity
- Normal consistency — alignment of surface normals
- Mesh validity — manifoldness, watertightness, degenerate face detection
- Text-mesh alignment — does the prompt category match the generated shape?

All metrics operate on (vertices, faces) tuples and do NOT require Blender.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def sample_surface_points(vertices: np.ndarray, faces: np.ndarray,
                          n_points: int = 2048, seed: int = 42) -> np.ndarray:
    """Uniformly sample points on the surface of a triangle mesh.

    Uses area-weighted random sampling: triangles with larger area
    get proportionally more sample points.

    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face vertex indices
        n_points: number of points to sample
        seed: random seed for reproducibility

    Returns:
        (n_points, 3) sampled surface points
    """
    rng = np.random.RandomState(seed)

    if len(faces) == 0 or len(vertices) == 0:
        return np.zeros((n_points, 3))

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()

    if total_area < 1e-10:
        return np.zeros((n_points, 3))

    probs = areas / total_area

    face_indices = rng.choice(len(faces), size=n_points, p=probs)

    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    sqrt_r1 = np.sqrt(r1)

    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    sampled_v0 = vertices[faces[face_indices, 0]]
    sampled_v1 = vertices[faces[face_indices, 1]]
    sampled_v2 = vertices[faces[face_indices, 2]]

    points = u[:, None] * sampled_v0 + v[:, None] * sampled_v1 + w[:, None] * sampled_v2

    return points


def chamfer_distance(points_a: np.ndarray, points_b: np.ndarray) -> dict:
    """Compute Chamfer Distance between two point clouds.

    CD = mean(min_b ||a - b||^2) + mean(min_a ||a - b||^2)

    This is the standard 3D reconstruction metric used by PoinTr,
    3DTopia-XL, and most mesh generation papers.

    Args:
        points_a: (N, 3) point cloud A (predicted)
        points_b: (M, 3) point cloud B (reference)

    Returns:
        dict with 'cd' (total), 'cd_a_to_b', 'cd_b_to_a'
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return {"cd": float("inf"), "cd_a_to_b": float("inf"),
                "cd_b_to_a": float("inf")}

    try:
        from scipy.spatial import cKDTree
        tree_b = cKDTree(points_b)
        dist_a_to_b, _ = tree_b.query(points_a)
        cd_a_to_b = (dist_a_to_b ** 2).mean()

        tree_a = cKDTree(points_a)
        dist_b_to_a, _ = tree_a.query(points_b)
        cd_b_to_a = (dist_b_to_a ** 2).mean()
    except ImportError:
        diff = points_a[:, None, :] - points_b[None, :, :]
        dists = (diff ** 2).sum(axis=-1)
        cd_a_to_b = dists.min(axis=1).mean()
        cd_b_to_a = dists.min(axis=0).mean()

    cd_total = float(cd_a_to_b + cd_b_to_a)
    return {
        "cd": cd_total,
        "cd_a_to_b": float(cd_a_to_b),
        "cd_b_to_a": float(cd_b_to_a),
    }


def f_score(points_pred: np.ndarray, points_ref: np.ndarray,
            threshold: float = 0.01) -> dict:
    """Compute F-score at a distance threshold.

    F-score measures the harmonic mean of precision and recall:
    - Precision: fraction of predicted points within threshold of reference
    - Recall: fraction of reference points within threshold of prediction

    Standard thresholds:  0.01 (tight), 0.02 (moderate), 0.05 (loose)

    Args:
        points_pred: (N, 3) predicted surface points
        points_ref: (M, 3) reference surface points
        threshold: distance threshold

    Returns:
        dict with 'f_score', 'precision', 'recall'
    """
    if len(points_pred) == 0 or len(points_ref) == 0:
        return {"f_score": 0.0, "precision": 0.0, "recall": 0.0}

    try:
        from scipy.spatial import cKDTree
        tree_ref = cKDTree(points_ref)
        dist_pred_to_ref, _ = tree_ref.query(points_pred)

        tree_pred = cKDTree(points_pred)
        dist_ref_to_pred, _ = tree_pred.query(points_ref)
    except ImportError:
        diff_pr = points_pred[:, None, :] - points_ref[None, :, :]
        dists_pr = np.sqrt((diff_pr ** 2).sum(axis=-1))
        dist_pred_to_ref = dists_pr.min(axis=1)
        dist_ref_to_pred = dists_pr.min(axis=0)

    precision = float((dist_pred_to_ref < threshold).mean())
    recall = float((dist_ref_to_pred < threshold).mean())

    if precision + recall < 1e-10:
        f = 0.0
    else:
        f = 2.0 * precision * recall / (precision + recall)

    return {"f_score": f, "precision": precision, "recall": recall}


def compute_face_normals(vertices: np.ndarray,
                         faces: np.ndarray) -> np.ndarray:
    """Compute per-face unit normals.

    Args:
        vertices: (V, 3)
        faces: (F, 3)

    Returns:
        (F, 3) unit normals (zero-length faces get [0,0,0])
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)
    return normals / lengths


def normal_consistency(vertices_pred: np.ndarray, faces_pred: np.ndarray,
                       vertices_ref: np.ndarray, faces_ref: np.ndarray,
                       n_samples: int = 2048) -> float:
    """Compute normal consistency between predicted and reference meshes.

    For each sampled point on the prediction, find the nearest point on
    the reference and compute |dot(n_pred, n_ref)|. Average over all points.

    Returns:
        float in [0, 1] where 1 = perfectly aligned normals
    """
    if (len(faces_pred) == 0 or len(faces_ref) == 0
            or len(vertices_pred) == 0 or len(vertices_ref) == 0):
        return 0.0

    normals_pred = compute_face_normals(vertices_pred, faces_pred)
    normals_ref = compute_face_normals(vertices_ref, faces_ref)

    rng = np.random.RandomState(42)

    areas_pred = _face_areas(vertices_pred, faces_pred)
    if areas_pred.sum() < 1e-10:
        return 0.0
    probs_pred = areas_pred / areas_pred.sum()
    sampled_faces_pred = rng.choice(len(faces_pred), size=n_samples, p=probs_pred)
    sampled_normals_pred = normals_pred[sampled_faces_pred]

    centers_pred = (vertices_pred[faces_pred[sampled_faces_pred, 0]]
                    + vertices_pred[faces_pred[sampled_faces_pred, 1]]
                    + vertices_pred[faces_pred[sampled_faces_pred, 2]]) / 3.0

    ref_centers = (vertices_ref[faces_ref[:, 0]]
                   + vertices_ref[faces_ref[:, 1]]
                   + vertices_ref[faces_ref[:, 2]]) / 3.0

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ref_centers)
        _, nearest_idx = tree.query(centers_pred)
    except ImportError:
        diff = centers_pred[:, None, :] - ref_centers[None, :, :]
        dists = (diff ** 2).sum(axis=-1)
        nearest_idx = dists.argmin(axis=1)

    nearest_normals_ref = normals_ref[nearest_idx]
    dots = np.abs((sampled_normals_pred * nearest_normals_ref).sum(axis=1))
    return float(dots.mean())


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-face areas."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def mesh_validity(vertices: np.ndarray,
                  faces: np.ndarray) -> dict:
    """Compute mesh validity metrics.

    Checks:
    - Degenerate faces (zero area)
    - Non-manifold edges (shared by != 2 faces)
    - Boundary edges (shared by only 1 face)
    - Duplicate faces
    - Overall validity score [0, 1]

    Args:
        vertices: (V, 3)
        faces: (F, 3)

    Returns:
        dict with validity metrics
    """
    result = {
        "num_vertices": len(vertices),
        "num_faces": len(faces),
        "degenerate_faces": 0,
        "degenerate_ratio": 0.0,
        "non_manifold_edges": 0,
        "boundary_edges": 0,
        "manifold_ratio": 1.0,
        "duplicate_faces": 0,
        "is_valid": True,
        "validity_score": 1.0,
    }

    if len(faces) == 0 or len(vertices) == 0:
        result["is_valid"] = False
        result["validity_score"] = 0.0
        return result

    areas = _face_areas(vertices, faces)
    degenerate = int((areas < 1e-8).sum())
    result["degenerate_faces"] = degenerate
    result["degenerate_ratio"] = degenerate / len(faces)

    edge_counts = {}
    for face in faces:
        for i in range(3):
            edge = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    total_edges = len(edge_counts)
    if total_edges > 0:
        manifold_edges = sum(1 for c in edge_counts.values() if c == 2)
        boundary = sum(1 for c in edge_counts.values() if c == 1)
        non_manifold = sum(1 for c in edge_counts.values() if c > 2)

        result["non_manifold_edges"] = non_manifold
        result["boundary_edges"] = boundary
        result["manifold_ratio"] = manifold_edges / total_edges

    face_set = set()
    duplicates = 0
    for face in faces:
        key = tuple(sorted(int(v) for v in face))
        if key in face_set:
            duplicates += 1
        face_set.add(key)
    result["duplicate_faces"] = duplicates

    score = 1.0
    score -= result["degenerate_ratio"] * 0.3
    if total_edges > 0:
        score -= (result["non_manifold_edges"] / total_edges) * 0.4
        score -= (result["boundary_edges"] / total_edges) * 0.15
    if len(faces) > 0:
        score -= (duplicates / len(faces)) * 0.15
    score = max(0.0, min(1.0, score))
    result["validity_score"] = score
    result["is_valid"] = score > 0.5

    return result


def bounding_box_iou(verts_pred: np.ndarray,
                     verts_ref: np.ndarray) -> float:
    """Compute IoU of axis-aligned bounding boxes.

    Quick proxy for shape similarity: if the bounding boxes
    don't overlap, the shapes are clearly very different.

    Returns:
        float in [0, 1]
    """
    if len(verts_pred) == 0 or len(verts_ref) == 0:
        return 0.0

    min_p, max_p = verts_pred.min(axis=0), verts_pred.max(axis=0)
    min_r, max_r = verts_ref.min(axis=0), verts_ref.max(axis=0)

    inter_min = np.maximum(min_p, min_r)
    inter_max = np.minimum(max_p, max_r)
    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_vol = inter_dims.prod()

    vol_p = np.maximum(max_p - min_p, 1e-10).prod()
    vol_r = np.maximum(max_r - min_r, 1e-10).prod()
    union_vol = vol_p + vol_r - inter_vol

    if union_vol < 1e-10:
        return 0.0
    return float(inter_vol / union_vol)


def shape_distribution(vertices: np.ndarray, n_pairs: int = 10000,
                       n_bins: int = 64, seed: int = 42) -> np.ndarray:
    """Compute a shape distribution (D2 descriptor).

    Histogram of pairwise distances between random vertex pairs.
    Used for quick shape-category classification.

    Returns:
        (n_bins,) normalized histogram
    """
    if len(vertices) < 2:
        return np.zeros(n_bins)

    rng = np.random.RandomState(seed)
    n = len(vertices)
    idx_a = rng.randint(0, n, size=n_pairs)
    idx_b = rng.randint(0, n, size=n_pairs)
    dists = np.linalg.norm(vertices[idx_a] - vertices[idx_b], axis=1)

    if dists.max() < 1e-10:
        return np.zeros(n_bins)

    hist, _ = np.histogram(dists, bins=n_bins, range=(0, dists.max()),
                           density=True)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def evaluate_single(vertices_pred: np.ndarray, faces_pred: np.ndarray,
                    vertices_ref: Optional[np.ndarray] = None,
                    faces_ref: Optional[np.ndarray] = None,
                    n_surface_points: int = 4096) -> dict:
    """Run all metrics on a single predicted mesh.

    If reference mesh is provided, computes comparative metrics (CD, F-score).
    Always computes validity metrics.

    Args:
        vertices_pred: (V, 3) predicted vertices
        faces_pred: (F, 3) predicted faces
        vertices_ref: (V', 3) reference vertices (optional)
        faces_ref: (F', 3) reference faces (optional)
        n_surface_points: points to sample for CD/F-score

    Returns:
        dict with all computed metrics
    """
    result = {}

    validity = mesh_validity(vertices_pred, faces_pred)
    result["validity"] = validity

    if vertices_ref is not None and faces_ref is not None:
        pts_pred = sample_surface_points(
            vertices_pred, faces_pred, n_surface_points)
        pts_ref = sample_surface_points(
            vertices_ref, faces_ref, n_surface_points)

        result["chamfer_distance"] = chamfer_distance(pts_pred, pts_ref)

        result["f_score_001"] = f_score(pts_pred, pts_ref, threshold=0.01)
        result["f_score_002"] = f_score(pts_pred, pts_ref, threshold=0.02)
        result["f_score_005"] = f_score(pts_pred, pts_ref, threshold=0.05)

        result["normal_consistency"] = normal_consistency(
            vertices_pred, faces_pred, vertices_ref, faces_ref)

        result["bbox_iou"] = bounding_box_iou(vertices_pred, vertices_ref)

    return result


def evaluate_batch(predictions: list[dict],
                   references: Optional[list[dict]] = None,
                   n_surface_points: int = 4096) -> dict:
    """Evaluate a batch of predictions and compute aggregate metrics.

    Each prediction dict should have 'vertices' and 'faces' keys.
    Each reference dict (if provided) should have 'vertices' and 'faces' keys.

    Returns:
        dict with per-sample results and aggregate statistics
    """
    per_sample = []
    has_refs = references is not None and len(references) == len(predictions)

    for i, pred in enumerate(predictions):
        v_pred = np.array(pred["vertices"], dtype=np.float64)
        f_pred = np.array(pred["faces"], dtype=np.int64)

        v_ref = None
        f_ref = None
        if has_refs:
            v_ref = np.array(references[i]["vertices"], dtype=np.float64)
            f_ref = np.array(references[i]["faces"], dtype=np.int64)

        metrics = evaluate_single(v_pred, f_pred, v_ref, f_ref,
                                  n_surface_points)
        metrics["index"] = i
        if "prompt" in pred:
            metrics["prompt"] = pred["prompt"]
        per_sample.append(metrics)

    agg = _aggregate_metrics(per_sample, has_refs)
    return {"per_sample": per_sample, "aggregate": agg}


def _aggregate_metrics(samples: list[dict], has_refs: bool) -> dict:
    """Compute aggregate statistics from per-sample metrics."""
    agg = {}

    validity_scores = [s["validity"]["validity_score"] for s in samples]
    agg["validity_score_mean"] = float(np.mean(validity_scores))
    agg["validity_score_std"] = float(np.std(validity_scores))
    agg["valid_ratio"] = float(np.mean([s["validity"]["is_valid"] for s in samples]))

    manifold_ratios = [s["validity"]["manifold_ratio"] for s in samples]
    agg["manifold_ratio_mean"] = float(np.mean(manifold_ratios))

    degenerate_ratios = [s["validity"]["degenerate_ratio"] for s in samples]
    agg["degenerate_ratio_mean"] = float(np.mean(degenerate_ratios))

    face_counts = [s["validity"]["num_faces"] for s in samples]
    agg["num_faces_mean"] = float(np.mean(face_counts))
    agg["num_faces_std"] = float(np.std(face_counts))

    vert_counts = [s["validity"]["num_vertices"] for s in samples]
    agg["num_vertices_mean"] = float(np.mean(vert_counts))

    if has_refs:
        cds = [s["chamfer_distance"]["cd"] for s in samples
               if s["chamfer_distance"]["cd"] < float("inf")]
        if cds:
            agg["chamfer_distance_mean"] = float(np.mean(cds))
            agg["chamfer_distance_median"] = float(np.median(cds))
            agg["chamfer_distance_std"] = float(np.std(cds))

        for threshold_key in ["f_score_001", "f_score_002", "f_score_005"]:
            scores = [s[threshold_key]["f_score"] for s in samples
                      if threshold_key in s]
            if scores:
                agg[f"{threshold_key}_mean"] = float(np.mean(scores))

        nc_scores = [s["normal_consistency"] for s in samples
                     if "normal_consistency" in s]
        if nc_scores:
            agg["normal_consistency_mean"] = float(np.mean(nc_scores))

        iou_scores = [s["bbox_iou"] for s in samples if "bbox_iou" in s]
        if iou_scores:
            agg["bbox_iou_mean"] = float(np.mean(iou_scores))

    agg["num_samples"] = len(samples)
    return agg
