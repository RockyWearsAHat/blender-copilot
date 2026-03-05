"""Comprehensive mesh quality test against the inference server.

Tests various prompts and checks for:
- Correct vertex/face counts
- Bounding box proportions
- Mesh validity (manifold, no degenerate faces)
- "Ball of triangles" corruption pattern
- Vertex spread / dimension filling
"""
import json
import sys
import time
import requests
import numpy as np

SERVER = "http://localhost:8420"

TEST_CASES = [
    {
        "prompt": "a cube",
        "expect_verts_range": (8, 24),
        "expect_faces_range": (6, 24),
        "expect_bbox_cubic": True,
        "description": "Perfect cube",
    },
    {
        "prompt": "a sphere",
        "expect_verts_range": (20, 800),
        "expect_faces_range": (20, 2000),
        "expect_bbox_cubic": True,
        "description": "Sphere",
    },
    {
        "prompt": "a cylinder",
        "expect_verts_range": (16, 500),
        "expect_faces_range": (12, 1000),
        "description": "Cylinder",
    },
    {
        "prompt": "a donut",
        "expect_verts_range": (20, 800),
        "expect_faces_range": (20, 2000),
        "description": "Donut/torus",
    },
    {
        "prompt": "low poly car",
        "expect_verts_range": (20, 1200),
        "expect_faces_range": (20, 2400),
        "expect_bbox_elongated": True,
        "description": "Low poly car",
    },
    {
        "prompt": "a chair",
        "expect_verts_range": (16, 800),
        "expect_faces_range": (12, 2000),
        "description": "Chair",
    },
]


def compute_mesh_stats(vertices, faces):
    """Analyze mesh geometry thoroughly."""
    verts = np.array(vertices)
    face_arr = np.array(faces)

    stats = {
        "num_verts": len(verts),
        "num_faces": len(face_arr),
    }

    if len(verts) == 0:
        return stats

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_dims = bbox_max - bbox_min
    stats["bbox_min"] = bbox_min.tolist()
    stats["bbox_max"] = bbox_max.tolist()
    stats["bbox_dims"] = bbox_dims.tolist()
    stats["bbox_volume"] = float(np.prod(bbox_dims + 1e-10))

    centroid = verts.mean(axis=0)
    dists_from_center = np.linalg.norm(verts - centroid, axis=1)
    stats["mean_dist_from_center"] = float(dists_from_center.mean())
    stats["max_dist_from_center"] = float(dists_from_center.max())
    stats["std_dist_from_center"] = float(dists_from_center.std())

    max_possible_dist = np.linalg.norm(bbox_dims) / 2
    if max_possible_dist > 1e-6:
        stats["vertex_spread_ratio"] = float(dists_from_center.mean() / max_possible_dist)
    else:
        stats["vertex_spread_ratio"] = 0.0

    # Near-duplicate vertices
    if len(verts) > 1:
        from scipy.spatial import cKDTree
        tree = cKDTree(verts)
        pairs = tree.query_pairs(r=0.001)
        stats["near_duplicate_vert_pairs"] = len(pairs)
        stats["near_duplicate_ratio"] = len(pairs) / max(1, len(verts))

    if len(face_arr) > 0:
        v0 = verts[face_arr[:, 0]]
        v1 = verts[face_arr[:, 1]]
        v2 = verts[face_arr[:, 2]]
        crosses = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(crosses, axis=1)

        stats["total_surface_area"] = float(areas.sum())
        stats["mean_face_area"] = float(areas.mean())
        stats["min_face_area"] = float(areas.min())
        stats["max_face_area"] = float(areas.max())
        stats["degenerate_faces"] = int((areas < 1e-8).sum())
        stats["degenerate_ratio"] = float((areas < 1e-8).mean())

        # Duplicate faces
        face_keys = set()
        dup_count = 0
        for f in face_arr:
            key = tuple(sorted(f))
            if key in face_keys:
                dup_count += 1
            face_keys.add(key)
        stats["duplicate_faces"] = dup_count

        # Edge analysis
        edge_counts = {}
        for f in face_arr:
            for i in range(3):
                e = tuple(sorted((int(f[i]), int(f[(i+1) % 3]))))
                edge_counts[e] = edge_counts.get(e, 0) + 1

        total_edges = len(edge_counts)
        manifold = sum(1 for c in edge_counts.values() if c == 2)
        boundary = sum(1 for c in edge_counts.values() if c == 1)
        non_manifold = sum(1 for c in edge_counts.values() if c > 2)

        stats["total_edges"] = total_edges
        stats["manifold_edges"] = manifold
        stats["boundary_edges"] = boundary
        stats["non_manifold_edges"] = non_manifold
        stats["manifold_ratio"] = manifold / max(1, total_edges)

        # Normal consistency
        normals = crosses / (np.linalg.norm(crosses, axis=1, keepdims=True) + 1e-10)
        face_centers = (v0 + v1 + v2) / 3
        outward = face_centers - centroid
        dots = (normals * outward).sum(axis=1)
        stats["outward_normal_ratio"] = float((dots > 0).mean())

        if len(verts) > 0 and len(face_arr) > 0:
            verts_per_face = len(verts) / len(face_arr)
            stats["verts_per_face_ratio"] = verts_per_face

            # Face overlap detection
            if len(face_centers) > 1:
                from scipy.spatial import cKDTree as cKDTree2
                fc_tree = cKDTree2(face_centers)
                nearby_pairs = fc_tree.query_pairs(r=0.01)
                stats["overlapping_face_pairs"] = len(nearby_pairs)
                stats["overlap_ratio"] = len(nearby_pairs) / max(1, len(face_arr))

    return stats


def test_prompt(prompt, params, description):
    """Test a single prompt."""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"PROMPT: {prompt!r}")
    print(f"{'='*70}")

    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_faces": 512,
        "top_k": 64,
        "top_p": 0.95,
        "cfg_scale": 2.0,
    }

    try:
        resp = requests.post(f"{SERVER}/generate/mesh", json=payload, timeout=120)
        data = resp.json()
    except Exception as e:
        print(f"  ERROR: Request failed: {e}")
        return {"prompt": prompt, "status": "REQUEST_FAILED", "error": str(e)}

    if "error" in data:
        print(f"  SERVER ERROR: {data['error']}")
        return {
            "prompt": prompt, "status": "SERVER_ERROR",
            "error": data["error"],
            "token_count": data.get("token_count", 0),
            "unique_tokens": data.get("unique_tokens", 0),
        }

    objects = data.get("objects", [])
    if not objects:
        print(f"  ERROR: No objects returned")
        return {"prompt": prompt, "status": "NO_OBJECTS"}

    mesh = objects[0].get("mesh", {})
    vertices = mesh.get("vertices", [])
    faces = mesh.get("faces", [])

    print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
    print(f"  Generation time: {data.get('generation_time', '?')}s")
    print(f"  Token count: {data.get('token_count', '?')}")

    if not vertices or not faces:
        print(f"  ERROR: Empty mesh!")
        return {"prompt": prompt, "status": "EMPTY_MESH"}

    stats = compute_mesh_stats(vertices, faces)

    print(f"\n  === Geometry Stats ===")
    print(f"  BBox dims: [{stats['bbox_dims'][0]:.3f}, {stats['bbox_dims'][1]:.3f}, {stats['bbox_dims'][2]:.3f}]")
    print(f"  BBox volume: {stats['bbox_volume']:.4f}")
    print(f"  Surface area: {stats.get('total_surface_area', 0):.4f}")
    print(f"  Vertex spread ratio: {stats.get('vertex_spread_ratio', 0):.3f}")
    print(f"  Near-dup verts: {stats.get('near_duplicate_vert_pairs', 0)} ({stats.get('near_duplicate_ratio', 0):.1%})")

    print(f"\n  === Face Quality ===")
    print(f"  Degenerate faces: {stats.get('degenerate_faces', 0)} ({stats.get('degenerate_ratio', 0):.1%})")
    print(f"  Duplicate faces: {stats.get('duplicate_faces', 0)}")
    print(f"  Overlapping face pairs: {stats.get('overlapping_face_pairs', 0)} ({stats.get('overlap_ratio', 0):.1%})")

    print(f"\n  === Topology ===")
    print(f"  Manifold ratio: {stats.get('manifold_ratio', 0):.1%}")
    print(f"  Boundary edges: {stats.get('boundary_edges', 0)}")
    print(f"  Non-manifold edges: {stats.get('non_manifold_edges', 0)}")
    print(f"  Outward normal ratio: {stats.get('outward_normal_ratio', 0):.1%}")
    print(f"  Verts/face ratio: {stats.get('verts_per_face_ratio', 0):.3f}")

    issues = []

    # Ball of triangles
    if stats.get("overlap_ratio", 0) > 0.3:
        issues.append(f"BALL_OF_TRIS: {stats['overlap_ratio']:.0%} face overlap")
    if stats.get("verts_per_face_ratio", 0) < 0.15:
        issues.append(f"LOW_VERT_FACE_RATIO: {stats['verts_per_face_ratio']:.3f}")
    if stats.get("degenerate_ratio", 0) > 0.1:
        issues.append(f"HIGH_DEGENERATE: {stats['degenerate_ratio']:.0%}")
    if stats.get("near_duplicate_ratio", 0) > 0.3:
        issues.append(f"MANY_DUPLICATE_VERTS: {stats['near_duplicate_ratio']:.0%}")
    if stats.get("non_manifold_edges", 0) > stats.get("total_edges", 1) * 0.1:
        issues.append(f"NON_MANIFOLD: {stats['non_manifold_edges']}/{stats['total_edges']}")
    if stats.get("vertex_spread_ratio", 0) < 0.2:
        issues.append(f"LOW_SPREAD: verts clustered ({stats['vertex_spread_ratio']:.3f})")

    # Expected properties
    vr = params.get("expect_verts_range")
    fr = params.get("expect_faces_range")
    if vr and not (vr[0] <= len(vertices) <= vr[1]):
        issues.append(f"VERTS_OUT_OF_RANGE: {len(vertices)} not in [{vr[0]},{vr[1]}]")
    if fr and not (fr[0] <= len(faces) <= fr[1]):
        issues.append(f"FACES_OUT_OF_RANGE: {len(faces)} not in [{fr[0]},{fr[1]}]")

    if params.get("expect_bbox_cubic"):
        dims = sorted(stats["bbox_dims"])
        if dims[0] > 0 and dims[2] / dims[0] > 3.0:
            issues.append(f"NOT_CUBIC: aspect {dims[2]/dims[0]:.1f}:1")

    if params.get("expect_bbox_elongated"):
        dims = sorted(stats["bbox_dims"])
        if dims[0] > 0 and dims[2] / dims[0] < 1.5:
            issues.append(f"NOT_ELONGATED: aspect {dims[2]/dims[0]:.1f}:1")

    if issues:
        print(f"\n  !!! ISSUES DETECTED !!!")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print(f"\n  OK - No major issues detected")

    return {
        "prompt": prompt,
        "status": "OK" if not issues else "ISSUES",
        "issues": issues,
        "stats": stats,
        "num_verts": len(vertices),
        "num_faces": len(faces),
        "generation_time": data.get("generation_time"),
    }


def main():
    try:
        health = requests.get(f"{SERVER}/health", timeout=5).json()
        print(f"Server: {health.get('status')}, model: {health.get('model_type')}, "
              f"step: {health.get('step')}, params: {health.get('model_params_m')}M")
    except Exception as e:
        print(f"Server not available: {e}")
        sys.exit(1)

    results = []
    for tc in TEST_CASES:
        result = test_prompt(tc["prompt"], tc, tc["description"])
        results.append(result)
        time.sleep(0.5)

    # Summary
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    ok = sum(1 for r in results if r["status"] == "OK")
    issues_count = sum(1 for r in results if r["status"] == "ISSUES")
    errors = sum(1 for r in results if r["status"] not in ("OK", "ISSUES"))

    print(f"  OK: {ok}/{len(results)}")
    print(f"  Issues: {issues_count}/{len(results)}")
    print(f"  Errors: {errors}/{len(results)}")

    all_issues = []
    for r in results:
        if r.get("issues"):
            for iss in r["issues"]:
                all_issues.append(f"  [{r['prompt']}] {iss}")

    if all_issues:
        print(f"\n  All issues:")
        for iss in all_issues:
            print(iss)

    ball_of_tris = [r for r in results if any("BALL_OF_TRIS" in i for i in r.get("issues", []))]
    if ball_of_tris:
        print(f"\n  BALL-OF-TRIANGLES detected in: {[r['prompt'] for r in ball_of_tris]}")

    return 0 if errors == 0 and not ball_of_tris else 1


if __name__ == "__main__":
    sys.exit(main())
