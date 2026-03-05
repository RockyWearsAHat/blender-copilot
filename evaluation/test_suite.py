"""Frozen test suite for geometric evaluation.

Provides a fixed set of prompts with expected shape properties for
reproducible evaluation across training runs.

Categories:
- primitives: sphere, cube, cylinder, cone, torus
- furniture: chair, table, shelf, lamp
- vehicles: car, boat, airplane
- nature: tree, rock, leaf
- architecture: house, tower, bridge
- miscellaneous: sword, cup, gear

Each test case defines:
- prompt: text prompt for generation
- category: semantic category
- expected_properties: rough shape expectations for sanity checks
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

TEST_SUITE_PATH = Path(__file__).parent.parent / "data" / "eval" / "test_suite.json"
LOW_POLY_STYLE_SUITE_PATH = Path(__file__).parent.parent / "data" / "eval" / "low_poly_style_suite.json"

TEST_CASES = [
    # ── Primitives ──
    {
        "id": "prim_sphere",
        "prompt": "sphere",
        "category": "primitive",
        "expected": {
            "aspect_ratio_range": [0.7, 1.3],
            "min_faces": 8,
            "symmetry_hint": "spherical",
        },
    },
    {
        "id": "prim_cube",
        "prompt": "cube",
        "category": "primitive",
        "expected": {
            "aspect_ratio_range": [0.7, 1.3],
            "min_faces": 6,
            "symmetry_hint": "cubic",
        },
    },
    {
        "id": "prim_cylinder",
        "prompt": "cylinder",
        "category": "primitive",
        "expected": {
            "aspect_ratio_range": [0.3, 3.0],
            "min_faces": 8,
            "symmetry_hint": "axial_z",
        },
    },
    {
        "id": "prim_cone",
        "prompt": "cone",
        "category": "primitive",
        "expected": {
            "aspect_ratio_range": [0.3, 3.0],
            "min_faces": 6,
        },
    },
    {
        "id": "prim_torus",
        "prompt": "torus",
        "category": "primitive",
        "expected": {
            "aspect_ratio_range": [0.2, 0.8],
            "min_faces": 16,
            "symmetry_hint": "toroidal",
        },
    },
    {
        "id": "prim_flat_box",
        "prompt": "flat wide box",
        "category": "primitive",
        "expected": {
            "aspect_ratio_range": [0.1, 0.5],
            "min_faces": 6,
        },
    },
    {
        "id": "prim_tall_cylinder",
        "prompt": "tall narrow cylinder",
        "category": "primitive",
        "expected": {
            "aspect_ratio_range": [1.5, 10.0],
            "min_faces": 8,
        },
    },
    # ── Furniture ──
    {
        "id": "furn_chair",
        "prompt": "chair",
        "category": "furniture",
        "expected": {
            "min_faces": 20,
        },
    },
    {
        "id": "furn_table",
        "prompt": "table",
        "category": "furniture",
        "expected": {
            "min_faces": 12,
        },
    },
    {
        "id": "furn_shelf",
        "prompt": "bookshelf",
        "category": "furniture",
        "expected": {
            "min_faces": 12,
        },
    },
    {
        "id": "furn_lamp",
        "prompt": "desk lamp",
        "category": "furniture",
        "expected": {
            "min_faces": 10,
        },
    },
    # ── Vehicles ──
    {
        "id": "veh_car",
        "prompt": "car",
        "category": "vehicle",
        "expected": {
            "min_faces": 30,
        },
    },
    {
        "id": "veh_boat",
        "prompt": "small boat",
        "category": "vehicle",
        "expected": {
            "min_faces": 20,
        },
    },
    {
        "id": "veh_airplane",
        "prompt": "airplane",
        "category": "vehicle",
        "expected": {
            "min_faces": 20,
        },
    },
    # ── Nature ──
    {
        "id": "nat_tree",
        "prompt": "tree",
        "category": "nature",
        "expected": {
            "min_faces": 10,
        },
    },
    {
        "id": "nat_rock",
        "prompt": "rock",
        "category": "nature",
        "expected": {
            "min_faces": 8,
        },
    },
    {
        "id": "nat_leaf",
        "prompt": "leaf",
        "category": "nature",
        "expected": {
            "min_faces": 4,
        },
    },
    # ── Architecture ──
    {
        "id": "arch_house",
        "prompt": "simple house",
        "category": "architecture",
        "expected": {
            "min_faces": 12,
        },
    },
    {
        "id": "arch_tower",
        "prompt": "tower",
        "category": "architecture",
        "expected": {
            "min_faces": 10,
            "aspect_ratio_range": [1.5, 10.0],
        },
    },
    {
        "id": "arch_bridge",
        "prompt": "bridge",
        "category": "architecture",
        "expected": {
            "min_faces": 12,
        },
    },
    # ── Objects ──
    {
        "id": "obj_sword",
        "prompt": "sword",
        "category": "object",
        "expected": {
            "min_faces": 8,
        },
    },
    {
        "id": "obj_cup",
        "prompt": "coffee cup",
        "category": "object",
        "expected": {
            "min_faces": 10,
        },
    },
    {
        "id": "obj_gear",
        "prompt": "gear",
        "category": "object",
        "expected": {
            "min_faces": 16,
        },
    },
    {
        "id": "obj_key",
        "prompt": "key",
        "category": "object",
        "expected": {
            "min_faces": 8,
        },
    },
    {
        "id": "obj_bottle",
        "prompt": "bottle",
        "category": "object",
        "expected": {
            "min_faces": 10,
        },
    },
    # ── Descriptive / compositional ──
    {
        "id": "comp_mushroom",
        "prompt": "mushroom",
        "category": "nature",
        "expected": {
            "min_faces": 8,
        },
    },
    {
        "id": "comp_hammer",
        "prompt": "hammer",
        "category": "object",
        "expected": {
            "min_faces": 8,
        },
    },
    {
        "id": "comp_skull",
        "prompt": "skull",
        "category": "organic",
        "expected": {
            "min_faces": 20,
        },
    },
    {
        "id": "comp_chess_pawn",
        "prompt": "chess pawn",
        "category": "object",
        "expected": {
            "min_faces": 10,
        },
    },
    {
        "id": "comp_crown",
        "prompt": "crown",
        "category": "object",
        "expected": {
            "min_faces": 12,
        },
    },
]


LOW_POLY_STYLE_CASES = [
    {
        "id": "lp_stylized_mountain",
        "prompt": "stylized low poly mountain terrain",
        "category": "style_low_poly",
        "domain": "modeling",
        "expected": {
            "min_faces": 120,
            "max_faces": 2500,
            "style_hint": "stylized",
        },
    },
    {
        "id": "lp_stylized_rock",
        "prompt": "faceted stylized low poly rock",
        "category": "style_low_poly",
        "domain": "modeling",
        "expected": {
            "min_faces": 60,
            "max_faces": 1500,
            "style_hint": "stylized",
        },
    },
    {
        "id": "lp_stylized_hills",
        "prompt": "flat shaded low poly hills",
        "category": "style_low_poly",
        "domain": "modeling",
        "expected": {
            "min_faces": 100,
            "max_faces": 2200,
            "style_hint": "stylized",
        },
    },
    {
        "id": "lp_retro_mountain",
        "prompt": "ps1 retro low poly mountain",
        "category": "style_low_poly",
        "domain": "modeling",
        "expected": {
            "min_faces": 120,
            "max_faces": 3000,
            "style_hint": "retro",
        },
    },
    {
        "id": "lp_retro_cliff",
        "prompt": "retro n64 smooth low poly cliff",
        "category": "style_low_poly",
        "domain": "modeling",
        "expected": {
            "min_faces": 80,
            "max_faces": 2200,
            "style_hint": "retro",
        },
    },
    {
        "id": "lp_retro_dunes",
        "prompt": "smooth low poly dunes with low-res texture",
        "category": "style_low_poly",
        "domain": "modeling",
        "expected": {
            "min_faces": 90,
            "max_faces": 2600,
            "style_hint": "retro",
        },
    },
]


def _default_domain_for_category(category: str) -> str:
    category = (category or "").lower()
    if category in ("primitive", "object", "organic"):
        return "modeling"
    if category in ("furniture", "architecture", "vehicle", "nature"):
        return "scene_assembly"
    return "modeling"


def load_test_suite(path: Optional[Path] = None) -> list[dict]:
    """Load the frozen test suite from disk or return built-in defaults.

    If a custom test suite JSON exists on disk, loads that. Otherwise
    returns the built-in TEST_CASES and optionally saves them to disk.

    Args:
        path: Optional path to JSON file. Defaults to data/eval/test_suite.json

    Returns:
        List of test case dicts
    """
    path = path or TEST_SUITE_PATH

    if path.exists():
        try:
            with open(path) as f:
                cases = json.load(f)
            logger.info(f"Loaded {len(cases)} test cases from {path}")
            return cases
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}, using built-in suite")

    return TEST_CASES


def load_low_poly_style_suite(path: Optional[Path] = None) -> list[dict]:
    """Load the frozen low-poly style suite from disk or built-in defaults."""
    path = path or LOW_POLY_STYLE_SUITE_PATH

    if path.exists():
        try:
            with open(path) as f:
                cases = json.load(f)
            logger.info(f"Loaded {len(cases)} low-poly style cases from {path}")
            return cases
        except Exception as e:
            logger.warning(f"Failed to load low-poly suite {path}: {e}, using built-in suite")

    return LOW_POLY_STYLE_CASES


def save_test_suite(cases: Optional[list[dict]] = None,
                    path: Optional[Path] = None) -> Path:
    """Save the test suite to disk for reproducibility.

    Args:
        cases: List of test case dicts. Defaults to built-in TEST_CASES.
        path: Output path. Defaults to data/eval/test_suite.json

    Returns:
        Path to saved file
    """
    cases = cases or TEST_CASES
    path = path or TEST_SUITE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(cases, f, indent=2)

    logger.info(f"Saved {len(cases)} test cases to {path}")
    return path


def save_low_poly_style_suite(cases: Optional[list[dict]] = None,
                              path: Optional[Path] = None) -> Path:
    """Save the low-poly style suite to disk for reproducibility."""
    cases = cases or LOW_POLY_STYLE_CASES
    path = path or LOW_POLY_STYLE_SUITE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(cases, f, indent=2)

    logger.info(f"Saved {len(cases)} low-poly style test cases to {path}")
    return path


def check_shape_expectations(vertices: np.ndarray,
                             expected: dict,
                             faces: Optional[np.ndarray] = None) -> dict:
    """Check if generated mesh meets shape expectations.

    Returns dict of checks: {check_name: (passed: bool, detail: str)}
    """
    results = {}

    if len(vertices) == 0:
        results["non_empty"] = (False, "Mesh has no vertices")
        return results

    results["non_empty"] = (True, f"{len(vertices)} vertices")

    if "min_faces" in expected:
        if faces is not None and len(faces) > 0:
            n_faces = len(faces)
        else:
            n_faces = len(vertices) // 3
        passed = n_faces >= expected["min_faces"]
        results["min_faces"] = (passed,
                                f"{n_faces} faces (need >= {expected['min_faces']})")

    if "max_faces" in expected:
        if faces is not None and len(faces) > 0:
            n_faces = len(faces)
        else:
            n_faces = len(vertices) // 3
        passed = n_faces <= expected["max_faces"]
        results["max_faces"] = (passed,
                                f"{n_faces} faces (need <= {expected['max_faces']})")

    if "aspect_ratio_range" in expected:
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        extents = bbox_max - bbox_min
        extents = np.maximum(extents, 1e-6)

        sorted_ext = np.sort(extents)
        if sorted_ext[-1] > 1e-6:
            aspect = sorted_ext[0] / sorted_ext[-1]
            lo, hi = expected["aspect_ratio_range"]
            passed = lo <= aspect <= hi
            results["aspect_ratio"] = (passed,
                                       f"aspect={aspect:.2f} (expected [{lo}, {hi}])")

    return results


def run_test_suite(generate_fn, test_cases: Optional[list[dict]] = None,
                   max_faces: int = 512) -> dict:
    """Run the full test suite through a generation function.

    Args:
        generate_fn: Callable(prompt, max_faces) -> (vertices, faces)
            where vertices is (V, 3) and faces is (F, 3)
        test_cases: Optional custom test cases. Defaults to built-in.
        max_faces: Max faces per generation (lower = faster eval)

    Returns:
        dict with per-case results and summary statistics
    """
    from evaluation.metrics import evaluate_single

    cases = test_cases or load_test_suite()
    results = []

    for case in cases:
        case_result = {
            "id": case["id"],
            "prompt": case["prompt"],
            "category": case["category"],
            "domain": case.get("domain", _default_domain_for_category(case.get("category", ""))),
        }

        try:
            verts, faces = generate_fn(case["prompt"], max_faces)
            verts = np.array(verts, dtype=np.float64)
            faces = np.array(faces, dtype=np.int64)

            metrics = evaluate_single(verts, faces)
            case_result["metrics"] = metrics

            if "expected" in case:
                checks = check_shape_expectations(
                    verts, case["expected"], faces=faces)
                case_result["expectation_checks"] = {
                    k: {"passed": v[0], "detail": v[1]}
                    for k, v in checks.items()
                }
                case_result["expectations_met"] = all(
                    v[0] for v in checks.values())
            else:
                case_result["expectations_met"] = True

            case_result["generated"] = True
            case_result["num_vertices"] = len(verts)
            case_result["num_faces"] = len(faces)

        except Exception as e:
            logger.error(f"Generation failed for '{case['prompt']}': {e}")
            case_result["generated"] = False
            case_result["error"] = str(e)
            case_result["expectations_met"] = False

        results.append(case_result)

    summary = _summarize_test_results(results)
    return {"results": results, "summary": summary}


def _summarize_test_results(results: list[dict]) -> dict:
    """Compute summary statistics from test suite results."""
    total = len(results)
    generated = sum(1 for r in results if r.get("generated", False))
    expectations_met = sum(1 for r in results
                          if r.get("expectations_met", False))

    validity_scores = []
    face_counts = []
    for r in results:
        if r.get("generated") and "metrics" in r:
            validity_scores.append(
                r["metrics"]["validity"]["validity_score"])
            face_counts.append(r.get("num_faces", 0))

    by_category = {}
    by_domain = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "generated": 0, "valid": 0}
        by_category[cat]["total"] += 1
        if r.get("generated"):
            by_category[cat]["generated"] += 1
        if r.get("expectations_met"):
            by_category[cat]["valid"] += 1

        domain = r.get("domain", _default_domain_for_category(cat))
        if domain not in by_domain:
            by_domain[domain] = {"total": 0, "generated": 0, "valid": 0}
        by_domain[domain]["total"] += 1
        if r.get("generated"):
            by_domain[domain]["generated"] += 1
        if r.get("expectations_met"):
            by_domain[domain]["valid"] += 1

    summary = {
        "total_cases": total,
        "generated_successfully": generated,
        "generation_rate": generated / max(total, 1),
        "expectations_met": expectations_met,
        "expectations_rate": expectations_met / max(total, 1),
        "by_category": by_category,
        "by_domain": by_domain,
    }

    if validity_scores:
        summary["validity_score_mean"] = float(np.mean(validity_scores))
        summary["validity_score_min"] = float(np.min(validity_scores))

    if face_counts:
        summary["face_count_mean"] = float(np.mean(face_counts))
        summary["face_count_std"] = float(np.std(face_counts))

    return summary
