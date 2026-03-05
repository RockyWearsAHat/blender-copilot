#!/usr/bin/env python3
"""Training data quality gate — validates cache is training-ready.

This is called automatically before training starts to ensure:
1. Minimum sample count is met
2. Label quality distribution is acceptable
3. No broken/corrupt cache files
4. Source diversity meets threshold
5. Token integrity is verified
6. Materials data exists (if materials training enabled)

If the gate fails, training is blocked with a clear error message
telling the user to run enrich_training_data.py.

Usage:
    python scripts/validate_training_quality.py           # Full validation
    python scripts/validate_training_quality.py --quick    # Fast spot-check (50 files)
    python scripts/validate_training_quality.py --json     # Machine-readable output

Can also be imported:
    from scripts.validate_training_quality import validate_training_data
    ok, report = validate_training_data()
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
CACHE_DIR = BASE / "data" / "processed" / ".mesh_cache"
MATERIALS_PATH = BASE / "data" / "datasets" / "geometry" / "materials_train.jsonl"

# ── Thresholds ────────────────────────────────────────────────────────

MIN_SAMPLES = 100            # Minimum total samples
MIN_UNIQUE_LABELS = 50       # Minimum unique labels
MAX_LABEL_DOMINANCE = 0.10   # No single label > 10% of total
MIN_SOURCES = 2              # At least 2 data sources
MAX_SOURCE_DOMINANCE = 0.95  # No single source > 95% (BlendSwap ~89%)
MAX_BROKEN_PCT = 0.02        # Max 2% broken samples
MIN_MATERIALS = 50           # Minimum material samples (if enabled)
MIN_MEAN_TOKEN_LEN = 50      # Mean token length must be > 50
MAX_EMPTY_LABEL_PCT = 0.01   # Max 1% empty labels

# Known weak labels
WEAK_LABELS = {
    "object", "mesh", "thing", "model", "untitled", "cube", "sphere",
    "cylinder", "plane", "default", "shape", "3d object", "3d mesh",
    "multi object scene composition",
}


def validate_training_data(
    quick: bool = False,
    check_materials: bool = True,
) -> tuple[bool, dict]:
    """Validate training data quality.

    Returns:
        (passed, report_dict) — passed is True if all checks pass.
    """
    pt_files = sorted(CACHE_DIR.glob("*.pt"))
    if not pt_files:
        return False, {
            "passed": False,
            "error": f"No .pt files found in {CACHE_DIR}",
            "fix": "Run: python scripts/data_pipeline.py --local",
        }

    # Sample or check all
    if quick:
        sample_files = random.sample(pt_files, min(50, len(pt_files)))
    else:
        sample_files = pt_files

    total_samples = 0
    labels = []
    token_lens = []
    sources = Counter()
    broken = 0
    empty_labels = 0
    weak_labels = 0
    tiers = Counter()
    has_quality_tier = 0

    for f in sample_files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
            samples = data if isinstance(data, list) else [data]
            total_samples += len(samples)
            for s in samples:
                label = s.get("label", "")
                labels.append(label)

                tokens = s.get("mesh_tokens", s.get("tokens"))
                if tokens is not None:
                    tl = len(tokens) if hasattr(tokens, "__len__") else 0
                    token_lens.append(tl)
                    if isinstance(tokens, torch.Tensor):
                        if torch.isnan(tokens.float()).any():
                            broken += 1
                else:
                    broken += 1

                src = s.get("data_source", s.get("source", "unknown"))
                sources[src] += 1

                if not label.strip():
                    empty_labels += 1
                elif label.strip().lower() in WEAK_LABELS:
                    weak_labels += 1

                tier = s.get("quality_tier")
                if tier:
                    tiers[tier] += 1
                    has_quality_tier += 1
        except Exception:
            broken += 1

    # Scale up if we sampled
    scale = len(pt_files) / len(sample_files) if quick else 1.0
    est_total = int(total_samples * scale)

    label_counter = Counter(l.strip().lower() for l in labels if l.strip())
    unique_labels = len(label_counter)
    most_common_label, most_common_count = label_counter.most_common(1)[0] if label_counter else ("", 0)
    label_dominance = most_common_count / max(1, total_samples)

    most_common_source = sources.most_common(1)[0] if sources else ("unknown", 0)
    source_dominance = most_common_source[1] / max(1, total_samples)

    broken_pct = broken / max(1, total_samples)
    empty_pct = empty_labels / max(1, total_samples)
    weak_pct = weak_labels / max(1, total_samples)

    tl_arr = np.array(token_lens) if token_lens else np.array([0])
    mean_tokens = float(tl_arr.mean()) if len(tl_arr) > 0 else 0

    # Materials check
    materials_count = 0
    if check_materials and MATERIALS_PATH.exists():
        with open(MATERIALS_PATH) as f:
            materials_count = sum(1 for line in f if line.strip())

    # ── Run checks ────────────────────────────────────────────────
    issues = []

    if est_total < MIN_SAMPLES:
        issues.append(f"Too few samples: {est_total} < {MIN_SAMPLES}")

    if unique_labels < MIN_UNIQUE_LABELS:
        issues.append(f"Too few unique labels: {unique_labels} < {MIN_UNIQUE_LABELS}")

    if label_dominance > MAX_LABEL_DOMINANCE:
        issues.append(
            f"Label dominance too high: '{most_common_label}' is "
            f"{label_dominance:.1%} (max {MAX_LABEL_DOMINANCE:.0%})"
        )

    if len(sources) < MIN_SOURCES:
        issues.append(f"Too few sources: {len(sources)} < {MIN_SOURCES}")

    if source_dominance > MAX_SOURCE_DOMINANCE:
        issues.append(
            f"Source dominance too high: '{most_common_source[0]}' is "
            f"{source_dominance:.1%} (max {MAX_SOURCE_DOMINANCE:.0%})"
        )

    if broken_pct > MAX_BROKEN_PCT:
        issues.append(f"Too many broken samples: {broken_pct:.1%} (max {MAX_BROKEN_PCT:.0%})")

    if empty_pct > MAX_EMPTY_LABEL_PCT:
        issues.append(f"Too many empty labels: {empty_pct:.1%} (max {MAX_EMPTY_LABEL_PCT:.0%})")

    if mean_tokens < MIN_MEAN_TOKEN_LEN:
        issues.append(f"Mean token length too low: {mean_tokens:.0f} < {MIN_MEAN_TOKEN_LEN}")

    if check_materials and materials_count < MIN_MATERIALS:
        issues.append(f"Too few materials: {materials_count} < {MIN_MATERIALS}")

    passed = len(issues) == 0

    report = {
        "passed": passed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cache_files": len(pt_files),
        "samples_checked": total_samples,
        "estimated_total": est_total,
        "unique_labels": unique_labels,
        "sources": dict(sources),
        "broken_samples": broken,
        "broken_pct": round(broken_pct, 4),
        "empty_labels": empty_labels,
        "weak_labels": weak_labels,
        "label_dominance": round(label_dominance, 4),
        "label_dominance_label": most_common_label,
        "source_dominance": round(source_dominance, 4),
        "source_dominance_source": most_common_source[0],
        "mean_token_length": round(mean_tokens, 1),
        "materials_count": materials_count,
        "quality_tiers": dict(tiers) if tiers else "not_present",
        "has_quality_metadata": has_quality_tier > 0,
        "issues": issues,
        "top_10_labels": [
            {"label": lbl, "count": cnt}
            for lbl, cnt in label_counter.most_common(10)
        ],
    }

    if not passed:
        report["fix"] = (
            "Run: python scripts/enrich_training_data.py --apply\n"
            "This will score and weight all samples (zero data removed)."
        )

    return passed, report


def main():
    parser = argparse.ArgumentParser(description="Validate training data quality")
    parser.add_argument("--quick", action="store_true", help="Fast spot-check")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--no-materials", action="store_true", help="Skip materials check")
    args = parser.parse_args()

    passed, report = validate_training_data(
        quick=args.quick,
        check_materials=not args.no_materials,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=" * 60)
        print(f"TRAINING DATA QUALITY GATE: {'PASSED ✓' if passed else 'FAILED ✗'}")
        print("=" * 60)
        print(f"Cache files:      {report['cache_files']}")
        print(f"Total samples:    {report['estimated_total']}")
        print(f"Unique labels:    {report['unique_labels']}")
        print(f"Sources:          {len(report['sources'])}")
        print(f"Broken:           {report['broken_samples']} ({report['broken_pct']:.1%})")
        print(f"Empty labels:     {report['empty_labels']}")
        print(f"Weak labels:      {report['weak_labels']}")
        print(f"Mean token len:   {report['mean_token_length']:.0f}")
        print(f"Materials:        {report['materials_count']}")
        print(f"Quality tiers:    {report['quality_tiers']}")

        if report.get("issues"):
            print(f"\nISSUES ({len(report['issues'])}):")
            for issue in report["issues"]:
                print(f"  ✗ {issue}")
            print(f"\nFIX: {report.get('fix', '')}")
        else:
            print("\nAll checks passed. Training data is ready.")

        print(f"\nTop 10 labels:")
        for item in report.get("top_10_labels", []):
            print(f"  [{item['count']}x] {item['label']}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
