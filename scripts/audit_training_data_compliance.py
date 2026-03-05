#!/usr/bin/env python3
"""Strict data compliance audit for geometry training.

Checks:
1) train.jsonl source composition (synthetic vs non-synthetic)
2) processed real-source JSON coverage in .mesh_cache (path-hash compatible)
3) car keyword density in processed real-source metadata

Exits non-zero when thresholds are not met.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEO_DIR = ROOT / "data" / "datasets" / "geometry"
PROC_DIR = ROOT / "data" / "processed"
CACHE_DIR = PROC_DIR / ".mesh_cache"

REAL_SOURCES = [
    "objaverse", "blendswap", "blender_official",
    "smutbase", "open3dlab", "github", "youtube",
]

CAR_RE = re.compile(
    r"\b(car|vehicle|automobile|supercar|sports car|race car|sedan|coupe|suv|truck|van|bus|lamborghini|ferrari|porsche|tesla|mustang|bmw|mercedes|audi|nissan|toyota)\b",
    re.IGNORECASE,
)


def _audit_train_jsonl(train_path: Path) -> dict:
    src_counts = Counter()
    total = 0
    if not train_path.exists():
        return {"total": 0, "src_counts": src_counts, "real_pct": 0.0}

    with train_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            src = str(d.get("source", "unknown"))
            src_counts[src] += 1
            total += 1

    real = sum(v for k, v in src_counts.items() if not k.startswith("synthetic_"))
    real_pct = (100.0 * real / total) if total else 0.0
    return {
        "total": total,
        "src_counts": src_counts,
        "real_pct": real_pct,
    }


def _audit_cache_coverage() -> dict:
    source_stats: dict[str, tuple[int, int]] = {}
    for src in REAL_SOURCES:
        src_dir = PROC_DIR / src
        if not src_dir.exists():
            source_stats[src] = (0, 0)
            continue

        files = [p for p in src_dir.glob("*.json") if not p.name.endswith(".meta.json")]
        hits = 0
        for p in files:
            key = hashlib.md5(str(p).encode()).hexdigest()[:16]
            cp = CACHE_DIR / f"{key}.pt"
            if cp.exists() and cp.stat().st_size > 200:
                hits += 1
        source_stats[src] = (len(files), hits)

    total_json = sum(v[0] for v in source_stats.values())
    total_hits = sum(v[1] for v in source_stats.values())
    coverage = (100.0 * total_hits / total_json) if total_json else 0.0
    return {
        "per_source": source_stats,
        "total_json": total_json,
        "total_hits": total_hits,
        "coverage_pct": coverage,
    }


def _audit_car_density() -> dict:
    total = 0
    car_hits = 0
    per_source = {}

    for src in REAL_SOURCES:
        src_dir = PROC_DIR / src
        files = [p for p in src_dir.glob("*.json") if not p.name.endswith(".meta.json")] if src_dir.exists() else []
        src_total = 0
        src_car = 0
        for p in files:
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            md = d.get("metadata", {}) if isinstance(d, dict) else {}
            fields = [
                str(d.get("label", "")),
                str(md.get("name", "")),
                str(md.get("description", "")),
                " ".join(map(str, md.get("tags", []) or [])),
                " ".join(map(str, md.get("categories", []) or [])),
            ]
            text = " ".join(fields)
            if CAR_RE.search(text):
                src_car += 1
            src_total += 1
        per_source[src] = (src_total, src_car)
        total += src_total
        car_hits += src_car

    car_pct = (100.0 * car_hits / total) if total else 0.0
    return {
        "per_source": per_source,
        "total": total,
        "car_hits": car_hits,
        "car_pct": car_pct,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit geometry data compliance")
    parser.add_argument("--min-real-pct", type=float, default=5.0,
                        help="Minimum non-synthetic share in train.jsonl")
    parser.add_argument("--min-cache-coverage", type=float, default=90.0,
                        help="Minimum %% of processed real JSONs covered by cache")
    parser.add_argument("--min-car-pct", type=float, default=10.0,
                        help="Minimum %% of car-related samples in processed real JSONs")
    args = parser.parse_args()

    train = _audit_train_jsonl(GEO_DIR / "train.jsonl")
    cache = _audit_cache_coverage()
    cars = _audit_car_density()

    print("=== Data Compliance Audit ===")
    print(f"train.jsonl samples: {train['total']}")
    print(f"train.jsonl real (non-synthetic): {train['real_pct']:.2f}%")
    print(f"cache coverage (real JSON -> .pt): {cache['coverage_pct']:.2f}% ({cache['total_hits']}/{cache['total_json']})")
    print(f"car density (processed real JSON): {cars['car_pct']:.2f}% ({cars['car_hits']}/{cars['total']})")

    print("\n-- cache per source --")
    for src, (n, h) in cache["per_source"].items():
        pct = (100.0 * h / n) if n else 0.0
        print(f"{src:16s} json={n:5d} cache={h:5d} ({pct:6.2f}%)")

    print("\n-- car density per source --")
    for src, (n, c) in cars["per_source"].items():
        pct = (100.0 * c / n) if n else 0.0
        print(f"{src:16s} total={n:5d} car={c:5d} ({pct:6.2f}%)")

    failures = []
    if train["real_pct"] < args.min_real_pct:
        failures.append(
            f"train.jsonl real_pct {train['real_pct']:.2f} < {args.min_real_pct:.2f}"
        )
    if cache["coverage_pct"] < args.min_cache_coverage:
        failures.append(
            f"cache coverage {cache['coverage_pct']:.2f} < {args.min_cache_coverage:.2f}"
        )
    if cars["car_pct"] < args.min_car_pct:
        failures.append(
            f"car density {cars['car_pct']:.2f} < {args.min_car_pct:.2f}"
        )

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(" -", f)
        return 2

    print("\nPASS: compliance thresholds met")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
