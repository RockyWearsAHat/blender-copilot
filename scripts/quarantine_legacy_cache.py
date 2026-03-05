#!/usr/bin/env python3
"""Quarantine legacy cache entries and keep only modern high-quality samples.

Default mode is dry-run. Use --apply to move legacy .pt files into a quarantine
folder, preserving recoverability.

"Truly new" criteria (customizable):
- non-empty mesh_tokens
- label is not generic/vague
- includes v5 fields: label_confidence, topology_quality, original_face_count
- includes v6 fields: composition, scene_complexity_score, workflow_supervision
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch


GENERIC_LABELS = {
    "object",
    "3d object",
    "rendered asset",
    "detailed mesh",
    "game model",
    "model",
    "item",
    "thing",
    "untitled",
    "default",
}

V5_FIELDS = ("label_confidence", "topology_quality", "original_face_count")
V6_FIELDS = ("composition", "scene_complexity_score", "workflow_supervision")


def _iter_samples(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [s for s in data if isinstance(s, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _has_nonempty_mesh_tokens(sample: dict[str, Any]) -> bool:
    mesh_tokens = sample.get("mesh_tokens")
    return isinstance(mesh_tokens, torch.Tensor) and mesh_tokens.numel() > 0


def _label_is_generic(sample: dict[str, Any]) -> bool:
    label = str(sample.get("label", "")).strip().lower()
    return label in GENERIC_LABELS or not label


def _has_fields(sample: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return all(field in sample for field in fields)


def classify_file(pt_path: Path, require_v5: bool, require_v6: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, [f"load_error:{type(exc).__name__}"]

    samples = _iter_samples(data)
    if not samples:
        return False, ["no_samples"]

    for sample in samples:
        if not _has_nonempty_mesh_tokens(sample):
            reasons.append("empty_mesh_tokens")
            break

    for sample in samples:
        if _label_is_generic(sample):
            reasons.append("generic_or_empty_label")
            break

    if require_v5:
        for sample in samples:
            if not _has_fields(sample, V5_FIELDS):
                reasons.append("missing_v5_fields")
                break

    if require_v6:
        for sample in samples:
            if not _has_fields(sample, V6_FIELDS):
                reasons.append("missing_v6_fields")
                break

    keep = not reasons
    return keep, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description="Quarantine legacy cache .pt files")
    parser.add_argument("--cache-dir", default="data/processed/.mesh_cache")
    parser.add_argument(
        "--quarantine-dir",
        default="data/processed/.mesh_cache_quarantine",
        help="Directory where legacy files are moved",
    )
    parser.add_argument("--apply", action="store_true", help="Move files instead of dry-run")
    parser.add_argument(
        "--require-v5",
        action="store_true",
        help="Require v5 cache fields (label_confidence/topology_quality/original_face_count)",
    )
    parser.add_argument(
        "--require-v6",
        action="store_true",
        help="Require v6 cache fields (composition/scene_complexity_score/workflow_supervision)",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of files processed")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    quarantine_dir = Path(args.quarantine_dir)

    files = sorted(cache_dir.glob("*.pt"))
    if args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        print(f"No .pt files found in {cache_dir}")
        return 1

    keep_count = 0
    quarantine_count = 0
    reason_counts: dict[str, int] = {}

    if args.apply:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    for pt_path in files:
        keep, reasons = classify_file(pt_path, args.require_v5, args.require_v6)
        if keep:
            keep_count += 1
            continue

        quarantine_count += 1
        for reason in sorted(set(reasons)):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        if args.apply:
            dst = quarantine_dir / pt_path.name
            shutil.move(str(pt_path), str(dst))

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] scanned_files={len(files)}")
    print(f"[{mode}] keep={keep_count}")
    print(f"[{mode}] quarantine={quarantine_count}")
    if reason_counts:
        print("Reasons:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {reason}: {count}")

    if args.apply:
        print(f"Moved legacy files to: {quarantine_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
