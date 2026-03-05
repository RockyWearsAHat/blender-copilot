#!/usr/bin/env python3
"""Mark raw sources as parsed-complete and optionally prune heavy raw binaries.

Default mode is dry-run. This script:
1) Verifies raw->processed coverage with a conservative pending estimate
2) Writes data/raw/<source>/.parsed_complete.json markers
3) Optionally deletes heavy raw binaries while preserving progress/metadata files

Usage:
  python scripts/mark_and_prune_raw.py --sources blendswap smutbase open3dlab objaverse --write-marker
  python scripts/mark_and_prune_raw.py --sources blendswap smutbase open3dlab objaverse --write-marker --prune --apply
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"

ALL_SOURCES = [
    "blender_official",
    "blendswap",
    "smutbase",
    "open3dlab",
    "github",
    "objaverse",
    "thingiverse",
    "sketchfab",
    "wikimedia",
    "objaverse_xl",
    "youtube",
]

RAW_MODEL_EXTS = {
    ".blend", ".glb", ".obj", ".stl", ".fbx", ".ply", ".abc", ".dae", ".usdz",
    ".zip", ".rar", ".7z", ".tar", ".gz",
}


def source_raw_dir(source: str) -> Path:
    return RAW / source


def iter_raw_models(raw_dir: Path):
    if not raw_dir.exists():
        return
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in RAW_MODEL_EXTS:
            yield p


def stems(paths: list[Path]) -> set[str]:
    return {p.stem for p in paths}


def coverage_stats(source: str) -> dict:
    raw_dir = source_raw_dir(source)
    proc_dir = PROC / source
    raw_files = list(iter_raw_models(raw_dir)) if raw_dir.exists() else []
    proc_json = list(proc_dir.glob("*.json")) if proc_dir.exists() else []
    proc_invalid = list(proc_dir.glob("*.invalid")) if proc_dir.exists() else []

    raw_stems = stems(raw_files)
    done_stems = stems(proc_json) | stems(proc_invalid)
    pending = sorted(raw_stems - done_stems)

    return {
        "source": source,
        "raw_dir_exists": raw_dir.exists(),
        "processed_dir_exists": proc_dir.exists(),
        "raw_count": len(raw_files),
        "processed_json_count": len(proc_json),
        "processed_invalid_count": len(proc_invalid),
        "pending_estimate": len(pending),
        "pending_sample": pending[:10],
    }


def write_marker(source: str, stats: dict, apply: bool):
    marker = source_raw_dir(source) / ".parsed_complete.json"
    payload = {
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "freeze_download": True,
        "raw_count": stats["raw_count"],
        "processed_json_count": stats["processed_json_count"],
        "processed_invalid_count": stats["processed_invalid_count"],
        "pending_estimate": stats["pending_estimate"],
        "note": "Raw parsing completed; pipeline should skip network downloads for this source unless explicitly overridden.",
    }
    if apply:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(payload, indent=2))
    return marker, payload


def prune_raw(source: str, apply: bool) -> tuple[int, int]:
    raw_dir = source_raw_dir(source)
    if not raw_dir.exists():
        return 0, 0

    removed = 0
    bytes_freed = 0
    for f in iter_raw_models(raw_dir):
        try:
            size = f.stat().st_size
        except Exception:
            size = 0
        if apply:
            try:
                f.unlink(missing_ok=True)
            except Exception:
                continue
        removed += 1
        bytes_freed += size

    if apply:
        for d in sorted(raw_dir.rglob("*"), reverse=True):
            if d.is_dir():
                try:
                    next(d.iterdir())
                except StopIteration:
                    try:
                        d.rmdir()
                    except Exception:
                        pass
                except Exception:
                    pass

    return removed, bytes_freed


def fmt_gb(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB"


def main():
    parser = argparse.ArgumentParser(description="Mark parsed raw sources and optionally prune raw binaries")
    parser.add_argument("--sources", nargs="*", default=["blendswap", "smutbase", "open3dlab", "objaverse"])
    parser.add_argument("--write-marker", action="store_true", help="Write .parsed_complete.json markers")
    parser.add_argument("--prune", action="store_true", help="Delete heavy raw model/archive files")
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow marker/prune even if pending_estimate > 0")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run)")
    args = parser.parse_args()

    chosen = [s for s in args.sources if s in ALL_SOURCES]
    if not chosen:
        print("No valid sources selected")
        return

    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Sources: {chosen}")

    total_removed = 0
    total_bytes = 0

    for source in chosen:
        stats = coverage_stats(source)
        print(f"\n[{source}] raw={stats['raw_count']} processed={stats['processed_json_count']} invalid={stats['processed_invalid_count']} pending≈{stats['pending_estimate']}")
        if stats["pending_estimate"] > 0 and not args.allow_incomplete:
            print("  skip: pending items detected (use --allow-incomplete to override)")
            if stats["pending_sample"]:
                print(f"  sample pending: {stats['pending_sample']}")
            continue

        if args.write_marker:
            marker, _ = write_marker(source, stats, apply=args.apply)
            print(f"  marker: {marker}")

        if args.prune:
            removed, bytes_freed = prune_raw(source, apply=args.apply)
            total_removed += removed
            total_bytes += bytes_freed
            print(f"  prune: files={removed} reclaim≈{fmt_gb(bytes_freed)}")

    if args.prune:
        print(f"\nTOTAL prune: files={total_removed} reclaim≈{fmt_gb(total_bytes)}")


if __name__ == "__main__":
    main()
