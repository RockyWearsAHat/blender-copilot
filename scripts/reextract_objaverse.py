#!/usr/bin/env python3
"""Re-extract objaverse items through Blender for full material data.

The original objaverse items were extracted via mesh_extractor.py (trimesh),
which only captures geometry — no materials, no UVs, no face_material_indices.

This script:
1. Downloads the original GLB files from objaverse (via the objaverse API)
2. Re-extracts each through blend_extractor.py (Blender headless)
3. Replaces the existing impoverished JSON in data/processed/objaverse/

The Blender extractor captures:
  - Full material node trees (Principled BSDF, textures, etc.)
  - Per-face material indices
  - UV coordinates (all layers)
  - Vertex colors
  - Normals, smooth flags
  - Modifiers, shape keys, armatures
  - Image thumbnails (base64, 512px)

Requirements:
  - Blender 4.x installed and available as 'blender' in PATH
    (or set BLENDER_PATH env var)
  - objaverse pip package: pip install objaverse
  - ~50 GB free disk space for temp GLB files (cleaned up after)

Usage:
    python scripts/reextract_objaverse.py                  # Re-extract all 442
    python scripts/reextract_objaverse.py --limit 10       # First 10 only
    python scripts/reextract_objaverse.py --dry-run        # Preview only
    python scripts/reextract_objaverse.py --uid <uid>      # Specific item
    python scripts/reextract_objaverse.py --skip-existing  # Only missing-material items
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
OBJAVERSE_JSON_DIR = BASE / "data" / "processed" / "objaverse"
BLEND_EXTRACTOR = BASE / "processing" / "blend_extractor.py"

# Where to cache downloaded GLBs
TEMP_GLB_DIR = BASE / "data" / "raw" / "objaverse" / "_reextract_cache"


def _find_blender() -> str:
    """Find the Blender executable."""
    env_path = os.environ.get("BLENDER_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    # Common macOS locations
    candidates = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/Applications/Blender.app/Contents/MacOS/blender",
        shutil.which("blender"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c

    raise RuntimeError(
        "Blender not found. Install Blender and either:\n"
        "  - Add it to PATH\n"
        "  - Set BLENDER_PATH environment variable\n"
        "  - Install at /Applications/Blender.app (macOS)"
    )


def _download_glb(uid: str, output_dir: Path) -> Path | None:
    """Download a single GLB from objaverse by UID."""
    try:
        import objaverse
    except ImportError:
        logger.error("objaverse not installed. Run: pip install objaverse")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already cached
    cached = output_dir / f"{uid}.glb"
    if cached.exists():
        return cached

    try:
        objects = objaverse.load_objects(
            uids=[uid],
            download_processes=1,
        )
        if uid in objects:
            src_path = Path(objects[uid])
            if src_path.exists():
                shutil.copy2(src_path, cached)
                return cached
    except Exception as e:
        logger.warning(f"Failed to download {uid}: {e}")

    return None


def _extract_via_blender(glb_path: Path, output_dir: Path,
                         blender_path: str, uid: str,
                         metadata: dict) -> bool:
    """Run blend_extractor.py via Blender headless to extract full data."""
    output_json = output_dir / f"{uid}.json"

    cmd = [
        blender_path,
        "--background",
        "--python", str(BLEND_EXTRACTOR),
        "--",
        "--input", str(glb_path),
        "--output", str(output_dir),
        "--filename", f"{uid}.json",
    ]

    # Pass metadata if blend_extractor supports it
    if metadata:
        meta_file = output_dir / f".{uid}_meta.json"
        with open(meta_file, "w") as f:
            json.dump(metadata, f)
        cmd.extend(["--metadata", str(meta_file)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout per file
        )
        if result.returncode != 0:
            logger.warning(f"Blender extraction failed for {uid}: {result.stderr[:200]}")
            return False

        # Verify the output exists and has materials
        if output_json.exists():
            try:
                data = json.load(open(output_json))
                n_objects = len(data.get("objects", []))
                has_mats = any(
                    bool(o.get("materials"))
                    for o in data.get("objects", [])
                )
                n_faces = sum(
                    len(o.get("mesh", {}).get("faces", []))
                    for o in data.get("objects", [])
                )
                logger.info(
                    f"  {uid}: {n_objects} objects, {n_faces} faces, "
                    f"materials={'YES' if has_mats else 'NO'}"
                )
                return True
            except Exception as e:
                logger.warning(f"Output JSON invalid for {uid}: {e}")
                return False
        else:
            logger.warning(f"No output JSON for {uid}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Blender timed out for {uid}")
        return False
    except Exception as e:
        logger.warning(f"Extraction error for {uid}: {e}")
        return False


def _get_uid_from_json(json_path: Path) -> str:
    """Extract the objaverse UID from a JSON filename."""
    return json_path.stem


def _has_poor_materials(json_path: Path) -> bool:
    """Check if a JSON was extracted by trimesh (poor materials)."""
    try:
        data = json.load(open(json_path))
        for obj in data.get("objects", []):
            mesh = obj.get("mesh", {})
            # Trimesh extraction: no FMI, no UV layers, no face_smooth
            if not mesh.get("face_material_indices") and not mesh.get("uv_layers"):
                return True
        return False
    except Exception:
        return True


def reextract(limit: int | None = None,
              dry_run: bool = False,
              skip_existing: bool = False,
              specific_uid: str | None = None) -> dict:
    """Main re-extraction loop."""
    blender_path = _find_blender()
    logger.info(f"Using Blender at: {blender_path}")

    # Collect targets
    if specific_uid:
        targets = [OBJAVERSE_JSON_DIR / f"{specific_uid}.json"]
        targets = [t for t in targets if t.exists()]
        if not targets:
            logger.error(f"No JSON found for UID {specific_uid}")
            return {}
    else:
        targets = sorted(OBJAVERSE_JSON_DIR.glob("*.json"))

    if skip_existing:
        targets = [t for t in targets if _has_poor_materials(t)]
        logger.info(f"Filtered to {len(targets)} items with poor materials")

    if limit is not None:
        targets = targets[:limit]

    logger.info(f"Re-extracting {len(targets)} objaverse items")

    stats = {
        "total": len(targets),
        "downloaded": 0,
        "extracted": 0,
        "failed_download": 0,
        "failed_extract": 0,
        "skipped": 0,
    }

    if dry_run:
        for t in targets[:20]:
            uid = _get_uid_from_json(t)
            poor = _has_poor_materials(t)
            logger.info(f"  Would re-extract: {uid} (poor_materials={poor})")
        if len(targets) > 20:
            logger.info(f"  ... and {len(targets) - 20} more")
        return stats

    # Load objaverse metadata for enrichment
    try:
        import objaverse
        annotations = objaverse.load_annotations()
        logger.info(f"Loaded {len(annotations)} objaverse annotations")
    except Exception:
        annotations = {}

    t0 = time.time()
    TEMP_GLB_DIR.mkdir(parents=True, exist_ok=True)

    for idx, json_path in enumerate(targets):
        uid = _get_uid_from_json(json_path)

        # Download GLB
        glb_path = _download_glb(uid, TEMP_GLB_DIR)
        if glb_path is None:
            stats["failed_download"] += 1
            continue
        stats["downloaded"] += 1

        # Get metadata
        meta = annotations.get(uid, {})

        # Backup original JSON
        backup_path = json_path.with_suffix(".json.bak")
        if not backup_path.exists():
            shutil.copy2(json_path, backup_path)

        # Extract via Blender
        success = _extract_via_blender(
            glb_path, OBJAVERSE_JSON_DIR, blender_path, uid, meta
        )
        if success:
            stats["extracted"] += 1
        else:
            stats["failed_extract"] += 1
            # Restore backup on failure
            if backup_path.exists():
                shutil.copy2(backup_path, json_path)

        # Progress
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(targets) - idx - 1) / rate
            logger.info(
                f"[{idx + 1}/{len(targets)}] "
                f"extracted={stats['extracted']}, "
                f"failed={stats['failed_download'] + stats['failed_extract']}, "
                f"ETA={eta:.0f}s"
            )

        # Clean up GLB to save space (keep cache for retries)
        # Uncomment to save disk: glb_path.unlink(missing_ok=True)

    elapsed = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"Re-extraction complete in {elapsed:.0f}s")
    logger.info(f"  Downloaded: {stats['downloaded']}/{stats['total']}")
    logger.info(f"  Extracted:  {stats['extracted']}/{stats['total']}")
    logger.info(f"  Failed DL:  {stats['failed_download']}")
    logger.info(f"  Failed ext: {stats['failed_extract']}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Re-extract objaverse items through Blender for full material data")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max items to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without downloading/extracting")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Only re-extract items with poor (trimesh) extraction")
    parser.add_argument("--uid", type=str, default=None,
                        help="Process a specific objaverse UID")
    args = parser.parse_args()

    reextract(
        limit=args.limit,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        specific_uid=args.uid,
    )


if __name__ == "__main__":
    main()
