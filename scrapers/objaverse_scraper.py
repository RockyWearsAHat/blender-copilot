"""Download annotated 3D models from Objaverse via HuggingFace.

Objaverse is a massive open dataset of 800K+ 3D objects with text annotations,
CC licensing, and GLB format. This is the PRIMARY data source for training
because:
  - No authentication required
  - Python API for easy batch download
  - Text annotations already paired with 3D models
  - Permissive licensing (CC-BY, CC-0)

Usage:
    python -m scrapers.objaverse_scraper --output data/raw/objaverse
    python -m scrapers.objaverse_scraper --output data/raw/objaverse --max 10000
    python -m scrapers.objaverse_scraper --output data/raw/objaverse --categories vehicle,furniture
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

from .utils import setup_logging, load_config, ensure_dir, save_metadata

logger = logging.getLogger(__name__)


def download_objaverse_models(output_dir: Path, config: dict,
                               max_models: int = 50000,
                               categories: list[str] | None = None):
    """Download 3D models from Objaverse using the objaverse Python package.

    Downloads GLB files and their annotations. Uses objaverse's built-in
    multiprocess downloading for speed.
    """
    try:
        import objaverse
    except ImportError:
        logger.error("objaverse package not installed. Run: pip install objaverse")
        return

    ensure_dir(output_dir)
    models_dir = ensure_dir(output_dir / "models")
    metadata_dir = ensure_dir(output_dir / "metadata")

    # Load progress to skip already downloaded
    progress_file = output_dir / ".progress.json"
    downloaded = set()
    if progress_file.exists():
        with open(progress_file) as f:
            downloaded = set(json.load(f))
    logger.info(f"Resuming with {len(downloaded)} already downloaded models")

    # Get annotations (text descriptions, categories, tags)
    logger.info("Loading Objaverse annotations...")
    annotations = objaverse.load_annotations()
    logger.info(f"Found {len(annotations)} annotated objects")

    # Filter by categories if specified
    if categories:
        cat_set = set(c.lower() for c in categories)
        filtered = {}
        for uid, ann in annotations.items():
            tags = [t.get("name", "").lower() for t in ann.get("tags", [])]
            cats = [c.get("name", "").lower() for c in ann.get("categories", [])]
            if any(t in cat_set or c in cat_set for t in tags for c in cats):
                filtered[uid] = ann
            elif any(cat in ann.get("name", "").lower() for cat in cat_set):
                filtered[uid] = ann
        logger.info(f"Filtered to {len(filtered)} objects matching categories: {categories}")
        annotations = filtered

    # Sort by download count (popular models tend to be higher quality)
    sorted_uids = sorted(
        annotations.keys(),
        key=lambda uid: annotations[uid].get("downloadCount", 0),
        reverse=True,
    )

    # Filter out already downloaded
    to_download = [uid for uid in sorted_uids if uid not in downloaded]
    to_download = to_download[:max_models]
    logger.info(f"Will download {len(to_download)} models (skipping {len(downloaded)} done)")

    if not to_download:
        logger.info("Nothing to download!")
        return

    # Download in batches using objaverse's built-in downloader
    batch_size = 100
    total_downloaded = 0

    for i in range(0, len(to_download), batch_size):
        batch_uids = to_download[i:i + batch_size]
        logger.info(f"Downloading batch {i // batch_size + 1} "
                     f"({i + 1}-{min(i + batch_size, len(to_download))} "
                     f"of {len(to_download)})")

        try:
            # objaverse.load_objects returns {uid: local_path}
            objects = objaverse.load_objects(
                uids=batch_uids,
                download_processes=4,
            )

            for uid, local_path in objects.items():
                if local_path is None:
                    continue

                # Copy/link the file to our output directory
                src = Path(local_path)
                if not src.exists():
                    continue

                dst = models_dir / f"{uid}.glb"
                if not dst.exists():
                    # Symlink to avoid duplication (objaverse caches internally)
                    try:
                        dst.symlink_to(src.resolve())
                    except OSError:
                        # Fallback to copy if symlinks not supported
                        import shutil
                        shutil.copy2(src, dst)

                # Save metadata
                ann = annotations.get(uid, {})
                meta = {
                    "source": "objaverse",
                    "uid": uid,
                    "name": ann.get("name", ""),
                    "description": ann.get("description", ""),
                    "tags": [t.get("name", "") for t in ann.get("tags", [])],
                    "categories": [c.get("name", "") for c in ann.get("categories", [])],
                    "license": ann.get("license", ""),
                    "download_count": ann.get("downloadCount", 0),
                    "like_count": ann.get("likeCount", 0),
                    "animation_count": ann.get("animationCount", 0),
                    "vertex_count": ann.get("vertexCount", 0),
                    "face_count": ann.get("faceCount", 0),
                    "format": "glb",
                }
                save_metadata(metadata_dir, uid, meta)

                downloaded.add(uid)
                total_downloaded += 1

        except Exception as e:
            logger.error(f"Batch download failed: {e}")
            continue

        # Save progress after each batch
        with open(progress_file, "w") as f:
            json.dump(list(downloaded), f)

        logger.info(f"Progress: {total_downloaded} downloaded in this run, "
                     f"{len(downloaded)} total")

    logger.info(f"Objaverse download complete. {total_downloaded} new models downloaded.")


def main():
    parser = argparse.ArgumentParser(
        description="Download 3D models from Objaverse (800K+ CC-licensed models)")
    parser.add_argument("--output", default="data/raw/objaverse",
                        help="Output directory for downloaded models")
    parser.add_argument("--max", type=int, default=None,
                        help="Max models to download (default: from config)")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated categories to filter by")
    args = parser.parse_args()

    setup_logging("objaverse")
    config = load_config()

    max_models = args.max or config.get("scraping", {}).get("objaverse", {}).get(
        "max_models", 50000)
    categories = args.categories.split(",") if args.categories else None

    output_dir = ensure_dir(args.output)
    download_objaverse_models(output_dir, config, max_models, categories)


if __name__ == "__main__":
    main()
