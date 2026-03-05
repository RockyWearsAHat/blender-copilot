"""Download 3D models from Objaverse-XL — ALL sources, NO caps.

Objaverse-XL contains 10M+ 3D objects across 4 sources:
  - Sketchfab:    ~800K  GLB files (via HuggingFace CDN)
  - GitHub:       ~5.2M  objects (.blend, .glb, .obj, .stl, .fbx, .ply, etc.)
  - Thingiverse:  ~3.7M  STL files
  - Smithsonian:  ~2.4K  GLB files

This scraper downloads AS MUCH AS POSSIBLE with aggressive dedup:
  - SHA-256 based dedup from Objaverse-XL metadata
  - File-on-disk dedup (skip already downloaded)
  - Cross-source dedup via global hash registry

Usage:
    python -m scrapers.objaverse_scraper --output data/raw/objaverse
    python -m scrapers.objaverse_scraper --source sketchfab
    python -m scrapers.objaverse_scraper --source github --file-types blend,glb,obj
"""

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from .utils import setup_logging, load_config, ensure_dir, save_metadata

logger = logging.getLogger(__name__)

# File types our pipeline can process (mesh_extractor + blend_extractor)
PROCESSABLE_TYPES = {"glb", "gltf", "obj", "stl", "ply", "blend", "off", "3ds"}

# Cached Objaverse v1 annotations (name, description, tags, categories)
_v1_annotations = None


def _get_v1_annotations() -> dict:
    """Load Objaverse v1 annotations (cached). 798K Sketchfab entries with
    rich text: name, description, tags, categories."""
    global _v1_annotations
    if _v1_annotations is None:
        try:
            import objaverse
            _v1_annotations = objaverse.load_annotations()
            logger.info(f"Loaded {len(_v1_annotations)} Objaverse v1 annotations")
        except Exception as e:
            logger.warning(f"Could not load v1 annotations: {e}")
            _v1_annotations = {}
    return _v1_annotations


def _enrich_meta_from_v1(meta: dict, uid: str) -> dict:
    """Enrich metadata dict with name/description/tags/categories from v1."""
    v1 = _get_v1_annotations()
    ann = v1.get(uid)
    if not ann:
        return meta
    if ann.get("name"):
        meta["name"] = ann["name"]
    if ann.get("description"):
        meta["description"] = ann["description"][:500]
    if ann.get("tags"):
        meta["tags"] = [t["name"] if isinstance(t, dict) else t
                        for t in ann["tags"][:10]]
    if ann.get("categories"):
        meta["categories"] = [c["name"] if isinstance(c, dict) else c
                              for c in ann["categories"][:5]]
    return meta

# Priority order for sources (best ROI first)
SOURCE_PRIORITY = ["sketchfab", "smithsonian", "github", "thingiverse"]

# File types worth downloading from each source
SOURCE_FILE_FILTERS = {
    "sketchfab": None,  # All GLBs, no filtering needed
    "github": {"glb", "gltf", "obj", "stl", "ply", "blend"},
    "thingiverse": {"stl", "obj"},
    "smithsonian": None,  # All GLBs
}


def _load_hash_registry(output_dir: Path) -> set:
    """Load the set of SHA-256 hashes we've already downloaded."""
    registry_file = output_dir / ".hash_registry.json"
    if registry_file.exists():
        try:
            with open(registry_file) as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def _save_hash_registry(output_dir: Path, hashes: set):
    """Persist the hash registry."""
    registry_file = output_dir / ".hash_registry.json"
    tmp = registry_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(list(hashes), f)
    tmp.rename(registry_file)


def _load_progress(output_dir: Path) -> set:
    """Load set of already-processed file identifiers."""
    progress_file = output_dir / ".progress.json"
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def _save_progress(output_dir: Path, progress: set):
    """Persist progress (set of processed file identifiers)."""
    progress_file = output_dir / ".progress.json"
    tmp = progress_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(list(progress), f)
    tmp.rename(progress_file)


def _get_uid_from_identifier(file_identifier: str, source: str) -> str:
    """Extract a clean UID from a file identifier."""
    if source == "sketchfab":
        parts = file_identifier.rstrip("/").split("/")
        for part in reversed(parts):
            if len(part) > 10:
                return part[:32]
        return file_identifier[-32:]
    elif source == "github":
        return file_identifier.replace("/", "_").replace(":", "_")[-80:]
    elif source == "thingiverse":
        return file_identifier.rstrip("/").split("/")[-1]
    elif source == "smithsonian":
        parts = file_identifier.rstrip("/").split("/")
        return parts[-1].replace(".glb", "") if parts else file_identifier[-32:]
    return file_identifier[-32:]


def download_source(
    source_name: str,
    output_dir: str | Path,
    max_objects: int | None = None,
    file_types: set | None = None,
    processes: int = 4,
    batch_offset: int = 0,
):
    """Download objects from a single Objaverse-XL source.

    Args:
        source_name: One of 'sketchfab', 'github', 'thingiverse', 'smithsonian'
        output_dir: Base output directory (e.g. data/raw/objaverse)
        max_objects: Max objects to download (None = unlimited)
        file_types: Set of file extensions to filter by (e.g. {'glb', 'blend'})
        processes: Number of parallel download processes
        batch_offset: Skip this many objects (for resuming)
    """
    try:
        import objaverse.xl as oxl
    except ImportError as e:
        logger.error(f"objaverse import failed: {e}. Run: pip install objaverse")
        return 0
    except Exception as e:
        logger.error(f"objaverse.xl initialization failed: {e}")
        return 0

    output_dir = Path(output_dir)
    source_dir = ensure_dir(output_dir / source_name)
    models_dir = ensure_dir(source_dir / "models")
    metadata_dir = ensure_dir(source_dir / "metadata")

    progress = _load_progress(source_dir)
    hash_registry = _load_hash_registry(output_dir)

    logger.info(f"=== Downloading from {source_name.upper()} ===")
    logger.info(f"  Output: {source_dir}")
    logger.info(f"  Already processed: {len(progress)} objects")
    logger.info(f"  Known hashes: {len(hash_registry)}")

    downloader = oxl.downloaders[source_name]
    logger.info(f"  Loading {source_name} annotations...")
    annotations = downloader.get_annotations()
    logger.info(f"  Total available: {len(annotations)} objects")

    # Filter by file type if specified
    type_filter = file_types or SOURCE_FILE_FILTERS.get(source_name)
    if type_filter:
        annotations = annotations[annotations["fileType"].isin(type_filter)]
        logger.info(
            f"  After file type filter ({type_filter}): "
            f"{len(annotations)} objects"
        )

    # Filter out already-processed
    not_processed = annotations[
        ~annotations["fileIdentifier"].isin(progress)
    ]
    logger.info(f"  Not yet processed: {len(not_processed)} objects")

    # Filter out known hashes (cross-source dedup)
    if hash_registry:
        not_duped = not_processed[
            ~not_processed["sha256"].isin(hash_registry)
        ]
        dupes_skipped = len(not_processed) - len(not_duped)
        if dupes_skipped > 0:
            logger.info(f"  Skipping {dupes_skipped} duplicates (hash match)")
        not_processed = not_duped

    # Apply offset for batching
    if batch_offset > 0:
        not_processed = not_processed.iloc[batch_offset:]

    # Apply max if set
    if max_objects is not None:
        not_processed = not_processed.head(max_objects)

    if len(not_processed) == 0:
        logger.info(f"  Nothing new to download from {source_name}!")
        return 0

    logger.info(f"  Downloading {len(not_processed)} objects...")

    # Download using objaverse-xl API
    downloaded_count = 0
    batch_size = min(500, len(not_processed))

    for batch_start in range(0, len(not_processed), batch_size):
        batch = not_processed.iloc[batch_start:batch_start + batch_size]
        logger.info(
            f"  Batch {batch_start // batch_size + 1}: "
            f"objects {batch_start + 1}-"
            f"{min(batch_start + batch_size, len(not_processed))} "
            f"of {len(not_processed)}"
        )

        try:
            results = downloader.download_objects(
                objects=batch,
                download_dir=str(output_dir / ".objaverse_cache"),
                processes=min(processes, len(batch)),
            )

            for file_id, local_path in results.items():
                if local_path is None or not Path(local_path).exists():
                    progress.add(file_id)
                    continue

                row = batch[batch["fileIdentifier"] == file_id]
                if row.empty:
                    continue

                sha256 = row.iloc[0].get("sha256", "")
                file_type = row.iloc[0].get("fileType", "glb")

                # Check hash dedup
                if sha256 and sha256 in hash_registry:
                    progress.add(file_id)
                    continue

                uid = _get_uid_from_identifier(file_id, source_name)
                safe_name = "".join(
                    c if c.isalnum() or c in "-_." else "_"
                    for c in uid
                )
                dst = models_dir / f"{safe_name}.{file_type}"

                if not dst.exists():
                    try:
                        shutil.copy2(local_path, dst)
                    except Exception as e:
                        logger.debug(f"  Copy failed for {uid}: {e}")
                        progress.add(file_id)
                        continue

                # Save metadata
                meta = {
                    "source": f"objaverse_xl_{source_name}",
                    "file_identifier": file_id,
                    "sha256": sha256,
                    "file_type": file_type,
                    "uid": uid,
                    "license": row.iloc[0].get("license", ""),
                }
                try:
                    raw_metadata = row.iloc[0].get("metadata", "{}")
                    if isinstance(raw_metadata, str):
                        extra = json.loads(raw_metadata)
                    elif isinstance(raw_metadata, dict):
                        extra = raw_metadata
                    else:
                        extra = {}
                    meta.update(extra)
                except Exception:
                    pass

                # Enrich with v1 annotations (name, desc, tags, categories)
                if source_name == "sketchfab":
                    meta = _enrich_meta_from_v1(meta, uid)

                save_metadata(str(metadata_dir), safe_name, meta)

                if sha256:
                    hash_registry.add(sha256)
                progress.add(file_id)
                downloaded_count += 1

        except Exception as e:
            logger.error(f"  Batch download failed: {e}")

        # Save progress after each batch
        _save_progress(source_dir, progress)
        _save_hash_registry(output_dir, hash_registry)
        logger.info(
            f"  Progress: {downloaded_count} new downloads, "
            f"{len(progress)} total processed"
        )

    logger.info(
        f"=== {source_name.upper()} complete: "
        f"{downloaded_count} new, {len(progress)} total processed ==="
    )
    return downloaded_count


def download_all_sources(
    output_dir: str | Path = "data/raw/objaverse",
    max_per_source: dict | None = None,
    processes: int = 4,
):
    """Download from ALL Objaverse-XL sources in priority order.

    Args:
        output_dir: Base output directory
        max_per_source: Optional dict of source_name -> max_objects.
                       None means unlimited for all sources.
        processes: Download parallelism
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    total = 0
    for source in SOURCE_PRIORITY:
        limit = None
        if max_per_source:
            limit = max_per_source.get(source)
        try:
            count = download_source(
                source, output_dir,
                max_objects=limit,
                processes=processes,
            )
            total += count
        except Exception as e:
            logger.error(f"{source} failed: {e}")
            continue

    logger.info(
        f"\n=== ALL SOURCES COMPLETE: {total} total new downloads ==="
    )
    return total


def enrich_existing_metadata(metadata_dir: str | Path) -> int:
    """Backfill existing Sketchfab metadata files with v1 annotations.

    Reads each .meta.json, checks if 'name' is missing, and if so
    enriches from objaverse.load_annotations().
    Returns count of files enriched.
    """
    metadata_dir = Path(metadata_dir)
    if not metadata_dir.exists():
        return 0

    v1 = _get_v1_annotations()
    if not v1:
        logger.warning("No v1 annotations available for enrichment")
        return 0

    enriched = 0
    files = list(metadata_dir.glob("*.meta.json"))
    for mf in files:
        try:
            with open(mf) as f:
                meta = json.load(f)
        except Exception:
            continue

        if meta.get("name"):
            continue

        uid = meta.get("uid", mf.stem.replace(".meta", ""))
        ann = v1.get(uid)
        if not ann:
            continue

        if ann.get("name"):
            meta["name"] = ann["name"]
        if ann.get("description"):
            meta["description"] = ann["description"][:500]
        if ann.get("tags"):
            meta["tags"] = [t["name"] if isinstance(t, dict) else t
                            for t in ann["tags"][:10]]
        if ann.get("categories"):
            meta["categories"] = [c["name"] if isinstance(c, dict) else c
                                  for c in ann["categories"][:5]]

        with open(mf, "w") as f:
            json.dump(meta, f, indent=2)
        enriched += 1

    logger.info(f"Enriched {enriched}/{len(files)} metadata files in {metadata_dir}")
    return enriched


def download_objaverse_batch(
    output_dir: str | Path = "data/raw/objaverse",
    batch_size: int = 500,
    source: str = "sketchfab",
    processes: int = 4,
) -> int:
    """Download a batch of models. Used by BackgroundDataPuller.

    Always checks for truly new data — skips anything already downloaded
    or with a known hash.

    Returns number of files downloaded in this batch.
    """
    return download_source(
        source_name=source,
        output_dir=output_dir,
        max_objects=batch_size,
        processes=processes,
    )


# Legacy API compatibility
def download_objaverse_models(output_dir, config=None, max_models=None,
                               categories=None):
    """Legacy wrapper — redirects to XL download."""
    logger.info("Redirecting to Objaverse-XL (Sketchfab source)...")
    return download_source(
        source_name="sketchfab",
        output_dir=str(output_dir),
        max_objects=max_models,
        processes=4,
    )


# Also expose as download_objaverse for BackgroundDataPuller compat
download_objaverse = download_objaverse_models


def main():
    parser = argparse.ArgumentParser(
        description="Download 3D models from Objaverse-XL (10M+ objects)")
    parser.add_argument(
        "--output", default="data/raw/objaverse",
        help="Output directory",
    )
    parser.add_argument(
        "--source", default="all",
        choices=["all", "sketchfab", "github", "thingiverse", "smithsonian"],
        help="Which source to download from (default: all)",
    )
    parser.add_argument(
        "--max", type=int, default=None,
        help="Max objects to download per source (default: unlimited)",
    )
    parser.add_argument(
        "--file-types", type=str, default=None,
        help="Comma-separated file types to download (e.g. glb,blend,obj)",
    )
    parser.add_argument(
        "--processes", type=int, default=4,
        help="Download parallelism (default: 4)",
    )
    args = parser.parse_args()

    setup_logging("objaverse_xl")

    file_types = None
    if args.file_types:
        file_types = set(args.file_types.split(","))

    output_dir = ensure_dir(args.output)

    if args.source == "all":
        max_per_source = None
        if args.max:
            max_per_source = {s: args.max for s in SOURCE_PRIORITY}
        download_all_sources(
            output_dir, max_per_source=max_per_source,
            processes=args.processes,
        )
    else:
        download_source(
            args.source, output_dir,
            max_objects=args.max,
            file_types=file_types,
            processes=args.processes,
        )


if __name__ == "__main__":
    main()
