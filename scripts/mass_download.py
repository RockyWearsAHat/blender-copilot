"""Mass download pipeline: popular-first from ALL sources.

Downloads 3D models from 4 sources, prioritizing popular items with
rich metadata. Then extracts meshes and builds training cache.

Sources (priority order):
  1. Objaverse Sketchfab (800K models, sorted by likes, require tags+name)
  2. SmutBase + Open3DLab (character models with titles/tags)
  3. BlendSwap (CC-0/CC-BY, sorted by downloads, requires login)
  4. GitHub (repos with .blend files, has description/topics)

Usage:
    python scripts/mass_download.py                    # All sources
    python scripts/mass_download.py --source objaverse # Just Objaverse
    python scripts/mass_download.py --max 5000         # Cap per source
    python scripts/mass_download.py --extract-only     # Skip download, just extract+cache
"""
import argparse
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent


def download_objaverse_popular(max_models=10000, min_likes=1, processes=4):
    """Download Objaverse models sorted by popularity (most liked first).
    
    Only downloads models with rich metadata: name + tags + description.
    Sorts by likeCount descending so we get the best models first.
    """
    try:
        import objaverse
        import objaverse.xl as oxl
    except ImportError:
        logger.error("objaverse not installed. Run: pip install objaverse")
        return 0

    output_dir = BASE / "data" / "raw" / "objaverse"
    source_dir = output_dir / "sketchfab"
    models_dir = source_dir / "models"
    metadata_dir = source_dir / "metadata"
    for d in [output_dir, source_dir, models_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    progress_file = source_dir / ".progress.json"
    progress = set()
    if progress_file.exists():
        try:
            progress = set(json.load(open(progress_file)))
        except Exception:
            pass

    logger.info("Loading Objaverse v1 annotations (800K models)...")
    anns = objaverse.load_annotations()
    logger.info(f"Loaded {len(anns)} annotations")

    rich = []
    for uid, ann in anns.items():
        if not ann.get("tags") or not ann.get("name"):
            continue
        likes = ann.get("likeCount", 0)
        if likes < min_likes:
            continue
        if not ann.get("isDownloadable", True):
            continue
        if uid in progress:
            continue
        rich.append((uid, ann, likes))

    rich.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"Found {len(rich)} popular models with rich metadata "
                f"(min {min_likes} likes, not yet downloaded)")

    if max_models:
        rich = rich[:max_models]
    logger.info(f"Will download up to {len(rich)} models (most popular first)")

    if not rich:
        logger.info("Nothing new to download!")
        return 0

    top5 = rich[:5]
    for uid, ann, likes in top5:
        tags = [t["name"] if isinstance(t, dict) else t for t in (ann.get("tags") or [])[:5]]
        logger.info(f"  Top: {likes} likes - {ann['name'][:60]} tags={tags}")

    uids_to_download = [uid for uid, _, _ in rich]

    downloaded = 0
    batch_size = 100
    for batch_start in range(0, len(uids_to_download), batch_size):
        batch_uids = uids_to_download[batch_start:batch_start + batch_size]
        logger.info(f"Batch {batch_start // batch_size + 1}: "
                    f"downloading {len(batch_uids)} models "
                    f"({batch_start + 1}-{batch_start + len(batch_uids)} "
                    f"of {len(uids_to_download)})")

        try:
            results = objaverse.load_objects(
                uids=batch_uids,
                download_processes=min(processes, len(batch_uids)),
            )

            for uid, local_path in results.items():
                if local_path is None or not Path(local_path).exists():
                    progress.add(uid)
                    continue

                ann = anns[uid]
                safe_name = uid[:32]
                ext = Path(local_path).suffix or ".glb"
                dst = models_dir / f"{safe_name}{ext}"

                if not dst.exists():
                    try:
                        shutil.copy2(local_path, dst)
                    except Exception as e:
                        logger.debug(f"Copy failed for {uid}: {e}")
                        progress.add(uid)
                        continue

                tags = ann.get("tags", [])
                tag_names = [t["name"] if isinstance(t, dict) else t for t in tags[:15]]
                cats = ann.get("categories", [])
                cat_names = [c["name"] if isinstance(c, dict) else c for c in cats[:5]]

                meta = {
                    "source": "objaverse_xl_sketchfab",
                    "uid": uid,
                    "name": ann.get("name", ""),
                    "description": (ann.get("description") or "")[:500],
                    "tags": tag_names,
                    "categories": cat_names,
                    "likeCount": ann.get("likeCount", 0),
                    "viewCount": ann.get("viewCount", 0),
                    "faceCount": ann.get("faceCount", 0),
                    "vertexCount": ann.get("vertexCount", 0),
                    "license": ann.get("license", {}).get("label", "") if isinstance(ann.get("license"), dict) else str(ann.get("license", "")),
                    "isDownloadable": True,
                }

                meta_path = metadata_dir / f"{safe_name}.meta.json"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

                progress.add(uid)
                downloaded += 1

        except Exception as e:
            logger.error(f"Batch download failed: {e}")

        objaverse_cache = Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs"
        if objaverse_cache.exists():
            for glb_file in objaverse_cache.rglob("*.glb"):
                try:
                    glb_file.unlink()
                except Exception:
                    pass

        with open(progress_file, "w") as f:
            json.dump(list(progress), f)

        logger.info(f"Progress: {downloaded} new downloads, "
                    f"{len(progress)} total processed")

    logger.info(f"Objaverse complete: {downloaded} new models downloaded")
    return downloaded


def download_smutbase(max_models=500):
    """Download character models from SmutBase and Open3DLab."""
    try:
        from scrapers.smutbase_scraper import scrape_batch
    except ImportError:
        logger.warning("SmutBase scraper not available")
        return 0

    total = 0
    for site in ["smutbase", "open3dlab"]:
        raw_dir = BASE / "data" / "raw" / site
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading from {site} (popular first)...")
        try:
            count = scrape_batch(
                site_key=site,
                output_dir=str(raw_dir),
                batch_size=min(max_models, 200),
                sort_by="popular",
            )
            total += count
            logger.info(f"{site}: downloaded {count} models")
        except Exception as e:
            logger.warning(f"{site} failed: {e}")
            import traceback
            traceback.print_exc()

    return total


def download_blendswap(max_models=100):
    """Download popular BlendSwap models (CC-0/CC-BY)."""
    try:
        from scrapers.blendswap_scraper import (
            create_session, get_blend_detail, BASE_URL,
        )
        from scrapers.utils import (
            ensure_dir, load_progress, save_progress,
            download_file, save_metadata,
        )
    except ImportError:
        logger.warning("BlendSwap scraper not available")
        return 0

    raw_dir = BASE / "data" / "raw" / "blendswap"
    raw_dir.mkdir(parents=True, exist_ok=True)

    progress_file = raw_dir / ".progress"
    progress = load_progress(progress_file)
    logger.info(f"BlendSwap: {len(progress)} already processed")

    session = create_session()

    import re
    from bs4 import BeautifulSoup

    downloaded = 0
    for page in range(1, 500):
        if downloaded >= max_models:
            break

        url = f"{BASE_URL}/blends/{page}/mostDownloads"
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"BlendSwap page {page} failed: {e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        blend_data = {}

        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            m = re.search(r"/blend/(\d+)", href)
            if not m:
                continue
            bid = m.group(1)
            if "like" in href:
                continue

            if bid not in blend_data:
                blend_url = href if href.startswith("http") else BASE_URL + href
                blend_data[bid] = {"id": bid, "title": "", "url": blend_url, "license": ""}

            title_el = link.select_one("h3, h4, h5") or link
            title = (title_el.get_text(strip=True) if title_el else "").strip()
            if title and len(title) >= 3 and not blend_data[bid]["title"]:
                blend_data[bid]["title"] = title

            if not blend_data[bid]["license"]:
                parent = link.parent
                while parent and parent.name not in ("body", None):
                    txt = parent.get_text()
                    for tag in ["CC-0", "CC-BY-NC", "CC-BY-SA", "CC-BY", "GAL"]:
                        if tag in txt:
                            blend_data[bid]["license"] = tag
                            break
                    if blend_data[bid]["license"]:
                        break
                    parent = parent.parent

        listings = [v for v in blend_data.values() if v["title"]]

        if not listings:
            logger.info(f"BlendSwap: no more listings at page {page}")
            break

        for listing in listings:
            if downloaded >= max_models:
                break

            bid = listing["id"]
            if bid in progress:
                continue

            lic = listing.get("license", "")
            if not any(tag in lic for tag in ["CC-0", "CC-BY", "GAL"]):
                save_progress(progress_file, bid)
                progress.add(bid)
                continue

            detail = get_blend_detail(listing["url"], session)
            if not detail or not detail.get("download_url"):
                save_progress(progress_file, bid)
                progress.add(bid)
                continue

            pop_dir = ensure_dir(raw_dir / "popular")
            out_path = pop_dir / f"{bid}.blend"
            success = download_file(
                detail["download_url"], out_path,
                max_size_mb=200,
                session=session,
            )
            if success:
                save_metadata(str(pop_dir), bid, {
                    **listing, **detail, "source": "blendswap",
                })
                downloaded += 1
                dl_count = (detail.get("stats") or {}).get("downloads", "?")
                logger.info(f"BlendSwap #{downloaded}: {listing['title'][:50]} "
                            f"({dl_count} downloads)")

            save_progress(progress_file, bid)
            progress.add(bid)
            time.sleep(2)

    logger.info(f"BlendSwap complete: {downloaded} new models")
    return downloaded


def download_github(max_repos=200):
    """Download .blend files from GitHub repos."""
    try:
        from scrapers.github_scraper import scrape_github
        from scrapers.utils import ensure_dir, load_config
    except ImportError:
        logger.warning("GitHub scraper not available")
        return 0

    raw_dir = BASE / "data" / "raw" / "github"
    raw_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    config.setdefault("scraping", {}).setdefault("github", {})["max_repos"] = max_repos

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        try:
            from dotenv import load_dotenv
            load_dotenv(BASE / ".env")
            token = os.environ.get("GITHUB_TOKEN", "")
        except ImportError:
            pass

    logger.info(f"GitHub: downloading .blend files (token={'YES' if token else 'NO'})...")
    try:
        scrape_github(raw_dir, config, token or None)
    except Exception as e:
        logger.warning(f"GitHub scrape failed: {e}")
        return 0

    return 1


def extract_and_cache():
    """Extract meshes from raw downloads and build training cache."""
    from processing.mesh_extractor import process_directory as extract_dir
    from scrapers.utils import load_config

    config = load_config()

    raw_sources = {
        "objaverse": ("data/raw/objaverse/sketchfab/models", "data/processed/objaverse"),
    }

    total_extracted = 0
    for source_name, (raw_rel, proc_rel) in raw_sources.items():
        raw_dir = BASE / raw_rel
        proc_dir = BASE / proc_rel
        meta_dir = BASE / raw_rel.replace("/models", "/metadata")

        if not raw_dir.exists():
            continue

        raw_files = list(raw_dir.glob("*"))
        raw_files = [f for f in raw_files if f.suffix.lower() in
                     {".glb", ".gltf", ".obj", ".stl", ".ply", ".blend", ".off", ".3ds"}]

        already = set()
        if proc_dir.exists():
            for jf in proc_dir.glob("*.json"):
                if not jf.name.endswith(".meta.json"):
                    already.add(jf.stem)

        new_files = [f for f in raw_files if f.stem not in already]
        if not new_files:
            logger.info(f"{source_name}: all {len(raw_files)} files already extracted")
            continue

        logger.info(f"{source_name}: extracting {len(new_files)} new files...")
        proc_dir.mkdir(parents=True, exist_ok=True)

        try:
            extract_dir(
                raw_dir, proc_dir,
                metadata_dir=meta_dir if meta_dir.exists() else None,
                config=config,
            )
            total_extracted += len(new_files)
        except Exception as e:
            logger.warning(f"{source_name} extraction failed: {e}")

    logger.info(f"Extraction complete: {total_extracted} new files processed")
    return total_extracted


def rebuild_cache():
    """Rebuild the mesh training cache from all processed JSONs."""
    logger.info("Rebuilding mesh cache...")
    import subprocess
    result = subprocess.run(
        [sys.executable, str(BASE / "scripts" / "rebuild_cache.py")],
        cwd=str(BASE),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("Cache rebuild complete!")
        for line in result.stdout.split("\n")[-5:]:
            if line.strip():
                logger.info(f"  {line.strip()}")
    else:
        logger.warning(f"Cache rebuild failed:\n{result.stderr[-500:]}")


def main():
    parser = argparse.ArgumentParser(
        description="Mass download 3D models from all sources, popular first")
    parser.add_argument("--source", choices=["objaverse", "smutbase", "blendswap", "github", "all"],
                        default="all", help="Which source to download from")
    parser.add_argument("--max", type=int, default=None,
                        help="Max models per source")
    parser.add_argument("--min-likes", type=int, default=1,
                        help="Min likes for Objaverse (default: 1)")
    parser.add_argument("--processes", type=int, default=4,
                        help="Download parallelism")
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip downloads, only extract and cache")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip extraction/caching after download")
    args = parser.parse_args()

    sources = [args.source] if args.source != "all" else ["objaverse", "smutbase", "blendswap", "github"]
    totals = {}

    if not args.extract_only:
        for source in sources:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"  SOURCE: {source.upper()}")
            logger.info(f"{'=' * 60}\n")

            if source == "objaverse":
                max_m = args.max or 10000
                totals[source] = download_objaverse_popular(
                    max_models=max_m,
                    min_likes=args.min_likes,
                    processes=args.processes,
                )

            elif source == "smutbase":
                max_m = args.max or 500
                totals[source] = download_smutbase(max_models=max_m)

            elif source == "blendswap":
                max_m = args.max or 100
                totals[source] = download_blendswap(max_models=max_m)

            elif source == "github":
                max_m = args.max or 200
                totals[source] = download_github(max_repos=max_m)

    if not args.no_extract:
        logger.info(f"\n{'=' * 60}")
        logger.info("  EXTRACTING & CACHING")
        logger.info(f"{'=' * 60}\n")
        extract_and_cache()
        rebuild_cache()

    logger.info(f"\n{'=' * 60}")
    logger.info("  DOWNLOAD SUMMARY")
    logger.info(f"{'=' * 60}")
    for src, count in totals.items():
        logger.info(f"  {src}: {count} new models")
    logger.info(f"  TOTAL: {sum(totals.values())} new models")
    logger.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
