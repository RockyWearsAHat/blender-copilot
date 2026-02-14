"""Scrape official Blender demo/test/splash .blend files.

Blender maintains a treasure trove of production-quality .blend files at
download.blender.org — splash screens made by world-class artists, EEVEE/Cycles
demos, geometry-nodes showcases, sculpt-mode demos, physics simulations, and
full benchmark scenes. All freely downloadable, no auth required.

Sources (all free, no login):
  1. download.blender.org/demo/  — Apache directory listing (recursive)
     - splash/     — Splash-screen .blend files (production quality)
     - eevee/      — EEVEE render demos
     - cycles/     — Cycles render demos
     - test/       — Benchmark/test scenes (BMW, Classroom, etc.)
     - geometry-nodes/  — Geometry nodes demos (30+ files)
     - sculpt_mode/     — Sculpting demos
     - physics/         — Physics/cloth/fluid demos
     - rendering/       — Shading demos
     - bbb/             — Big Buck Bunny production files
  2. studio.blender.org/characters/ — Free character rigs (CC-BY)
  3. SVN benchmarks — benchmark .blend files

Usage:
    python -m scrapers.blender_official --output data/raw/blender_official
    python -m scrapers.blender_official --output data/raw/blender_official --max-size 500
"""

import argparse
import json
import logging
import re
import time
import zipfile
import tempfile
import shutil
from pathlib import Path
from html.parser import HTMLParser

from .utils import setup_logging, ensure_dir, download_file, save_metadata

logger = logging.getLogger(__name__)

# ── Known high-quality .blend file URLs ──────────────────────────────────
# These are curated from Blender's official demo page. The directory listing
# scraper will also find these, but having them explicit ensures we never miss
# the best files even if the directory structure changes.

CURATED_FILES = [
    # ─── Splash Screens (production-quality showcase scenes) ───
    {"url": "https://download.blender.org/demo/splash/blender-4.5-splash.blend",
     "category": "splash", "name": "Blender 4.5 Dogwalk", "quality": "showcase"},
    {"url": "https://download.blender.org/demo/splash/blender-4.3-splash.blend",
     "category": "splash", "name": "Blender 4.3 House of Chores", "quality": "showcase"},
    {"url": "https://download.blender.org/demo/splash/blender-4.2-splash.zip",
     "category": "splash", "name": "Blender 4.2 Gold", "quality": "showcase"},
    {"url": "https://download.blender.org/demo/splash/blender-4.1-splash.blend",
     "category": "splash", "name": "Blender 4.1 Lynxsdesign", "quality": "showcase"},
    {"url": "https://download.blender.org/demo/splash/blender-4.0-splash.blend",
     "category": "splash", "name": "Blender 4.0 Gaku", "quality": "showcase"},
    {"url": "https://download.blender.org/demo/splash/blender-3.6-splash.zip",
     "category": "splash", "name": "Blender 3.6 Pet Projects", "quality": "showcase"},
    {"url": "https://download.blender.org/demo/splash/blender-3.5-splash.blend",
     "category": "splash", "name": "Blender 3.5 Cozy Kitchen", "quality": "showcase"},
    {"url": "https://download.blender.org/demo/splash/blender-3.4-splash.zip",
     "category": "splash", "name": "Blender 3.4 Charge", "quality": "showcase"},
    {"url": "https://download.blender.org/demo/splash/blender-3.3-splash.blend",
     "category": "splash", "name": "Blender 3.3 Scanlands", "quality": "showcase"},
    {"url": "https://download.blender.org/demo/splash/blender-3.1-splash.zip",
     "category": "splash", "name": "Blender 3.1 Secret Deer", "quality": "showcase"},

    # ─── EEVEE Demos ───
    {"url": "https://download.blender.org/demo/eevee/mandelbrot_grow.blend",
     "category": "eevee", "name": "Mandelbrot Grow"},
    {"url": "https://download.blender.org/demo/eevee/skeleton-arm-xray.blend",
     "category": "eevee", "name": "Skeleton Arm X-Ray"},

    # ─── Cycles Demos ───
    {"url": "https://download.blender.org/demo/cycles/monster_under_the_bed_sss_demo_by_metin_seven.blend",
     "category": "cycles", "name": "Monster Under the Bed"},
    {"url": "https://download.blender.org/demo/cycles/lone-monk_cycles_and_exposure-node_demo.blend",
     "category": "cycles", "name": "Lone Monk"},
    {"url": "https://download.blender.org/demo/cycles/flat-archiviz.blend",
     "category": "cycles", "name": "Italian Flat Archviz"},
    {"url": "https://download.blender.org/demo/cycles/loft.blend",
     "category": "cycles", "name": "Loft"},

    # ─── Test / Benchmark Scenes ───
    {"url": "https://download.blender.org/demo/test/BMW27.blend.zip",
     "category": "benchmark", "name": "BMW 27", "quality": "benchmark"},
    {"url": "https://download.blender.org/demo/test/BMW27_2.blend.zip",
     "category": "benchmark", "name": "BMW 27 v2", "quality": "benchmark"},
    {"url": "https://download.blender.org/demo/test/classroom.zip",
     "category": "benchmark", "name": "Classroom", "quality": "benchmark"},
    {"url": "https://download.blender.org/demo/test/pabellon_barcelona_v1.scene_.zip",
     "category": "benchmark", "name": "Barcelona Pavillion", "quality": "benchmark"},
    {"url": "https://download.blender.org/demo/test/ForYou.blend",
     "category": "test", "name": "For You"},
    {"url": "https://download.blender.org/demo/test/splash-pokedstudio.blend.zip",
     "category": "test", "name": "Racing Car Splash"},
    {"url": "https://download.blender.org/demo/test/stylized_levi.zip",
     "category": "test", "name": "Stylized Levi"},
    {"url": "https://download.blender.org/demo/test/AtvBuggy.zip",
     "category": "test", "name": "ATV Buggy"},

    # ─── Physics Demos ───
    {"url": "https://download.blender.org/demo/physics/fluid-simulation_flip_vs_apic_solver.blend",
     "category": "physics", "name": "Fluid APIC vs FLIP"},
    {"url": "https://download.blender.org/demo/cloth_internal_air_pressure.blend",
     "category": "physics", "name": "Cloth Internal Air Pressure"},
    {"url": "https://download.blender.org/demo/cloth_brushes_demo.blend",
     "category": "physics", "name": "Cloth Brushes Demo"},

    # ─── Sculpt Mode Demos ───
    {"url": "https://download.blender.org/demo/sculpt_mode/01_sculpt_grab_silhouette.blend",
     "category": "sculpt", "name": "Sculpt Grab Silhouette"},
    {"url": "https://download.blender.org/demo/sculpt_mode/02_elastic_deform_snake_hook.blend",
     "category": "sculpt", "name": "Elastic Deform Snake Hook"},
    {"url": "https://download.blender.org/demo/sculpt_mode/03_multires_displacement_smear.blend",
     "category": "sculpt", "name": "Multires Displacement Smear"},
    {"url": "https://download.blender.org/demo/sculpt_mode/auto_masking_options.blend",
     "category": "sculpt", "name": "Auto-Masking Options"},
    {"url": "https://download.blender.org/demo/sculpt_mode/color_attribute_painting.blend",
     "category": "sculpt", "name": "Color Attribute Painting"},

    # ─── Geometry Nodes Demos ───
    {"url": "https://download.blender.org/demo/geometry-nodes/SDF_mixer_kitbukoros.blend",
     "category": "geometry_nodes", "name": "SDF Boolean Mixer"},
    {"url": "https://download.blender.org/demo/geometry-nodes/tree_leaves_moss_geo-nodes-demo.blend",
     "category": "geometry_nodes", "name": "Tree Leaves Moss"},
    {"url": "https://download.blender.org/demo/geometry-nodes/flower_scattering.blend",
     "category": "geometry_nodes", "name": "Flower Scattering"},
    {"url": "https://download.blender.org/demo/geometry-nodes/pebble_scattering.blend",
     "category": "geometry_nodes", "name": "Pebble Scattering"},
    {"url": "https://download.blender.org/demo/geometry-nodes/blender-geometry-nodes_procedural-buildings.blend",
     "category": "geometry_nodes", "name": "Procedural Buildings"},
    {"url": "https://download.blender.org/demo/geometry-nodes/chocolate.blend",
     "category": "geometry_nodes", "name": "Chocolate Donut"},
    {"url": "https://download.blender.org/demo/geometry-nodes/hair_nodes-female_hair_styles.blend",
     "category": "geometry_nodes", "name": "Female Hair Styles"},
    {"url": "https://download.blender.org/demo/geometry-nodes/hair_nodes-animal_fur_examples.blend",
     "category": "geometry_nodes", "name": "Animal Fur Examples"},
    {"url": "https://download.blender.org/demo/geometry-nodes/gizmo_array.blend",
     "category": "geometry_nodes", "name": "Gizmo Array"},
    {"url": "https://download.blender.org/demo/geometry-nodes/transform_socket-pizza_delivery.blend",
     "category": "geometry_nodes", "name": "Pizza Delivery"},

    # ─── Rendering / Shading ───
    {"url": "https://download.blender.org/demo/rendering/repeat_zone_fractal_raymarch.blend",
     "category": "rendering", "name": "Fractal Raymarch"},
    {"url": "https://download.blender.org/demo/rendering/radial_tiling_texture.blend",
     "category": "rendering", "name": "Radial Tiling Texture"},

    # ─── Misc ───
    {"url": "https://download.blender.org/demo/Blender-282.blend",
     "category": "misc", "name": "Blender 2.82 Demo"},
    {"url": "https://download.blender.org/demo/color_vortex.blend",
     "category": "misc", "name": "Color Vortex"},
    {"url": "https://download.blender.org/demo/greasepencil-bike.blend",
     "category": "misc", "name": "Grease Pencil Bike"},

    # ─── Asset Bundles ───
    {"url": "https://www.blender.org/download/demo/asset-bundles/human-base-meshes/human-base-meshes-bundle-v1.4.1.zip",
     "category": "assets", "name": "Human Base Meshes", "quality": "showcase"},

    # ─── Big Buck Bunny production files ───
    {"url": "https://download.blender.org/demo/bbb/blender.zip",
     "category": "production", "name": "Big Buck Bunny (full production)", "quality": "showcase"},
]

# Directories to crawl recursively for additional .blend files
CRAWL_DIRECTORIES = [
    "https://download.blender.org/demo/eevee/",
    "https://download.blender.org/demo/cycles/",
    "https://download.blender.org/demo/test/",
    "https://download.blender.org/demo/geometry-nodes/",
    "https://download.blender.org/demo/sculpt_mode/",
    "https://download.blender.org/demo/physics/",
    "https://download.blender.org/demo/rendering/",
    "https://download.blender.org/demo/splash/",
]


class ApacheDirParser(HTMLParser):
    """Parse Apache directory listing HTML pages to extract file links."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for attr, value in attrs:
                if attr == "href" and value and not value.startswith("?"):
                    self.links.append(value)


def crawl_apache_directory(base_url: str, extensions: set[str] | None = None,
                           max_depth: int = 3, depth: int = 0) -> list[str]:
    """Recursively crawl an Apache directory listing for file URLs.

    Args:
        base_url: URL of the directory to crawl.
        extensions: Set of file extensions to include (e.g. {'.blend', '.zip'}).
        max_depth: Maximum recursion depth.
        depth: Current recursion depth.

    Returns:
        List of absolute URLs to files.
    """
    import requests

    if depth > max_depth:
        return []

    if extensions is None:
        extensions = {".blend", ".zip"}

    if not base_url.endswith("/"):
        base_url += "/"

    try:
        resp = requests.get(base_url, timeout=30, headers={
            "User-Agent": "BlenderModelTraining/0.1 (research; open-source)"
        })
        # Retry on rate limit
        if resp.status_code == 429:
            wait = 10
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = int(retry_after)
                except ValueError:
                    pass
            logger.info(f"Rate limited on crawl, waiting {wait}s...")
            time.sleep(wait)
            resp = requests.get(base_url, timeout=30, headers={
                "User-Agent": "BlenderModelTraining/0.1 (research; open-source)"
            })
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to crawl {base_url}: {e}")
        return []

    parser = ApacheDirParser()
    parser.feed(resp.text)

    files = []
    for link in parser.links:
        # Skip parent directory and special links
        if link in ("../", "/", "#") or link.startswith("?"):
            continue

        full_url = base_url + link

        # Is it a subdirectory?
        if link.endswith("/"):
            sub_files = crawl_apache_directory(full_url, extensions,
                                               max_depth, depth + 1)
            files.extend(sub_files)
        else:
            # Check extension
            lower = link.lower()
            if any(lower.endswith(ext) for ext in extensions):
                files.append(full_url)

    return files


def extract_blend_from_zip(zip_path: Path, output_dir: Path) -> list[Path]:
    """Extract .blend files from a ZIP archive.

    Returns list of extracted .blend file paths.
    """
    extracted = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in zf.infolist():
                if info.filename.lower().endswith('.blend') and not info.is_dir():
                    # Extract to output_dir with flat naming
                    safe_name = Path(info.filename).name
                    out_path = output_dir / safe_name
                    if out_path.exists():
                        continue

                    # Extract to temp, then move
                    with zf.open(info) as src:
                        with open(out_path, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                    extracted.append(out_path)
                    logger.debug(f"  Extracted: {safe_name}")
    except Exception as e:
        logger.warning(f"Failed to extract {zip_path.name}: {e}")

    return extracted


def download_blender_official(output_dir: Path, max_size_mb: float = 500,
                               crawl: bool = True, curated_only: bool = False):
    """Download official Blender demo/test/splash .blend files.

    Args:
        output_dir: Base output directory.
        max_size_mb: Maximum file size to download (MB).
        crawl: If True, also crawl Apache directory listings for additional files.
        curated_only: If True, only download curated list (faster, smaller).
    """
    models_dir = ensure_dir(output_dir / "models")
    metadata_dir = ensure_dir(output_dir / "metadata")
    zip_dir = ensure_dir(output_dir / ".zips")  # Temp dir for zip files

    # Track progress
    progress_file = output_dir / ".progress.json"
    downloaded = set()
    if progress_file.exists():
        with open(progress_file) as f:
            downloaded = set(json.load(f))

    # ── Phase 1: Download curated files ──────────────────────────────────
    logger.info(f"Phase 1: Downloading {len(CURATED_FILES)} curated files...")

    all_urls = {}  # url -> metadata
    for entry in CURATED_FILES:
        all_urls[entry["url"]] = entry

    # ── Phase 2: Crawl directories for additional files ──────────────────
    if crawl and not curated_only:
        logger.info("Phase 2: Crawling Apache directory listings...")
        for dir_url in CRAWL_DIRECTORIES:
            logger.info(f"  Crawling {dir_url}")
            found = crawl_apache_directory(dir_url)
            for url in found:
                if url not in all_urls:
                    # Derive name/category from URL
                    filename = url.split("/")[-1]
                    parts = url.replace("https://download.blender.org/demo/", "").split("/")
                    category = parts[0] if len(parts) > 1 else "misc"
                    name = Path(filename).stem.replace("-", " ").replace("_", " ").title()

                    all_urls[url] = {
                        "url": url,
                        "category": category,
                        "name": name,
                    }
            time.sleep(0.5)  # Be nice to the server

    total = len(all_urls)
    logger.info(f"Found {total} total files to download")

    # ── Phase 3: Download all ────────────────────────────────────────────
    success = 0
    skipped = 0
    errors = 0

    for i, (url, meta) in enumerate(all_urls.items()):
        filename = url.split("/")[-1]
        # Use a safe ID derived from URL
        file_id = re.sub(r'[^\w\-.]', '_', filename)

        if file_id in downloaded:
            skipped += 1
            continue

        is_zip = filename.lower().endswith(".zip")
        dst = (zip_dir if is_zip else models_dir) / filename

        logger.info(f"[{i + 1}/{total}] Downloading: {meta.get('name', filename)}")

        ok = download_file(url, dst, max_size_mb=max_size_mb,
                           rate_limit_seconds=2.0)
        if not ok:
            errors += 1
            downloaded.add(file_id)  # Don't retry
            continue

        # If ZIP, extract .blend files
        if is_zip and dst.exists():
            extracted = extract_blend_from_zip(dst, models_dir)
            logger.info(f"  Extracted {len(extracted)} .blend file(s) from {filename}")
            if not extracted:
                # ZIP might not contain .blend files (e.g., textures only)
                logger.debug(f"  No .blend files in {filename}")

        # Save metadata
        save_metadata(metadata_dir, file_id, {
            "source": "blender_official",
            "url": url,
            "name": meta.get("name", filename),
            "category": meta.get("category", "unknown"),
            "quality": meta.get("quality", "demo"),
            "license": "CC-BY or CC-0 (Blender official)",
            "format": "blend",
        })

        downloaded.add(file_id)
        success += 1

        # Save progress periodically
        if success % 5 == 0:
            with open(progress_file, "w") as f:
                json.dump(list(downloaded), f)

    # Final progress save
    with open(progress_file, "w") as f:
        json.dump(list(downloaded), f)

    # Count actual .blend files
    blend_count = len(list(models_dir.glob("*.blend")))

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Blender Official Download Complete")
    logger.info(f"  Downloaded: {success} files")
    logger.info(f"  Skipped:    {skipped} (already done)")
    logger.info(f"  Errors:     {errors}")
    logger.info(f"  .blend files ready: {blend_count}")
    logger.info(f"  Output: {models_dir}")
    logger.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Download official Blender demo/test/splash files")
    parser.add_argument("--output", default="data/raw/blender_official",
                        help="Output directory")
    parser.add_argument("--max-size", type=float, default=500,
                        help="Max file size in MB (default: 500)")
    parser.add_argument("--curated-only", action="store_true",
                        help="Only download curated list (skip crawling)")
    parser.add_argument("--no-crawl", action="store_true",
                        help="Don't crawl directory listings")
    args = parser.parse_args()

    setup_logging("blender_official")
    output_dir = ensure_dir(args.output)
    download_blender_official(output_dir,
                               max_size_mb=args.max_size,
                               crawl=not args.no_crawl,
                               curated_only=args.curated_only)


if __name__ == "__main__":
    main()
