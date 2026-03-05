#!/usr/bin/env python3
"""Full-scale downloader for BlendSwap (subscription) + SmutBase + Open3DLab.

Runs in three sequential phases:
  1. BlendSwap   — all 24 categories × up to --bs-pages pages each (all licenses)
  2. SmutBase    — all pages (free CDN + premium if SMUTBASE_EMAIL set)
  3. Open3DLab   — all pages (same infrastructure as SmutBase)

After each source completes, runs:
  • scripts/extract_blends.py → Blender headless extraction → JSON
  • scripts/rebuild_cache.py  → rebuild .pt training cache

Usage:
    nohup python scripts/rip_blendswap_smutbase.py > /tmp/rip_all.log 2>&1 &

Optional flags:
    --bs-pages   N   max pages per BlendSwap category (default 100)
    --sm-pages   N   max SmutBase pages (default 500)
    --o3d-pages  N   max Open3DLab pages (default 250)
    --no-extract     skip Blender extraction (just download)
    --no-rebuild     skip pt cache rebuild
    --blendswap-only, --smutbase-only, --open3dlab-only
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# ── project root on path ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

logger = logging.getLogger("rip_all")

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def setup_logging():
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Also write to file
    fh = logging.FileHandler("/tmp/rip_all.log")
    fh.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)


# ────────────────────────────────────────────────────────────────────────
# Phase 1 — BlendSwap
# ────────────────────────────────────────────────────────────────────────

def run_blendswap(max_pages: int):
    """Download all BlendSwap categories with all licenses (subscription mode)."""
    from scrapers.blendswap_scraper import (
        create_session, scrape_category, CATEGORIES,
    )
    from scrapers.utils import ensure_dir, load_progress, setup_logging as _sl
    _sl("blendswap")

    output_dir = ROOT / "data" / "raw" / "blendswap"
    ensure_dir(output_dir)
    progress = load_progress(output_dir / ".progress")
    session = create_session()

    config: dict = {}  # no license filter — all_licenses=True overrides

    categories = list(CATEGORIES.keys())
    logger.info(
        f"[BlendSwap] Starting: {len(categories)} categories × "
        f"up to {max_pages} pages each | all licenses"
    )

    for cat in categories:
        cat_id = CATEGORIES[cat]
        try:
            scrape_category(
                cat,
                cat_id,
                output_dir,
                max_pages,
                session,
                progress,
                config,
                all_licenses=True,   # ← subscription: skip license filter
            )
        except KeyboardInterrupt:
            logger.warning("[BlendSwap] Interrupted — progress saved.")
            raise
        except Exception as e:
            logger.error(f"[BlendSwap] Category '{cat}' failed: {e}")
            continue

    logger.info("[BlendSwap] Done.")


# ────────────────────────────────────────────────────────────────────────
# Phase 2+3 — SmutBase / Open3DLab
# ────────────────────────────────────────────────────────────────────────

def run_smutbase_site(site_key: str, max_pages: int):
    """Scrape all pages of SmutBase or Open3DLab."""
    from scrapers.smutbase_scraper import scrape_site
    from scrapers.utils import setup_logging as _sl
    _sl(site_key)

    output_dir = str(ROOT / "data" / "raw" / site_key)

    # smutba.se serves ~16–20 items per page
    # 4000 models ÷ 18 per page ≈ 222 pages for SmutBase
    # Open3DLab has 2000+ models ÷ 18 ≈ 111 pages
    logger.info(
        f"[{site_key}] Starting: up to {max_pages} pages | "
        f"premium={'enabled as ' + os.environ.get('OPEN3DLAB_USER','') if os.environ.get('OPEN3DLAB_USER') else 'disabled (set OPEN3DLAB_USER in .env)'}"
    )

    try:
        scrape_site(
            site_key=site_key,
            output_dir=output_dir,
            max_pages=max_pages,
            download_files=True,
        )
    except KeyboardInterrupt:
        logger.warning(f"[{site_key}] Interrupted — progress saved.")
        raise
    except Exception as e:
        logger.error(f"[{site_key}] Scraping failed: {e}")

    logger.info(f"[{site_key}] Done.")


# ────────────────────────────────────────────────────────────────────────
# Post-phase: Blender extraction
# ────────────────────────────────────────────────────────────────────────

def run_extraction(input_dir: str, output_dir: str, label: str):
    """Run scripts/extract_blends.py on a directory of .blend files."""
    extract_script = ROOT / "scripts" / "extract_blends.py"
    if not extract_script.exists():
        logger.warning(f"[extract] {extract_script} not found — skipping")
        return

    # Count available .blend files
    blend_files = list(Path(input_dir).rglob("*.blend"))
    if not blend_files:
        logger.info(f"[extract/{label}] No .blend files in {input_dir} — skip")
        return

    logger.info(f"[extract/{label}] {len(blend_files)} .blend files → {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(extract_script),
        "--input", input_dir,
        "--output", output_dir,
    ]

    try:
        result = subprocess.run(cmd, timeout=7200)   # 2 hour max
        if result.returncode == 0:
            json_count = len(list(Path(output_dir).glob("*.json")))
            logger.info(f"[extract/{label}] Complete — {json_count} JSONs in {output_dir}")
        else:
            logger.warning(f"[extract/{label}] Exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        logger.warning(f"[extract/{label}] Timeout after 2h — partial results saved")
    except Exception as e:
        logger.error(f"[extract/{label}] Failed: {e}")


# ────────────────────────────────────────────────────────────────────────
# Post-phase: rebuild pt cache
# ────────────────────────────────────────────────────────────────────────

def run_rebuild():
    """Rebuild the .pt training cache from all processed JSONs."""
    rebuild_script = ROOT / "scripts" / "rebuild_cache.py"
    if not rebuild_script.exists():
        logger.warning("[rebuild] rebuild_cache.py not found — skipping")
        return

    logger.info("[rebuild] Rebuilding .pt training cache …")
    try:
        result = subprocess.run(
            [sys.executable, str(rebuild_script)],
            timeout=3600,
        )
        if result.returncode == 0:
            logger.info("[rebuild] Cache rebuild complete.")
        else:
            logger.warning(f"[rebuild] Exited with code {result.returncode}")
    except Exception as e:
        logger.error(f"[rebuild] Failed: {e}")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Rip BlendSwap + SmutBase + Open3DLab")
    parser.add_argument("--bs-pages",      type=int, default=100,
                        help="Max pages per BlendSwap category (default 100)")
    parser.add_argument("--sm-pages",      type=int, default=500,
                        help="Max SmutBase pages (default 500)")
    parser.add_argument("--o3d-pages",     type=int, default=250,
                        help="Max Open3DLab pages (default 250)")
    parser.add_argument("--no-extract",    action="store_true",
                        help="Skip Blender extraction after download")
    parser.add_argument("--no-rebuild",    action="store_true",
                        help="Skip pt cache rebuild")
    parser.add_argument("--blendswap-only",  action="store_true")
    parser.add_argument("--smutbase-only",   action="store_true")
    parser.add_argument("--open3dlab-only",  action="store_true")
    args = parser.parse_args()

    only_one = args.blendswap_only or args.smutbase_only or args.open3dlab_only
    do_bs  = args.blendswap_only  or not only_one
    do_sm  = args.smutbase_only   or not only_one
    do_o3d = args.open3dlab_only  or not only_one

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("  RIP ALL — BlendSwap + SmutBase + Open3DLab")
    logger.info(f"  BlendSwap: {do_bs} | SmutBase: {do_sm} | Open3DLab: {do_o3d}")
    logger.info(f"  BS pages: {args.bs_pages} | SM pages: {args.sm_pages} | O3D pages: {args.o3d_pages}")
    logger.info("=" * 60)

    # ── Phase 1: BlendSwap ─────────────────────────────────────────────
    if do_bs:
        logger.info("\n── Phase 1: BlendSwap ──────────────────────────────────")
        run_blendswap(args.bs_pages)

        if not args.no_extract:
            # BlendSwap downloads .blend files by category subdirectory
            bs_raw = str(ROOT / "data" / "raw" / "blendswap")
            bs_proc = str(ROOT / "data" / "processed" / "blendswap")
            run_extraction(bs_raw, bs_proc, "blendswap")

    # ── Phase 2: SmutBase ─────────────────────────────────────────────
    if do_sm:
        logger.info("\n── Phase 2: SmutBase ───────────────────────────────────")
        run_smutbase_site("smutbase", args.sm_pages)

        if not args.no_extract:
            sm_raw   = str(ROOT / "data" / "raw" / "smutbase" / "files")
            sm_proc  = str(ROOT / "data" / "processed" / "smutbase")
            run_extraction(sm_raw, sm_proc, "smutbase")

    # ── Phase 3: Open3DLab ────────────────────────────────────────────
    if do_o3d:
        logger.info("\n── Phase 3: Open3DLab ──────────────────────────────────")
        run_smutbase_site("open3dlab", args.o3d_pages)

        if not args.no_extract:
            o3d_raw   = str(ROOT / "data" / "raw" / "open3dlab" / "files")
            o3d_proc  = str(ROOT / "data" / "processed" / "open3dlab")
            run_extraction(o3d_raw, o3d_proc, "open3dlab")

    # ── Final: rebuild cache ──────────────────────────────────────────
    if not args.no_rebuild:
        logger.info("\n── Final: Rebuild .pt cache ────────────────────────────")
        run_rebuild()

    elapsed = (time.time() - t0) / 3600
    logger.info(f"\n✓ All done in {elapsed:.1f}h")


if __name__ == "__main__":
    main()
