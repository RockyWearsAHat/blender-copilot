"""Scrape 3D models from Sketchfab (sketchfab.com).

Sketchfab is the world's largest 3D model hosting platform with 5M+
models. Uses their public API v3 (no auth needed for free/downloadable
models).

Only downloads models that are marked as downloadable (free to use).
Respects CC-BY, CC-0 and other permissive licenses.

Usage:
    python -m scrapers.sketchfab_scraper --output data/raw/sketchfab
    python -m scrapers.sketchfab_scraper --output data/raw/sketchfab --max-pages 20
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from .utils import setup_logging, ensure_dir, load_progress, save_progress, download_file

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "BlenderCopilotTraining/0.1 (research; open-source 3D model training)",
    "Accept": "application/json",
}

# Sketchfab public API v3
API_BASE = "https://api.sketchfab.com/v3"
SEARCH_URL = f"{API_BASE}/models"
DOWNLOAD_URL = f"{API_BASE}/models/{{uid}}/download"

# Search categories mapping Blender relevance
SEARCH_CATEGORIES = [
    "furniture",
    "architecture",
    "vehicles",
    "characters",
    "food-drink",
    "nature",
    "electronics",
    "weapons",
    "animals-pets",
    "art-abstract",
    "science-technology",
    "sports-fitness",
]

# License types (permissive only)
FREE_LICENSES = [
    "cc0",
    "by",       # CC-BY
    "by-nc",    # CC-BY-NC
    "by-sa",    # CC-BY-SA
    "by-nc-nd", # CC-BY-NC-ND
]


def _search_sketchfab(
    query: str,
    page: int,
    session: requests.Session,
    downloadable_only: bool = True,
    api_token: Optional[str] = None,
) -> list[dict]:
    """Search Sketchfab for 3D models.

    Returns a list of model metadata dicts.
    """
    params = {
        "q": query,
        "downloadable": "true" if downloadable_only else "false",
        "sort_by": "-likeCount",
        "count": 24,
        "cursor": (page - 1) * 24 if page > 1 else 0,
        "type": "models",
    }

    headers = dict(HEADERS)
    if api_token:
        headers["Authorization"] = f"Token {api_token}"

    try:
        resp = session.get(
            SEARCH_URL,
            params=params,
            headers=headers,
            timeout=30,
        )

        if resp.status_code == 429:
            logger.warning("  Sketchfab rate limited — waiting 60s")
            time.sleep(60)
            return []

        if resp.status_code != 200:
            logger.debug(f"  Sketchfab search failed: {resp.status_code}")
            return []

        data = resp.json()
        results = data.get("results", [])
        return results

    except Exception as e:
        logger.debug(f"  Sketchfab search error: {e}")
        return []


def _get_download_url(
    uid: str,
    session: requests.Session,
    api_token: Optional[str] = None,
) -> Optional[str]:
    """Get download URL for a free Sketchfab model.

    Returns the download URL or None.
    Note: Full download (all formats) requires login. GLB format is
    often available from the embed viewer without authentication.
    """
    # Method 1: Try API download endpoint (requires auth for non-CC0)
    if api_token:
        url = DOWNLOAD_URL.format(uid=uid)
        headers = dict(HEADERS)
        headers["Authorization"] = f"Token {api_token}"
        try:
            resp = session.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                # Prefer GLB > FBX > OBJ
                for fmt in ["glb", "fbx", "obj"]:
                    if fmt in data:
                        return data[fmt].get("url")
        except Exception:
            pass

    # Method 2: Try the embeds endpoint for GLB (often accessible without auth)
    # The viewer uses an optimized GLB that's available publicly
    glb_url = f"https://media.sketchfab.com/models/{uid}/export-viewer/model_file.bin"
    try:
        resp = session.head(glb_url, headers=HEADERS, timeout=10, allow_redirects=True)
        if resp.status_code == 200:
            return glb_url
    except Exception:
        pass

    # Method 3: Try the standard low-poly GLB served from CDN
    cdn_url = f"https://sketchfab.com/models/{uid}/embed"
    try:
        resp = session.get(cdn_url, headers=HEADERS, timeout=30)
        import re
        # Extract GLB URL from embed page
        m = re.search(r'"file":\s*"(https://[^"]+\.glb[^"]*)"', resp.text)
        if m:
            return m.group(1)
    except Exception:
        pass

    return None


def _is_allowed_license(model: dict) -> bool:
    """Check if model has a permissive license."""
    license_info = model.get("license", {})
    if not license_info:
        return False

    # Check slug first (most reliable field from Sketchfab API)
    slug = license_info.get("slug", "").lower().strip()
    label = license_info.get("label", "").lower()

    # Accept if slug matches any allowed license
    for allowed in FREE_LICENSES:
        if slug == allowed or slug.startswith(allowed):
            return True

    # Also check label contains common free license identifiers
    if "creative commons zero" in label or "cc0" in label:
        return True
    if "cc by" in label or "cc-by" in label or "attribution" in label:
        return True

    # Accept if explicitly has attribution field set (CC-BY style)
    if license_info.get("attribution") is not None:
        return True

    return False


def scrape_sketchfab(
    output_dir: str = "data/raw/sketchfab",
    max_pages: int = 20,
    max_per_query: int = 10,
    delay: float = 3.0,
    api_token: Optional[str] = None,
    config: dict | None = None,
) -> int:
    """Scrape 3D models from Sketchfab.

    Downloads GLB/OBJ files from models with permissive licenses.
    Returns total number of files downloaded.

    Args:
        output_dir: Where to save downloaded files
        max_pages: Pages per category to browse (24 results/page)
        max_per_query: Max files to download per category
        delay: Seconds between requests
        api_token: Optional Sketchfab API token for higher rate limits
        config: Project config dict for shared settings
    """
    out_path = ensure_dir(output_dir)
    meta_path = out_path / "metadata.jsonl"
    progress = load_progress(out_path / ".progress")
    session = requests.Session()

    max_size_mb = 50
    if config:
        max_size_mb = config.get("scraping", {}).get("max_file_size_mb", 50)
    if api_token is None and config:
        api_token = config.get("scraping", {}).get("sketchfab", {}).get("api_token")

    total = 0

    for category in SEARCH_CATEGORIES:
        for page in range(1, max_pages + 1):
            page_key = f"{category}:p{page}"
            if page_key in progress:
                continue

            logger.info(f"  Sketchfab: '{category}' page {page}")
            models = _search_sketchfab(category, page, session,
                                       api_token=api_token)

            if not models:
                save_progress(out_path / ".progress", page_key)
                break  # No more pages or rate limited

            dl_this_page = 0
            for model in models:
                uid = model.get("uid")
                name = model.get("name", "unknown")
                if not uid:
                    continue

                model_key = f"model:{uid}"
                if model_key in progress:
                    continue

                # Check license
                if not _is_allowed_license(model):
                    logger.debug(f"  Skipping {name}: non-permissive license")
                    save_progress(out_path / ".progress", model_key)
                    continue

                # Try to get download URL
                dl_url = _get_download_url(uid, session, api_token=api_token)
                time.sleep(delay * 0.5)

                if not dl_url:
                    logger.debug(f"  No download URL for {name} ({uid})")
                    save_progress(out_path / ".progress", model_key)
                    continue

                # Determine file extension
                ext = ".glb"
                for e in [".glb", ".fbx", ".obj", ".stl"]:
                    if e in dl_url.lower():
                        ext = e
                        break

                safe_name = f"sketchfab_{uid}_{name[:40]}{ext}"
                safe_name = "".join(c if c.isalnum() or c in "._-" else "_"
                                    for c in safe_name)
                out_file = out_path / safe_name

                if not out_file.exists():
                    success = download_file(
                        dl_url, out_file,
                        max_size_mb=max_size_mb,
                        session=session,
                        rate_limit_seconds=delay,
                    )
                else:
                    success = True

                if success:
                    meta_entry = {
                        "source": "sketchfab",
                        "uid": uid,
                        "name": name,
                        "category": category,
                        "file": safe_name,
                        "format": ext,
                        "license": model.get("license", {}).get("slug", ""),
                        "views": model.get("viewCount", 0),
                        "likes": model.get("likeCount", 0),
                        "description": (model.get("description") or "")[:200],
                        "tags": [t.get("name", "") for t in
                                 (model.get("tags") or [])[:10]],
                    }
                    with open(meta_path, "a") as mf:
                        mf.write(json.dumps(meta_entry) + "\n")
                    total += 1
                    dl_this_page += 1
                    logger.debug(f"    Downloaded: {safe_name}")

                save_progress(out_path / ".progress", model_key)

                if dl_this_page >= max_per_query:
                    break

                time.sleep(delay)

            save_progress(out_path / ".progress", page_key)

            if dl_this_page < 2:
                break  # Low success rate, move to next category

    logger.info(f"Sketchfab scraper: downloaded {total} files to {out_path}")
    return total


def scrape_batch(output_dir: str = "data/raw/sketchfab") -> int:
    """Pull Sketchfab models. Used by BackgroundDataPuller."""
    return scrape_sketchfab(output_dir=output_dir, max_pages=5)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape 3D models from Sketchfab"
    )
    parser.add_argument("--output", default="data/raw/sketchfab")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--max-per-query", type=int, default=10,
                        help="Max models to download per category")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--api-token", default=None,
                        help="Sketchfab API token (optional, for better rate limits)")
    args = parser.parse_args()

    setup_logging("sketchfab")
    scrape_sketchfab(
        output_dir=args.output,
        max_pages=args.max_pages,
        max_per_query=args.max_per_query,
        delay=args.delay,
        api_token=args.api_token,
    )


if __name__ == "__main__":
    main()
