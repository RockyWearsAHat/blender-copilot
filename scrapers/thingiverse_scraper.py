"""Scrape 3D models from Thingiverse (thingiverse.com).

Thingiverse is the largest open-source 3D printing model repository
with 3M+ designs. Uses their public API (no auth required for browsing).

Free/open models only. Rate-limited to respect their servers.

Usage:
    python -m scrapers.thingiverse_scraper --output data/raw/thingiverse
    python -m scrapers.thingiverse_scraper --output data/raw/thingiverse --max-pages 50
"""

import argparse
import json
import logging
import time
from pathlib import Path

import requests

from .utils import setup_logging, ensure_dir, load_progress, save_progress, download_file

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "BlenderCopilotTraining/0.1 (research; open-source 3D model training)",
    "Accept": "application/json",
}

# Thingiverse public API base — browse without auth
API_BASE = "https://www.thingiverse.com/explore/featured/page:{page}/per_page:30"

# Curated list of Thingiverse search terms
SEARCH_TERMS = [
    "furniture", "architecture", "vehicle", "animal", "character",
    "tool", "jewelry", "toy", "decoration", "container",
    "plant", "food", "household", "sports", "electronics",
    "abstract", "art", "nature", "building", "weapon",
]

# Thingiverse API v2 search endpoint
SEARCH_API = "https://api.thingiverse.com/search/{query}?page={page}&per_page=30&sort=newest&type=things"
THING_API = "https://api.thingiverse.com/things/{thing_id}"
FILES_API = "https://api.thingiverse.com/things/{thing_id}/files"

# Also try the non-auth browse endpoint
BROWSE_API = "https://www.thingiverse.com/search?q={query}&type=things&sort=text&per_page=30&page={page}"


def _fetch_thingiverse_page(query: str, page: int, session: requests.Session) -> list[dict]:
    """Fetch a page of Thingiverse search results.

    Tries API first, falls back to HTML scraping.
    """
    # Method 1: Try the unofficial API (no token needed for reads)
    try:
        url = f"https://api.thingiverse.com/search/{requests.utils.quote(query)}"
        params = {"page": page, "per_page": 30, "sort": "newest"}
        resp = session.get(url, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("hits", [])
            if hits:
                return hits
    except Exception:
        pass

    # Method 2: HTML scrape with BeautifulSoup
    try:
        from bs4 import BeautifulSoup
        url = f"https://www.thingiverse.com/search?q={requests.utils.quote(query)}&type=things&sort=newest&page={page}"
        resp = session.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        things = []

        # Find thing cards
        for card in soup.find_all("div", class_=lambda x: x and "ThingCard" in x):
            thing_id = None
            name = ""
            href = ""

            link = card.find("a", href=True)
            if link:
                href = link.get("href", "")
                name = link.get("title", "") or link.get_text(strip=True)

            # Extract ID from href like /thing:12345
            import re
            m = re.search(r"/thing:(\d+)", href)
            if m:
                thing_id = int(m.group(1))

            if thing_id and name:
                things.append({
                    "id": thing_id,
                    "name": name,
                    "url": f"https://www.thingiverse.com{href}",
                })

        return things
    except Exception as e:
        logger.debug(f"HTML scrape failed for query='{query}' page={page}: {e}")
        return []


def _get_thing_files(thing_id: int, session: requests.Session) -> list[dict]:
    """Get downloadable files for a thing."""
    url = f"https://www.thingiverse.com/thing:{thing_id}/files"
    try:
        resp = session.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return []

        from bs4 import BeautifulSoup
        import re
        soup = BeautifulSoup(resp.text, "html.parser")

        files = []
        # Look for download links to .stl, .obj, .blend, .glb files
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            if any(href.lower().endswith(ext) for ext in [".stl", ".obj", ".blend", ".glb", ".3mf"]):
                name = link.get_text(strip=True) or Path(href).name
                files.append({
                    "name": name,
                    "url": href if href.startswith("http") else f"https://cdn.thingiverse.com{href}",
                })

        return files
    except Exception as e:
        logger.debug(f"Failed to get files for thing {thing_id}: {e}")
        return []


def scrape_thingiverse(
    output_dir: str = "data/raw/thingiverse",
    max_pages: int = 50,
    max_per_query: int = 5,
    delay: float = 2.0,
    preferred_formats: tuple = (".blend", ".glb", ".obj", ".stl"),
    config: dict | None = None,
) -> int:
    """Scrape 3D models from Thingiverse.

    Downloads mesh files organized by category.
    Returns total number of files downloaded.
    """
    out_path = ensure_dir(output_dir)
    meta_path = out_path / "metadata.jsonl"
    progress = load_progress(out_path / ".progress")

    session = requests.Session()
    session.headers.update(HEADERS)

    max_size_mb = 50  # Thingiverse files can be large
    if config:
        max_size_mb = config.get("scraping", {}).get("max_file_size_mb", 50)

    total = 0

    for query in SEARCH_TERMS:
        for page in range(1, max_pages + 1):
            page_key = f"{query}:p{page}"
            if page_key in progress:
                continue

            logger.info(f"  Thingiverse: '{query}' page {page}")

            things = _fetch_thingiverse_page(query, page, session)
            if not things:
                logger.debug(f"  No results for '{query}' page {page}")
                save_progress(out_path / ".progress", page_key)
                break  # No more pages

            downloaded_this_page = 0
            for thing in things:
                thing_id = thing.get("id") or thing.get("thing_id")
                name = thing.get("name", "unknown")
                if not thing_id:
                    continue

                thing_key = f"thing:{thing_id}"
                if thing_key in progress:
                    continue

                # Save metadata
                meta_entry = {
                    "source": "thingiverse",
                    "id": thing_id,
                    "name": name,
                    "url": thing.get("url", f"https://www.thingiverse.com/thing:{thing_id}"),
                    "query": query,
                }

                # Try to get downloadable files
                files = _get_thing_files(thing_id, session)
                time.sleep(delay * 0.5)

                downloaded = False
                for fmt in preferred_formats:
                    for file_info in files:
                        file_url = file_info.get("url", "")
                        file_name = file_info.get("name", "")
                        if not file_url or not file_url.lower().endswith(fmt):
                            continue

                        safe_name = f"thingiverse_{thing_id}_{Path(file_name).stem[:40]}{fmt}"
                        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)
                        out_file = out_path / safe_name

                        if out_file.exists():
                            downloaded = True
                            break

                        success = download_file(
                            file_url, out_file,
                            max_size_mb=max_size_mb,
                            session=session,
                            rate_limit_seconds=delay,
                        )
                        if success:
                            meta_entry["file"] = safe_name
                            meta_entry["format"] = fmt
                            with open(meta_path, "a") as mf:
                                mf.write(json.dumps(meta_entry) + "\n")
                            total += 1
                            downloaded_this_page += 1
                            downloaded = True
                            logger.debug(f"    Downloaded: {safe_name}")
                            break

                    if downloaded:
                        break

                save_progress(out_path / ".progress", thing_key)

                if downloaded_this_page >= max_per_query:
                    break

                time.sleep(delay)

            save_progress(out_path / ".progress", page_key)

            if downloaded_this_page < max_per_query // 2:
                break  # Low hit rate — move to next query

    logger.info(f"Thingiverse scraper: downloaded {total} files to {out_path}")
    return total


def scrape_batch(output_dir: str = "data/raw/thingiverse") -> int:
    """Pull Thingiverse models. Used by BackgroundDataPuller."""
    return scrape_thingiverse(output_dir=output_dir, max_pages=10)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape 3D models from Thingiverse"
    )
    parser.add_argument("--output", default="data/raw/thingiverse")
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between requests (default: 2.0)")
    args = parser.parse_args()

    setup_logging("thingiverse")
    scrape_thingiverse(
        output_dir=args.output,
        max_pages=args.max_pages,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
