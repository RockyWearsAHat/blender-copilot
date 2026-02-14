"""Scrape free .blend files from BlendSwap.com.

BlendSwap hosts thousands of free Blender models under CC-0 and CC-BY licenses.
This scraper downloads .blend files and their metadata (title, description,
category, tags, license).

Usage:
    python -m scrapers.blendswap_scraper --output data/raw/blendswap
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .utils import (
    setup_logging, load_config, ensure_dir, download_file,
    save_metadata, load_progress, save_progress
)

logger = logging.getLogger(__name__)

BASE_URL = "https://www.blendswap.com"

# BlendSwap category IDs (numeric — from /categories page)
CATEGORIES = {
    "animals": 1,
    "accessories": 2,
    "architecture": 3,
    "game-engine": 4,
    "characters": 5,
    "clothes": 6,
    "electronics": 7,
    "exterior": 8,
    "fantasy": 9,
    "food-drink": 10,
    "furniture": 11,
    "holidays": 12,
    "humans": 13,
    "interior": 14,
    "math-art": 15,
    "mechanical": 16,
    "music": 17,
    "nature": 18,
    "objects": 19,
    "sci-fi": 20,
    "sports": 21,
    "textures": 22,
    "vehicles": 23,
    "weapons-armor": 24,
}


def get_blend_listings(category_id: int, category_name: str, page: int,
                       session: requests.Session) -> list[dict]:
    """Fetch one page of .blend listings for a category.

    URLs follow the pattern:
      /blends/category/{id}       — page 1
      /blends/category/{id}/{page} — page 2+

    Returns list of dicts with keys: id, title, url, license.
    """
    if page == 1:
        url = f"{BASE_URL}/blends/category/{category_id}"
    else:
        url = f"{BASE_URL}/blends/category/{category_id}/{page}"

    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    listings = []

    # Parse listings — each blend has an <a> linking to /blend/{id} and
    # license text nearby (CC-0, CC-BY, etc.)
    for link in soup.select("a[href*='/blend/']"):
        href = link.get("href", "")
        blend_id_match = re.search(r"/blend/(\d+)", href)
        if not blend_id_match:
            continue

        blend_id = blend_id_match.group(1)

        # Get the title from the heading inside or near the link
        title_el = link.select_one("h3, h4, h5") or link
        title = title_el.get_text(strip=True) if title_el else "Untitled"

        # Skip preview image links (they link to same blend but have no title text)
        if not title or title == "Untitled" or "preview" in title.lower():
            # Try parent container
            parent = link.parent
            if parent:
                heading = parent.select_one("h3, h4, h5")
                if heading:
                    title = heading.get_text(strip=True)

        if not title or title == "Untitled":
            continue

        # Find license text near this listing
        license_text = ""
        parent = link.parent
        while parent and parent.name not in ("body", None):
            text = parent.get_text()
            for lic in ["CC-0", "CC-BY-NC", "CC-BY-SA", "CC-BY", "GAL"]:
                if lic in text:
                    license_text = lic
                    break
            if license_text:
                break
            parent = parent.parent

        blend_url = href if href.startswith("http") else BASE_URL + href

        listings.append({
            "id": blend_id,
            "title": title,
            "url": blend_url,
            "category": category_name,
            "license": license_text,
        })

    # Deduplicate (links appear multiple times — thumbnail + text)
    seen = set()
    unique = []
    for item in listings:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)

    return unique


def get_blend_detail(blend_url: str,
                     session: requests.Session) -> dict | None:
    """Fetch detail page for a single blend to get download link + metadata."""
    try:
        resp = session.get(blend_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to fetch detail {blend_url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Try to find download link
    download_link = None
    for a in soup.select("a[href*='download'], a.download-btn, a.btn-download"):
        href = a.get("href", "")
        if "download" in href.lower():
            if not href.startswith("http"):
                href = BASE_URL + href
            download_link = href
            break

    # Extract description
    desc_el = soup.select_one(".blend-description, .description, .detail-text")
    description = desc_el.get_text(strip=True) if desc_el else ""

    # Extract tags
    tags = []
    for tag_el in soup.select(".tag, .badge-tag, .blend-tag"):
        tags.append(tag_el.get_text(strip=True))

    # Extract stats
    stats = {}
    for stat_el in soup.select(".stat, .blend-stat"):
        text = stat_el.get_text(strip=True)
        if "vert" in text.lower():
            nums = re.findall(r"[\d,]+", text)
            if nums:
                stats["vertices"] = int(nums[0].replace(",", ""))
        if "face" in text.lower():
            nums = re.findall(r"[\d,]+", text)
            if nums:
                stats["faces"] = int(nums[0].replace(",", ""))

    return {
        "download_url": download_link,
        "description": description,
        "tags": tags,
        "stats": stats,
    }


def scrape_category(category_name: str, category_id: int, output_dir: Path,
                    max_pages: int, session: requests.Session, progress: set,
                    config: dict):
    """Scrape all pages of a category."""
    cat_dir = ensure_dir(output_dir / category_name)
    logger.info(f"Scraping category: {category_name} (id={category_id}, max {max_pages} pages)")

    max_size = config.get("scraping", {}).get("max_file_size_mb", 200)

    for page in range(1, max_pages + 1):
        logger.info(f"  Page {page}/{max_pages}")
        listings = get_blend_listings(category_id, category_name, page, session)

        if not listings:
            logger.info(f"  No more listings at page {page}")
            break

        for listing in listings:
            blend_id = listing["id"]
            if blend_id in progress:
                continue

            # Check license
            allowed_licenses = config.get("scraping", {}).get(
                "blendswap", {}).get("licenses", ["CC-0", "CC-BY"])
            license_ok = any(
                lic.lower() in listing.get("license", "").lower()
                for lic in allowed_licenses
            )
            if not license_ok:
                save_progress(output_dir / ".progress", blend_id)
                continue

            # Get detail page
            detail = get_blend_detail(listing["url"], session)
            if not detail or not detail.get("download_url"):
                save_progress(output_dir / ".progress", blend_id)
                continue

            # Download the .blend file
            filename = f"{blend_id}.blend"
            success = download_file(
                detail["download_url"],
                cat_dir / filename,
                max_size_mb=max_size,
            )

            if success:
                # Save metadata
                metadata = {
                    **listing,
                    **detail,
                    "source": "blendswap",
                }
                save_metadata(cat_dir, blend_id, metadata)

            save_progress(output_dir / ".progress", blend_id)
            time.sleep(2)  # Be respectful


def main():
    parser = argparse.ArgumentParser(description="Scrape .blend files from BlendSwap")
    parser.add_argument("--output", default="data/raw/blendswap",
                        help="Output directory")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Categories to scrape (default: all from config)")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Max pages per category (overrides config)")
    args = parser.parse_args()

    setup_logging("blendswap")
    config = load_config()

    output_dir = ensure_dir(args.output)
    progress = load_progress(output_dir / ".progress")
    logger.info(f"Resuming with {len(progress)} already processed")

    session = requests.Session()
    session.headers.update({
        "User-Agent": config.get("scraping", {}).get(
            "user_agent", "BlenderModelTraining/0.1")
    })

    bs_config = config.get("scraping", {}).get("blendswap", {})
    categories = args.categories or bs_config.get("categories", list(CATEGORIES.keys()))
    max_pages = args.max_pages or bs_config.get("max_pages_per_category", 50)

    for cat in categories:
        cat_key = cat.lower().strip()
        if cat_key in CATEGORIES:
            scrape_category(cat_key, CATEGORIES[cat_key],
                            output_dir, max_pages, session, progress, config)
        else:
            # Try fuzzy match
            matched = False
            for k in CATEGORIES:
                if cat_key in k or k in cat_key:
                    logger.info(f"Matched '{cat}' → '{k}'")
                    scrape_category(k, CATEGORIES[k],
                                    output_dir, max_pages, session, progress, config)
                    matched = True
                    break
            if not matched:
                logger.warning(f"Unknown category: {cat}. "
                               f"Available: {', '.join(CATEGORIES.keys())}")

    logger.info("BlendSwap scraping complete!")


if __name__ == "__main__":
    main()
