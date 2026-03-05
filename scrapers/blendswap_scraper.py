"""Scrape free .blend files from BlendSwap.com.

BlendSwap hosts thousands of free Blender models under CC-0 and CC-BY licenses.
This scraper downloads .blend files and their metadata (title, description,
category, tags, license).

Requires a free BlendSwap account — set BLENDSWAP_EMAIL and BLENDSWAP_PASSWORD
in .env (downloads require login).

Usage:
    python -m scrapers.blendswap_scraper --output data/raw/blendswap
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Iterator

import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

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


def create_session() -> requests.Session:
    """Create a requests session, logging in if credentials are available.

    BlendSwap requires login to download files. Set BLENDSWAP_EMAIL and
    BLENDSWAP_PASSWORD in .env.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    })

    email = os.environ.get("BLENDSWAP_EMAIL", "").strip()
    password = os.environ.get("BLENDSWAP_PASSWORD", "").strip()

    if email and password:
        _login(session, email, password)
    else:
        logger.warning(
            "No BLENDSWAP_EMAIL / BLENDSWAP_PASSWORD in .env — "
            "downloads will fail (login required). "
            "Set credentials in .env and restart."
        )

    return session


def _login(session: requests.Session, email: str, password: str) -> bool:
    """Log in to BlendSwap via POST /login with CSRF token."""
    login_url = f"{BASE_URL}/login"

    try:
        resp = session.get(login_url, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        csrf_input = soup.find("input", {"name": "csrf_token"})
        csrf_token = csrf_input["value"] if csrf_input else ""

        if not csrf_token:
            logger.warning("BlendSwap login failed: no CSRF token found")
            return False

        resp = session.post(
            login_url,
            data={
                "csrf_token": csrf_token,
                "email": email,
                "password": password,
                "remember": "on",
                "next": "",
            },
            headers={"Referer": login_url},
            timeout=30,
            allow_redirects=True,
        )

        if "/login" not in resp.url:
            logger.info(f"Logged in to BlendSwap as {email}")
            return True
        else:
            logger.warning(
                f"BlendSwap login failed for {email} — check credentials"
            )
            return False

    except Exception as e:
        logger.warning(f"BlendSwap login error: {e}")
        return False


def get_listing_items(session: requests.Session, category_name: str,
                      category_id: int, max_pages: int = 3) -> Iterator[dict]:
    """Iterate listings for a BlendSwap category sorted by popularity (likes).

    Fetches pages 1‥max_pages.  Each item yielded already contains url, id,
    title, license and category.  The caller is responsible for fetching
    the full detail (download_url) via get_blend_detail() when needed.

    Uses BlendSwap's ?sort_by=likes query param for popular-first ordering.
    """
    for page in range(1, max_pages + 1):
        listings = get_blend_listings(category_id, category_name, page,
                                      session, sort_by="likes")
        if not listings:
            break
        for listing in listings:
            yield listing
        time.sleep(0.5)


def get_blend_listings(category_id: int, category_name: str, page: int,
                       session: requests.Session,
                       sort_by: str = "likes") -> list[dict]:
    """Fetch one page of .blend listings for a category.

    URLs follow the pattern:
      /blends/category/{id}?sort_by=likes       — page 1
      /blends/category/{id}/{page}?sort_by=likes — page 2+

    sort_by: 'likes' (popular), 'date' (recent), 'name'

    Returns list of dicts with keys: id, title, url, license.
    """
    if page == 1:
        url = f"{BASE_URL}/blends/category/{category_id}?sort_by={sort_by}"
    else:
        url = f"{BASE_URL}/blends/category/{category_id}/{page}?sort_by={sort_by}"

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
    """Fetch detail page for a single blend to get download link + metadata.

    When logged in, the sidebar card-body contains a download button.
    When not logged in, it shows "You must be logged in to download."
    """
    try:
        resp = session.get(blend_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to fetch detail {blend_url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract blend ID from URL for constructing download path
    bid_match = re.search(r"/blend/(\d+)", blend_url)
    blend_id = bid_match.group(1) if bid_match else ""

    # Try to find download link (only visible when logged in)
    download_link = None

    # Method 1: direct link in sidebar card-body
    card_body = soup.select_one(".card-body.text-center, .card-body")
    if card_body:
        dl_btn = card_body.find("a", href=True)
        if dl_btn:
            href = dl_btn["href"]
            if not href.startswith("http"):
                href = BASE_URL + href
            download_link = href

    # Method 2: any link with download in href
    if not download_link:
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if "download" in href.lower() and blend_id in href:
                if not href.startswith("http"):
                    href = BASE_URL + href
                download_link = href
                break

    # Method 3: construct the likely URL pattern
    if not download_link and blend_id:
        # Common patterns: /blend/{id}/download or /blends/download/{id}
        # We'll try these when actually downloading
        download_link = f"{BASE_URL}/blend/{blend_id}/download"

    # Check if login is required
    if not download_link:
        login_text = soup.find(string=re.compile(r"logged in to download", re.I))
        if login_text:
            logger.debug(f"Blend {blend_id}: login required for download")

    # Extract description
    desc_el = soup.select_one(".blend-description, .description, .detail-text")
    if not desc_el:
        # Try the main content area
        main_col = soup.select_one(".col-lg-9, .col-md-8")
        if main_col:
            desc_el = main_col.find("p")
    description = desc_el.get_text(strip=True) if desc_el else ""

    # Extract tags
    tags = []
    for tag_el in soup.select(".tag, .badge-tag, .blend-tag, .badge"):
        tag_text = tag_el.get_text(strip=True)
        if tag_text and len(tag_text) < 50:
            tags.append(tag_text)

    # Extract stats from sidebar list
    stats = {}
    for li in soup.select(".list-group-item"):
        text = li.get_text(strip=True)
        if "Downloads" in text:
            nums = re.findall(r"\d[\d,]*", text)
            if nums:
                try:
                    stats["downloads"] = int(nums[0].replace(",", ""))
                except ValueError:
                    pass
        if "Likes" in text:
            nums = re.findall(r"\d[\d,]*", text)
            if nums:
                try:
                    stats["likes"] = int(nums[0].replace(",", ""))
                except ValueError:
                    pass

    # Extract license
    license_text = ""
    for li in soup.select(".list-group-item"):
        text = li.get_text(strip=True)
        if "License:" in text:
            license_text = text.replace("License:", "").strip()

    # Extract render engine
    render = ""
    for li in soup.select(".list-group-item"):
        text = li.get_text(strip=True)
        if "Render:" in text:
            render = text.replace("Render:", "").strip()

    return {
        "download_url": download_link,
        "description": description,
        "tags": tags,
        "stats": stats,
        "license": license_text,
        "render_engine": render,
    }


def scrape_category(category_name: str, category_id: int, output_dir: Path,
                    max_pages: int, session: requests.Session, progress: set,
                    config: dict, all_licenses: bool = False):
    """Scrape all pages of a category.

    all_licenses=True skips the CC license filter — use when you have a
    paid subscription that allows downloading all content.
    """
    cat_dir = ensure_dir(output_dir / category_name)
    logger.info(f"Scraping category: {category_name} (id={category_id}, max {max_pages} pages)")

    max_size = config.get("scraping", {}).get("max_file_size_mb", 500)

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

            # Check license — skip filter when using subscription (all_licenses)
            if not all_licenses:
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
                session=session,
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

    session = create_session()

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


def scrape_blendswap(output_dir: str, config: dict | None = None):
    """Entry point called by run.py cmd_scrape."""
    setup_logging("blendswap")
    if config is None:
        config = load_config()

    output_dir = ensure_dir(output_dir)
    progress = load_progress(output_dir / ".progress")
    logger.info(f"Resuming with {len(progress)} already processed")

    session = create_session()

    bs_config = config.get("scraping", {}).get("blendswap", {})
    categories = bs_config.get("categories", list(CATEGORIES.keys()))
    max_pages = bs_config.get("max_pages_per_category", 50)

    for cat in categories:
        cat_key = cat.lower().strip()
        if cat_key in CATEGORIES:
            scrape_category(cat_key, CATEGORIES[cat_key],
                            output_dir, max_pages, session, progress, config)
        else:
            logger.warning(f"Unknown category: {cat}")

    logger.info("BlendSwap scraping complete!")


if __name__ == "__main__":
    main()
