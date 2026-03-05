"""Scrape image captions and descriptions from Wikimedia Commons.

Wikimedia Commons hosts millions of CC-licensed images with rich
human-written descriptions — ideal for expanding our text tokenizer
vocabulary with real-world object/material/shape terminology.

We search for 3D-relevant categories and extract:
  - Image titles (descriptive filenames)
  - Descriptions (often multi-sentence)
  - Category tags (hierarchical labels)

All text is saved as JSONL for vocabulary expansion during training.

No authentication required — uses the public MediaWiki API.

Usage:
    python -m scrapers.wikimedia_scraper --output data/raw/wikimedia
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .utils import setup_logging, ensure_dir, load_progress, save_progress

logger = logging.getLogger(__name__)

API_URL = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": "BlenderCopilotTraining/0.1 (research; open-source 3D model training)",
}

# Categories rich in 3D/CG/object terminology
SEARCH_QUERIES = [
    "3D model blender rendering",
    "3D computer graphics wireframe",
    "3D rendering architecture building",
    "3D model vehicle car",
    "3D model character human",
    "3D model furniture interior",
    "3D model animal creature",
    "3D printed object",
    "computer generated image landscape",
    "low poly 3D model",
    "subdivision surface modeling",
    "mesh topology wireframe",
    "procedural texture material",
    "PBR material rendering",
    "architectural visualization rendering",
    "blender cycles render",
    "sculpted 3D model",
    "hard surface modeling",
    "3D model weapon armor",
    "3D model robot mechanical",
    "3D model tree plant nature",
    "3D model spaceship sci-fi",
    "isometric 3D rendering",
    "CAD model engineering",
    "polygon mesh geometry",
]

# Categories to browse directly
CATEGORIES = [
    "Category:3D computer graphics",
    "Category:Images created with Blender",
    "Category:Computer-generated images",
    "Category:3D rendering",
    "Category:Rendering techniques",
    "Category:Wireframe models",
    "Category:Computer-aided design",
    "Category:Mesh generation",
    "Category:3D modeling",
    "Category:Polygon meshes",
    "Category:Computer graphics algorithms",
    "Category:Procedural generation",
    "Category:Texture mapping",
    "Category:Normal mapping",
    "Category:UV mapping",
    "Category:3D animation",
    "Category:Architectural renders",
]


def _strip_html(text: str) -> str:
    """Remove HTML tags from Wikimedia descriptions."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def _clean_title(title: str) -> str:
    """Clean a Wikimedia file title into usable text."""
    title = title.replace("File:", "").replace("_", " ")
    title = re.sub(r"\.\w{2,4}$", "", title)  # Remove extension
    title = re.sub(r"\s+", " ", title).strip()
    return title


def search_images(query: str, limit: int = 50,
                  continue_token: str = "") -> tuple[list[dict], str]:
    """Search Wikimedia Commons for images matching a query.

    Returns (results, continue_token) where results is a list of dicts
    with 'title', 'description', 'categories' keys.
    """
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": "6",  # File namespace
        "gsrlimit": str(min(limit, 50)),
        "prop": "imageinfo",
        "iiprop": "extmetadata",
        "iiextmetadatafilter": "ImageDescription|ObjectName|Categories",
    }
    if continue_token:
        params["gsroffset"] = continue_token

    try:
        resp = requests.get(API_URL, params=params,
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Wikimedia search failed for '{query}': {e}")
        return [], ""

    results = []
    pages = data.get("query", {}).get("pages", {})

    for pid, page in pages.items():
        title = _clean_title(page.get("title", ""))
        ii = page.get("imageinfo", [{}])[0]
        ext = ii.get("extmetadata", {})

        desc = _strip_html(ext.get("ImageDescription", {}).get("value", ""))
        cats = ext.get("Categories", {}).get("value", "")

        # Skip if no useful text
        if not title and not desc:
            continue

        results.append({
            "title": title,
            "description": desc[:1000],
            "categories": cats,
            "source": "wikimedia_commons",
            "query": query,
        })

    # Pagination token
    cont = data.get("continue", {}).get("gsroffset", "")
    return results, str(cont) if cont else ""


def browse_category(category: str,
                    limit: int = 50) -> list[dict]:
    """Browse a Wikimedia Commons category for file descriptions."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": str(min(limit, 50)),
        "prop": "imageinfo",
        "iiprop": "extmetadata",
        "iiextmetadatafilter": "ImageDescription|Categories",
    }

    try:
        resp = requests.get(API_URL, params=params,
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Wikimedia category browse failed for '{category}': {e}")
        return []

    results = []
    pages = data.get("query", {}).get("pages", {})

    for pid, page in pages.items():
        title = _clean_title(page.get("title", ""))
        ii = page.get("imageinfo", [{}])[0]
        ext = ii.get("extmetadata", {})

        desc = _strip_html(ext.get("ImageDescription", {}).get("value", ""))
        cats = ext.get("Categories", {}).get("value", "")

        if not title and not desc:
            continue

        results.append({
            "title": title,
            "description": desc[:1000],
            "categories": cats,
            "source": "wikimedia_commons",
            "category": category,
        })

    return results


def scrape_wikimedia(output_dir: str = "data/raw/wikimedia",
                     max_per_query: int = 200,
                     max_per_category: int = 100) -> int:
    """Full scrape: search queries + category browsing.

    Returns total number of text entries saved.
    """
    out_path = ensure_dir(output_dir)
    texts_file = out_path / "captions.jsonl"
    progress = load_progress(out_path / ".progress")

    total = 0

    # 1. Search queries
    for query in SEARCH_QUERIES:
        if query in progress:
            continue

        logger.info(f"  Wikimedia search: '{query}'")
        fetched = 0
        cont = ""

        while fetched < max_per_query:
            results, cont = search_images(
                query, limit=50, continue_token=cont
            )
            if not results:
                break

            with open(texts_file, "a") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")

            fetched += len(results)
            total += len(results)

            if not cont:
                break
            time.sleep(1.0)  # Be polite to API

        save_progress(out_path / ".progress", query)
        time.sleep(0.5)

    # 2. Category browsing
    for cat in CATEGORIES:
        if cat in progress:
            continue

        logger.info(f"  Wikimedia category: {cat}")
        results = browse_category(cat, limit=max_per_category)

        if results:
            with open(texts_file, "a") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            total += len(results)

        save_progress(out_path / ".progress", cat)
        time.sleep(1.0)

    logger.info(f"Wikimedia Commons: saved {total} captions to {texts_file}")
    return total


def scrape_batch(output_dir: str = "data/raw/wikimedia",
                 batch_size: int = 200) -> int:
    """Pull a batch of captions. Used by BackgroundDataPuller.

    Returns number of new text entries.
    """
    return scrape_wikimedia(
        output_dir=output_dir,
        max_per_query=batch_size,
        max_per_category=50,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Scrape image captions from Wikimedia Commons"
    )
    parser.add_argument("--output", default="data/raw/wikimedia")
    parser.add_argument("--max-per-query", type=int, default=200)
    args = parser.parse_args()

    setup_logging("wikimedia")
    scrape_wikimedia(output_dir=args.output, max_per_query=args.max_per_query)


if __name__ == "__main__":
    main()
