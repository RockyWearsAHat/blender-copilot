"""Scrape 3D / CG terminology from public glossaries and documentation.

Expands the text tokenizer vocabulary with domain-specific terms that
users would type as prompts: "chamfer", "bevel", "subdivide", "manifold",
"topology", "UV unwrap", "PBR", etc.

Sources (all public, no auth required):
  - Wikipedia: Glossary of computer graphics (63K+ chars)
  - Wikipedia: 3D modeling article
  - Wikipedia: Polygon mesh article
  - Wikipedia: List of common 3D test models
  - Polycount Wiki (3D art terminology)

Text is saved as JSONL with term/definition pairs for vocabulary building.

Usage:
    python -m scrapers.terminology_scraper --output data/raw/terminology
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

HEADERS = {
    "User-Agent": "BlenderCopilotTraining/0.1 (research; open-source 3D model training)",
}

# Wikipedia articles to extract terminology from
WIKIPEDIA_ARTICLES = [
    "Glossary_of_computer_graphics",
    "3D_modeling",
    "Polygon_mesh",
    "Subdivision_surface",
    "Boolean_operations_on_polygons",
    "Constructive_solid_geometry",
    "Non-uniform_rational_B-spline",
    "B%C3%A9zier_curve",
    "UV_mapping",
    "Texture_mapping",
    "Normal_mapping",
    "Bump_mapping",
    "Displacement_mapping",
    "Physically_based_rendering",
    "Ray_tracing_(graphics)",
    "Rasterisation",
    "Radiosity_(computer_graphics)",
    "Ambient_occlusion",
    "Screen_space_ambient_occlusion",
    "Phong_shading",
    "Gouraud_shading",
    "Flat_shading",
    "Specular_highlight",
    "Fresnel_equations",
    "Subsurface_scattering",
    "Caustic_(optics)",
    "Global_illumination",
    "Path_tracing",
    "Mesh_generation",
    "Marching_cubes",
    "Delaunay_triangulation",
    "Voronoi_diagram",
    "Laplacian_smoothing",
    "Catmull%E2%80%93Clark_subdivision_surface",
    "Loop_subdivision_surface",
    "Edge_loop",
    "Topology_(electrical_circuits)",
    "Manifold",
    "Euler_characteristic",
    "Convex_hull",
    "Bounding_volume",
    "Octree",
    "Binary_space_partitioning",
    "Level_of_detail_(computer_graphics)",
    "Polygon_mesh",
    "Triangle_mesh",
    "Quad_(geometry)",
    "Vertex_(geometry)",
    "Edge_(geometry)",
    "Face_(geometry)",
    "Extrusion",
    "Chamfer_(geometry)",
    "Fillet_(mechanics)",
    "Taper",
    "Loft_(3D)",
    "Sweep_(geometry)",
    "Keyframe",
    "Skeletal_animation",
    "Inverse_kinematics",
    "Armature_(computer_animation)",
    "Rigging_(computer_graphics)",
    "Morph_target_animation",
    "Particle_system",
    "Hair_(modeling)",
    "Cloth_simulation",
    "Fluid_simulation",
    "Rigid_body_dynamics",
    "Soft_body_dynamics",
    "Depth_of_field",
    "Motion_blur",
    "Bloom_(shader_effect)",
    "Volumetric_lighting",
    "High-dynamic-range_rendering",
    "Tone_mapping",
    "Color_grading",
    "Compositing",
    "Render_pass",
    "Anti-aliasing",
    "Anisotropic_filtering",
    "Mipmap",
    "Shader",
    "Vertex_shader",
    "Fragment_shader",
    "Geometry_shader",
    "Compute_shader",
    "Signed_distance_function",
    "Metaball",
    "Isosurface",
    "Point_cloud",
    "Voxel",
    "NeRF",
    "Gaussian_splatting",
]

# Polycount Wiki pages with rich 3D art terminology
POLYCOUNT_PAGES = [
    "http://wiki.polycount.com/wiki/Terminology",
    "http://wiki.polycount.com/wiki/Subdivision_Surface_Modeling",
    "http://wiki.polycount.com/wiki/Hard_Surface",
    "http://wiki.polycount.com/wiki/Texture_Types",
    "http://wiki.polycount.com/wiki/Edge_Flow",
    "http://wiki.polycount.com/wiki/Topology",
    "http://wiki.polycount.com/wiki/UV_Layout",
    "http://wiki.polycount.com/wiki/Normal_Map",
    "http://wiki.polycount.com/wiki/Ambient_Occlusion",
    "http://wiki.polycount.com/wiki/Polycount",
]

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def _parse_wikitext_glossary(wikitext: str) -> list[dict]:
    """Parse Wikipedia glossary wikitext into term/definition pairs."""
    entries = []

    # Match {{term|...}} / {{defn|...}} patterns
    term_pattern = re.compile(
        r"\{\{(?:term|anchor)\|([^}]+)\}\}\s*\n\s*\{\{defn\|([^}]+)\}\}",
        re.IGNORECASE,
    )

    for match in term_pattern.finditer(wikitext):
        term = match.group(1).strip()
        definition = match.group(2).strip()

        # Clean wiki markup
        term = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", term)
        definition = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", definition)
        definition = re.sub(r"\{\{[^}]+\}\}", "", definition)
        definition = re.sub(r"<[^>]+>", "", definition)
        definition = definition.strip()

        if term and definition and len(definition) > 10:
            entries.append({
                "term": term,
                "definition": definition[:500],
                "source": "wikipedia_glossary",
            })

    # Also grab == Section == headers and following paragraph text
    section_pattern = re.compile(
        r"^===?\s*(.+?)\s*===?\s*$(.+?)(?=^===?|\Z)",
        re.MULTILINE | re.DOTALL,
    )

    for match in section_pattern.finditer(wikitext):
        heading = match.group(1).strip()
        body = match.group(2).strip()

        # Clean wiki markup from body
        body = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", body)
        body = re.sub(r"\{\{[^}]+\}\}", "", body)
        body = re.sub(r"<[^>]+>", "", body)
        body = re.sub(r"\n+", " ", body).strip()

        if heading and body and len(body) > 20:
            entries.append({
                "term": heading,
                "definition": body[:500],
                "source": "wikipedia_article",
            })

    return entries


def fetch_wikipedia_article(title: str) -> list[dict]:
    """Fetch a Wikipedia article and extract terminology entries."""
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "revisions|extracts",
        "rvprop": "content",
        "rvslots": "main",
        "exintro": False,
        "explaintext": True,
        "exsectionformat": "plain",
    }

    try:
        resp = requests.get(WIKIPEDIA_API, params=params,
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed for '{title}': {e}")
        return []

    pages = data.get("query", {}).get("pages", {})
    entries = []

    for pid, page in pages.items():
        if int(pid) < 0:
            logger.debug(f"Wikipedia article not found: {title}")
            continue

        # Try wikitext parsing first (for glossary pages)
        revisions = page.get("revisions", [])
        if revisions:
            wikitext = (revisions[0].get("slots", {})
                        .get("main", {}).get("*", ""))
            if wikitext:
                entries.extend(_parse_wikitext_glossary(wikitext))

        # Also grab the plain-text extract
        extract = page.get("extract", "")
        if extract and len(extract) > 50:
            clean_title = title.replace("_", " ").replace("%C3%A9", "e")
            clean_title = re.sub(r"%[0-9A-F]{2}", "", clean_title)

            # Split into paragraphs
            paragraphs = [p.strip() for p in extract.split("\n\n") if len(p.strip()) > 30]
            for i, para in enumerate(paragraphs[:10]):
                entries.append({
                    "term": clean_title if i == 0 else f"{clean_title} ({i})",
                    "definition": para[:500],
                    "source": "wikipedia_article",
                })

    return entries


def fetch_polycount_page(url: str) -> list[dict]:
    """Scrape a Polycount Wiki page for terminology."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return []
    except Exception as e:
        logger.warning(f"Polycount fetch failed for {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        return []

    entries = []

    # Extract dt/dd pairs (definition lists)
    for dt in content.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        if dd:
            term = dt.get_text(strip=True)
            definition = dd.get_text(strip=True)
            if term and definition and len(definition) > 10:
                entries.append({
                    "term": term,
                    "definition": definition[:500],
                    "source": "polycount_wiki",
                })

    # Extract headers + following paragraphs
    for heading in content.find_all(["h2", "h3", "h4"]):
        term = heading.get_text(strip=True)
        # Skip edit links and TOC
        if term.lower() in ("contents", "navigation menu", "edit"):
            continue

        # Gather following paragraphs until next heading
        paragraphs = []
        for sibling in heading.find_next_siblings():
            if sibling.name in ("h2", "h3", "h4"):
                break
            if sibling.name == "p":
                text = sibling.get_text(strip=True)
                if text and len(text) > 10:
                    paragraphs.append(text)

        if paragraphs:
            entries.append({
                "term": term,
                "definition": " ".join(paragraphs)[:500],
                "source": "polycount_wiki",
            })

    return entries


def scrape_terminology(output_dir: str = "data/raw/terminology") -> int:
    """Full scrape of all terminology sources.

    Returns total number of term entries saved.
    """
    out_path = ensure_dir(output_dir)
    terms_file = out_path / "terms.jsonl"
    progress = load_progress(out_path / ".progress")

    total = 0

    # 1. Wikipedia articles
    for article in WIKIPEDIA_ARTICLES:
        if article in progress:
            continue

        clean_name = article.replace("_", " ")[:50]
        logger.info(f"  Wikipedia: {clean_name}")

        entries = fetch_wikipedia_article(article)
        if entries:
            with open(terms_file, "a") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
            total += len(entries)
            logger.debug(f"    {len(entries)} entries from {clean_name}")

        save_progress(out_path / ".progress", article)
        time.sleep(1.0)  # Rate limit

    # 2. Polycount Wiki
    for url in POLYCOUNT_PAGES:
        page_key = f"polycount:{url.split('/')[-1]}"
        if page_key in progress:
            continue

        logger.info(f"  Polycount: {url.split('/')[-1]}")
        entries = fetch_polycount_page(url)
        if entries:
            with open(terms_file, "a") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
            total += len(entries)

        save_progress(out_path / ".progress", page_key)
        time.sleep(2.0)

    logger.info(f"Terminology scraper: saved {total} entries to {terms_file}")
    return total


def scrape_batch(output_dir: str = "data/raw/terminology") -> int:
    """Pull terminology. Used by BackgroundDataPuller.

    This is mostly one-shot (finite sources), but is idempotent
    thanks to progress tracking.
    """
    return scrape_terminology(output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Scrape 3D/CG terminology from public glossaries"
    )
    parser.add_argument("--output", default="data/raw/terminology")
    args = parser.parse_args()

    setup_logging("terminology")
    scrape_terminology(output_dir=args.output)


if __name__ == "__main__":
    main()
