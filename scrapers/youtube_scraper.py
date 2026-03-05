"""Scrape Blender tutorial transcripts from curated YouTube channels.

The real training value from YouTube is in the *video content itself* -
step-by-step narration of professional Blender workflows. .blend files
from descriptions are usually templates or assets, not the final result.

Strategy:
  1. Pull from CURATED CHANNELS - recognized pros, not random uploaders
  2. TRANSCRIPTS are the primary data - timestamped workflow narration
  3. .blend files are downloaded WHEN AVAILABLE, but templates/starter
     assets are skipped - we only want FINAL PRODUCTS
  4. Videos must meet quality thresholds (views, duration, relevance)

Curated channels are organized into tiers:
  - Tier 1: Legendary educators (Blender Guru, CG Geek, etc.)
  - Tier 2: Professional studios and recognized artists
  - Tier 3: Solid mid-size channels with proven quality

Usage:
    python -m scrapers.youtube_scraper --output data/raw/youtube
    python -m scrapers.youtube_scraper --channels "Blender Guru,CG Geek"
    python -m scrapers.youtube_scraper --discover
"""

import argparse
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs, unquote

from .utils import (
    setup_logging, load_config, ensure_dir,
    load_progress, save_progress, download_file,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Curated channel registry
# ═══════════════════════════════════════════════════════════════════
# (handle, display_name, tier, specialty)

CURATED_CHANNELS = [
    # Tier 1: Legendary educators
    ("@blenderguru",        "Blender Guru",       1, "modeling, materials, lighting"),
    ("@CGGeek",             "CG Geek",            1, "environments, VFX, procedural"),
    ("@LarsMezaka",         "Lars Mezaka",        1, "hard surface, vehicles, product"),
    ("@GrantAbbitt",        "Grant Abbitt",       1, "beginner, characters, sculpting"),
    ("@RoyalSkiesLLC",      "Royal Skies LLC",    1, "characters, anime, game-ready"),
    ("@CGCookie",           "CG Cookie",          1, "comprehensive, all topics"),
    ("@DuckyDGS",           "Ducky 3D",           1, "procedural, abstract, geo nodes"),
    # Tier 2: Professional studios / recognized artists
    ("@IanHubert2",         "Ian Hubert",         2, "VFX, lazy tutorials, environments"),
    ("@FlippedNormals",     "FlippedNormals",     2, "sculpting, characters, industry"),
    ("@SouthernShotty3D",   "SouthernShotty",     2, "stylized, game assets"),
    ("@CGMatter",           "CG Matter",          2, "quick tips, nodes, shading"),
    ("@DefaultCube",        "Default Cube",       2, "tips, tricks, workflow"),
    ("@Polyfjord",          "Polyfjord",          2, "isometric, stylized, animation"),
    ("@CrossMindStudio",    "CrossMind Studio",   2, "hard surface, mechanical"),
    ("@JoshGambrell",       "Josh Gambrell",      2, "hard surface, product vis"),
    ("@DerekElliott",       "Derek Elliott",      2, "environments, scenes"),
    ("@MaxHayArt",          "Max Hay",            2, "environments, procedural"),
    ("@CurtisHolt",         "Curtis Holt",        2, "news, tutorials, addons"),
    ("@CGBoost",            "CG Boost",           2, "structured courses, all levels"),
    ("@Imphenzia",          "Imphenzia",          2, "low poly, game dev, 10-min models"),
    ("@PieterSophwortel",   "Pieter Sophwortel",  2, "archviz, interiors"),
    ("@RyanKingArt",        "Ryan King Art",      2, "speed modeling, environments"),
    # Tier 3: Solid mid-size quality channels
    ("@KevBinge",           "Kev Binge",          3, "hard surface, vehicles"),
    ("@ArjanBrussee",       "Arjan Brussee",      3, "geometry nodes, procedural"),
    ("@JoeyCarson3D",       "Joey Carlino",       3, "stylized, characters"),
    ("@Thilakanathan",      "CG Figures",         3, "character modeling"),
    ("@CBaileyFilm",        "CBailey Film",       3, "VFX, compositing"),
    ("@DECODED",            "DECODED",            3, "geometry nodes, procedural"),
    ("@Grabbitt",           "Grabbitt",           3, "sculpting, organic"),
    ("@KaizerTutorials",    "Kaizer",             3, "hard surface, details"),
    ("@BlenderBros",        "Blender Bros",       3, "game assets, environments"),
    ("@olav3d",             "Olav3D",             3, "tutorials, quick tips"),
    ("@ChamferZone",        "Chamfer Zone",       3, "hard surface, mechanical"),
    ("@RealTimeTutorial",   "Bad Normals",        3, "game dev, optimization"),
]

_MIN_VIEWS_TIER = {1: 5_000, 2: 10_000, 3: 20_000}
_MIN_DURATION_SECS = 180        # Skip shorts / teasers
_MAX_DURATION_SECS = 14400      # Skip 4+ hour livestreams
_MIN_VIEWS_DISCOVERY = 50_000   # For discovering new channels

# ═══════════════════════════════════════════════════════════════════
# Transcript handling - the PRIMARY training data
# ═══════════════════════════════════════════════════════════════════

def get_transcript_with_timestamps(video_id):
    """Fetch transcript with timestamps. Returns dict or None."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        segments = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join(seg["text"] for seg in segments)
        lines = []
        for seg in segments:
            m, s = int(seg["start"] // 60), int(seg["start"] % 60)
            lines.append(f"[{m:02d}:{s:02d}] {seg['text']}")
        return {
            "full_text": full_text,
            "segments": segments,
            "timestamped_text": "\n".join(lines),
            "duration_from_transcript": (
                segments[-1]["start"] + segments[-1].get("duration", 0)
                if segments else 0),
        }
    except Exception as e:
        logger.debug(f"No transcript for {video_id}: {e}")
        return None


def extract_workflow_steps(segments):
    """Extract structured workflow steps from transcript segments."""
    keywords = [
        "add a", "create a", "let's make", "go ahead and",
        "next step", "first step", "now we", "select all", "tab into",
        "edit mode", "object mode", "sculpt mode", "extrude", "loop cut",
        "bevel", "subdivide", "boolean", "mirror", "array", "scale",
        "rotate", "grab", "move", "delete", "merge", "separate",
        "modifier", "material", "shader", "texture", "uv unwrap",
        "ctrl r", "ctrl b", "shift a", "apply", "smooth shade",
        "geometry nodes", "node editor", "principled",
    ]
    steps = []
    for seg in segments:
        low = seg["text"].lower()
        hits = [kw for kw in keywords if kw in low]
        if hits:
            steps.append({"time": seg["start"], "text": seg["text"],
                          "keywords": hits})
    return steps


def score_transcript_quality(transcript_data, video_meta):
    """Score transcript usefulness for training (0.0 - 1.0)."""
    if not transcript_data:
        return 0.0
    text = transcript_data["full_text"].lower()
    wc = len(text.split())
    if wc < 50:
        return 0.0

    blender_terms = [
        "blender", "mesh", "vertex", "vertices", "edge", "face", "polygon",
        "extrude", "loop cut", "bevel", "subdivide", "modifier", "boolean",
        "mirror", "array", "solidify", "subsurf", "subdivision surface",
        "material", "shader", "node", "principled", "texture", "uv",
        "sculpt", "geometry nodes", "procedural", "cycles", "eevee",
        "render", "viewport", "object mode", "edit mode", "tab",
        "scale", "rotate", "grab", "ctrl r", "shift a", "alt",
        "smooth shade", "flat shade", "normal", "topology",
        "retopology", "weight paint", "armature", "rigging",
    ]
    term_hits = sum(1 for t in blender_terms if t in text)
    density = term_hits / len(blender_terms)

    instructional = [
        "we're going to", "let's", "go ahead", "make sure",
        "you can see", "notice how", "the next step", "now we",
        "i'm going to", "you want to", "you need to",
        "select", "click", "drag", "press", "type",
    ]
    instr_hits = sum(1 for p in instructional if p in text)
    instr_density = min(instr_hits / 10.0, 1.0)

    dur = video_meta.get("duration", 600)
    wpm = (wc / max(dur, 1)) * 60
    wpm_score = 1.0 if 80 <= wpm <= 250 else 0.5

    if density < 0.05:
        return 0.1

    return round(min(
        density * 0.40 + instr_density * 0.35 +
        wpm_score * 0.15 + min(term_hits / 20.0, 1.0) * 0.10,
        1.0), 3)


# ═══════════════════════════════════════════════════════════════════
# .blend file handling - OPTIONAL supplement
# ═══════════════════════════════════════════════════════════════════

# Template indicators - these suggest the .blend is a starter/asset, not output
_TEMPLATE_KEYWORDS = [
    "starter file", "template", "base mesh", "starting point",
    "asset pack", "free asset", "pre-made", "premade",
    "kit", "kitbash", "download below to follow along",
    "follow along file", "starting file",
    # Additional template/starter patterns
    "start file", "startup file", "begin file",
    "reference file", "practice file", "exercise file",
    "resource pack", "resource file", "resources below",
    "before we start", "use this file", "grab the file",
    "free model pack", "scene setup", "prepared scene",
    "pre-built", "prebuilt", "pre-configured",
    "follow along with", "download to follow",
]

# Final product indicators - presence of these OVERRIDES template detection
_FINAL_PRODUCT_KEYWORDS = [
    "finished file", "final file", "final result",
    "completed project", "final project", "end result",
    "finished project", "final scene", "completed scene",
    "the result", "what we made", "what we built",
    "final render", "finished model", "completed model",
]

# ── Blend file link patterns ────────────────────────────────────────

# Regex patterns for common file hosting services
_BLEND_LINK_PATTERNS = [
    # Direct .blend links
    re.compile(r'(https?://\S+\.blend)\b', re.IGNORECASE),
    # Google Drive
    re.compile(r'(https?://drive\.google\.com/\S+)', re.IGNORECASE),
    # Dropbox
    re.compile(r'(https?://(?:www\.)?dropbox\.com/\S+)', re.IGNORECASE),
    # Mega
    re.compile(r'(https?://mega\.nz/\S+)', re.IGNORECASE),
    # Gumroad (many free .blend downloads)
    re.compile(r'(https?://(?:www\.)?gumroad\.com/l/\S+)', re.IGNORECASE),
    re.compile(r'(https?://\S+\.gumroad\.com/l/\S+)', re.IGNORECASE),
    # MediaFire
    re.compile(r'(https?://(?:www\.)?mediafire\.com/\S+)', re.IGNORECASE),
    # GitHub releases/raw links
    re.compile(r'(https?://github\.com/\S+\.blend\S*)', re.IGNORECASE),
    re.compile(r'(https?://raw\.githubusercontent\.com/\S+\.blend\S*)', re.IGNORECASE),
    # Patreon (often has free tiers with .blend)
    re.compile(r'(https?://(?:www\.)?patreon\.com/posts/\S+)', re.IGNORECASE),
    # Generic zip/rar that might contain .blend
    re.compile(r'(https?://\S+\.(?:zip|rar|7z))\b', re.IGNORECASE),
    # Blender Market
    re.compile(r'(https?://(?:www\.)?blendermarket\.com/\S+)', re.IGNORECASE),
    # CGTrader free
    re.compile(r'(https?://(?:www\.)?cgtrader\.com/free-3d-models/\S+)', re.IGNORECASE),
    # Sketchfab (downloadable)
    re.compile(r'(https?://sketchfab\.com/3d-models/\S+)', re.IGNORECASE),
]

# Keywords that suggest the link is a .blend file download
_BLEND_CONTEXT_KEYWORDS = [
    'blend file', '.blend', 'project file', 'download file',
    'source file', 'blender file', 'download the project',
    'free download', 'get the file', 'link below',
    'download link', 'project download', 'scene file',
]


def extract_blend_links(description: str) -> list[dict]:
    """Extract potential .blend file download links from a video description.

    Returns list of dicts with 'url', 'source_type', and 'confidence'.
    """
    if not description:
        return []

    links = []
    seen_urls = set()
    desc_lower = description.lower()

    # Check if description mentions blend files at all
    has_blend_context = any(kw in desc_lower for kw in _BLEND_CONTEXT_KEYWORDS)
    is_template = any(kw in desc_lower for kw in _TEMPLATE_KEYWORDS)
    is_final = any(kw in desc_lower for kw in _FINAL_PRODUCT_KEYWORDS)
    # Final product keywords override template detection
    if is_final:
        is_template = False

    for pattern in _BLEND_LINK_PATTERNS:
        for match in pattern.finditer(description):
            url = match.group(1).rstrip('.,;:!?)>"\']')
            if url in seen_urls:
                continue
            seen_urls.add(url)

            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Determine source type and confidence
            if url.lower().endswith('.blend'):
                links.append({
                    'url': url,
                    'source_type': 'direct_blend',
                    'confidence': 1.0,
                    'is_likely_template': is_template,
                })
            elif 'drive.google.com' in domain:
                links.append({
                    'url': _normalize_gdrive_url(url),
                    'source_type': 'google_drive',
                    'confidence': 0.7 if has_blend_context else 0.3,
                    'is_likely_template': is_template,
                })
            elif 'dropbox.com' in domain:
                dl_url = url.replace('?dl=0', '?dl=1')
                if '?dl=' not in dl_url:
                    dl_url += '?dl=1'
                links.append({
                    'url': dl_url,
                    'source_type': 'dropbox',
                    'confidence': 0.7 if has_blend_context else 0.3,
                    'is_likely_template': is_template,
                })
            elif 'gumroad.com' in domain:
                links.append({
                    'url': url,
                    'source_type': 'gumroad',
                    'confidence': 0.6 if has_blend_context else 0.2,
                    'is_likely_template': is_template,
                })
            elif 'mega.nz' in domain:
                links.append({
                    'url': url,
                    'source_type': 'mega',
                    'confidence': 0.6 if has_blend_context else 0.3,
                    'is_likely_template': is_template,
                })
            elif 'mediafire.com' in domain:
                links.append({
                    'url': url,
                    'source_type': 'mediafire',
                    'confidence': 0.6 if has_blend_context else 0.3,
                    'is_likely_template': is_template,
                })
            elif 'github.com' in domain:
                links.append({
                    'url': url,
                    'source_type': 'github',
                    'confidence': 0.8 if '.blend' in url.lower() else 0.4,
                    'is_likely_template': is_template,
                })
            elif url.lower().endswith(('.zip', '.rar', '.7z')):
                links.append({
                    'url': url,
                    'source_type': 'archive',
                    'confidence': 0.5 if has_blend_context else 0.1,
                    'is_likely_template': is_template,
                })
            else:
                links.append({
                    'url': url,
                    'source_type': 'other',
                    'confidence': 0.3 if has_blend_context else 0.1,
                    'is_likely_template': is_template,
                })

    # Sort by confidence (best first)
    links.sort(key=lambda x: x['confidence'], reverse=True)
    return links


def _normalize_gdrive_url(url: str) -> str:
    """Convert Google Drive share links to direct download links."""
    parsed = urlparse(url)
    if 'drive.google.com' not in parsed.netloc:
        return url

    # Handle /file/d/FILE_ID/view format
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        return f'https://drive.google.com/uc?export=download&id={file_id}'

    # Handle ?id=FILE_ID format
    qs = parse_qs(parsed.query)
    if 'id' in qs:
        file_id = qs['id'][0]
        return f'https://drive.google.com/uc?export=download&id={file_id}'

    return url


def download_blend_files(links: list[dict], output_dir: Path,
                         video_id: str) -> list[str]:
    """Attempt to download .blend files from extracted links.

    Returns list of successfully downloaded file paths.
    """
    downloaded = []
    blend_dir = ensure_dir(output_dir / "blend_files")

    for i, link_info in enumerate(links):
        url = link_info['url']
        source_type = link_info['source_type']
        confidence = link_info['confidence']

        # Skip low-confidence links for non-direct types
        if confidence < 0.3 and source_type != 'direct_blend':
            continue

        # Skip complex services we can't easily download from
        if source_type in ('gumroad', 'patreon', 'blendermarket',
                           'cgtrader', 'sketchfab', 'mega'):
            # Just record the link for manual follow-up
            logger.debug(f"  Skipping {source_type} link (needs browser): {url}")
            continue

        filename = f"{video_id}_{i}"
        if source_type == 'direct_blend':
            filename += '.blend'
        elif source_type == 'archive':
            ext = Path(urlparse(url).path).suffix or '.zip'
            filename += ext
        else:
            filename += '.blend'

        out_path = blend_dir / filename

        try:
            success = download_file(
                url, out_path,
                max_size_mb=200,
                rate_limit_seconds=2.0,
            )
            if success and out_path.exists():
                # Verify it's actually a blend file or valid archive
                if _is_valid_download(out_path):
                    downloaded.append(str(out_path))
                    logger.info(
                        f"  Downloaded blend file: {filename} "
                        f"(from {source_type})"
                    )
                else:
                    out_path.unlink(missing_ok=True)
                    logger.debug(
                        f"  Downloaded file is not a valid blend/archive: "
                        f"{filename}"
                    )
        except Exception as e:
            logger.debug(f"  Failed to download {url}: {e}")

    return downloaded


def _is_valid_download(path: Path) -> bool:
    """Check if a downloaded file is a valid .blend or archive."""
    try:
        with open(path, 'rb') as f:
            magic = f.read(8)

        # .blend file magic
        if magic[:7] == b'BLENDER':
            return True

        # ZIP archive (might contain .blend)
        if magic[:4] == b'PK\x03\x04':
            return True

        # RAR archive
        if magic[:7] == b'Rar!\x1a\x07\x00' or magic[:8] == b'Rar!\x1a\x07\x01\x00':
            return True

        # 7z archive
        if magic[:6] == b'7z\xbc\xaf\x27\x1c':
            return True

        # HTML page (not a real file download)
        if b'<html' in magic.lower() or b'<!doctype' in magic.lower():
            return False

        return False
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
# Channel-based video discovery
# ═══════════════════════════════════════════════════════════════════

def get_channel_videos(handle, max_results=200):
    """Fetch recent videos from a specific YouTube channel handle."""
    results = []
    try:
        cmd = [
            "yt-dlp",
            f"https://www.youtube.com/{handle}/videos",
            "--dump-json",
            "--no-download",
            "--flat-playlist",
            "--playlist-end", str(max_results),
            "--quiet",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        for line in proc.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                results.append({
                    "id": data.get("id", ""),
                    "title": data.get("title", ""),
                    "url": data.get("url",
                        f"https://www.youtube.com/watch?v={data.get('id', '')}"),
                    "channel": data.get("channel", data.get("uploader", "")),
                    "channel_handle": handle,
                    "duration": data.get("duration"),
                    "view_count": data.get("view_count"),
                    "description": data.get("description", ""),
                })
            except json.JSONDecodeError:
                continue
    except subprocess.TimeoutExpired:
        logger.warning(f"Channel fetch timed out for {handle}")
    except FileNotFoundError:
        logger.error("yt-dlp not installed. Run: pip install yt-dlp")
    except Exception as e:
        logger.warning(f"Channel fetch failed for {handle}: {e}")
    return results


def filter_video(video, tier):
    """Check if a video passes quality thresholds for its tier."""
    dur = video.get("duration") or 0
    views = video.get("view_count") or 0
    title = (video.get("title") or "").lower()

    if dur < _MIN_DURATION_SECS or dur > _MAX_DURATION_SECS:
        return False
    if views < _MIN_VIEWS_TIER.get(tier, 10_000):
        return False

    skip_patterns = [
        "unboxing", "giveaway", "drama", "rant", "news",
        "live stream", "q&a", "podcast", "vlog",
        "review", "comparison",
    ]
    if any(p in title for p in skip_patterns):
        return False

    blender_hints = [
        "blender", "modeling", "tutorial", "sculpt", "shader",
        "material", "geometry node", "procedural", "render",
        "animation", "rigging", "timelapse", "speed model",
        "hard surface", "low poly", "how to", "beginner",
        "advanced", "donut", "archviz", "environment",
    ]
    if not any(h in title for h in blender_hints):
        return False

    return True


def discover_quality_channels(seed_queries=None, min_views=None):
    """Find new Blender tutorial channels by searching popular videos.

    Returns list of (handle, name, avg_views) for channels not already curated.
    """
    min_views = min_views or _MIN_VIEWS_DISCOVERY
    known_handles = {ch[0].lower() for ch in CURATED_CHANNELS}

    queries = seed_queries or [
        "blender tutorial 2024",
        "blender modeling tutorial",
        "blender geometry nodes",
        "blender hard surface modeling",
        "blender sculpting tutorial",
    ]

    channel_stats = {}
    for q in queries:
        logger.info(f"Discovery search: {q}")
        videos = _search_youtube(q, max_results=100)
        for v in videos:
            views = v.get("view_count") or 0
            ch = v.get("channel", "")
            if not ch or views < min_views:
                continue
            handle_guess = f"@{ch.replace(' ', '')}"
            if handle_guess.lower() in known_handles:
                continue
            if ch not in channel_stats:
                channel_stats[ch] = {"views": [], "handle_guess": handle_guess}
            channel_stats[ch]["views"].append(views)
        time.sleep(2)

    candidates = []
    for name, stats in channel_stats.items():
        avg = sum(stats["views"]) / len(stats["views"])
        if avg >= min_views and len(stats["views"]) >= 2:
            candidates.append((stats["handle_guess"], name, int(avg)))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


# ═══════════════════════════════════════════════════════════════════
# Legacy search wrapper (kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════════

def _search_youtube(query, max_results=500):
    """Search YouTube for videos matching a query. Uses yt-dlp."""

    results = []
    try:
        # Use yt-dlp to search and extract metadata without downloading
        cmd = [
            "yt-dlp",
            f"ytsearch{min(max_results, 100)}:{query}",
            "--dump-json",
            "--no-download",
            "--flat-playlist",
            "--quiet",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        for line in proc.stdout.strip().split("\n"):
            if line.strip():
                try:
                    data = json.loads(line)
                    results.append({
                        "id": data.get("id", ""),
                        "title": data.get("title", ""),
                        "url": data.get("url", f"https://www.youtube.com/watch?v={data.get('id', '')}"),
                        "channel": data.get("channel", data.get("uploader", "")),
                        "duration": data.get("duration"),
                        "view_count": data.get("view_count"),
                        "description": data.get("description", ""),
                    })
                except json.JSONDecodeError:
                    continue

    except subprocess.TimeoutExpired:
        logger.warning(f"Search timed out for: {query}")
    except FileNotFoundError:
        logger.error("yt-dlp not installed. Run: pip install yt-dlp")
    except Exception as e:
        logger.warning(f"Search failed for '{query}': {e}")

    return results


def get_transcript(video_id: str) -> str | None:
    """Fetch transcript/captions for a YouTube video."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Join all text segments
        full_text = " ".join(
            segment["text"] for segment in transcript_list
        )
        return full_text

    except Exception as e:
        logger.debug(f"No transcript for {video_id}: {e}")
        return None


def get_detailed_metadata(video_id: str) -> dict | None:
    """Get detailed video metadata using yt-dlp."""
    try:
        cmd = [
            "yt-dlp",
            f"https://www.youtube.com/watch?v={video_id}",
            "--dump-json",
            "--no-download",
            "--quiet",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout.strip())
    except Exception as e:
        logger.debug(f"Failed to get metadata for {video_id}: {e}")
    return None


def process_video(video, output_dir, download_blends=True):
    """Process a single video: transcript first, blend files optional.

    Returns dict with transcript_saved, blend_files, quality_score.
    """
    result = {"transcript_saved": False, "blend_files": [],
              "quality_score": 0.0, "workflow_steps": []}

    video_id = video.get("id", "")
    if not video_id:
        return result

    output_file = output_dir / f"{video_id}.json"

    detail = get_detailed_metadata(video_id)
    tags = []
    full_description = video.get("description", "")
    if detail:
        tags = detail.get("tags", [])
        full_description = detail.get("description", full_description)
        video["description"] = full_description
        video["like_count"] = detail.get("like_count")
        video["comment_count"] = detail.get("comment_count")
        if not video.get("duration"):
            video["duration"] = detail.get("duration")

    transcript_data = get_transcript_with_timestamps(video_id)

    quality = score_transcript_quality(transcript_data, video)
    result["quality_score"] = quality

    if quality < 0.15 and not transcript_data:
        logger.debug(f"Skipping {video_id}: no transcript, low quality")
        return result

    steps = []
    if transcript_data and transcript_data.get("segments"):
        steps = extract_workflow_steps(transcript_data["segments"])
        result["workflow_steps"] = steps

    blend_links = extract_blend_links(full_description)
    blend_files_downloaded = []
    if download_blends and blend_links:
        non_template = [l for l in blend_links if not l.get("is_likely_template")]
        template_count = len(blend_links) - len(non_template)
        if template_count:
            logger.info(
                f"  Skipped {template_count} template/starter .blend links "
                f"(only downloading final products)"
            )
        if non_template:
            logger.info(
                f"  {len(non_template)} non-template .blend links in: "
                f"{video.get('title', video_id)[:50]}"
            )
            blend_files_downloaded = download_blend_files(
                non_template, output_dir, video_id)
            result["blend_files"] = blend_files_downloaded

    record = {
        **video,
        "transcript": transcript_data["full_text"] if transcript_data else "",
        "timestamped_transcript": (
            transcript_data["timestamped_text"] if transcript_data else ""),
        "workflow_steps": steps,
        "quality_score": quality,
        "tags": tags,
        "source": "youtube",
        "blend_links": blend_links,
        "blend_files_downloaded": blend_files_downloaded,
        "has_blend_files": bool(blend_files_downloaded),
        "blend_links_template_flagged": sum(
            1 for l in blend_links if l.get("is_likely_template")),
    }

    with open(output_file, "w") as f:
        json.dump(record, f, indent=2)

    result["transcript_saved"] = True
    icon = "📦" if blend_files_downloaded else "📝"
    logger.info(
        f"  Saved [{icon} q={quality:.2f}]: "
        f"{video.get('title', video_id)[:60]}..."
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# Channel-based scraping orchestrator
# ═══════════════════════════════════════════════════════════════════

def scrape_channels(output_dir, channels=None, tier=None,
                    max_per_channel=200, download_blends=True,
                    progress=None):
    """Scrape videos from curated channels.

    Args:
        output_dir: Path for output JSON + blend files.
        channels: List of (handle, name, tier, specialty) tuples.
                  Defaults to CURATED_CHANNELS.
        tier: If set, only scrape channels of this tier or higher priority.
        max_per_channel: Max videos to fetch per channel.
        download_blends: Whether to download .blend files.
        progress: Set of already-processed video IDs.

    Returns:
        dict with stats.
    """
    if channels is None:
        channels = CURATED_CHANNELS
    if tier is not None:
        channels = [ch for ch in channels if ch[2] <= tier]
    if progress is None:
        progress = set()

    output_dir = Path(output_dir)
    stats = {
        "total_videos": 0,
        "transcripts_saved": 0,
        "blend_files": 0,
        "skipped_filter": 0,
        "skipped_existing": 0,
        "by_channel": {},
    }

    for handle, name, ch_tier, specialty in channels:
        logger.info(f"{'='*60}")
        logger.info(f"Channel: {name} ({handle}) [tier {ch_tier}]")
        logger.info(f"  Specialty: {specialty}")
        logger.info(f"{'='*60}")

        videos = get_channel_videos(handle, max_results=max_per_channel)
        logger.info(f"  Fetched {len(videos)} videos")

        ch_stats = {"fetched": len(videos), "processed": 0,
                    "transcripts": 0, "blends": 0, "filtered": 0}

        for video in videos:
            vid = video.get("id", "")
            if not vid:
                continue
            if vid in progress:
                stats["skipped_existing"] += 1
                continue

            if not filter_video(video, ch_tier):
                ch_stats["filtered"] += 1
                stats["skipped_filter"] += 1
                continue

            result = process_video(video, output_dir,
                                   download_blends=download_blends)
            ch_stats["processed"] += 1
            stats["total_videos"] += 1

            if result["transcript_saved"]:
                ch_stats["transcripts"] += 1
                stats["transcripts_saved"] += 1
            if result["blend_files"]:
                ch_stats["blends"] += len(result["blend_files"])
                stats["blend_files"] += len(result["blend_files"])

            progress.add(vid)
            time.sleep(1.5)

        stats["by_channel"][name] = ch_stats
        logger.info(
            f"  {name}: {ch_stats['processed']} processed, "
            f"{ch_stats['transcripts']} transcripts, "
            f"{ch_stats['blends']} blends, "
            f"{ch_stats['filtered']} filtered out"
        )

    return stats


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Scrape Blender tutorials from curated YouTube channels")
    parser.add_argument("--output", default="data/raw/youtube")
    parser.add_argument(
        "--channels", type=str, default=None,
        help="Comma-separated channel handles to scrape (default: all curated)")
    parser.add_argument(
        "--tier", type=int, default=None, choices=[1, 2, 3],
        help="Only scrape channels at this tier or better (1=best)")
    parser.add_argument(
        "--max-per-channel", type=int, default=200,
        help="Max videos to fetch per channel")
    parser.add_argument(
        "--no-blend-download", action="store_true",
        help="Skip downloading .blend files (transcript-only mode)")
    parser.add_argument(
        "--discover", action="store_true",
        help="Run channel discovery to find new quality channels")
    parser.add_argument(
        "--legacy-search", action="store_true",
        help="Use old keyword-search mode instead of channel-based")
    parser.add_argument(
        "--queries", nargs="*", default=None,
        help="Search queries for --legacy-search mode")
    parser.add_argument(
        "--max-per-query", type=int, default=None,
        help="Max results per query in --legacy-search mode")
    args = parser.parse_args()

    setup_logging("youtube")
    config = load_config()
    output_dir = ensure_dir(args.output)
    progress = load_progress(output_dir / ".progress")
    yt_config = config.get("scraping", {}).get("youtube", {})

    if args.discover:
        logger.info("Running channel discovery...")
        candidates = discover_quality_channels()
        logger.info(f"Found {len(candidates)} candidate channels:")
        for handle, name, avg_views in candidates[:30]:
            logger.info(f"  {handle:30s} {name:30s} avg={avg_views:,} views")
        disc_path = Path(args.output) / "discovered_channels.json"
        with open(disc_path, "w") as f:
            json.dump([{"handle": h, "name": n, "avg_views": v}
                       for h, n, v in candidates], f, indent=2)
        logger.info(f"Saved to {disc_path}")
        return

    if args.legacy_search:
        logger.info("Legacy keyword-search mode")
        queries = args.queries or yt_config.get("search_queries", [
            "blender modeling tutorial",
            "blender hard surface tutorial",
            "blender sculpting tutorial",
            "blender geometry nodes tutorial",
            "blender materials tutorial",
        ])
        max_per = args.max_per_query or yt_config.get("max_videos_per_query", 100)
        total_saved = 0
        for i, query in enumerate(queries):
            logger.info(f"[{i+1}/{len(queries)}] Searching: {query}")
            videos = _search_youtube(query, max_results=max_per)
            logger.info(f"  Found {len(videos)} videos")
            for video in videos:
                vid = video.get("id", "")
                if vid in progress:
                    continue
                result = process_video(video, output_dir,
                                       download_blends=not args.no_blend_download)
                if result["transcript_saved"]:
                    total_saved += 1
                save_progress(output_dir / ".progress", vid)
                time.sleep(1.5)
        logger.info(f"Legacy search complete. Saved {total_saved} transcripts.")
        return

    channels = None
    if args.channels:
        handles = [h.strip() for h in args.channels.split(",")]
        handles = [h if h.startswith("@") else f"@{h}" for h in handles]
        channels = [ch for ch in CURATED_CHANNELS
                    if ch[0] in handles]
        extra = yt_config.get("extra_channels", [])
        for h in handles:
            if not any(ch[0] == h for ch in (channels or [])):
                name = h.lstrip("@")
                channels.append((h, name, 2, "user-specified"))

    stats = scrape_channels(
        output_dir,
        channels=channels,
        tier=args.tier,
        max_per_channel=args.max_per_channel,
        download_blends=not args.no_blend_download,
        progress=progress,
    )

    summary_path = Path(args.output) / "scrape_summary.json"
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Scraping complete: {stats['transcripts_saved']} transcripts, "
        f"{stats['blend_files']} blend files from "
        f"{stats['total_videos']} videos "
        f"({stats['skipped_filter']} filtered, "
        f"{stats['skipped_existing']} already done)"
    )


# ═══════════════════════════════════════════════════════════════════
# Backward compatibility aliases
# ═══════════════════════════════════════════════════════════════════
search_youtube_videos = _search_youtube
get_transcript = lambda vid: (
    get_transcript_with_timestamps(vid) or {}
).get("full_text")


if __name__ == "__main__":
    main()
