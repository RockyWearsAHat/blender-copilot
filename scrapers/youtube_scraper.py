"""Scrape Blender tutorial transcripts and metadata from YouTube.

Extracts video transcripts (which describe modeling workflows), titles,
descriptions, and tags. These text descriptions will be paired with
.blend file data to create text→3D training pairs.

Usage:
    python -m scrapers.youtube_scraper --output data/raw/youtube
"""

import argparse
import json
import logging
import time
from pathlib import Path

from .utils import setup_logging, load_config, ensure_dir, load_progress, save_progress

logger = logging.getLogger(__name__)


def search_youtube_videos(query: str, max_results: int = 500) -> list[dict]:
    """Search YouTube for Blender tutorial videos.

    Uses yt-dlp for searching to avoid needing a YouTube API key.
    Returns list of video metadata dicts.
    """
    import subprocess
    import json

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
    import subprocess

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


def process_video(video: dict, output_dir: Path) -> bool:
    """Process a single video: fetch transcript and save."""
    video_id = video.get("id", "")
    if not video_id:
        return False

    output_file = output_dir / f"{video_id}.json"
    if output_file.exists():
        return True

    # Get transcript
    transcript = get_transcript(video_id)
    if not transcript:
        return False

    # Filter: must be about Blender modeling (not just mentions it)
    blender_keywords = [
        "blender", "modeling", "mesh", "vertex", "vertices", "polygon",
        "material", "shader", "uv", "texture", "sculpt", "modifier",
        "subdivision", "boolean", "extrude", "loop cut", "bevel",
    ]
    transcript_lower = transcript.lower()
    keyword_count = sum(1 for kw in blender_keywords if kw in transcript_lower)
    if keyword_count < 3:
        logger.debug(f"Skipping {video_id}: not enough Blender keywords ({keyword_count})")
        return False

    # Get detailed metadata if available
    detail = get_detailed_metadata(video_id)
    tags = []
    if detail:
        tags = detail.get("tags", [])
        video["description"] = detail.get("description", video.get("description", ""))
        video["like_count"] = detail.get("like_count")
        video["comment_count"] = detail.get("comment_count")

    # Save
    record = {
        **video,
        "transcript": transcript,
        "tags": tags,
        "source": "youtube",
        "blender_keyword_count": keyword_count,
    }

    with open(output_file, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"Saved transcript: {video.get('title', video_id)[:60]}...")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Blender tutorial transcripts from YouTube")
    parser.add_argument("--output", default="data/raw/youtube")
    parser.add_argument("--queries", nargs="*", default=None,
                        help="Custom search queries (overrides config)")
    parser.add_argument("--max-per-query", type=int, default=None)
    args = parser.parse_args()

    setup_logging("youtube")
    config = load_config()

    output_dir = ensure_dir(args.output)
    progress = load_progress(output_dir / ".progress")

    yt_config = config.get("scraping", {}).get("youtube", {})
    queries = args.queries or yt_config.get("search_queries", [
        "blender modeling tutorial",
        "blender hard surface modeling",
        "blender character modeling",
        "blender vehicle modeling",
        "blender architecture tutorial",
    ])
    max_per_query = args.max_per_query or yt_config.get("max_videos_per_query", 500)

    total_saved = 0
    for query in queries:
        logger.info(f"Searching: '{query}'")
        videos = search_youtube_videos(query, max_results=max_per_query)
        logger.info(f"  Found {len(videos)} videos")

        for video in videos:
            vid = video.get("id", "")
            if vid in progress:
                continue

            success = process_video(video, output_dir)
            if success:
                total_saved += 1

            save_progress(output_dir / ".progress", vid)
            time.sleep(1)  # Be respectful

    logger.info(f"YouTube scraping complete. Saved {total_saved} transcripts.")


if __name__ == "__main__":
    main()
