"""Scrape .blend files from GitHub repositories.

Searches GitHub for repositories containing .blend files, filtered by
permissive licenses. Downloads the .blend files and extracts metadata
from the repo (README, description, topics).

Usage:
    python -m scrapers.github_scraper --output data/raw/github
    python -m scrapers.github_scraper --token ghp_xxx --output data/raw/github
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import requests

from .utils import (
    setup_logging, load_config, ensure_dir, download_file,
    save_metadata, load_progress, save_progress
)

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
GITHUB_SEARCH_API = f"{GITHUB_API}/search/code"

# License SPDX IDs that allow training use
ALLOWED_LICENSES = {
    "mit", "apache-2.0", "gpl-2.0", "gpl-3.0", "lgpl-2.1", "lgpl-3.0",
    "bsd-2-clause", "bsd-3-clause", "unlicense", "cc0-1.0", "cc-by-4.0",
    "cc-by-sa-4.0", "isc", "artistic-2.0", "0bsd",
}


def search_blend_files(session: requests.Session, query: str,
                       page: int = 1, per_page: int = 30) -> list[dict]:
    """Search GitHub code for .blend files.

    Returns list of dicts with repo and file path information.
    """
    params = {
        "q": query,
        "per_page": per_page,
        "page": page,
    }
    try:
        resp = session.get(GITHUB_SEARCH_API, params=params, timeout=30)

        # Handle rate limiting
        if resp.status_code == 403:
            reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset_time - int(time.time()), 60)
            logger.warning(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            return []

        resp.raise_for_status()
        data = resp.json()
        return data.get("items", [])

    except Exception as e:
        logger.warning(f"Search failed: {e}")
        return []


def get_repo_info(session: requests.Session, owner: str, repo: str) -> dict | None:
    """Get repository metadata including license."""
    try:
        resp = session.get(f"{GITHUB_API}/repos/{owner}/{repo}", timeout=30)
        if resp.status_code == 403:
            time.sleep(60)
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def get_repo_blend_files(session: requests.Session, owner: str, repo: str,
                          default_branch: str = "main") -> list[dict]:
    """List all .blend files in a repository using the tree API."""
    try:
        resp = session.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{default_branch}",
            params={"recursive": "1"},
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        tree = resp.json().get("tree", [])
        return [
            item for item in tree
            if item.get("path", "").lower().endswith(".blend")
            and item.get("type") == "blob"
        ]
    except Exception:
        return []


def download_blend_from_repo(session: requests.Session, owner: str, repo: str,
                              file_path: str, branch: str,
                              output_path: Path, max_size_mb: float) -> bool:
    """Download a .blend file from a GitHub repo."""
    # Use raw.githubusercontent.com for direct download
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
    return download_file(url, output_path, max_size_mb=max_size_mb,
                         rate_limit_seconds=0.5)


def scrape_github(output_dir: Path, config: dict, token: str | None = None):
    """Main scraping loop for GitHub .blend files."""
    gh_config = config.get("scraping", {}).get("github", {})
    max_repos = gh_config.get("max_repos", 5000)
    max_size = config.get("scraping", {}).get("max_file_size_mb", 200)
    allowed = set(gh_config.get("licenses", ALLOWED_LICENSES))

    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"token {token}"
    session.headers["Accept"] = "application/vnd.github.v3+json"
    session.headers["User-Agent"] = config.get("scraping", {}).get(
        "user_agent", "BlenderModelTraining/0.1")

    progress = load_progress(output_dir / ".progress")
    logger.info(f"Resuming with {len(progress)} already processed repos")

    # Search queries targeting .blend files
    queries = [
        "extension:blend",
        "extension:blend blender model",
        "extension:blend 3d",
        "extension:blend character",
        "extension:blend vehicle",
        "extension:blend architecture",
    ]

    repos_seen = set()
    total_downloaded = 0

    for query in queries:
        if total_downloaded >= max_repos:
            break

        for page in range(1, 35):  # GitHub caps at 34 pages (1000 results)
            if total_downloaded >= max_repos:
                break

            logger.info(f"Query '{query}' page {page}")
            items = search_blend_files(session, query, page=page)

            if not items:
                break

            for item in items:
                repo_full = item.get("repository", {}).get("full_name", "")
                if not repo_full or repo_full in repos_seen:
                    continue
                repos_seen.add(repo_full)

                if repo_full in progress:
                    continue

                owner, repo = repo_full.split("/", 1)

                # Check license
                repo_info = get_repo_info(session, owner, repo)
                if not repo_info:
                    save_progress(output_dir / ".progress", repo_full)
                    continue

                license_info = repo_info.get("license") or {}
                license_key = (license_info.get("spdx_id") or "").lower()
                if license_key not in allowed and license_key != "noassertion":
                    logger.debug(f"Skipping {repo_full}: license={license_key}")
                    save_progress(output_dir / ".progress", repo_full)
                    continue

                # Get all .blend files in repo
                branch = repo_info.get("default_branch", "main")
                blend_files = get_repo_blend_files(session, owner, repo, branch)

                if not blend_files:
                    save_progress(output_dir / ".progress", repo_full)
                    continue

                # Download each .blend file
                repo_dir = ensure_dir(output_dir / owner / repo)
                for bf in blend_files:
                    file_path = bf["path"]
                    # Sanitize filename
                    safe_name = file_path.replace("/", "__")
                    out_path = repo_dir / safe_name

                    size_bytes = bf.get("size", 0)
                    if size_bytes > max_size * 1024 * 1024:
                        continue

                    success = download_blend_from_repo(
                        session, owner, repo, file_path, branch,
                        out_path, max_size,
                    )
                    if success:
                        total_downloaded += 1
                        save_metadata(repo_dir, safe_name, {
                            "source": "github",
                            "repo": repo_full,
                            "file_path": file_path,
                            "license": license_key,
                            "description": repo_info.get("description", ""),
                            "topics": repo_info.get("topics", []),
                            "stars": repo_info.get("stargazers_count", 0),
                        })

                save_progress(output_dir / ".progress", repo_full)
                time.sleep(2)  # Rate limiting

    logger.info(f"GitHub scraping complete. Downloaded {total_downloaded} files.")


def main():
    parser = argparse.ArgumentParser(description="Scrape .blend files from GitHub")
    parser.add_argument("--output", default="data/raw/github")
    parser.add_argument("--token", default=None,
                        help="GitHub personal access token (or set GITHUB_TOKEN env)")
    args = parser.parse_args()

    setup_logging("github")
    config = load_config()

    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning("No GitHub token — rate limited to 10 requests/min. "
                        "Set GITHUB_TOKEN or use --token for 30 req/min.")

    output_dir = ensure_dir(args.output)
    scrape_github(output_dir, config, token)


if __name__ == "__main__":
    main()
