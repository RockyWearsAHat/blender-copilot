"""Common utilities for all scrapers."""

import os
import time
import hashlib
import logging
import requests
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def setup_logging(name: str, level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level),
        format=f"%(asctime)s [{name}] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config():
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_hash(path: str | Path) -> str:
    """SHA-256 hash of a file for deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, output_path: str | Path,
                  max_size_mb: float = 200,
                  headers: dict | None = None,
                  rate_limit_seconds: float = 1.0,
                  max_retries: int = 3,
                  session: "requests.Session | None" = None) -> bool:
    """Download a file with size limit, rate limiting, and retry on 429.

    Args:
        session: Optional requests.Session with auth cookies.
                 If None, uses a bare requests.get().

    Returns True if download succeeded, False otherwise.
    """
    output_path = Path(output_path)
    if output_path.exists():
        logger.debug(f"Already exists: {output_path}")
        return True

    ensure_dir(output_path.parent)
    default_headers = {
        "User-Agent": "BlenderModelTraining/0.1 (research; open-source)"
    }
    if headers:
        default_headers.update(headers)

    getter = session or requests

    for attempt in range(max_retries):
        try:
            resp = getter.get(url, headers=default_headers, stream=True, timeout=60)

            # Handle rate limiting with exponential backoff
            if resp.status_code == 429:
                wait = (attempt + 1) * 5  # 5s, 10s, 15s
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = int(retry_after)
                    except ValueError:
                        pass
                logger.info(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue

            resp.raise_for_status()

            # Check content-length before downloading
            content_length = resp.headers.get("content-length")
            if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                logger.info(f"Skipping {url}: {int(content_length) / 1e6:.1f}MB > {max_size_mb}MB limit")
                return False

            # Stream to disk with size check
            downloaded = 0
            max_bytes = max_size_mb * 1024 * 1024
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        logger.info(f"Skipping {url}: exceeded {max_size_mb}MB during download")
                        tmp_path.unlink(missing_ok=True)
                        return False
                    f.write(chunk)

            tmp_path.rename(output_path)

            if downloaded < 10000:
                logger.warning(f"Rejected: {output_path.name} — only {downloaded} bytes (likely error page)")
                output_path.unlink(missing_ok=True)
                return False

            # Validate binary files aren't HTML error pages
            suffix = output_path.suffix.lower()
            if suffix in ('.blend', '.glb', '.stl', '.obj', '.ply', '.rar', '.zip'):
                with open(output_path, 'rb') as vf:
                    magic = vf.read(20)
                if magic.lstrip()[:5] == b'<!doc' or magic.lstrip()[:5] == b'<html':
                    logger.warning(
                        f"Rejected: {output_path.name} — HTML instead of {suffix} "
                        f"(auth required?)")
                    output_path.unlink(missing_ok=True)
                    return False

            logger.info(f"Downloaded: {output_path.name} ({downloaded / 1e6:.1f}MB)")

            if rate_limit_seconds > 0:
                time.sleep(rate_limit_seconds)
            return True

        except Exception as e:
            if attempt < max_retries - 1:
                logger.debug(f"Attempt {attempt + 1} failed for {url}: {e}")
                time.sleep((attempt + 1) * 2)
            else:
                logger.warning(f"Failed to download {url}: {e}")

    return False


def save_metadata(output_dir: str | Path, filename: str, metadata: dict):
    """Save metadata JSON alongside a downloaded file."""
    import json
    meta_path = Path(output_dir) / (filename + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_progress(progress_file: str | Path) -> set:
    """Load set of already-processed URLs/IDs from a progress file."""
    path = Path(progress_file)
    if not path.exists():
        return set()
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())


def save_progress(progress_file: str | Path, item_id: str):
    """Append an item to the progress file."""
    path = Path(progress_file)
    ensure_dir(path.parent)
    with open(path, "a") as f:
        f.write(item_id + "\n")


def is_blend_file(path: str | Path) -> bool:
    """Check if a file is a valid .blend file by reading magic bytes."""
    try:
        with open(path, "rb") as f:
            magic = f.read(7)
            return magic == b"BLENDER"
    except Exception:
        return False


class GlobalHashRegistry:
    """Cross-scraper file hash deduplication.

    Maintains a persistent set of SHA-256 hashes across ALL scrapers so
    the same 3D model is never processed twice, even if it appears on
    multiple sites under different names.

    Usage:
        registry = GlobalHashRegistry("data/raw")
        if registry.is_new(file_path):
            # truly new file — process it
            registry.add_file(file_path)
        else:
            # duplicate — skip
            os.remove(file_path)
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self._file = self.base_dir / ".global_hashes.json"
        self._hashes = self._load()

    def _load(self) -> set:
        if self._file.exists():
            try:
                import json
                with open(self._file) as f:
                    return set(json.load(f))
            except Exception:
                pass
        return set()

    def save(self):
        import json
        ensure_dir(self._file.parent)
        tmp = self._file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(list(self._hashes), f)
        tmp.rename(self._file)

    def has_hash(self, sha: str) -> bool:
        return sha in self._hashes

    def add_hash(self, sha: str):
        self._hashes.add(sha)

    def is_new(self, path: str | Path) -> bool:
        """Check if a file is truly new (not seen before by any scraper)."""
        sha = file_hash(path)
        return sha not in self._hashes

    def add_file(self, path: str | Path) -> str:
        """Register a file's hash. Returns the hash."""
        sha = file_hash(path)
        self._hashes.add(sha)
        return sha

    def check_and_add(self, path: str | Path) -> bool:
        """Check if file is new AND register it if so. Returns True if new."""
        sha = file_hash(path)
        if sha in self._hashes:
            return False
        self._hashes.add(sha)
        return True

    @property
    def count(self) -> int:
        return len(self._hashes)

