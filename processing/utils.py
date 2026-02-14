"""Common utilities for the processing pipeline."""

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
