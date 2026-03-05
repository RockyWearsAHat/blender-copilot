"""Quality filters for scraped 3D model metadata.

Centralized quality gates applied before downloading any model.
Keeps garbage, spam, and low-effort content out of training data.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Minimum community validation ────────────────────────────────────────
# Models with very few downloads haven't been vetted by anyone.
MIN_DOWNLOADS = 10

# ── Maximum file size (MB) ──────────────────────────────────────────────
# Huge files slow down the pipeline and are often scenes, not reusable models.
MAX_FILE_SIZE_MB = 1000

# ── Title / description blocklist ───────────────────────────────────────
# Patterns that indicate low-quality, spam, joke, or off-topic uploads.
# Case-insensitive regex fragments joined with |
_BLOCKLIST_PATTERNS = [
    # Bodily functions / gross-out
]

_BLOCKLIST_RE = re.compile("|".join(_BLOCKLIST_PATTERNS), re.IGNORECASE) if _BLOCKLIST_PATTERNS else None

# ── Useful categories for 3D model training ─────────────────────────────
# We want models that teach the network about real geometry:
# architecture, vehicles, characters, furniture, mechanical parts, etc.
# Skip pure textures / HDRI / reference-image-only uploads.
SKIP_CATEGORIES = {
    "textures",  # no geometry, just images
}


def passes_quality_filter(
    title: str = "",
    description: str = "",
    tags: list[str] | None = None,
    downloads: int | None = None,
    likes: int | None = None,
    category: str = "",
    file_size_mb: float | None = None,
    source: str = "",
) -> tuple[bool, str]:
    """Check whether a model passes quality gates.

    Returns (passed: bool, reason: str).
    If passed is False, reason explains why it was rejected.
    """
    tags = tags or []
    combined_text = f"{title} {description} {' '.join(tags)}"

    # 1. Blocklist check on title + description + tags
    if _BLOCKLIST_RE:
        match = _BLOCKLIST_RE.search(combined_text)
        if match:
            return False, f"blocklist match: '{match.group()}'"

    # 2. Minimum downloads (community validation)
    # None means unknown (parser couldn't find count) — allow through
    if downloads is not None and downloads > 0 and downloads < MIN_DOWNLOADS:
        return False, f"too few downloads ({downloads} < {MIN_DOWNLOADS})"

    # 3. File size cap
    if file_size_mb is not None and file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"file too large ({file_size_mb:.0f}MB > {MAX_FILE_SIZE_MB}MB)"

    # 4. Skip useless categories
    if category.lower() in SKIP_CATEGORIES:
        return False, f"skipped category: {category}"

    # 5. Title too short (likely spam or placeholder)
    if len(title.strip()) < 3:
        return False, f"title too short: '{title}'"

    return True, "ok"
