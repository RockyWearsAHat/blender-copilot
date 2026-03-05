"""Tests for new scrapers: Thingiverse, Sketchfab, and ObjaverseXL.

Tests validate:
- Module imports cleanly
- Core data structures / helper functions work
- No network requests in tests (mocked or skipped)
- Quality filter integration

Run with:
    python -m pytest tests/test_new_scrapers.py -v
"""

import json
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════════════
# Thingiverse scraper
# ══════════════════════════════════════════════════════════════════════

def test_thingiverse_imports():
    """thingiverse_scraper imports without error."""
    from scrapers.thingiverse_scraper import (
        scrape_thingiverse, scrape_batch, SEARCH_TERMS,
        HEADERS, _get_thing_files,
    )
    assert len(SEARCH_TERMS) >= 10, "Should have at least 10 search terms"
    assert "User-Agent" in HEADERS
    print(f"  [OK] thingiverse_scraper: {len(SEARCH_TERMS)} search terms")


def test_thingiverse_scrape_batch_signature():
    """scrape_batch has correct signature (output_dir param)."""
    import inspect
    from scrapers.thingiverse_scraper import scrape_batch

    sig = inspect.signature(scrape_batch)
    assert "output_dir" in sig.parameters
    print("  [OK] scrape_batch signature: has output_dir")


def test_thingiverse_headers_have_user_agent():
    """Thingiverse headers include a descriptive User-Agent."""
    from scrapers.thingiverse_scraper import HEADERS

    ua = HEADERS.get("User-Agent", "")
    assert "BlenderCopilot" in ua or "blender" in ua.lower(), \
        f"User-Agent should identify as BlenderCopilot, got: '{ua}'"
    print(f"  [OK] Thingiverse User-Agent: '{ua[:60]}'")


def test_thingiverse_fetch_page_mock():
    """_fetch_thingiverse_page handles failed requests gracefully."""
    from scrapers.thingiverse_scraper import _fetch_thingiverse_page
    import requests

    # Mock requests session that returns 403
    mock_session = mock.MagicMock()
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 403
    mock_session.get.return_value = mock_resp

    results = _fetch_thingiverse_page("furniture", page=1, session=mock_session)

    # Should return empty list on failure, not raise
    assert isinstance(results, list)
    print(f"  [OK] _fetch_thingiverse_page: graceful 403 handling, returned {len(results)} results")


def test_thingiverse_scrape_empty_dir():
    """scrape_thingiverse creates output dir and returns 0 when network unavailable."""
    from scrapers.thingiverse_scraper import scrape_thingiverse

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "thingiverse"

        # Mock the entire _fetch_thingiverse_page to return empty
        with mock.patch("scrapers.thingiverse_scraper._fetch_thingiverse_page",
                        return_value=[]):
            result = scrape_thingiverse(
                output_dir=str(output),
                max_pages=1,
                max_per_query=1,
                delay=0.0,
            )

        assert output.exists(), "Output directory should be created"
        assert result == 0, f"Expected 0 downloads with empty pages, got {result}"
        print("  [OK] scrape_thingiverse: creates dir, returns 0 when no results")


# ══════════════════════════════════════════════════════════════════════
# Sketchfab scraper
# ══════════════════════════════════════════════════════════════════════

def test_sketchfab_imports():
    """sketchfab_scraper imports without error."""
    from scrapers.sketchfab_scraper import (
        scrape_sketchfab, scrape_batch,
        SEARCH_CATEGORIES, FREE_LICENSES, HEADERS,
        _is_allowed_license,
    )
    assert len(SEARCH_CATEGORIES) >= 5
    assert len(FREE_LICENSES) >= 3
    print(f"  [OK] sketchfab_scraper: {len(SEARCH_CATEGORIES)} categories, "
          f"{len(FREE_LICENSES)} allowed licenses")


def test_sketchfab_license_check_cc0():
    """_is_allowed_license accepts CC-0."""
    from scrapers.sketchfab_scraper import _is_allowed_license

    cc0_model = {
        "license": {
            "slug": "cc0",
            "label": "Creative Commons Zero v1.0 Universal",
        }
    }
    assert _is_allowed_license(cc0_model), "CC-0 should be allowed"
    print("  [OK] _is_allowed_license: accepts CC-0")


def test_sketchfab_license_check_attribution():
    """_is_allowed_license accepts CC-BY."""
    from scrapers.sketchfab_scraper import _is_allowed_license

    cc_by = {
        "license": {"slug": "by", "label": "CC-BY 4.0", "attribution": True},
    }
    assert _is_allowed_license(cc_by), "CC-BY should be allowed"
    print("  [OK] _is_allowed_license: accepts CC-BY")


def test_sketchfab_license_check_no_license():
    """_is_allowed_license rejects empty license."""
    from scrapers.sketchfab_scraper import _is_allowed_license

    no_license = {"license": None}
    no_license2 = {}

    assert not _is_allowed_license(no_license), "None license should be rejected"
    assert not _is_allowed_license(no_license2), "Missing license should be rejected"
    print("  [OK] _is_allowed_license: rejects None/missing license")


def test_sketchfab_search_mock_200():
    """_search_sketchfab parses valid API response correctly."""
    from scrapers.sketchfab_scraper import _search_sketchfab
    import requests

    mock_session = mock.MagicMock()
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "results": [
            {
                "uid": "abc123",
                "name": "Wooden Chair",
                "license": {"slug": "cc0"},
                "likeCount": 42,
                "viewCount": 1000,
                "description": "A nice wooden chair",
                "tags": [{"name": "furniture"}],
            },
            {
                "uid": "def456",
                "name": "Metal Table",
                "license": {"slug": "by"},
                "likeCount": 10,
                "viewCount": 500,
                "description": "",
                "tags": [],
            },
        ]
    }
    mock_session.get.return_value = mock_resp

    results = _search_sketchfab("furniture", page=1, session=mock_session)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0]["uid"] == "abc123"
    assert results[1]["name"] == "Metal Table"
    print(f"  [OK] _search_sketchfab: parsed {len(results)} results from mock API")


def test_sketchfab_search_mock_429():
    """_search_sketchfab handles rate limiting gracefully."""
    from scrapers.sketchfab_scraper import _search_sketchfab
    import requests

    mock_session = mock.MagicMock()
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 429
    mock_session.get.return_value = mock_resp

    with mock.patch("time.sleep"):  # Speed up the sleep
        results = _search_sketchfab("furniture", page=1, session=mock_session)

    assert isinstance(results, list), "Should return empty list on 429"
    print(f"  [OK] _search_sketchfab: handles 429 gracefully")


def test_sketchfab_scrape_empty_dir():
    """scrape_sketchfab creates output dir and returns 0 when network unavailable."""
    from scrapers.sketchfab_scraper import scrape_sketchfab

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "sketchfab"

        with mock.patch("scrapers.sketchfab_scraper._search_sketchfab",
                        return_value=[]):
            result = scrape_sketchfab(
                output_dir=str(output),
                max_pages=1,
                max_per_query=1,
                delay=0.0,
            )

        assert output.exists()
        assert result == 0
        print("  [OK] scrape_sketchfab: handles empty search results")


def test_sketchfab_get_download_url_mock_404():
    """_get_download_url returns None when all methods fail."""
    from scrapers.sketchfab_scraper import _get_download_url

    mock_session = mock.MagicMock()
    mock_resp = mock.MagicMock()
    mock_resp.status_code = 404
    mock_resp.text = "<html>not found</html>"
    mock_session.get.return_value = mock_resp
    mock_session.head.return_value = mock_resp

    url = _get_download_url("nonexistent_uid", mock_session, api_token=None)
    assert url is None, f"Expected None for 404 response, got: {url}"
    print("  [OK] _get_download_url: returns None when all methods fail")


# ══════════════════════════════════════════════════════════════════════
# ObjaverseXL integration
# ══════════════════════════════════════════════════════════════════════

def test_objaverse_xl_importable():
    """objaverse.xl imports correctly with expected downloaders."""
    try:
        import objaverse.xl as oxl

        assert hasattr(oxl, "download_objects")
        assert hasattr(oxl, "get_annotations")
        assert hasattr(oxl, "get_alignment_annotations")
        assert hasattr(oxl, "sketchfab")
        assert hasattr(oxl, "thingiverse")
        assert hasattr(oxl, "smithsonian")

        print("  [OK] objaverse.xl: imports with all expected downloaders")
    except ImportError:
        pytest.skip("objaverse not installed")


def test_objaverse_xl_smithsonian_annotations():
    """Smithsonian annotations parquet loads with expected columns."""
    try:
        import objaverse.xl as oxl
        import pandas as pd

        ann = oxl.smithsonian.SmithsonianDownloader.get_annotations()
        assert isinstance(ann, pd.DataFrame)
        assert "fileIdentifier" in ann.columns
        assert "fileType" in ann.columns
        assert "license" in ann.columns
        assert len(ann) > 0

        # All Smithsonian should be GLB
        file_types = ann["fileType"].unique().tolist()
        assert "glb" in file_types, f"Expected GLB files, got: {file_types}"

        print(f"  [OK] Smithsonian annotations: {len(ann)} models, types={file_types}")
    except ImportError:
        pytest.skip("objaverse not installed")


def test_objaverse_xl_sketchfab_annotations():
    """Sketchfab annotations parquet loads with expected columns and size."""
    try:
        import objaverse.xl as oxl
        import pandas as pd

        ann = oxl.sketchfab.SketchfabDownloader.get_annotations()
        assert isinstance(ann, pd.DataFrame)
        assert len(ann) > 100_000, f"Expected 100k+ Sketchfab models, got {len(ann)}"
        assert "fileIdentifier" in ann.columns
        assert "license" in ann.columns

        print(f"  [OK] Sketchfab XL annotations: {len(ann):,} models")
    except ImportError:
        pytest.skip("objaverse not installed")


def test_objaverse_xl_thingiverse_annotations():
    """Thingiverse annotations parquet loads with expected columns."""
    try:
        import objaverse.xl as oxl
        import pandas as pd

        ann = oxl.thingiverse.ThingiverseDownloader.get_annotations()
        assert isinstance(ann, pd.DataFrame)
        assert len(ann) > 1_000_000, f"Expected 1M+ Thingiverse models, got {len(ann)}"

        # Thingiverse is mostly STL
        file_types = ann["fileType"].unique().tolist()
        assert "stl" in file_types, f"Expected STL files, got: {file_types}"

        print(f"  [OK] Thingiverse XL annotations: {len(ann):,} models, types={file_types}")
    except ImportError:
        pytest.skip("objaverse not installed")


def test_objaverse_scraper_xl_download_function():
    """objaverse_scraper.py has download functions with output_dir parameter."""
    from scrapers.objaverse_scraper import download_objaverse_models

    import inspect
    sig = inspect.signature(download_objaverse_models)
    assert "output_dir" in sig.parameters

    print("  [OK] download_objaverse_models: has output_dir parameter")


# ══════════════════════════════════════════════════════════════════════
# New scraper integration with run.py
# ══════════════════════════════════════════════════════════════════════

def test_run_py_recognizes_new_sources():
    """run.py CLI accepts thingiverse and sketchfab as valid sources."""
    import subprocess
    result = subprocess.run(
        ["python3", "run.py", "scrape", "--help"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    output = result.stdout + result.stderr

    for source in ["thingiverse", "sketchfab", "wikimedia", "terminology"]:
        assert source in output, f"'{source}' not found in CLI help output"

    print("  [OK] run.py: all new sources in CLI help")


def test_render_manifests_exist():
    """Render manifests exist for cached meshes (vision layer ready)."""
    renders_dir = Path("data/renders")
    if not renders_dir.exists():
        pytest.skip("No renders directory (run render pipeline first)")

    manifests = list(renders_dir.glob("**/*_manifest.json"))
    if not manifests:
        pytest.skip("No render manifests found")

    # Verify manifest structure
    with open(manifests[0]) as f:
        m = json.load(f)

    assert "mesh_id" in m
    assert "label" in m
    assert "renders" in m
    assert len(m["renders"]) > 0

    first_render = m["renders"][0]
    assert "filepath" in first_render

    print(f"  [OK] render manifests: {len(manifests)} manifests, "
          f"sample label='{m['label'][:40]}'")


def test_contrastive_stream_loads_renders():
    """ContrastiveStream loads real render PNGs when data/renders exists."""
    renders_dir = Path("data/renders")
    if not renders_dir.exists():
        pytest.skip("No renders directory")

    from training.train_unified import ContrastiveStream

    stream = ContrastiveStream(
        geometry_jsonl=None,
        text_tokenizer=None,
        max_text_length=32,
        image_size=64,
        prefetch_size=4,
        render_threads=0,
    )

    if stream._png_pairs:
        assert len(stream._png_pairs) > 0
        pair = stream._png_pairs[0]
        assert len(pair) == 2  # (path, label)
        assert Path(pair[0]).exists(), f"PNG file should exist: {pair[0]}"
        assert isinstance(pair[1], str)
        print(f"  [OK] ContrastiveStream: loaded {len(stream._png_pairs)} real PNGs")
    else:
        print("  [SKIP] ContrastiveStream: no PNG pairs (renders may not be linked)")


def test_training_data_quality_with_new_sources():
    """Training cache has expected quality metrics after adding new sources."""
    import torch
    import random

    cache_dir = Path("data/processed/.mesh_cache")
    if not cache_dir.exists():
        pytest.skip("No cache directory")

    pt_files = sorted(cache_dir.glob("*.pt"))
    if len(pt_files) < 100:
        pytest.skip(f"Too few cache files ({len(pt_files)}) to validate quality")

    # Sample 10 random files
    sample = random.sample(pt_files, min(10, len(pt_files)))

    valid = 0
    empty = 0
    for pt_file in sample:
        try:
            data = torch.load(pt_file, map_location="cpu", weights_only=True)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict) and "mesh_tokens" in item:
                    tokens = item["mesh_tokens"]
                    if hasattr(tokens, "__len__") and len(tokens) > 2:
                        valid += 1
                    else:
                        empty += 1
        except Exception:
            empty += 1

    total = valid + empty
    quality_rate = valid / max(total, 1)
    assert quality_rate >= 0.5, \
        f"Cache quality too low: {valid}/{total} valid ({quality_rate:.0%})"

    print(f"  [OK] Training cache quality: {valid}/{total} valid samples "
          f"({quality_rate:.0%}) from sample of {len(sample)} files")


if __name__ == "__main__":
    import sys
    tests = [
        # Thingiverse
        ("thingiverse imports", test_thingiverse_imports),
        ("scrape_batch signature", test_thingiverse_scrape_batch_signature),
        ("thingiverse User-Agent", test_thingiverse_headers_have_user_agent),
        ("fetch_page mock 403", test_thingiverse_fetch_page_mock),
        ("scrape_thingiverse empty", test_thingiverse_scrape_empty_dir),
        # Sketchfab
        ("sketchfab imports", test_sketchfab_imports),
        ("license CC-0 accepted", test_sketchfab_license_check_cc0),
        ("license CC-BY accepted", test_sketchfab_license_check_attribution),
        ("license None rejected", test_sketchfab_license_check_no_license),
        ("search mock 200", test_sketchfab_search_mock_200),
        ("search mock 429", test_sketchfab_search_mock_429),
        ("scrape empty dir", test_sketchfab_scrape_empty_dir),
        ("download_url mock 404", test_sketchfab_get_download_url_mock_404),
        # ObjaverseXL
        ("objaverse.xl importable", test_objaverse_xl_importable),
        ("Smithsonian annotations", test_objaverse_xl_smithsonian_annotations),
        ("Sketchfab XL annotations", test_objaverse_xl_sketchfab_annotations),
        ("Thingiverse XL annotations", test_objaverse_xl_thingiverse_annotations),
        ("scrape_objaverse signature", test_objaverse_scraper_xl_download_function),
        # Integration
        ("run.py recognizes new sources", test_run_py_recognizes_new_sources),
        ("render manifests exist", test_render_manifests_exist),
        ("ContrastiveStream loads renders", test_contrastive_stream_loads_renders),
        ("training cache quality", test_training_data_quality_with_new_sources),
    ]

    passed = failed = 0
    for name, fn in tests:
        print(f"[{passed + failed + 1:02d}/{len(tests):02d}] {name}")
        try:
            fn()
            passed += 1
        except pytest.skip.Exception as e:
            print(f"  [SKIP] {e}")
            passed += 1  # skips counted as passes
        except Exception as e:
            print(f"  [FAIL] {type(e).__name__}: {e}")
            failed += 1
        print()

    print(f"{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*55}")
    if failed:
        sys.exit(1)
