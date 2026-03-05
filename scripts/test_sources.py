#!/usr/bin/env python3
"""Quick smoke-test: pull 1 item from each data source and report status.

Usage:
    python scripts/test_sources.py
    python scripts/test_sources.py --include blendswap smutbase open3dlab
    python scripts/test_sources.py --skip blender_official objaverse
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("test_sources")

# Sources that are slow to initialise (skip by default)
SLOW_SOURCES = {"objaverse"}   # loads ~800 MB annotation index


def test_source(name: str, iter_fn, raw_dir: Path, proc_dir: Path) -> dict:
    result = {"source": name, "status": "?", "name": "", "url": "", "dl_url": "", "error": ""}
    try:
        gen = iter_fn(raw_dir, proc_dir)
        item = next(gen, None)
        if item is None:
            result["status"] = "EMPTY"
        else:
            meta = item.get("metadata", {})
            result["status"] = "OK"
            result["name"] = str(meta.get("name", ""))[:60]
            result["url"] = str(item.get("source_url", ""))[:70]
            result["dl_url"] = str(item.get("download_url") or "None")[:70]
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def main():
    parser = argparse.ArgumentParser(description="Smoke-test data source iterators")
    parser.add_argument("--include", nargs="*", default=None,
                        help="Only test these sources")
    parser.add_argument("--skip", nargs="*", default=list(SLOW_SOURCES),
                        help=f"Skip these sources (default: {sorted(SLOW_SOURCES)})")
    args = parser.parse_args()

    from scripts.data_pipeline import SOURCE_ITERS
    raw_dir  = Path("data/raw")
    proc_dir = Path("data/processed")

    sources = list(SOURCE_ITERS.keys())
    if args.include:
        sources = [s for s in sources if s in args.include]
    if args.skip:
        sources = [s for s in sources if s not in args.skip]

    print(f"\n{'='*70}")
    print(f"  Testing {len(sources)} sources: {sources}")
    print(f"{'='*70}\n")

    results = []
    for name in sources:
        print(f"--- {name} ...", flush=True)
        r = test_source(name, SOURCE_ITERS[name], raw_dir, proc_dir)
        results.append(r)
        if r["status"] == "OK":
            print(f"  [OK]  name='{r['name']}'")
            print(f"        url={r['url']}")
            print(f"        dl ={r['dl_url']}")
        elif r["status"] == "EMPTY":
            print(f"  [EMPTY]  no items yielded (check if raw files exist / site accessible)")
        else:
            print(f"  [ERROR]  {r['error']}")
        print()

    ok    = [r for r in results if r["status"] == "OK"]
    empty = [r for r in results if r["status"] == "EMPTY"]
    bad   = [r for r in results if r["status"] == "ERROR"]

    print(f"{'='*70}")
    print(f"  SUMMARY: {len(ok)} OK  |  {len(empty)} EMPTY  |  {len(bad)} ERROR")
    if bad:
        print(f"\n  Errors:")
        for r in bad:
            print(f"    {r['source']}: {r['error']}")
    print(f"{'='*70}\n")

    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
