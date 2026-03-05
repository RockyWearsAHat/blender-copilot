#!/usr/bin/env python3
"""Download one smutbase file ≤ X MB, skipping models with separate texture ZIPs.
Extract it and verify full-res PNG is stored.

Usage:
    .venv/bin/python scripts/_smutbase_image_test.py
"""
import os, sys, json, subprocess, time, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BLENDER   = "/Applications/Blender.app/Contents/MacOS/Blender"
EXTRACTOR = ROOT / "processing/blend_extractor.py"
DEST_DIR  = ROOT / "data/raw/smutbase_fresh"
OUT_DIR   = ROOT / "data/processed/smutbase_fresh"
MAX_MB    = 80   # skip files larger than this to avoid huge downloads

def smutbase_download_compact() -> Path | None:
    from scrapers.smutbase_scraper import (
        create_session, get_listing_page,
        get_project_details, download_project_file,
    )
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Already have a suitable file?
    existing = [b for b in DEST_DIR.glob("*.blend")
                if "Domina" not in b.name and b.stat().st_size < MAX_MB * 1024 * 1024]
    if existing:
        print(f"Using cached: {existing[0].name}")
        return existing[0]

    session = create_session("smutbase")
    skip = {"Domina", "Neytiri"}  # known problematic

    for page in (1, 2):
        projects = get_listing_page(
            session, "https://smutba.se", page=page,
            software_tag="blender", sort_by="popular",
        )
        print(f"[page {page}] {len(projects)} projects")
        for proj in projects:
            title = proj["title"]
            if any(s in title for s in skip):
                print(f"  SKIP {title}")
                continue

            details = get_project_details(session, proj["url"], proj["project_id"])
            if not details:
                continue

            # Estimate size from metadata to avoid huge downloads
            size_str = details.get("filesize", "") or ""
            approx_mb = _parse_size_mb(size_str)
            if approx_mb and approx_mb > MAX_MB:
                print(f"  SKIP {title} ({size_str}, too large)")
                skip.add(title.split()[0])   # skip variations of this name
                continue

            print(f"  Downloading: {title} ({size_str})")
            result = download_project_file(session, details, DEST_DIR)
            if result:
                p = Path(result)
                if p.exists() and p.stat().st_size > 50_000:
                    print(f"  OK: {p.name} ({p.stat().st_size//1024}KB)")
                    return p
                elif p.exists():
                    p.unlink()

    return None


def _parse_size_mb(size_str: str) -> float | None:
    import re
    size_str = size_str.strip()
    m = re.match(r"([\d.]+)\s*(MB|GB|KB)", size_str, re.I)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2).upper()
    if unit == "GB": return val * 1024
    if unit == "MB": return val
    if unit == "KB": return val / 1024
    return None


def extract_and_audit(blend_path: Path) -> dict:
    label = blend_path.stem[:50]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / f"_tmp_{label}"
    tmp.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting {blend_path.name} ...")
    t0 = time.time()
    result = subprocess.run(
        [BLENDER, "--background", "--python", str(EXTRACTOR),
         "--", "--input", str(blend_path), "--output", str(tmp)],
        capture_output=True, text=True, timeout=300,
    )
    elapsed = round(time.time() - t0, 1)
    produced = sorted(tmp.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not produced or result.returncode != 0:
        print(f"FAIL code={result.returncode} ({elapsed}s)")
        print(result.stderr[-400:])
        return {}
    dest = OUT_DIR / f"{label}.json"
    produced[-1].rename(dest)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"OK ({elapsed}s, {dest.stat().st_size//1024}KB): {dest.name}")

    d = json.load(open(dest))
    imgs = d.get("images") or {}
    objs = d.get("objects") or []
    has_png   = any(v.get("image_data") for v in imgs.values())
    has_thumb = any(v.get("thumbnail") for v in imgs.values())
    png_info = [
        f"{k} ({v.get('image_data_size',['?','?'])[0]}x"
        f"{v.get('image_data_size',['?','?'])[1]}, "
        f"{(v.get('image_data_bytes') or 0)//1024}KB)"
        for k, v in list(imgs.items())[:3] if v.get("image_data")
    ]
    mods = [
        f"{o['name']}: " + ",".join(m.get("type","?") for m in (o.get("modifiers") or []))
        for o in objs if o.get("modifiers")
    ]
    return {
        "objects": len(objs),
        "images_total": len(imgs),
        "has_full_png": has_png,
        "has_thumbnail": has_thumb,
        "png_samples": png_info,
        "modifier_examples": mods[:3],
    }


def main():
    print("=" * 60)
    print("smutbase image test — download + extract with full-res PNG")
    print("=" * 60)

    blend = smutbase_download_compact()
    if not blend:
        print("\nFAIL: no suitable smutbase file found")
        sys.exit(1)

    audit = extract_and_audit(blend)
    if not audit:
        print("\nFAIL: extraction failed")
        sys.exit(1)

    ok = lambda b: "✅" if b else "❌"
    print("\n=== RESULTS ===")
    print(f"  objects       : {audit['objects']}")
    print(f"  images        : {audit['images_total']}")
    print(f"  full_res_PNG  : {ok(audit['has_full_png'])}", end="")
    if audit["png_samples"]:
        print(f"  → {audit['png_samples']}")
    else:
        print()
    print(f"  thumbnail     : {ok(audit['has_thumbnail'])}")
    print(f"  modifiers     : {ok(bool(audit['modifier_examples']))} {audit['modifier_examples'][:2]}")
    if not audit["has_full_png"] and not audit["has_thumbnail"]:
        print("\n  ⚠️  No images at all — model likely uses only external textures.")
        print("     Choose a different model that packs textures into the .blend.")


if __name__ == "__main__":
    main()
