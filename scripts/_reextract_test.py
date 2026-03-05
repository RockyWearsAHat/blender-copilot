#!/usr/bin/env python3
"""Re-extract selected .blend files with the UPDATED blend_extractor.py.

This tests the full new pipeline:
  .blend → blend_extractor.py (full-res PNG, modifiers, rigid_body)
         → data/processed/test_reextract/*.json
         → build_master_cache  → data/master_cache/test_reextract/<hash>.pt
         → build_training_cache → data/training_cache/test_v3 (appended)

Usage:
    .venv/bin/python scripts/_reextract_test.py

Selected test files:
  - chocolate_donut_001  : Subdivision + Solidify modifiers + HDRI world + icing texture
  - geometric_surface_patterns_000 : GeometryNodes modifier
  - 31825_000             : blendswap-style scene with multiple mesh objects
  - 3fc08fd6544d4add8ce55f6c5e2bc872_001 : complex scene

After this script, reinstall BlenderAICopilot.zip and use Reconstruct Full Scene
to see full-res textures + modifiers in the viewport.
"""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
EXTRACTOR = BASE / "processing" / "blend_extractor.py"

# ── chosen test .blend files ──────────────────────────────────────────────────
TEST_BLENDS = [
    BASE / "data/datasets/collapse_traces/chocolate_donut_001/final.blend",
    BASE / "data/datasets/collapse_traces/geometric_surface_patterns_000/final.blend",
    BASE / "data/datasets/collapse_traces/31825_000/final.blend",
    BASE / "data/datasets/collapse_traces/2bb529bd1f1f42cf9cdfa8aff56b35f6_000/final.blend",
]

OUT_JSON_DIR = BASE / "data/processed/test_reextract"
MASTER_CACHE_DIR = BASE / "data/master_cache/test_reextract"
TRAINING_CACHE_DIR = BASE / "data/training_cache/test_v3"

# ─────────────────────────────────────────────────────────────────────────────


def run_blender_extract(blend_path: Path, out_dir: Path, timeout: int = 300) -> bool:
    """Run Blender headless to extract one .blend file.

    blend_extractor.py names output files by .blend stem, so `final.blend`
    → `final.json`.  We run each file into a private tmp subdir, then rename
    the result to `{parent_dir_name}.json` so files don't overwrite each other.
    """
    name = blend_path.parent.name          # e.g. "chocolate_donut_001"
    out_json = out_dir / f"{name}.json"    # desired final path
    if out_json.exists():
        print(f"  SKIP (already extracted): {out_json.name}")
        return True

    # Use a temp subdir so `final.json` doesn't collide between runs
    tmp_dir = out_dir / f"_tmp_{name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Extracting {name} ...")
    cmd = [
        BLENDER, "--background",
        "--python", str(EXTRACTOR),
        "--",
        "--input", str(blend_path),
        "--output", str(tmp_dir),
    ]
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        elapsed = round(time.time() - t0, 1)

        # Find the produced JSON (stem == blend stem, usually "final")
        produced = sorted(tmp_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if produced and result.returncode == 0:
            produced[-1].rename(out_json)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            sz = round(out_json.stat().st_size / 1024)
            print(f"  OK ({elapsed}s, {sz}KB): {out_json.name}")
            return True

        print(f"  FAIL (code={result.returncode}, {elapsed}s): {blend_path.name}")
        if result.stderr:
            print("  STDERR:", result.stderr[-500:])
        return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT ({timeout}s): {blend_path.name}")
        return False


def verify_json_new_fields(json_path: Path) -> dict:
    """Check that the extracted JSON has the new fields we added."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    results = {
        "has_images": bool(data.get("images")),
        "has_full_res_png": False,
        "has_modifiers_on_any_obj": False,
        "has_rigid_body_on_any_obj": False,
        "object_count": len(data.get("objects", [])),
        "image_names": list(data.get("images", {}).keys()),
    }

    # Check images for full-res PNG
    for img_name, img_info in data.get("images", {}).items():
        if img_info.get("image_data"):
            results["has_full_res_png"] = True
            w, h = img_info.get("image_data_size", [0, 0])
            results.setdefault("png_images", []).append(
                f"{img_name} ({w}x{h}, {img_info.get('image_data_bytes',0)//1024}KB)"
            )

    # Check objects for modifiers / rigid body
    for obj in data.get("objects", []):
        if obj.get("modifiers"):
            results["has_modifiers_on_any_obj"] = True
            results.setdefault("modifier_examples", []).append(
                f"{obj['name']}: " + ", ".join(m["type"] for m in obj["modifiers"][:3])
            )
        if obj.get("rigid_body"):
            results["has_rigid_body_on_any_obj"] = True
            results.setdefault("rigid_body_examples", []).append(
                f"{obj['name']}: {obj['rigid_body'].get('type','?')}"
            )

    return results


def build_master_cache_for_source(source_name: str) -> bool:
    """Run build_master_cache.py for a specific source."""
    print(f"\n  Building master cache for source={source_name} ...")
    cmd = [
        sys.executable,
        str(BASE / "scripts/build_master_cache.py"),
        "--source", source_name,
        "--force",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    ok = result.returncode == 0
    if ok:
        entries = list((BASE / "data/master_cache" / source_name).glob("*.pt"))
        print(f"  Master cache: {len(entries)} entries in data/master_cache/{source_name}/")
    else:
        print(f"  FAIL: build_master_cache returned {result.returncode}")
        if result.stderr:
            print("  STDERR:", result.stderr[-500:])
    return ok


def rebuild_test_training_cache() -> bool:
    """Build a dedicated test_reextract training cache from the new master cache entries.

    This creates data/training_cache/test_reextract/ (not test_v3) so it
    doesn't disturb the existing overnight-run caches.  Load this task in the
    addon to verify modifiers + full-res textures.
    """
    build_script = BASE / "scripts/build_training_cache.py"
    if not build_script.exists():
        print("  build_training_cache.py not found – skipping")
        return False

    task_name = "test_reextract"
    task_cache_dir = BASE / "data/training_cache" / task_name
    print(f"\n  Building training cache task='{task_name}' ...")
    cmd = [
        sys.executable,
        str(build_script),
        "--source", task_name,
        "--task", task_name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        batches = list(task_cache_dir.glob("*.pt"))
        print(f"  Training cache: {len(batches)} file(s) in data/training_cache/{task_name}/")
        if result.stdout:
            tail = result.stdout[-400:].strip()
            if tail:
                print("  " + tail.replace("\n", "\n  "))
        return True
    print(f"  FAIL: build_training_cache returned {result.returncode}")
    if result.stderr:
        print(result.stderr[-500:])
    return False


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 1: Re-extract .blend files with updated blend_extractor.py")
    print("=" * 70)

    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    extracted = []
    for blend_path in TEST_BLENDS:
        if not blend_path.exists():
            print(f"  MISSING: {blend_path}")
            continue
        ok = run_blender_extract(blend_path, OUT_JSON_DIR, timeout=300)
        if ok:
            extracted.append(blend_path.parent.name)

    if not extracted:
        print("\nNo files extracted – aborting.")
        sys.exit(1)

    print(f"\nExtracted {len(extracted)} file(s).")

    print("\n" + "=" * 70)
    print("STEP 2: Verify new fields in extracted JSONs")
    print("=" * 70)
    for json_path in sorted(OUT_JSON_DIR.glob("*.json")):
        v = verify_json_new_fields(json_path)
        print(f"\n  {json_path.name}:")
        print(f"    objects          : {v['object_count']}")
        print(f"    has_images       : {v['has_images']} {v['image_names']}")
        print(f"    full_res_png     : {v['has_full_res_png']}", end="")
        if v.get("png_images"):
            print(f"  →  {v['png_images']}")
        else:
            print()
        print(f"    modifiers        : {v['has_modifiers_on_any_obj']}", end="")
        if v.get("modifier_examples"):
            print(f"  →  {v['modifier_examples']}")
        else:
            print()
        print(f"    rigid_body       : {v['has_rigid_body_on_any_obj']}", end="")
        if v.get("rigid_body_examples"):
            print(f"  →  {v['rigid_body_examples']}")
        else:
            print()

    print("\n" + "=" * 70)
    print("STEP 3: Build master cache for test_reextract source")
    print("=" * 70)
    mc_ok = build_master_cache_for_source("test_reextract")

    print("\n" + "=" * 70)
    print("STEP 4: Rebuild test training cache")
    print("=" * 70)
    tc_ok = rebuild_test_training_cache()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Extracted JSONs : {len(extracted)} files in {OUT_JSON_DIR}")
    print(f"  Master cache    : {'OK' if mc_ok else 'FAILED'} — data/master_cache/test_reextract/")
    print(f"  Training cache  : {'OK' if tc_ok else 'SKIPPED'} — data/training_cache/test_reextract/")
    print()
    print("Next steps:")
    print("  1. Install BlenderAICopilot.zip (already built)")
    print("  2. In Blender: open addon prefs → set Queue Task = 'test_reextract'")
    print("  3. Load Queue → navigate to an item → click Reconstruct Full Scene")
    print("  4. Expected: full-res textures + modifier stack in the viewport")


if __name__ == "__main__":
    main()
