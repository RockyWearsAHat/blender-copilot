#!/usr/bin/env python3
"""Fresh end-to-end test: pick 5 source JSONs, build v3 master cache + training cache.

This bypasses the existing master cache entirely so we test the full v3 pipeline
from scratch. Output goes to data/master_cache_test/ and data/training_cache/test_v3/.
"""
import json
import glob
import hashlib
import os
import sys
import shutil
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

# ── Step 1: Pick 5 diverse source JSONs ──────────────────────────────

def pick_sources(n=5):
    """Return 5 hardcoded diverse source files (small, fast to load)."""
    base = ROOT / "data" / "processed"
    picks = []
    candidates = [
        ("objaverse", base / "objaverse" / "26adfd631730494ab0fc80c806eada77.json"),
        ("blendswap", base / "blendswap" / "31621.json"),
        ("blender_official", base / "blender_official" / "candy_bounce.json"),
        ("open3dlab", None),  # will find first small one
        ("objaverse", base / "objaverse" / "da1301f7148f421698d6b9dc1858d45b.json"),
        ("smutbase", base / "smutbase" / "GenericCartoonFemale.json"),
    ]
    for src, path in candidates:
        if len(picks) >= n:
            break
        if path is None:
            # Find first small file from this source
            import glob as g
            for f in sorted(g.glob(str(base / src / "*.json"))):
                if os.path.getsize(f) < 5_000_000:
                    path = Path(f)
                    break
        if path is None or not path.exists():
            continue
        try:
            j = json.load(open(path))
            objs = j.get("objects", [])
            total_faces = sum(len(o.get("mesh", {}).get("faces", [])) for o in objs)
            tags = j.get("metadata", {}).get("tags", [])
            label = j.get("label", j.get("metadata", {}).get("name", ""))
            has_mats = any(o.get("materials") for o in objs)
            picks.append({
                "src": src,
                "file": str(path),
                "objs": len(objs),
                "faces": total_faces,
                "label": str(label)[:80],
                "tags": tags[:8],
                "has_mats": has_mats,
            })
        except Exception as e:
            print(f"  Skip {path}: {e}")
    return picks[:n]


# ── Step 2: Build v3 master cache for just these files ───────────────

def build_test_master_cache(picks, out_dir):
    """Build master cache entries for picked files only."""
    from scripts.build_master_cache import convert_source_file, CACHE_VERSION

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in picks:
        filepath = Path(p["file"])
        src = p["src"]
        src_dir = out_dir / src
        src_dir.mkdir(parents=True, exist_ok=True)

        try:
            entry = convert_source_file(filepath, src)
            if entry is None:
                print(f"  SKIP (convert returned None): {filepath.name}")
                continue

            file_hash = hashlib.md5(str(filepath).encode()).hexdigest()[:16]
            cache_path = src_dir / f"{file_hash}.pt"
            torch.save(entry, cache_path)

            rel_path = f"{src}/{file_hash}.pt"
            n_obj = len(entry.get("objects", []))
            version = entry.get("_version", "?")
            anims = len(entry.get("animations", []))
            mat_names = entry.get("material_names", [])
            per_obj = entry.get("per_object_labels", [])

            print(f"  OK  v{version} [{src}] {filepath.name}")
            print(f"      {n_obj} objects, {p['faces']} faces")
            print(f"      animations: {anims}, materials: {mat_names[:5]}")
            print(f"      per_object_labels: {len(per_obj)}")
            if per_obj:
                for pol in per_obj[:3]:
                    print(f"        obj '{pol.get('name','')}' text_label='{pol.get('text_label','')}' file_label='{pol.get('file_label','')}'")

            entries.append({
                "path": rel_path,
                "source": src,
                "source_file": str(filepath.name),
                "n_objects": n_obj,
            })
        except Exception as e:
            print(f"  FAIL [{src}] {filepath.name}: {e}")
            import traceback; traceback.print_exc()

    # Write index
    index = {
        "version": CACHE_VERSION,
        "built_at": time.time(),
        "entries": entries,
    }
    torch.save(index, out_dir / "index.pt")
    print(f"\nMaster cache index: {len(entries)} entries at {out_dir / 'index.pt'}")
    return entries


# ── Step 3: Build training cache from test master cache ──────────────

def build_test_training_cache(master_dir, out_dir):
    """Build training cache from the test master cache."""
    from scripts.build_training_cache import build_training_cache, DEFAULT_CONFIG

    # Temporarily override paths
    import scripts.build_training_cache as btc
    orig_master = btc.MASTER_CACHE_DIR
    orig_train = btc.TRAINING_CACHE_DIR

    btc.MASTER_CACHE_DIR = master_dir
    btc.TRAINING_CACHE_DIR = out_dir.parent  # parent because task_name becomes subdir

    config = dict(DEFAULT_CONFIG)
    config["task_name"] = out_dir.name
    config["min_faces"] = 20  # Lower threshold for test
    config["max_per_label"] = 5
    config["max_per_source_file"] = 10
    config["require_materials"] = False
    config["require_english"] = True

    try:
        stats = build_training_cache(config, dry_run=False)
    finally:
        btc.MASTER_CACHE_DIR = orig_master
        btc.TRAINING_CACHE_DIR = orig_train

    return stats


# ── Step 4: Inspect results ──────────────────────────────────────────

def inspect_training_cache(cache_dir):
    """Print label details for all items in the test training cache."""
    batch_files = sorted(cache_dir.glob("batch_*.pt"))
    if not batch_files:
        print("  No batch files found!")
        return

    total = 0
    for bf in batch_files:
        items = torch.load(bf, weights_only=False)
        for item in items:
            total += 1
            label = item.get("label", "???")
            src = item.get("data_source", "?")
            ref = item.get("master_cache_ref", {})
            n_tokens = len(item.get("mesh_tokens", []))
            n_faces = (n_tokens - 2) // 9 if n_tokens > 2 else 0
            sc = item.get("scene_context", {})
            mats = sc.get("materials", []) if isinstance(sc, dict) else []
            mat_names = [m.get("name", "") for m in mats if isinstance(m, dict)]
            print(f"  [{total}] label=\"{label}\"")
            print(f"       src={src} faces~{n_faces} mats={mat_names[:4]}")
            print(f"       ref={ref.get('source_file','')} obj#{ref.get('object_index','?')}")
            print()

    print(f"Total training items: {total}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FRESH V3 PIPELINE TEST — 5 items end-to-end")
    print("=" * 70)

    master_dir = ROOT / "data" / "master_cache_test"
    train_dir = ROOT / "data" / "training_cache" / "test_v3"

    # Clean previous test
    if master_dir.exists():
        shutil.rmtree(master_dir)
    if train_dir.exists():
        shutil.rmtree(train_dir)

    # Step 1: Pick sources
    print("\n── Step 1: Picking 5 source files ──")
    picks = pick_sources(5)
    for i, p in enumerate(picks):
        print(f"  {i+1}. [{p['src']}] {os.path.basename(p['file'])}")
        print(f"     {p['objs']} objs, {p['faces']} faces, mats={p['has_mats']}")
        print(f"     label: {p['label']}")
        print(f"     tags: {p['tags']}")

    # Step 2: Build master cache
    print("\n── Step 2: Building v3 master cache ──")
    entries = build_test_master_cache(picks, master_dir)
    if not entries:
        print("ERROR: No master cache entries built!")
        sys.exit(1)

    # Step 3: Build training cache
    print("\n── Step 3: Building training cache ──")
    stats = build_test_training_cache(master_dir, train_dir)
    print(f"\nTraining cache stats: {dict(stats) if stats else 'empty'}")

    # Step 4: Inspect
    print("\n── Step 4: Training items produced ──")
    inspect_training_cache(train_dir)

    # Report paths
    print("\n" + "=" * 70)
    print(f"Master cache: {master_dir}")
    print(f"Training cache: {train_dir}")
    batch_files = sorted(train_dir.glob("batch_*.pt"))
    print(f"Batch files: {len(batch_files)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
