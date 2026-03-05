#!/usr/bin/env python3
"""
Targeted re-extraction test:
  1. Download candy_bounce.blend from download.blender.org (public, no auth)
     → verify: full-res PNG images, modifiers (BEVEL), rigid_body physics
  2. Download 1 recent smutbase .blend via the scraper (uses .env credentials)
     → verify: full-res PNG images (was previously only thumbnail quality)
  3. Rebuild master cache entries for both
  4. Summarise results

Usage:
    .venv/bin/python scripts/_test_image_rb.py
"""

import json
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path

# ── bootstrap sys.path so we can import scrapers/ ────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BLENDER       = "/Applications/Blender.app/Contents/MacOS/Blender"
EXTRACTOR     = ROOT / "processing/blend_extractor.py"
BUILD_MASTER  = ROOT / "scripts/build_master_cache.py"

# candy_bounce.blend — publicly hosted Blender demo with BEVEL modifiers,
# rigid-body physics, and node-based materials that reference images.
CANDY_URL     = "https://download.blender.org/demo/geometry-nodes/fields/candy_bounce.blend"
CANDY_DEST    = ROOT / "data/raw/blender_official/candy_bounce.blend"

OUT_CANDY     = ROOT / "data/processed/blender_official"  # existing folder
OUT_SMUTBASE  = ROOT / "data/processed/smutbase_fresh"    # new temp folder
OUT_RB        = ROOT / "data/processed/test_rb"           # rigid body test


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, timeout: int = 120) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  Already downloaded: {dest.name} ({dest.stat().st_size // 1024}KB)")
        return True
    print(f"  Downloading {url} ...")
    cmd = ["curl", "-L", "--silent", "--show-error", "-o", str(dest), url]
    result = subprocess.run(cmd, timeout=timeout)
    if result.returncode == 0 and dest.exists() and dest.stat().st_size > 10_000:
        print(f"  OK: {dest.name} ({dest.stat().st_size // 1024}KB)")
        return True
    print(f"  FAIL: {dest}")
    return False


def run_blender_extract(blend_path: Path, out_dir: Path, label: str,
                        timeout: int = 300) -> Path | None:
    """Extract one .blend and return the produced JSON path."""
    tmp_dir = out_dir / f"_tmp_{label}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done (by looking for file in out_dir)
    existing = list(out_dir.glob(f"{label}.json"))
    if existing:
        print(f"  SKIP (already extracted): {existing[0].name}")
        return existing[0]

    cmd = [
        BLENDER, "--background",
        "--python", str(EXTRACTOR),
        "--", "--input", str(blend_path), "--output", str(tmp_dir),
    ]
    print(f"  Extracting {blend_path.name} ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = round(time.time() - t0, 1)

    produced = sorted(tmp_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if produced and result.returncode == 0:
        dest_json = out_dir / f"{label}.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        produced[-1].rename(dest_json)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  OK ({elapsed}s, {dest_json.stat().st_size // 1024}KB): {dest_json.name}")
        return dest_json

    print(f"  FAIL code={result.returncode} ({elapsed}s)")
    tail = (result.stderr or "")[-600:]
    if tail:
        print("  STDERR:", tail)
    return None


def build_master_cache(source_name: str) -> bool:
    """Run build_master_cache.py for one source folder."""
    print(f"  Building master cache: {source_name} ...")
    result = subprocess.run(
        [sys.executable, str(BUILD_MASTER), "--source", source_name, "--force"],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        entries = list((ROOT / "data/master_cache" / source_name).glob("*.pt"))
        print(f"  {len(entries)} entries in data/master_cache/{source_name}/")
        return True
    print(f"  FAIL: {result.stderr[-300:]}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Rigid-body scene script (runs inside Blender to create the .blend)
# ─────────────────────────────────────────────────────────────────────────────

_RB_SCENE_PY = '''
import bpy, sys
from pathlib import Path

bpy.ops.wm.read_factory_settings(use_empty=True)

# Ground — PASSIVE rigid body
bpy.ops.mesh.primitive_plane_add(size=10, location=(0,0,0))
g = bpy.context.active_object; g.name = "Ground"
bpy.ops.rigidbody.world_add()
bpy.ops.rigidbody.object_add()
g.rigid_body.type = "PASSIVE"; g.rigid_body.friction = 0.7; g.rigid_body.restitution = 0.3
m = bpy.data.materials.new("GroundMat"); m.use_nodes = True
m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.3,0.6,0.3,1)
g.data.materials.append(m)

# Sphere — ACTIVE rigid body
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(0,0,3))
s = bpy.context.active_object; s.name = "BouncingSphere"
bpy.ops.rigidbody.object_add()
s.rigid_body.type = "ACTIVE"; s.rigid_body.mass = 2.5
s.rigid_body.friction = 0.5; s.rigid_body.restitution = 0.8
m2 = bpy.data.materials.new("SphereMat"); m2.use_nodes = True
m2.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.9,0.2,0.2,1)
s.data.materials.append(m2)

# Cube with BEVEL modifier — ACTIVE rigid body
bpy.ops.mesh.primitive_cube_add(size=0.8, location=(1.5,0,5))
c = bpy.context.active_object; c.name = "BouncingCube"
bpy.ops.rigidbody.object_add()
c.rigid_body.type = "ACTIVE"; c.rigid_body.mass = 1.0
c.rigid_body.friction = 0.4; c.rigid_body.restitution = 0.6
bv = c.modifiers.new("Bevel","BEVEL"); bv.width = 0.05; bv.segments = 2
m3 = bpy.data.materials.new("CubeMat"); m3.use_nodes = True
m3.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.2,0.4,0.9,1)
c.data.materials.append(m3)

# Camera + sun
bpy.ops.object.camera_add(location=(7,-7,5))
bpy.context.active_object.rotation_euler = (1.1,0,0.8)
bpy.context.scene.camera = bpy.context.active_object
bpy.ops.object.light_add(type="SUN",location=(5,5,10))
bpy.context.active_object.data.energy = 3.0

out = Path(sys.argv[sys.argv.index("--")+1]) if "--" in sys.argv else Path("/tmp")
out.mkdir(parents=True, exist_ok=True)
bp = out / "rigid_body_test.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(bp))
print("SAVED:", bp)
'''


def step_rigid_body() -> dict | None:
    """Create a synthetic rigid-body .blend with Blender, then extract it."""
    print("\n" + "=" * 70)
    print("STEP 2 — Rigid body physics (synthetic test scene)")
    print("=" * 70)

    rb_blend = OUT_RB / "rigid_body_test.blend"

    if not rb_blend.exists():
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tf:
            tf.write(_RB_SCENE_PY)
            scene_script = tf.name

        print("  Creating rigid-body scene in Blender ...")
        OUT_RB.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            [BLENDER, "--background", "--python", scene_script, "--", str(OUT_RB)],
            capture_output=True, text=True, timeout=90,
        )
        os.unlink(scene_script)
        if res.returncode != 0 or not rb_blend.exists():
            print(f"  ❌  Scene creation failed (code={res.returncode})")
            print(res.stderr[-400:])
            return None
        print(f"  ✅  Saved: {rb_blend.name}")

    jp = run_blender_extract(rb_blend, OUT_RB, "rigid_body_test")
    return audit_json(jp, "rigid_body_test") if jp else None


def audit_json(json_path: Path, label: str) -> dict:
    """Return summary of extracted JSON fields."""
    d = json.load(open(json_path, encoding="utf-8"))
    objs   = d.get("objects") or []
    imgs   = d.get("images") or {}

    result = {
        "objects": len(objs),
        "images":  len(imgs),
        "has_full_png": False,
        "has_thumbnail": False,
        "modifier_summary": [],
        "rigid_body_objects": [],
        "png_details": [],
    }

    for img_name, iv in imgs.items():
        if iv.get("image_data"):
            result["has_full_png"] = True
            w, h = iv.get("image_data_size", [0, 0])
            kb = (iv.get("image_data_bytes") or 0) // 1024
            result["png_details"].append(f"{img_name} ({w}×{h}, {kb}KB)")
        if iv.get("thumbnail"):
            result["has_thumbnail"] = True

    for o in objs:
        mods = o.get("modifiers") or []
        if mods:
            result["modifier_summary"].append(
                f"{o['name']}: " + ", ".join(m.get("type","?") for m in mods)
            )
        if o.get("rigid_body"):
            result["rigid_body_objects"].append(
                f"{o['name']} ({o['rigid_body'].get('type','?')})"
            )

    return result


def print_audit(label: str, r: dict):
    ok   = lambda b: "✅" if b else "❌"
    print(f"\n  {label}:")
    print(f"    objects       : {r['objects']}")
    print(f"    images        : {r['images']}")
    print(f"    full_res_PNG  : {ok(r['has_full_png'])}", end="")
    if r["png_details"]:
        print(f"  → {r['png_details'][:3]}")
    else:
        print()
    print(f"    thumbnail     : {ok(r['has_thumbnail'])}")
    if r["modifier_summary"]:
        print(f"    modifiers     : ✅ {r['modifier_summary'][:4]}")
    else:
        print(f"    modifiers     : ❌  (none in scene)")
    if r["rigid_body_objects"]:
        print(f"    rigid_body    : ✅ {r['rigid_body_objects']}")
    else:
        print(f"    rigid_body    : ❌  (none in scene)")


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — candy_bounce.blend
# ─────────────────────────────────────────────────────────────────────────────

def step_candy():
    print("\n" + "=" * 70)
    print("STEP 1 — candy_bounce.blend  (full-res PNG + modifiers)")
    print("=" * 70)

    if not download_file(CANDY_URL, CANDY_DEST):
        return None

    json_path = run_blender_extract(CANDY_DEST, OUT_CANDY, "candy_bounce_fresh")
    if json_path:
        return audit_json(json_path, "candy_bounce_fresh")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 — smutbase: download ONE file that packs textures
# ─────────────────────────────────────────────────────────────────────────────

def step_smutbase():
    print("\n" + "=" * 70)
    print("STEP 3 — smutbase (full-res PNG image extraction)")
    print("=" * 70)

    OUT_SMUTBASE.mkdir(parents=True, exist_ok=True)
    tmp_blend_dir = ROOT / "data/raw/smutbase_fresh"
    tmp_blend_dir.mkdir(parents=True, exist_ok=True)

    # Use any existing blend EXCEPT Domina (external textures only — no packed images)
    existing_blends = [b for b in tmp_blend_dir.glob("*.blend") if "Domina" not in b.name]
    if existing_blends:
        blend_path = existing_blends[0]
        print(f"  Already have: {blend_path.name}")
    else:
        blend_path = _smutbase_download_one(tmp_blend_dir)

    if not blend_path or not blend_path.exists():
        print("  ❌  Could not obtain a smutbase .blend file")
        return None

    label = blend_path.stem[:50]
    json_path = run_blender_extract(blend_path, OUT_SMUTBASE, label)
    if json_path:
        return audit_json(json_path, label)
    return None


def _smutbase_download_one(dest_dir: Path) -> Path | None:
    """Login to smutbase and download the first available .blend file."""
    try:
        from scrapers.smutbase_scraper import create_session, get_listing_page, \
            get_project_details, download_project_file
    except ImportError as e:
        print(f"  Import error: {e}")
        return None

    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")

    session = create_session("smutbase")
    print("  Fetching listing page 1 ...")
    projects = get_listing_page(session, "https://smutba.se", page=1,
                                software_tag="blender", sort_by="popular")
    if not projects:
        print("  No projects returned (login may have failed)")
        return None

    print(f"  Got {len(projects)} projects, trying best candidates ...")
    for proj in projects[:10]:
        title = proj["title"]
        if "Domina" in title:   # has only external textures, skip
            print(f"  → Skipping {title} (external textures)")
            continue
        pid = proj["project_id"]
        print(f"  → {title}")
        details = get_project_details(session, proj["url"], pid)
        if not details:
            continue
        blend_path = download_project_file(session, details, dest_dir)
        if blend_path and Path(blend_path).exists() and Path(blend_path).stat().st_size > 50_000:
            print(f"  Downloaded: {Path(blend_path).name}")
            return Path(blend_path)

    print("  Could not download any project with packed textures")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    candy_audit    = step_candy()
    rb_audit       = step_rigid_body()
    smutbase_audit = step_smutbase()

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    if candy_audit:
        print_audit("candy_bounce_fresh (PNG + modifiers)", candy_audit)
    else:
        print("\n  candy_bounce: ❌  extraction failed")

    if rb_audit:
        print_audit("rigid_body_test (physics)", rb_audit)
    else:
        print("\n  rigid_body_test: ❌  extraction failed")

    if smutbase_audit:
        print_audit("smutbase (packed textures → full-res PNG)", smutbase_audit)
    else:
        print("\n  smutbase: ❌  extraction failed or skipped")


if __name__ == "__main__":
    main()
