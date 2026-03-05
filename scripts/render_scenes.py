"""Orchestrator for scene renders — renders .blend files using their NATIVE camera.

Scans processed JSONs for files with an active camera and a resolvable
source .blend path, then calls Blender headless to render the scene AS-IS.
The rendered image + camera/lighting metadata are stored back into the
corresponding .pt cache files so training can learn what scene configurations
produce what output images.

This is DIFFERENT from render_cache.py:
  - render_cache.py:   Decodes mesh tokens → clean scene → 8 orbiting views
  - render_scenes.py:  Opens ORIGINAL .blend → uses its camera/lights → 1 render

Usage:
    python scripts/render_scenes.py
    python scripts/render_scenes.py --max-samples 50 --size 512
    python scripts/render_scenes.py --skip-existing --workers 2
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIRS = [
    PROJECT_ROOT / "data" / "processed" / d
    for d in [
        "blender_official", "blendswap", "github",
        "smutbase", "open3dlab", "youtube",
    ]
]

CACHE_DIR = PROJECT_ROOT / "data" / "processed" / ".mesh_cache"
RENDERS_DIR = PROJECT_ROOT / "data" / "renders" / "scenes"
SCENE_RENDER_SCRIPT = PROJECT_ROOT / "processing" / "blender_scene_render.py"


def load_config():
    import yaml
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_blender():
    candidates = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "/usr/local/bin/blender",
        "/usr/bin/blender",
        "/snap/bin/blender",
    ]
    import shutil
    path_blender = shutil.which("blender")
    if path_blender:
        candidates.insert(0, path_blender)
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def resolve_blend_path(source_file):
    """Resolve a source_file reference to an actual .blend path."""
    if not source_file:
        return None

    p = Path(source_file)

    if p.is_absolute() and p.exists() and p.suffix == ".blend":
        return str(p)

    relative = PROJECT_ROOT / p
    if relative.exists() and relative.suffix == ".blend":
        return str(relative)

    if p.suffix == ".blend":
        for search_dir in [
            PROJECT_ROOT / "data" / "raw" / "blender_official" / "models",
            PROJECT_ROOT / "data" / "raw" / "blendswap",
            PROJECT_ROOT / "data" / "raw" / "github",
            PROJECT_ROOT / "data" / "raw" / "smutbase",
        ]:
            if search_dir.exists():
                found = list(search_dir.rglob(p.name))
                if found:
                    return str(found[0])

    return None


def compute_cache_key(json_path):
    """Same key as rebuild_cache.py uses."""
    return hashlib.md5(str(json_path).encode()).hexdigest()[:16]


def scan_renderable_scenes():
    """Find all processed JSONs that have a camera and a resolvable .blend."""
    scenes = []

    for processed_dir in PROCESSED_DIRS:
        if not processed_dir.exists():
            continue

        for json_path in sorted(processed_dir.glob("*.json")):
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except Exception:
                continue

            if not data.get("active_camera"):
                continue

            source_file = data.get("source_file", "")
            blend_path = resolve_blend_path(source_file)
            if not blend_path:
                continue

            cache_key = compute_cache_key(json_path)
            scene_id = json_path.stem

            scenes.append({
                "json_path": str(json_path),
                "blend_path": blend_path,
                "cache_key": cache_key,
                "scene_id": scene_id,
                "source": processed_dir.name,
                "camera": data["active_camera"],
                "render_settings": data.get("render", {}),
            })

    return scenes


def render_scene(blender_path, blend_path, scene_id, output_dir,
                 size, max_samples, timeout=180):
    """Call Blender to render a scene using its native camera."""
    cmd = [
        blender_path,
        str(blend_path),
        "--background",
        "--python", str(SCENE_RENDER_SCRIPT),
        "--",
        "--output", str(output_dir),
        "--scene-id", scene_id,
        "--size", str(size),
        "--max-samples", str(max_samples),
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        duration = time.time() - start

        if result.returncode == 2:
            return "no_camera", duration, None

        if result.returncode != 0:
            stderr = result.stderr[-300:] if result.stderr else ""
            return "error", duration, f"Exit {result.returncode}: {stderr}"

        manifest_path = Path(output_dir) / f"{scene_id}_manifest.json"
        if manifest_path.exists():
            return "success", duration, str(manifest_path)
        else:
            return "error", duration, "No manifest created"

    except subprocess.TimeoutExpired:
        return "error", time.time() - start, f"Timeout ({timeout}s)"
    except Exception as e:
        return "error", time.time() - start, str(e)


def load_scene_render_as_tensor(output_dir, scene_id):
    """Load the scene render PNG as a uint8 tensor."""
    try:
        from PIL import Image
    except ImportError:
        return None

    img_path = Path(output_dir) / f"{scene_id}_scene.png"
    if not img_path.exists():
        return None

    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr)


def load_scene_manifest(output_dir, scene_id):
    """Load manifest with camera/lighting metadata."""
    manifest_path = Path(output_dir) / f"{scene_id}_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


def update_cache_with_scene_render(cache_key, scene_id, output_dir):
    """Embed scene render + metadata into the .pt cache file."""
    pt_path = CACHE_DIR / f"{cache_key}.pt"
    if not pt_path.exists():
        return False

    image_tensor = load_scene_render_as_tensor(output_dir, scene_id)
    manifest = load_scene_manifest(output_dir, scene_id)

    if image_tensor is None or manifest is None:
        return False

    scene_render_data = {
        "scene_render_image": image_tensor,
        "scene_camera": manifest.get("camera", {}),
        "scene_lights": manifest.get("lights", []),
        "scene_world": manifest.get("world"),
        "scene_compositor": manifest.get("compositor"),
        "scene_engine": manifest.get("render", {}).get("engine", ""),
        "scene_resolution": manifest.get("render", {}).get("resolution", []),
        "scene_objects_summary": manifest.get("scene_objects", {}),
    }

    try:
        data = torch.load(pt_path, weights_only=False, map_location="cpu")

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item["scene_render"] = scene_render_data
        elif isinstance(data, dict):
            data["scene_render"] = scene_render_data

        torch.save(data, pt_path)
        return True
    except Exception as e:
        print(f"  Warning: failed to update {cache_key}.pt: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Render .blend scenes using their native camera"
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of scenes to render")
    parser.add_argument("--size", type=int, default=512,
                        help="Max render resolution")
    parser.add_argument("--render-samples", type=int, default=32,
                        help="Max Cycles/EEVEE samples (low = fast)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel Blender processes")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip scenes already rendered")
    parser.add_argument("--update-cache", action="store_true",
                        help="Embed renders into .pt cache files")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Timeout per render in seconds")
    parser.add_argument("--blender", type=str, default=None,
                        help="Path to Blender executable")
    args = parser.parse_args()

    blender_path = args.blender or find_blender()
    if not blender_path:
        print("ERROR: Blender not found. Install or pass --blender")
        sys.exit(1)

    print(f"Scene Render Pipeline")
    print(f"  Blender:     {blender_path}")
    print(f"  Resolution:  {args.size}x{args.size} max")
    print(f"  Samples:     {args.render_samples}")
    print(f"  Output:      {RENDERS_DIR}")
    print()

    print(f"  Scanning for scenes with cameras...")
    scenes = scan_renderable_scenes()
    print(f"  Found {len(scenes)} renderable scenes")

    if not scenes:
        print("  No scenes with cameras found. Nothing to render.")
        return

    if args.skip_existing:
        filtered = []
        for s in scenes:
            manifest = RENDERS_DIR / s["scene_id"] / f"{s['scene_id']}_manifest.json"
            if manifest.exists():
                continue
            filtered.append(s)
        skipped = len(scenes) - len(filtered)
        scenes = filtered
        if skipped:
            print(f"  Skipping {skipped} already-rendered scenes")

    if args.max_samples and len(scenes) > args.max_samples:
        scenes = scenes[:args.max_samples]

    total = len(scenes)
    print(f"  Rendering {total} scenes...")
    print(f"  {'=' * 60}")

    success_count = 0
    skip_count = 0
    error_count = 0
    total_time = 0.0
    updated_cache = 0
    start_time = time.time()

    for i, scene in enumerate(scenes):
        scene_id = scene["scene_id"]
        output_dir = RENDERS_DIR / scene_id

        status, duration, info = render_scene(
            blender_path, scene["blend_path"], scene_id,
            str(output_dir), args.size, args.render_samples,
            timeout=args.timeout,
        )

        total_time += duration

        if status == "success":
            success_count += 1
            status_str = "OK"

            if args.update_cache:
                if update_cache_with_scene_render(
                    scene["cache_key"], scene_id, str(output_dir)
                ):
                    updated_cache += 1

        elif status == "no_camera":
            skip_count += 1
            status_str = "SKIP"
        else:
            error_count += 1
            status_str = "ERR"

        done = i + 1
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0

        label = scene_id[:30]
        print(f"  [{done:>4}/{total}] {status_str:>4} {label:<32s} "
              f"{duration:5.1f}s  {scene['source']}  "
              f"ETA: {eta / 60:.0f}m")

    wall_time = time.time() - start_time

    print()
    print(f"  {'=' * 60}")
    print(f"  SCENE RENDER STATISTICS")
    print(f"  {'=' * 60}")
    print(f"  Total scenes:    {total}")
    print(f"  Successful:      {success_count}")
    print(f"  No camera:       {skip_count}")
    print(f"  Errors:          {error_count}")
    print(f"  Wall time:       {wall_time:.1f}s ({wall_time / 60:.1f}m)")
    if success_count > 0:
        avg = total_time / success_count
        print(f"  Avg render time: {avg:.1f}s per scene")
    if args.update_cache:
        print(f"  Cache updated:   {updated_cache}")
    print(f"  Output:          {RENDERS_DIR}")
    print()

    stats = {
        "total": total,
        "success": success_count,
        "no_camera": skip_count,
        "errors": error_count,
        "wall_time_seconds": wall_time,
        "size": args.size,
        "render_samples": args.render_samples,
        "cache_updated": updated_cache if args.update_cache else None,
    }
    stats_path = RENDERS_DIR / "scene_render_stats.json"
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
