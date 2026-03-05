"""Orchestrator for batch rendering mesh cache files via Blender headless.

Reads .pt cache files from data/processed/.mesh_cache/, decodes mesh tokens
back to vertices+faces using MeshTokenizer, exports temporary JSONs, and
calls Blender in --background mode to produce multi-view renders.

Usage:
    python scripts/render_cache.py --max-samples 100 --workers 2
    python scripts/render_cache.py --skip-existing --size 512 --views 8

Requires Blender installed at /Applications/Blender.app (macOS) or
'blender' on PATH.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from processing.mesh_tokenizer import MeshTokenizer


def load_config():
    """Load project config.yaml."""
    import yaml
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_blender():
    """Find the Blender executable."""
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


def decode_mesh_tokens(mesh_tokens, tokenizer):
    """Decode mesh token tensor back to vertices and faces.

    Args:
        mesh_tokens: torch.Tensor of token IDs
        tokenizer: MeshTokenizer instance

    Returns:
        (vertices, faces) lists or (None, None) on failure
    """
    tokens = mesh_tokens.tolist()
    try:
        vertices, faces = tokenizer.decode_tokens(tokens)
    except Exception as e:
        print(f"  Failed to decode tokens: {e}")
        return None, None

    if not vertices or not faces:
        return None, None

    return vertices, faces


def export_mesh_json(vertices, faces, label, mesh_id, output_path,
                     materials=None, face_material_indices=None,
                     uv_layers=None, vertex_color_layers=None,
                     face_smooth=None, images=None):
    """Export mesh data to a temporary JSON file for Blender.

    Includes full material node trees, per-face material assignments,
    UV coordinates, vertex colors, and image thumbnails so the renderer
    can rebuild the exact Blender material setup.
    """
    data = {
        "vertices": vertices,
        "faces": faces,
        "label": label,
        "mesh_id": mesh_id,
    }
    if materials:
        data["materials"] = materials
    if face_material_indices:
        data["face_material_indices"] = face_material_indices
    if uv_layers:
        data["uv_layers"] = uv_layers
    if vertex_color_layers:
        data["vertex_color_layers"] = vertex_color_layers
    if face_smooth:
        data["face_smooth"] = face_smooth
    if images:
        data["images"] = images

    with open(output_path, "w") as f:
        json.dump(data, f)

    return output_path


def render_single_mesh(blender_path, render_script, json_path, output_dir,
                       engine, timeout=300, mesh_id=None,
                       full_width=2560, full_height=1440, full_samples=128,
                       vp_width=1920, vp_height=1080, vp_samples=8,
                       skip_full=False, skip_viewport=False):
    """Call Blender headless to render one mesh. Returns (success, manifest_path, duration)."""
    cmd = [
        blender_path,
        "--background",
        "--factory-startup",
        "--python", str(render_script),
        "--",
        "--input",      str(json_path),
        "--output",     str(output_dir),
        "--width",      str(full_width),
        "--height",     str(full_height),
        "--samples",    str(full_samples),
        "--vp-width",   str(vp_width),
        "--vp-height",  str(vp_height),
        "--vp-samples", str(vp_samples),
        "--engine", engine,
    ]
    if skip_full:
        cmd.append("--skip-full")
    if skip_viewport:
        cmd.append("--skip-viewport")

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

        if result.returncode != 0:
            stderr_snippet = result.stderr[-500:] if result.stderr else ""
            return False, None, duration, f"Blender exit code {result.returncode}: {stderr_snippet}"

        # Use provided mesh_id (from cache data), NOT temp filename
        if mesh_id is None:
            mesh_id = Path(json_path).stem
        manifest_path = Path(output_dir) / f"{mesh_id}_manifest.json"
        if manifest_path.exists():
            return True, str(manifest_path), duration, None
        else:
            return False, None, duration, "Manifest not created"

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return False, None, duration, f"Timeout after {timeout}s"
    except Exception as e:
        duration = time.time() - start
        return False, None, duration, str(e)


def load_renders_as_tensors(output_dir, mesh_id):
    """Load rendered PNGs as uint8 tensors for embedding in .pt cache files.

    Loads the full render + all 14 viewport views. Returns a list of tensors
    (variable resolution) rather than a stacked tensor since full vs viewport
    renders may differ in size.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    render_dir = Path(output_dir)
    tensors = {}

    # Full quality render
    full_path = render_dir / f"{mesh_id}_full.png"
    if full_path.exists():
        img = Image.open(full_path).convert("RGB")
        tensors["full"] = torch.from_numpy(np.array(img, dtype=np.uint8))

    # 14 viewport views (view00 … view13)
    vp = []
    for i in range(14):
        p = render_dir / f"{mesh_id}_view{i:02d}.png"
        if p.exists():
            img = Image.open(p).convert("RGB")
            vp.append(torch.from_numpy(np.array(img, dtype=np.uint8)))

    if vp:
        # All viewport views are the same size — safe to stack
        tensors["viewport"] = torch.stack(vp)

    if not tensors:
        return None
    return tensors


def process_one_sample(args_tuple):
    """Process a single cache file: decode, export JSON, call Blender.

    This runs in a worker process.
    """
    (pt_path, renders_base_dir, blender_path, render_script,
     engine, tokenizer_config, skip_existing, timeout,
     full_width, full_height, full_samples,
     vp_width, vp_height, vp_samples) = args_tuple

    mesh_id = Path(pt_path).stem
    output_dir = Path(renders_base_dir) / mesh_id

    if skip_existing:
        manifest = output_dir / f"{mesh_id}_manifest.json"
        # Require zero-padded view00 naming (new format). Old renders use view0.png
        # (no zero-padding) and should NOT be skipped — they need re-rendering.
        has_new_views = (output_dir / f"{mesh_id}_view00.png").exists()
        if manifest.exists() and has_new_views:
            return {
                "mesh_id": mesh_id,
                "status": "skipped",
                "duration": 0.0,
            }

    tokenizer = MeshTokenizer(
        vocab_size=tokenizer_config["vocab_size"],
        coord_range=tuple(tokenizer_config["coord_range"]),
        max_faces=tokenizer_config.get("max_faces", 4000),
    )

    try:
        data = torch.load(pt_path, weights_only=False, map_location="cpu")
    except Exception as e:
        return {
            "mesh_id": mesh_id,
            "status": "error",
            "error": f"Failed to load .pt: {e}",
            "duration": 0.0,
        }

    if isinstance(data, list):
        if not data:
            return {
                "mesh_id": mesh_id,
                "status": "error",
                "error": "Empty .pt file",
                "duration": 0.0,
            }
        item = data[0]
    elif isinstance(data, dict):
        item = data
    else:
        return {
            "mesh_id": mesh_id,
            "status": "error",
            "error": f"Unexpected .pt format: {type(data).__name__}",
            "duration": 0.0,
        }

    mesh_tokens = item.get("mesh_tokens")
    if mesh_tokens is None:
        return {
            "mesh_id": mesh_id,
            "status": "error",
            "error": "No mesh_tokens in cache file",
            "duration": 0.0,
        }

    vertices, faces = decode_mesh_tokens(mesh_tokens, tokenizer)
    if vertices is None:
        return {
            "mesh_id": mesh_id,
            "status": "error",
            "error": "Token decode returned empty mesh",
            "duration": 0.0,
        }

    label = item.get("label", "")

    # Full material, UV, and face-assignment data from scene_context
    materials             = None
    face_material_indices = None
    uv_layers             = None
    vertex_color_layers   = None
    face_smooth           = None
    images                = None
    scene_ctx = item.get("scene_context")
    if scene_ctx and isinstance(scene_ctx, dict):
        materials             = scene_ctx.get("materials")
        face_material_indices = scene_ctx.get("face_material_indices")
        uv_layers             = scene_ctx.get("uv_layers")
        vertex_color_layers   = scene_ctx.get("vertex_color_layers")
        face_smooth           = scene_ctx.get("face_smooth")
        images                = scene_ctx.get("images")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                     dir=str(Path(renders_base_dir)),
                                     prefix=f"{mesh_id}_") as tmp:
        tmp_json_path = tmp.name
        export_mesh_json(vertices, faces, label, mesh_id, tmp_json_path,
                         materials=materials,
                         face_material_indices=face_material_indices,
                         uv_layers=uv_layers,
                         vertex_color_layers=vertex_color_layers,
                         face_smooth=face_smooth,
                         images=images)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        success, manifest_path, duration, error = render_single_mesh(
            blender_path, render_script, tmp_json_path, str(output_dir),
            engine, timeout=timeout, mesh_id=mesh_id,
            full_width=full_width, full_height=full_height, full_samples=full_samples,
            vp_width=vp_width, vp_height=vp_height, vp_samples=vp_samples,
        )
    finally:
        try:
            os.unlink(tmp_json_path)
        except OSError:
            pass

    result = {
        "mesh_id": mesh_id,
        "status": "success" if success else "error",
        "duration": duration,
        "n_vertices": len(vertices),
        "n_faces": len(faces),
        "label": label[:80],
    }
    if error:
        result["error"] = error
    if manifest_path:
        result["manifest"] = manifest_path

    return result


def update_cache_with_renders(cache_dir, renders_base_dir, mesh_ids):
    """Optionally update .pt cache files to include rendered image tensors."""
    updated = 0
    for mesh_id in mesh_ids:
        pt_path = Path(cache_dir) / f"{mesh_id}.pt"
        render_dir = Path(renders_base_dir) / mesh_id

        if not pt_path.exists() or not render_dir.exists():
            continue

        images = load_renders_as_tensors(str(render_dir), mesh_id)
        if images is None:
            continue

        try:
            data = torch.load(pt_path, weights_only=False, map_location="cpu")
            if isinstance(data, list) and data:
                data[0]["render_images"] = images
            elif isinstance(data, dict):
                data["render_images"] = images
            torch.save(data, pt_path)
            updated += 1
        except Exception as e:
            print(f"  Warning: failed to update {mesh_id}.pt: {e}")

    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Batch render mesh cache files via Blender headless"
    )
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Path to .mesh_cache directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for renders (default: data/renders/)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of cache items to render")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of parallel Blender processes")
    # Full render resolution/samples
    parser.add_argument("--full-width",   type=int, default=2560)
    parser.add_argument("--full-height",  type=int, default=1440)
    parser.add_argument("--full-samples", type=int, default=128)
    # Viewport render resolution/samples (14 views)
    parser.add_argument("--vp-width",   type=int, default=1920)
    parser.add_argument("--vp-height",  type=int, default=1080)
    parser.add_argument("--vp-samples", type=int, default=8)
    parser.add_argument("--engine", default="BLENDER_EEVEE_NEXT",
                        choices=["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"],
                        help="Render engine (applies to full render)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip meshes that already have renders")
    parser.add_argument("--update-cache", action="store_true",
                        help="Update .pt cache files with rendered image tensors")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per Blender render in seconds (default 300)")
    parser.add_argument("--blender", type=str, default=None,
                        help="Path to Blender executable")
    args = parser.parse_args()

    config = load_config()

    tok_config = {
        "vocab_size": config["tokenization"]["vocab_size"],
        "coord_range": config["tokenization"]["coordinate_range"],
        "max_faces": config["tokenization"]["max_faces"],
    }

    cache_dir = args.cache_dir or str(PROJECT_ROOT / "data" / "processed" / ".mesh_cache")
    renders_dir = args.output_dir or str(PROJECT_ROOT / "data" / "renders")
    render_script = str(PROJECT_ROOT / "processing" / "blender_render.py")

    blender_path = args.blender or find_blender()
    if not blender_path:
        print("ERROR: Blender not found. Install Blender or pass --blender /path/to/blender")
        sys.exit(1)

    print(f"Blender Batch Render Pipeline")
    print(f"  Blender:      {blender_path}")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Output dir:   {renders_dir}")
    print(f"  Full render:  {args.full_width}x{args.full_height} @ {args.full_samples}smp")
    print(f"  VP renders:   {args.vp_width}x{args.vp_height} @ {args.vp_samples}smp  (14 views)")
    print(f"  Workers:      {args.workers}")
    print(f"  Engine:       {args.engine}")
    print()

    pt_files = sorted(Path(cache_dir).glob("*.pt"))
    if not pt_files:
        print(f"ERROR: No .pt files found in {cache_dir}")
        sys.exit(1)

    if args.max_samples:
        pt_files = pt_files[:args.max_samples]

    total = len(pt_files)
    print(f"  Found {total} cache files to process")

    Path(renders_dir).mkdir(parents=True, exist_ok=True)

    task_args = [
        (str(pt_path), renders_dir, blender_path, render_script,
         args.engine, tok_config, args.skip_existing, args.timeout,
         args.full_width, args.full_height, args.full_samples,
         args.vp_width, args.vp_height, args.vp_samples)
        for pt_path in pt_files
    ]

    results = []
    success_count = 0
    error_count = 0
    skip_count = 0
    total_render_time = 0.0
    start_time = time.time()
    successful_ids = []

    print()
    print(f"  Rendering {total} meshes with {args.workers} workers...")
    print(f"  {'='*60}")

    if args.workers <= 1:
        for i, task in enumerate(task_args):
            result = process_one_sample(task)
            results.append(result)

            if result["status"] == "success":
                success_count += 1
                total_render_time += result["duration"]
                successful_ids.append(result["mesh_id"])
            elif result["status"] == "skipped":
                skip_count += 1
            else:
                error_count += 1

            done = i + 1
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0

            status_char = "OK" if result["status"] == "success" else (
                "SKIP" if result["status"] == "skipped" else "ERR")
            label_str = result.get("label", "")[:40]
            print(f"  [{done:>5}/{total}] {status_char:>4} {result['mesh_id'][:16]}  "
                  f"{result['duration']:5.1f}s  {label_str}  "
                  f"ETA: {eta/60:.0f}m")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one_sample, task): i
                       for i, task in enumerate(task_args)}

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "mesh_id": pt_files[idx].stem,
                        "status": "error",
                        "error": str(e),
                        "duration": 0.0,
                    }
                results.append(result)

                if result["status"] == "success":
                    success_count += 1
                    total_render_time += result["duration"]
                    successful_ids.append(result["mesh_id"])
                elif result["status"] == "skipped":
                    skip_count += 1
                else:
                    error_count += 1

                done = success_count + error_count + skip_count
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0

                status_char = "OK" if result["status"] == "success" else (
                    "SKIP" if result["status"] == "skipped" else "ERR")
                label_str = result.get("label", "")[:40]
                print(f"  [{done:>5}/{total}] {status_char:>4} {result['mesh_id'][:16]}  "
                      f"{result['duration']:5.1f}s  {label_str}  "
                      f"ETA: {eta/60:.0f}m")

    wall_time = time.time() - start_time

    if args.update_cache and successful_ids:
        print()
        print(f"  Updating {len(successful_ids)} cache files with render tensors...")
        updated = update_cache_with_renders(cache_dir, renders_dir, successful_ids)
        print(f"  Updated {updated} cache files.")

    print()
    print(f"  {'='*60}")
    print(f"  RENDER STATISTICS")
    print(f"  {'='*60}")
    print(f"  Total samples:   {total}")
    print(f"  Successful:      {success_count}")
    print(f"  Skipped:         {skip_count}")
    print(f"  Errors:          {error_count}")
    print(f"  Wall time:       {wall_time:.1f}s ({wall_time/60:.1f}m)")
    if success_count > 0:
        avg_time = total_render_time / success_count
        print(f"  Avg render time: {avg_time:.1f}s per mesh")
        print(f"  Throughput:      {success_count/wall_time*3600:.0f} meshes/hr")
    print(f"  Output:          {renders_dir}")

    if error_count > 0:
        print()
        print(f"  First 10 errors:")
        err_results = [r for r in results if r["status"] == "error"]
        for r in err_results[:10]:
            err_msg = r.get("error", "unknown")[:100]
            print(f"    {r['mesh_id'][:20]}: {err_msg}")

    errors_path = Path(renders_dir) / "render_errors.json"
    error_list = [r for r in results if r["status"] == "error"]
    if error_list:
        with open(errors_path, "w") as f:
            json.dump(error_list, f, indent=2)
        print(f"\n  Error log: {errors_path}")

    stats_path = Path(renders_dir) / "render_stats.json"
    stats = {
        "total": total,
        "success": success_count,
        "skipped": skip_count,
        "errors": error_count,
        "wall_time_seconds": wall_time,
        "avg_render_time": total_render_time / max(success_count, 1),
        "full_width":   args.full_width,
        "full_height":  args.full_height,
        "full_samples": args.full_samples,
        "vp_width":     args.vp_width,
        "vp_height":    args.vp_height,
        "vp_samples":   args.vp_samples,
        "viewport_views": 14,
        "engine": args.engine,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
