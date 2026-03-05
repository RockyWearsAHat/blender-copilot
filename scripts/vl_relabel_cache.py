#!/usr/bin/env python3
"""
vl_relabel_cache.py — Relabel .pt cache items using Qwen VL vision model.

For each item that has renders, sends the best view image to a local Ollama
VL model and writes the new label back into:
  1. The manifest JSON  (manifest["vl_label"])
  2. The .pt cache file (item["label"] is REPLACED with the VL label)

Usage:
    python scripts/vl_relabel_cache.py                        # all rendered items
    python scripts/vl_relabel_cache.py --max 500              # first 500
    python scripts/vl_relabel_cache.py --skip-existing        # only unlabeled
    python scripts/vl_relabel_cache.py --dry-run              # no writes
"""

import argparse
import base64
import json
import os
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RENDERS_DIR  = ROOT / "data" / "renders"
CACHE_DIR    = ROOT / "data" / "processed" / ".mesh_cache"

# ── Labeling prompt ───────────────────────────────────────────────────────
# Views sent: full render (if available), then upper-front-right diagonal,
# right side, and front — the most informative trio.
_VIEW_LABELS = [
    "[1] Full-quality render (native lighting)",
    "[2] Upper-front-right diagonal (45° azimuth, +45° elevation — best overall shape)",
    "[3] Right side view (eye-level)",
    "[4] Front view (eye-level)",
]

LABELING_PROMPT = """\
PURPOSE: Generate a TRAINING LABEL for a text-to-3D mesh AI model.
This label becomes the text prompt the model learns to associate with this 3D shape.
It must read like something a user would actually type in a 3D model generator.

GOOD LABELS: "medieval sword", "wooden dining chair", "iron knight helmet",
             "sci-fi space station", "cartoon bear head", "low-poly pine tree"
BAD LABELS: "3D object", "rendered asset", "detailed mesh", "game model",
            "nicely crafted piece" — too vague, too technical, or overly elaborate

IMAGES PROVIDED (1 to 4 views in order — upper diagonals and side views are most useful):
{image_manifest}

NOTE on perspective views: these are valid geometry captures but some angles may look at
the floor, the underside, or empty space. Ignore any view that appears black or uninformative.

Current label (may be inaccurate): "{current_label}"

RULES:
1. Base the label on actual geometry visible in the images — silhouette, structure, form.
2. The current label is a cross-reference hint only; override it if the geometry clearly shows something different.
3. Do NOT infer object type from any filename, folder name, or ID string.
4. If the shape is ambiguous use neutral terms: 'humanoid figure', 'quadruped animal', 'mechanical part'.
5. 3-7 words. No explanation. No quotes. No trailing punctuation.

Output ONLY the label."""


# ── HTTP helper ───────────────────────────────────────────────────────────
def _http_post(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"API error {e.code}: {err[:300]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama connection error: {e}") from e


def label_images(image_paths: list[Path], current_label: str, model: str,
                 ollama_url: str, timeout: int = 180) -> tuple[str, float]:
    """Call VL model on up to 4 views, return (new_label, elapsed_s)."""
    content: list[dict] = []
    manifest_lines: list[str] = []

    for i, img_path in enumerate(image_paths[:4]):
        try:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}})
            view_label = _VIEW_LABELS[i] if i < len(_VIEW_LABELS) else f"[{i+1}] view"
            manifest_lines.append(f"  {view_label}")
        except Exception:
            continue

    if not content:
        raise ValueError("No readable image paths provided")

    image_manifest = "\n".join(manifest_lines)
    prompt_text = LABELING_PROMPT.format(
        image_manifest=image_manifest,
        current_label=current_label,
    )
    content.append({"type": "text", "text": prompt_text})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
    }

    t0 = time.perf_counter()
    result = _http_post(f"{ollama_url}/v1/chat/completions", payload, timeout=timeout)
    elapsed = time.perf_counter() - t0

    label = result["choices"][0]["message"]["content"].strip().strip(".,;:")
    return label, elapsed


# Keep old name as alias so any callers still work
def label_image(image_path: Path, current_label: str, model: str,
               ollama_url: str, timeout: int = 120) -> tuple[str, float]:
    return label_images([image_path], current_label, model, ollama_url, timeout)


# ── View selection ────────────────────────────────────────────────────────
# 14-view index map:
#  00=front 01=back 02=right 03=left 04=top 05=bottom
#  06-09=upper diagonals (best)  10-13=lower diagonals (may hit floor)
# Order: full render > upper_front_right > right > front
_PREFERRED_VIEW_INDICES = (6, 2, 0, 7)  # upper-front-right, right, front, upper-back-right


def best_view_paths(render_dir: Path, mesh_id: str) -> list[Path]:
    """Return up to 4 views for labeling: full render + best viewport angles.
    Order matches _VIEW_LABELS in the prompt (full, upper-front-right, right, front).
    """
    paths: list[Path] = []

    # Full quality render first (most informative for material+shape)
    full = render_dir / f"{mesh_id}_full.png"
    if full.exists():
        paths.append(full)

    # Preferred viewport views (zero-padded new naming)
    for idx in _PREFERRED_VIEW_INDICES:
        p = render_dir / f"{mesh_id}_view{idx:02d}.png"
        if p.exists() and p not in paths:
            paths.append(p)
        if len(paths) >= 4:
            break

    # Old 4-view naming fallback
    if not paths:
        for idx in (1, 0, 2, 3):
            p = render_dir / f"{mesh_id}_view{idx}.png"
            if p.exists():
                paths.append(p)

    # Last resort: any PNG
    if not paths:
        pngs = sorted(render_dir.glob(f"{mesh_id}_view*.png"))
        if pngs:
            paths.append(pngs[0])

    return paths


def best_view_path(render_dir: Path, mesh_id: str) -> Optional[Path]:
    """Backwards-compat: return single best view path."""
    views = best_view_paths(render_dir, mesh_id)
    return views[0] if views else None


# ── Cache update ──────────────────────────────────────────────────────────
def update_pt_label(pt_path: Path, new_label: str) -> bool:
    """Replace the label in a .pt cache file. Returns True on success."""
    try:
        import torch
        data = torch.load(pt_path, weights_only=False, map_location="cpu")
        if isinstance(data, list) and data:
            data[0]["label"] = new_label
        elif isinstance(data, dict):
            data["label"] = new_label
        else:
            return False
        torch.save(data, pt_path)
        return True
    except Exception as e:
        print(f"    WARNING: failed to update {pt_path.name}: {e}", file=sys.stderr)
        return False


def update_manifest_label(manifest_path: Path, new_label: str) -> bool:
    """Write vl_label into manifest JSON."""
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        data["vl_label"]      = new_label
        data["original_label"] = data.get("vl_label", data.get("label", ""))
        data["label"]          = new_label
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"    WARNING: failed to update manifest {manifest_path.name}: {e}",
              file=sys.stderr)
        return False


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Relabel .pt cache items using Qwen VL renders."
    )
    parser.add_argument("--max",   type=int, default=None, help="Max items to process")
    parser.add_argument("--model", type=str, default="qwen2.5vl:32b")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip items that already have a vl_label in manifest")
    parser.add_argument("--dry-run", action="store_true",
                        help="Query VL model but do not write changes")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--renders-dir", type=str, default=str(RENDERS_DIR))
    parser.add_argument("--cache-dir",   type=str, default=str(CACHE_DIR))
    args = parser.parse_args()

    renders_dir = Path(args.renders_dir)
    cache_dir   = Path(args.cache_dir)

    # Warm VL model once so subsequent calls skip cold-boot
    try:
        sys.path.insert(0, str(ROOT))
        from processing.qwen_client import warm_model
        warm_model(model=args.model)
        print(f"Warmed VL model: {args.model}")
    except Exception as e:
        print(f"Warning: could not warm model ({e}), using cold calls")

    # Gather all manifests
    manifests = sorted(renders_dir.glob("*/*_manifest.json"))
    if not manifests:
        print(f"No manifests found in {renders_dir}")
        sys.exit(1)

    if args.skip_existing:
        filtered = []
        for mp in manifests:
            try:
                with open(mp) as f:
                    d = json.load(f)
                if "vl_label" not in d:
                    filtered.append(mp)
            except Exception:
                filtered.append(mp)
        print(f"Found {len(manifests):,} manifests, {len(filtered):,} without vl_label")
        manifests = filtered
    else:
        print(f"Found {len(manifests):,} manifests to process")

    if args.max:
        manifests = manifests[:args.max]
        print(f"Capped at {args.max}")

    # Stats
    processed = 0
    updated_pt = 0
    updated_manifest = 0
    errors = 0
    timings = []
    start_wall = time.time()

    print(f"\nRelabeling with {args.model}  {'[DRY RUN]' if args.dry_run else ''}")
    print(f"{'#':>6}  {'Old label':<40}  {'New label':<40}  {'t':>5}")
    print("-" * 100)

    for i, manifest_path in enumerate(manifests):
        mesh_id    = manifest_path.parent.name
        render_dir = manifest_path.parent
        pt_path    = cache_dir / f"{mesh_id}.pt"

        try:
            with open(manifest_path) as f:
                mdata = json.load(f)
            current_label = mdata.get("label", "")
        except Exception as e:
            print(f"{i+1:>6}  [manifest read error: {e}]")
            errors += 1
            continue

        view_paths = best_view_paths(render_dir, mesh_id)
        if not view_paths:
            print(f"{i+1:>6}  {current_label[:40]:<40}  [no view PNG found]")
            errors += 1
            continue

        try:
            new_label, elapsed = label_images(
                view_paths, current_label, args.model,
                args.ollama_url, args.timeout,
            )
            timings.append(elapsed)
        except Exception as e:
            print(f"{i+1:>6}  {current_label[:40]:<40}  [VL ERROR: {str(e)[:50]}]")
            errors += 1
            continue

        changed = new_label.lower() != current_label.lower()
        marker  = " *" if changed else "  "
        old_d   = current_label[:38] + ".." if len(current_label) > 40 else current_label
        new_d   = new_label[:38]     + ".." if len(new_label)     > 40 else new_label
        nv      = len(view_paths)
        print(f"{i+1:>6}  {old_d:<40}  {new_d:<40}  {elapsed:>4.1f}s  {nv}v{marker}")
        processed += 1

        if not args.dry_run:
            if update_manifest_label(manifest_path, new_label):
                updated_manifest += 1
            if pt_path.exists():
                if update_pt_label(pt_path, new_label):
                    updated_pt += 1

        # Progress ETA every 50 items
        if (i + 1) % 50 == 0 and timings:
            mean_t = sum(timings) / len(timings)
            remaining = len(manifests) - (i + 1)
            eta_m = mean_t * remaining / 60
            elapsed_wall = time.time() - start_wall
            print(f"  --- {i+1}/{len(manifests)} done  "
                  f"mean={mean_t:.1f}s  ETA≈{eta_m:.0f}m  "
                  f"wall={elapsed_wall/60:.1f}m ---")

    # Summary
    wall = time.time() - start_wall
    print("\n" + "=" * 100)
    print("RELABELING SUMMARY")
    print("=" * 100)
    print(f"  Processed:          {processed:,} / {len(manifests):,}")
    print(f"  Errors:             {errors:,}")
    if args.dry_run:
        print(f"  DRY RUN — no files written")
    else:
        print(f"  Manifests updated:  {updated_manifest:,}")
        print(f"  Cache .pt updated:  {updated_pt:,}")
    if timings:
        mean_t = sum(timings) / len(timings)
        print(f"  Mean time/item:     {mean_t:.1f}s")
        print(f"  Wall time:          {wall/60:.1f}m")
        total_cache = len(list(cache_dir.glob("*.pt")))
        est_hours = mean_t * total_cache / 3600
        print(f"  Est. full cache ({total_cache:,} items): {est_hours:.1f}h at {mean_t:.1f}s/item")


if __name__ == "__main__":
    main()
