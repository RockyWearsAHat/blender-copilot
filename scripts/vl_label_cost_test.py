#!/usr/bin/env python3
"""
vl_label_cost_test.py — Benchmark Qwen VL relabeling cost for the training cache.

Samples N items from data/renders/, sends view images to a local Qwen VL
model via Ollama, and estimates total time to relabel the full dataset.

Usage:
    python scripts/vl_label_cost_test.py
    python scripts/vl_label_cost_test.py --sample 50 --view 1
    python scripts/vl_label_cost_test.py --model qwen2.5vl:7b --views all
    python scripts/vl_label_cost_test.py --sample 100 --output results.json
"""

import argparse
import base64
import json
import os
import random
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RENDERS_DIR = ROOT / "data" / "renders"

# ── Labeling prompt ───────────────────────────────────────────────────────
LABELING_PROMPT = """\
You are labeling 3D mesh objects for a text-to-mesh AI training dataset.

Look at this rendered 3D object and provide a concise, accurate label.

Requirements:
- 2 to 6 words maximum
- Describe the specific object type (e.g. "wooden chair" not just "furniture")
- Include key distinctive features if visible (e.g. "round dining table", "curved sword blade")
- Use simple, common English words
- Do NOT include render quality notes, polygon count, or artistic style
- Do NOT use adjectives like "beautifully", "detailed", "rendered", "3D"

Current label (may be wrong or vague): "{current_label}"

Respond with ONLY the new label — no explanation, no punctuation at end."""


# ── HTTP helper (same pattern as ai_engine.py) ────────────────────────────
def _http_post(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"API error {e.code}: {err[:400]}") from e
    except urllib.error.URLError as e:
        if "Connection refused" in str(e) or "No connection" in str(e):
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure it is running:\n"
                "  ollama serve"
            ) from e
        raise RuntimeError(f"Connection error: {str(e)[:300]}") from e
    return json.loads(body)


def _encode_image(path: Path) -> str:
    """Base64-encode a PNG image."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Core labeling call ────────────────────────────────────────────────────
def label_with_vl(
    image_paths: list[Path],
    current_label: str,
    model: str,
    ollama_url: str,
    timeout: int = 90,
) -> tuple[str, float]:
    """Send image(s) to Qwen VL and return (new_label, elapsed_seconds)."""
    content = []

    # Attach each view as an image_url block
    for img_path in image_paths:
        b64 = _encode_image(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })

    # Add the text prompt
    content.append({
        "type": "text",
        "text": LABELING_PROMPT.format(current_label=current_label),
    })

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

    new_label = result["choices"][0]["message"]["content"].strip().strip(".,;:")
    return new_label, elapsed


# ── Manifest loading ──────────────────────────────────────────────────────
def load_manifests(renders_dir: Path) -> list[dict]:
    manifests = []
    for manifest_path in renders_dir.glob("*/*_manifest.json"):
        try:
            with open(manifest_path) as f:
                data = json.load(f)
            render_dir = manifest_path.parent
            data["_render_dir"] = render_dir
            # Build view_path lookup by view_index
            # Supports both new zero-padded names (view00) and old names (view0)
            view_map: dict[int, Path] = {}
            for r in data.get("renders", []):
                idx = r.get("view_index", -1)
                fname = r.get("filename", "")
                if fname:
                    p = render_dir / fname
                    if p.exists():
                        view_map[idx] = p
            # Also scan directory for any view PNGs not in manifest
            if not view_map:
                for png in sorted(render_dir.glob("*_view*.png")):
                    name = png.stem  # e.g. hash_view02 or hash_view1
                    import re
                    m = re.search(r'_view(\d+)$', name)
                    if m:
                        view_map[int(m.group(1))] = png
            if view_map:
                data["_view_map"] = view_map
                manifests.append(data)
        except Exception:
            continue
    return manifests


def get_views(manifest: dict, view_spec: str) -> list[Path]:
    """Return list of view Paths based on view_spec.

    New naming: view00=front, view01=back, view02=right, view03=left,
                view04=top, view05=bottom, view06-09=upper diagonals,
                view10-13=lower diagonals.
    Also supports old 4-view naming (view0..view3) for backwards compat.
    """
    view_map: dict[int, Path] = manifest.get("_view_map", {})
    if not view_map:
        return []

    if view_spec == "all":
        return [view_map[k] for k in sorted(view_map)]
    elif view_spec == "best":
        # Prefer: upper-front-right diagonal (idx 6) → right side (idx 2) → front (idx 0)
        for preferred in (6, 2, 0, 1, 3):
            if preferred in view_map:
                return [view_map[preferred]]
        return [list(view_map.values())[0]]
    else:
        idx = int(view_spec)
        if idx in view_map:
            return [view_map[idx]]
        return [list(view_map.values())[0]]


# ── Main test ─────────────────────────────────────────────────────────────
def run_test(args) -> None:
    # 1. Check Ollama is reachable
    print(f"Checking Ollama at {args.ollama_url} …", flush=True)
    try:
        req = urllib.request.Request(f"{args.ollama_url}/api/tags")
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
            tags_data = json.loads(r.read())
        model_names = [m["name"] for m in tags_data.get("models", [])]
        print(f"  Available models: {', '.join(model_names) if model_names else '(none listed)'}")
        matched = [m for m in model_names if args.model.split(":")[0] in m]
        if not matched:
            print(f"  WARNING: model '{args.model}' not found in Ollama. Available: {model_names}")
            print("  Proceeding anyway — Ollama may pull it on first request.")
        else:
            print(f"  Model '{matched[0]}' is available. ✓")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    # 2. Load manifests
    print(f"\nLoading manifests from {RENDERS_DIR} …", flush=True)
    if not RENDERS_DIR.exists():
        print(f"  ERROR: {RENDERS_DIR} does not exist.")
        sys.exit(1)
    all_manifests = load_manifests(RENDERS_DIR)
    total_available = len(all_manifests)
    print(f"  Found {total_available:,} rendered items with usable views.")

    if total_available == 0:
        print("  No rendered items found. Run:  python run.py render")
        sys.exit(1)

    # 3. Sample
    n = min(args.sample, total_available)
    if args.seed is not None:
        random.seed(args.seed)
    sample = random.sample(all_manifests, n)
    print(f"\nSampling {n} items (views: {args.views}) from {total_available:,} total.\n")

    # 4. Run inference
    results = []
    timings = []
    errors = 0

    print(f"{'#':>4}  {'Old label':<45}  {'New label':<45}  {'Time':>6}")
    print("-" * 110)

    for i, manifest in enumerate(sample):
        mesh_id = manifest.get("mesh_id", "?")
        current_label = manifest.get("label", "unknown")
        views = get_views(manifest, args.views)

        if not views:
            print(f"{i+1:>4}  {current_label:<45}  [no views found]")
            errors += 1
            continue

        try:
            new_label, elapsed = label_with_vl(
                views, current_label, args.model, args.ollama_url,
                timeout=args.timeout,
            )
            timings.append(elapsed)
            changed = new_label.lower() != current_label.lower()
            change_marker = " *" if changed else "  "
            cur_display = current_label[:43] + ".." if len(current_label) > 45 else current_label
            new_display = new_label[:43] + ".." if len(new_label) > 45 else new_label
            print(f"{i+1:>4}  {cur_display:<45}  {new_display:<45}  {elapsed:>5.1f}s{change_marker}")

            results.append({
                "mesh_id": mesh_id,
                "original_label": current_label,
                "new_label": new_label,
                "changed": changed,
                "elapsed_s": round(elapsed, 2),
                "render_dir": str(manifest.get("_render_dir", "")),
                "n_faces": manifest.get("n_faces"),
                "n_vertices": manifest.get("n_vertices"),
                "views_used": [str(v) for v in views],
            })

        except Exception as e:
            print(f"{i+1:>4}  {current_label:<45}  [ERROR: {str(e)[:60]}]")
            errors += 1
            results.append({
                "mesh_id": mesh_id,
                "original_label": current_label,
                "new_label": None,
                "changed": False,
                "elapsed_s": None,
                "error": str(e),
            })

    # 5. Summary
    print("\n" + "=" * 110)
    print("SUMMARY")
    print("=" * 110)

    if timings:
        mean_t = sum(timings) / len(timings)
        median_t = sorted(timings)[len(timings) // 2]
        min_t = min(timings)
        max_t = max(timings)

        successful = len(timings)
        changed_count = sum(1 for r in results if r.get("changed"))
        change_pct = 100 * changed_count / successful if successful else 0

        total_items = total_available
        est_total_s = mean_t * total_items
        est_total_h = est_total_s / 3600
        est_total_m = est_total_s / 60

        images_per_call = len(get_views(sample[0], args.views)) if sample else 1

        print(f"  Items tested:        {successful} / {n} (errors: {errors})")
        print(f"  Images per call:     {images_per_call}  (view spec: {args.views})")
        print(f"  Labels changed:      {changed_count} / {successful}  ({change_pct:.0f}%)")
        print()
        print(f"  Time per item:       min={min_t:.1f}s  median={median_t:.1f}s  mean={mean_t:.1f}s  max={max_t:.1f}s")
        print()
        print(f"  Total items available:  {total_items:,}")
        print(f"  Estimated total time:")
        print(f"    @ {mean_t:.1f}s/item  →  {est_total_m:,.0f} min  ({est_total_h:.1f} hours)")
        if images_per_call > 1:
            single_view_est = (mean_t / images_per_call) * total_items / 3600
            print(f"    (single-view estimate: ~{single_view_est:.1f} hours)")

        print()
        print(f"  Cost:  $0.00  (local Ollama — no API fees)")
        print()

        # Label quality sample
        changed_examples = [r for r in results if r.get("changed") and r.get("new_label")]
        if changed_examples:
            print("  Example improvements (* = changed):")
            for r in changed_examples[:8]:
                print(f"    {r['original_label']:<40}  →  {r['new_label']}")
    else:
        print(f"  All {errors} items failed. Check Ollama is running and model is loaded.")

    # 6. Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "config": {
                "model": args.model,
                "ollama_url": args.ollama_url,
                "sample_size": n,
                "views": args.views,
                "seed": args.seed,
            },
            "stats": {
                "total_available": total_available,
                "successful": len(timings),
                "errors": errors,
                "changed_count": sum(1 for r in results if r.get("changed")),
                "mean_time_s": round(sum(timings) / len(timings), 2) if timings else None,
                "median_time_s": round(sorted(timings)[len(timings) // 2], 2) if timings else None,
                "est_total_hours": round((sum(timings) / len(timings)) * total_available / 3600, 1) if timings else None,
            },
            "results": results,
        }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Full results saved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen VL relabeling cost for training cache items."
    )
    parser.add_argument(
        "--sample", type=int, default=20,
        help="Number of items to test (default: 20)",
    )
    parser.add_argument(
        "--model", type=str, default="qwen2.5vl",
        help="Ollama model name (default: qwen2.5vl)",
    )
    parser.add_argument(
        "--views", type=str, default="best",
        choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                 "10", "11", "12", "13", "best", "all"],
        help="Which rendered views to send (default: best = upper-front-right diagonal)",
    )
    parser.add_argument(
        "--ollama-url", type=str, default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Per-item timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save full JSON results",
    )

    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
