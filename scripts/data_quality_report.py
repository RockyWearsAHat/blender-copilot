#!/usr/bin/env python3
"""Comprehensive data quality report.

Covers every metric from the 2026-02-17 assessment:
  - Source attribution breakdown
  - Label entropy & unique-label count
  - Label frequency distribution (max, P99, P95)
  - Mesh complexity distribution (faces)
  - Genuine vs potential-duplicate ratio
  - Overall quality score

Usage:
    python scripts/data_quality_report.py
    python scripts/data_quality_report.py --top 20        # show top N over-cap labels
    python scripts/data_quality_report.py --cap 100       # warn when label > N samples
    python scripts/data_quality_report.py --fix-in-place  # apply cap + remove junk labels
"""
import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).parent.parent
CACHE_DIR = BASE / "data" / "processed" / ".mesh_cache"
SOURCE_DIRS = {
    "objaverse": BASE / "data" / "processed" / "objaverse",
    "blender_official": BASE / "data" / "processed" / "blender_official",
    "blendswap": BASE / "data" / "processed" / "blendswap",
    "smutbase": BASE / "data" / "processed" / "smutbase",
    "github": BASE / "data" / "processed" / "github",
    "open3dlab": BASE / "data" / "processed" / "open3dlab",
    "youtube": BASE / "data" / "processed" / "youtube",
}
# synthetic samples are generated in-process, not cached → estimate separately


# ── Colour helpers ────────────────────────────────────────────────────

def _c(code, text):
    return f"\033[{code}m{text}\033[0m"

RED    = lambda t: _c("31", t)
YELLOW = lambda t: _c("33", t)
GREEN  = lambda t: _c("32", t)
BOLD   = lambda t: _c("1",  t)
CYAN   = lambda t: _c("36", t)


# ── Metric helpers ───────────────────────────────────────────────────

def _entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def _max_entropy(n_unique: int) -> float:
    return math.log2(n_unique) if n_unique > 1 else 1.0


def _status(current, target_good, target_ok, higher_is_better=True):
    if higher_is_better:
        ok = current >= target_ok
        good = current >= target_good
    else:
        ok = current <= target_ok
        good = current <= target_good
    if good:
        return GREEN("🟢 Good")
    if ok:
        return YELLOW("🟡 Fair")
    return RED("🔴 Poor")


# ── Loader ─────────────────────────────────────────────────────────

def load_cache(verbose=True):
    pt_files = sorted(CACHE_DIR.glob("*.pt"))
    if not pt_files:
        print(RED("ERROR: no .pt files found in"), CACHE_DIR)
        sys.exit(1)
    if verbose:
        print(f"Loading {len(pt_files)} cache files …", end="", flush=True)

    samples = []
    for f in pt_files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
            items = data if isinstance(data, list) else [data]
            for it in items:
                if not isinstance(it, dict):
                    continue
                mt = it.get("mesh_tokens")
                if mt is None or not isinstance(mt, torch.Tensor) or len(mt) < 3:
                    continue
                samples.append({
                    "label":       it.get("label", "").strip(),
                    "data_source": it.get("data_source", "unknown"),
                    "face_count":  max(0, (len(mt) - 2) // 9),
                    "token_hash":  hash(tuple(mt[:60].tolist())),  # fast proxy
                    "quality_weight": float(it.get("quality_weight", 0.5)
                                            if isinstance(it.get("quality_weight"), (int, float, torch.Tensor))
                                            else 0.5),
                })
        except Exception as e:
            if verbose:
                print(f"\n  [warn] {f.name}: {e}")

    if verbose:
        print(f" {len(samples)} samples")
    return samples


# ── Report ──────────────────────────────────────────────────────────

def run_report(args):
    samples = load_cache()

    labels      = [s["label"] for s in samples]
    sources     = [s["data_source"] for s in samples]
    faces       = [s["face_count"] for s in samples]
    hashes      = [s["token_hash"] for s in samples]
    qweights    = [s["quality_weight"] for s in samples]

    label_norm  = [l.lower().strip() for l in labels]
    label_counts: Counter = Counter(label_norm)
    source_counts: Counter = Counter(sources)
    total = len(samples)

    print()
    print(BOLD("=" * 65))
    print(BOLD("  DATA QUALITY REPORT"))
    print(BOLD("=" * 65))
    print(f"  Total samples : {total:,}")
    print(f"  Cache dir     : {CACHE_DIR}")
    print()

    # ── 1. Source attribution ────────────────────────────────────────
    print(BOLD("1. SOURCE ATTRIBUTION"))
    print("   " + "-" * 55)
    unknown_count = source_counts.get("unknown", 0)
    attributed_pct = 100 * (total - unknown_count) / max(total, 1)
    status = _status(attributed_pct, 95, 70)
    print(f"   Attributed samples : {total - unknown_count:,} / {total:,}  "
          f"({attributed_pct:.1f}%)  {status}")
    print()
    if source_counts:
        for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
            bar_len = int(30 * cnt / total)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"   {src:20s} {cnt:7,} ({100*cnt/total:5.1f}%) |{bar}|")
    print()

    # ── 2. Label uniqueness ─────────────────────────────────────────
    print(BOLD("2. LABEL UNIQUENESS"))
    print("   " + "-" * 55)
    n_unique = len(label_counts)
    unique_pct = 100 * n_unique / max(total, 1)
    status_unique = _status(n_unique, 8000, 5000)
    status_pct    = _status(unique_pct, 50, 30)
    print(f"   Unique labels  : {n_unique:,}  {status_unique}")
    print(f"   Unique ratio   : {unique_pct:.1f}%  {status_pct}")

    # Truly unique = labels that appear only once
    singletons = sum(1 for c in label_counts.values() if c == 1)
    print(f"   Singletons     : {singletons:,} ({100*singletons/max(n_unique,1):.1f}% of unique)")
    print()

    # ── 3. Label entropy ─────────────────────────────────────────────
    print(BOLD("3. LABEL ENTROPY"))
    print("   " + "-" * 55)
    h_bits = _entropy_bits(label_counts)
    h_max  = _max_entropy(n_unique)
    h_pct  = 100 * h_bits / h_max if h_max > 0 else 0.0
    status_h = _status(h_pct, 85, 70)
    print(f"   Entropy        : {h_bits:.2f} bits  ({h_pct:.0f}% of max={h_max:.2f})  {status_h}")
    print()

    # ── 4. Label frequency distribution ─────────────────────────────
    print(BOLD("4. LABEL FREQUENCY DISTRIBUTION"))
    print("   " + "-" * 55)
    freqs = sorted(label_counts.values(), reverse=True)
    p95_thresh = np.percentile(freqs, 95) if freqs else 0
    p99_thresh = np.percentile(freqs, 99) if freqs else 0
    max_freq   = freqs[0] if freqs else 0

    over_500   = sum(1 for c in freqs if c > 500)
    over_100   = sum(1 for c in freqs if c > 100)
    samples_over_50 = sum(c for c in label_counts.values() if c > 50)

    cap   = args.cap
    status_max = _status(max_freq, cap, cap * 3, higher_is_better=False)
    print(f"   Max frequency  : {max_freq:,}  {status_max}")
    print(f"   P99 frequency  : {p99_thresh:.0f}")
    print(f"   P95 frequency  : {p95_thresh:.0f}")
    print(f"   Labels > {cap:,}   : {over_100:,}")
    print(f"   Labels > 500   : {over_500:,}")
    print(f"   Samples in labels > 50  : {samples_over_50:,}  "
          f"({100*samples_over_50/max(total,1):.1f}%)")
    print()

    print(f"   Top {args.top} most-frequent labels:")
    for lbl, cnt in label_counts.most_common(args.top):
        bar = "█" * min(40, cnt // max(max_freq // 40, 1))
        flag = RED("  ← OVER CAP") if cnt > cap else ""
        print(f"     [{cnt:6,}] {lbl[:60]}{flag}")
    print()

    # ── 5. Mesh complexity ───────────────────────────────────────────
    print(BOLD("5. MESH COMPLEXITY DISTRIBUTION"))
    print("   " + "-" * 55)
    faces_arr = np.array(faces)
    buckets = [(0, 50), (50, 200), (200, 500), (500, 1000),
               (1000, 2000), (2000, 4000), (4000, 8001)]
    for lo, hi in buckets:
        cnt = int(np.sum((faces_arr >= lo) & (faces_arr < hi)))
        pct = 100 * cnt / max(total, 1)
        bar = "█" * int(pct / 2)
        print(f"   {lo:5}–{hi-1:<5} faces : {cnt:7,}  ({pct:5.1f}%)  {bar}")

    mean_f = float(faces_arr.mean()) if len(faces_arr) else 0
    median_f = float(np.median(faces_arr)) if len(faces_arr) else 0
    print(f"\n   Mean faces  : {mean_f:.0f}   Median : {median_f:.0f}")
    complex_count = int(np.sum(faces_arr >= 1000))
    complex_pct   = 100 * complex_count / max(total, 1)
    status_complex = _status(complex_pct, 20, 10)
    print(f"   ≥1000 faces : {complex_count:,}  ({complex_pct:.1f}%)  {status_complex}")
    print()

    # ── 6. Duplicate detection ───────────────────────────────────────
    print(BOLD("6. DEDUPLICATION"))
    print("   " + "-" * 55)
    unique_hashes = len(set(hashes))
    dup_count  = total - unique_hashes
    dup_pct    = 100 * dup_count / max(total, 1)
    status_dup = _status(dup_pct, 1, 5, higher_is_better=False)
    print(f"   Estimated duplicates : {dup_count:,} ({dup_pct:.1f}%)  {status_dup}")
    print()

    # ── 7. Quality weight distribution ──────────────────────────────
    print(BOLD("7. QUALITY WEIGHT DISTRIBUTION"))
    print("   " + "-" * 55)
    qw = np.array(qweights)
    print(f"   Mean  : {qw.mean():.3f}   Median : {np.median(qw):.3f}")
    print(f"   Stdev : {qw.std():.3f}   Min : {qw.min():.3f}   Max : {qw.max():.3f}")
    low_q = int(np.sum(qw < 0.4))
    print(f"   Low quality (< 0.4) : {low_q:,}  ({100*low_q/max(total,1):.1f}%)")
    print()

    # ── 8. Summary score ────────────────────────────────────────────
    print(BOLD("8. OVERALL QUALITY SCORE"))
    print("   " + "-" * 55)
    score_attrs   = int(attributed_pct    >= 95) * 25
    score_entropy = int(h_pct             >= 85) * 25
    score_unique  = int(n_unique          >= 8000) * 25
    score_cap     = int(max_freq          <= cap)  * 25
    total_score   = score_attrs + score_entropy + score_unique + score_cap
    col = GREEN if total_score >= 75 else (YELLOW if total_score >= 50 else RED)
    print(f"   Source attribution : {'25/25' if score_attrs  else ' 0/25'}")
    print(f"   Label entropy      : {'25/25' if score_entropy else ' 0/25'}")
    print(f"   Unique labels      : {'25/25' if score_unique  else ' 0/25'}")
    print(f"   Label cap OK       : {'25/25' if score_cap     else ' 0/25'}")
    print(f"\n   {col(f'TOTAL: {total_score}/100')}")
    print()

    # ── Fix-in-place mode ────────────────────────────────────────────
    if args.fix_in_place:
        _apply_fixes(cap)

    return total_score


def _apply_fixes(cap: int):
    """Remove samples with empty labels and cap over-frequent labels in-place."""
    print(BOLD("=" * 65))
    print(BOLD("  APPLYING IN-PLACE FIXES"))
    print(BOLD("=" * 65))

    # Count globally first
    from collections import Counter
    label_counts: Counter = Counter()
    all_files = sorted(CACHE_DIR.glob("*.pt"))
    for f in all_files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
            items = data if isinstance(data, list) else [data]
            for it in items:
                lbl = it.get("label", "").strip().lower()
                if lbl:
                    label_counts[lbl] += 1
        except Exception:
            continue

    over_cap = {lbl for lbl, cnt in label_counts.items() if cnt > cap}
    print(f"  Labels over cap ({cap}): {len(over_cap)}")

    seen: Counter = Counter()
    removed_cap = removed_empty = 0

    for f in all_files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
            items = data if isinstance(data, list) else [data]
        except Exception:
            continue

        orig_len = len(items)
        kept = []
        for it in items:
            lbl = it.get("label", "").strip()
            if not lbl or len(lbl) < 2:
                removed_empty += 1
                continue
            lbl_lower = lbl.lower()
            if lbl_lower in over_cap:
                if seen[lbl_lower] < cap:
                    kept.append(it)
                    seen[lbl_lower] += 1
                else:
                    removed_cap += 1
            else:
                kept.append(it)

        if len(kept) != orig_len:
            if kept:
                torch.save(kept, f)
            else:
                f.unlink()

    print(f"  Empty labels removed : {removed_empty:,}")
    print(f"  Label-cap removed    : {removed_cap:,}")
    remaining = sum(len(torch.load(f, map_location="cpu", weights_only=False)
                        if isinstance(torch.load(f, map_location="cpu", weights_only=False), list)
                        else [torch.load(f, map_location="cpu", weights_only=False)])
                    for f in sorted(CACHE_DIR.glob("*.pt")))
    print(f"  Remaining samples    : {remaining:,}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=15,
                   help="Number of top-frequency labels to display")
    p.add_argument("--cap", type=int, default=100,
                   help="Warn / cap threshold for label frequency (default: 100)")
    p.add_argument("--fix-in-place", action="store_true",
                   help="Apply cap and empty-label removal directly to cache")
    args = p.parse_args()

    score = run_report(args)
    sys.exit(0 if score >= 75 else 1)


if __name__ == "__main__":
    main()
