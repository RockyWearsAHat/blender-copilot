#!/usr/bin/env python3
"""Quality report for policy collapse/build traces.

This is the policy-side analogue of scripts/data_quality_report.py.
It answers: "Is my trace dataset big, diverse, and usable?" before training.

Usage:
  python scripts/policy_trace_quality_report.py --trace-root data/datasets/collapse_traces
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


_DEFAULT_KEYWORDS = "terrain,hill,hills,landscape,grass,grassy,mountain,valley,ground,forest"  # noqa: E501


def _entropy_bits(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def _load_mesh_json(dir_path: Path) -> dict:
    p = dir_path / "mesh.json"
    if not p.exists():
        return {}
    try:
        raw = p.read_text(encoding="utf-8")
        txt = raw.strip()
        if txt.startswith("```"):
            lines = txt.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            while lines and not lines[0].strip():
                lines = lines[1:]
            if lines and lines[-1].rstrip().endswith("```"):
                lines = lines[:-1]
            txt = "\n".join(lines).strip()
        return json.loads(txt)
    except Exception:
        return {}


def _trace_len(trace_path: Path) -> int:
    n = 0
    try:
        with trace_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
    except Exception:
        return 0
    return n


def _iter_trace_jsonl(trace_path: Path):
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                yield None
                continue
            if not isinstance(obj, dict):
                yield None
                continue
            yield obj


def _norm_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _keyword_hits(mesh_json: dict, keywords: list[str]) -> set[str]:
    hits: set[str] = set()
    label = _norm_text(str(mesh_json.get("label") or ""))
    texts: list[str] = [label]
    pv = mesh_json.get("prompt_variants")
    if isinstance(pv, list):
        texts.extend(_norm_text(str(x)) for x in pv if str(x).strip())
    blob = "\n".join(texts)
    for kw in keywords:
        if kw and kw in blob:
            hits.add(kw)
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="Policy trace dataset quality report")
    ap.add_argument("--trace-root", type=Path, default=Path("data/datasets/collapse_traces"))
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument(
        "--keywords",
        type=str,
        default=_DEFAULT_KEYWORDS,
        help="Comma-separated keywords to check in labels/prompt_variants",
    )
    ap.add_argument(
        "--validate-steps",
        action="store_true",
        help="Validate step JSON structure + monotonic collapse properties",
    )
    args = ap.parse_args()

    root = Path(args.trace_root)
    if not root.exists():
        print(f"ERROR: trace root not found: {root}")
        return 1

    trace_paths = sorted(root.glob("*/trace.jsonl"))
    if not trace_paths:
        print(f"ERROR: no trace.jsonl files under: {root}")
        return 1

    lens: list[int] = []
    labels: list[str] = []
    prompt_variants_total = 0
    empty_label = 0

    keywords = [k.strip().lower() for k in str(args.keywords).split(",") if k.strip()]
    keyword_trace_hits: Counter[str] = Counter()
    traces_with_any_keyword = 0

    # Step-level validation.
    bad_json_lines = 0
    bad_step_records = 0
    op_counts: Counter[str] = Counter()
    monotonic_vertex_ok = 0
    monotonic_vertex_total = 0
    modifier_nonincreasing_ok = 0
    modifier_nonincreasing_total = 0
    vertex_delta_abs: list[int] = []

    for tp in trace_paths:
        trace_len = _trace_len(tp)
        if trace_len <= 0:
            continue
        lens.append(trace_len)

        mj = _load_mesh_json(tp.parent)
        label = str(mj.get("label") or "").strip().lower()
        if not label:
            empty_label += 1
        labels.append(label)

        pv = mj.get("prompt_variants")
        if isinstance(pv, list):
            prompt_variants_total += sum(1 for x in pv if str(x).strip())

        if keywords:
            hits = _keyword_hits(mj, keywords)
            if hits:
                traces_with_any_keyword += 1
                for h in hits:
                    keyword_trace_hits[h] += 1

        if bool(args.validate_steps):
            for rec in _iter_trace_jsonl(tp):
                if rec is None:
                    bad_json_lines += 1
                    continue
                op = str(rec.get("op") or "").strip().lower()
                if op:
                    op_counts[op] += 1
                pre = rec.get("pre")
                post = rec.get("post")
                if not isinstance(pre, dict) or not isinstance(post, dict):
                    bad_step_records += 1
                    continue

                # Monotonic collapse sanity: post should not exceed pre.
                try:
                    pre_v = int(pre.get("vertex_count", 0))
                    post_v = int(post.get("vertex_count", 0))
                except Exception:
                    bad_step_records += 1
                    continue
                monotonic_vertex_total += 1
                if post_v <= pre_v:
                    monotonic_vertex_ok += 1
                vertex_delta_abs.append(abs(pre_v - post_v))

                try:
                    pre_m = int(pre.get("modifier_count", 0))
                    post_m = int(post.get("modifier_count", 0))
                except Exception:
                    pre_m = None
                    post_m = None
                if pre_m is not None and post_m is not None:
                    modifier_nonincreasing_total += 1
                    if post_m <= pre_m:
                        modifier_nonincreasing_ok += 1

    if not lens:
        print("ERROR: all traces unreadable/empty")
        return 1

    label_counts = Counter([label for label in labels if label])
    h = _entropy_bits(label_counts)
    total = len(lens)
    avg_len = sum(lens) / max(1, total)
    p50 = sorted(lens)[total // 2]
    p90 = sorted(lens)[int(0.9 * (total - 1))]

    print("=" * 72)
    print("POLICY TRACE QUALITY REPORT")
    print("=" * 72)
    print(f"Trace root          : {root}")
    print(f"Traces (non-empty)  : {total:,} / {len(trace_paths):,}")
    print(f"Steps per trace     : mean={avg_len:.1f}  p50={p50}  p90={p90}")
    print()

    print("Labels")
    print("------")
    print(f"Non-empty labels    : {sum(1 for label in labels if label):,}")
    print(f"Empty labels        : {empty_label:,}")
    print(f"Unique labels       : {len(label_counts):,}")
    print(f"Label entropy       : {h:.2f} bits")
    if label_counts:
        print(f"Top {int(args.top)} labels:")
        for lbl, cnt in label_counts.most_common(int(args.top)):
            print(f"  [{cnt:6,}] {lbl[:70]}")
    print()

    if prompt_variants_total > 0:
        print("Prompt variants")
        print("---------------")
        print(f"Total stored variants: {prompt_variants_total:,}")
        print(f"Avg variants/trace   : {prompt_variants_total / max(1, total):.2f}")

    if keywords:
        print("\nKeyword coverage")
        print("----------------")
        print(f"Keywords checked     : {', '.join(keywords)}")
        print(f"Traces w/ any keyword: {traces_with_any_keyword:,} / {total:,}")
        if keyword_trace_hits:
            for kw, cnt in keyword_trace_hits.most_common(int(args.top)):
                print(f"  [{cnt:6,}] {kw}")

    if bool(args.validate_steps):
        print("\nStep validation")
        print("---------------")
        print(f"Bad JSON lines       : {bad_json_lines:,}")
        print(f"Bad step records     : {bad_step_records:,}")
        if op_counts:
            print("Op histogram:")
            for op, cnt in op_counts.most_common(int(args.top)):
                print(f"  [{cnt:6,}] {op}")
        if monotonic_vertex_total > 0:
            pct = 100.0 * float(monotonic_vertex_ok) / float(monotonic_vertex_total)
            print(f"Vertex monotonic ok  : {monotonic_vertex_ok:,}/{monotonic_vertex_total:,} ({pct:.1f}%)")
        if modifier_nonincreasing_total > 0:
            pct = 100.0 * float(modifier_nonincreasing_ok) / float(modifier_nonincreasing_total)
            print(
                f"Modifier non-incr ok : {modifier_nonincreasing_ok:,}/{modifier_nonincreasing_total:,} ({pct:.1f}%)"
            )
        if vertex_delta_abs:
            vds = sorted(vertex_delta_abs)
            p50d = vds[len(vds) // 2]
            p90d = vds[int(0.9 * (len(vds) - 1))]
            print(f"|Δverts| per step    : mean={sum(vds)/len(vds):.1f}  p50={p50d}  p90={p90d}")

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
