#!/usr/bin/env python3
"""Apply a human validation verdict (approve/reject/skip) to a queue item.

This script is intended to be called from Blender (addon) via the external
venv Python, so Blender itself does not need torch installed.

It updates:
- the originating `data/processed/.mesh_cache/*.pt` entry (label/tags and/or quality_weight)
- appends an entry to `<queue_dir>/reviews.jsonl`

Usage:
  ./.venv/bin/python scripts/validator_apply_review.py \
    --queue-dir data/validation_queue \
    --item-id <id> \
    --verdict approve \
    --label "sports car" \
    --tags "car,vehicle,racing"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


def _split_tags(s: str) -> list[str]:
    parts = [p.strip() for p in (s or "").split(",")]
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if not p:
            continue
        low = p.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(p)
    return out


def _load_item_json(queue_dir: Path, item_id: str) -> dict[str, Any]:
    item_path = queue_dir / "items" / f"{item_id}.json"
    if not item_path.exists():
        raise SystemExit(f"Item JSON not found: {item_path}")
    return json.loads(item_path.read_text(encoding="utf-8"))


def _update_cache_item(*, cache_pt: Path, item_index: int, verdict: str, label: str | None, tags: list[str]) -> dict[str, Any]:
    obj = torch.load(cache_pt, map_location="cpu", weights_only=False)
    is_list = isinstance(obj, list)
    items = obj if is_list else [obj]

    if not (0 <= int(item_index) < len(items)):
        raise SystemExit(f"item_index out of range for {cache_pt.name}: {item_index} (len={len(items)})")

    it = items[int(item_index)]
    if not isinstance(it, dict):
        raise SystemExit(f"cache item is not a dict: {cache_pt.name}[{item_index}]")

    def _to_json(val):
        """Convert torch.Tensor or numpy values to JSON-safe types."""
        if isinstance(val, torch.Tensor):
            return val.item() if val.numel() == 1 else val.tolist()
        return val

    before = {
        "label": it.get("label"),
        "user_tags": it.get("user_tags"),
        "quality_weight": _to_json(it.get("quality_weight")),
    }

    ts = int(time.time())
    it["human_validated_at"] = ts
    it["human_verdict"] = verdict

    if verdict == "approve":
        if label is not None and label.strip():
            it["label"] = label.strip()
        if tags is not None:
            it["user_tags"] = tags
        # If it was low-quality by heuristic but human approved, bump weight.
        qw = it.get("quality_weight")
        try:
            qw_f = float(qw.item() if isinstance(qw, torch.Tensor) else qw)
        except Exception:
            qw_f = 0.5
        it["quality_weight"] = float(max(qw_f, 0.9))

    elif verdict == "reject":
        # Make rejected samples contribute zero loss for real_geometry.
        it["quality_weight"] = float(0.0)
        # Still capture tags/label edits if provided (useful for later triage),
        # but do not overwrite label unless user explicitly typed one.
        if label is not None and label.strip():
            it["label_reject_note"] = label.strip()
        if tags:
            it["user_tags"] = tags

    elif verdict == "skip":
        # No dataset mutation, only record the decision.
        pass

    after = {
        "label": it.get("label"),
        "user_tags": it.get("user_tags"),
        "quality_weight": _to_json(it.get("quality_weight")),
    }

    if is_list:
        items[int(item_index)] = it
        torch.save(items, cache_pt)
    else:
        torch.save(it, cache_pt)

    return {"before": before, "after": after}


def _append_review(queue_dir: Path, review: dict[str, Any]) -> None:
    path = queue_dir / "reviews.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(review) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queue-dir", type=Path, required=True)
    p.add_argument("--item-id", type=str, required=True)
    p.add_argument("--verdict", type=str, choices=["approve", "reject", "skip"], required=True)
    p.add_argument("--label", type=str, default="")
    p.add_argument("--tags", type=str, default="")
    args = p.parse_args()

    queue_dir = Path(args.queue_dir)
    item_id = args.item_id

    item = _load_item_json(queue_dir, item_id)

    cache_pt = Path(item.get("cache_pt", ""))
    if not cache_pt.exists():
        raise SystemExit(f"cache_pt not found on disk: {cache_pt}")

    item_index = int(item.get("item_index", -1))

    label = (args.label or "").strip()
    tags = _split_tags(args.tags)

    delta: dict[str, Any] | None = None
    if args.verdict in {"approve", "reject"}:
        delta = _update_cache_item(
            cache_pt=cache_pt,
            item_index=item_index,
            verdict=args.verdict,
            label=label,
            tags=tags,
        )

    review = {
        "ts": int(time.time()),
        "item_id": item_id,
        "verdict": args.verdict,
        "cache_pt": str(cache_pt),
        "item_index": item_index,
        "new_label": label,
        "new_tags": tags,
        "delta": delta,
    }

    _append_review(queue_dir, review)
    print(json.dumps({"ok": True, "review": review}))


if __name__ == "__main__":
    main()
