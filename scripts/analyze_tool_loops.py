#!/usr/bin/env python3
"""Summarize tool-loop reliability KPIs from addon chat logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STAT_RE = re.compile(
    r"Tool loop stats: total_calls=(\d+), repetitive_calls=(\d+), dead_end_rounds=(\d+)"
)


def iter_chat_logs(chat_dir: Path):
    for path in sorted(chat_dir.glob("chat_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            yield path, payload
        except Exception:
            continue


def extract_stats(payload: dict) -> list[dict]:
    out = []
    for msg in payload.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", ""))
        m = STAT_RE.search(content)
        if not m:
            continue
        out.append({
            "total_calls": int(m.group(1)),
            "repetitive_calls": int(m.group(2)),
            "dead_end_rounds": int(m.group(3)),
        })
    return out


def summarize(stats_rows: list[dict]) -> dict:
    if not stats_rows:
        return {
            "runs": 0,
            "mean_total_calls": 0.0,
            "mean_repetitive_calls": 0.0,
            "mean_dead_end_rounds": 0.0,
            "success_proxy_rate": 0.0,
        }

    runs = len(stats_rows)
    mean_total = sum(r["total_calls"] for r in stats_rows) / runs
    mean_repeat = sum(r["repetitive_calls"] for r in stats_rows) / runs
    mean_dead = sum(r["dead_end_rounds"] for r in stats_rows) / runs

    success_proxy = 0
    for r in stats_rows:
        if r["dead_end_rounds"] <= 1 and r["repetitive_calls"] <= 2:
            success_proxy += 1

    return {
        "runs": runs,
        "mean_total_calls": mean_total,
        "mean_repetitive_calls": mean_repeat,
        "mean_dead_end_rounds": mean_dead,
        "success_proxy_rate": success_proxy / runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Blender Copilot tool-loop reliability")
    parser.add_argument("--chat-dir", default="chat_logs", help="Directory containing chat_*.json logs")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    chat_dir = Path(args.chat_dir)
    if not chat_dir.exists():
        print(f"chat directory not found: {chat_dir}")
        return 1

    rows = []
    for _, payload in iter_chat_logs(chat_dir):
        rows.extend(extract_stats(payload))

    summary = summarize(rows)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("Tool-loop reliability summary")
        print(f"  runs: {summary['runs']}")
        print(f"  mean_total_calls: {summary['mean_total_calls']:.2f}")
        print(f"  mean_repetitive_calls: {summary['mean_repetitive_calls']:.2f}")
        print(f"  mean_dead_end_rounds: {summary['mean_dead_end_rounds']:.2f}")
        print(f"  success_proxy_rate: {summary['success_proxy_rate']:.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
