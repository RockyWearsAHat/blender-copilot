#!/usr/bin/env python3
"""Export a human validation queue from the mesh cache.

Goal:
- Create a folder of simple JSON meshes that can be loaded in Blender
  for rapid approve/reject validation.
- Include enough provenance to write edits back to the originating
  `data/processed/.mesh_cache/*.pt` entries.

Output layout:
  <out>/index.jsonl        # one line per item with metadata + item_json path
  <out>/items/<id>.json    # vertices/faces + provenance + label/tags fields

Typical usage:
  ./.venv/bin/python scripts/export_validation_queue.py \
    --out data/validation_queue \
    --max-items 500 \
    --min-quality-weight 0.4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from processing.mesh_tokenizer import MeshTokenizer


@dataclass(frozen=True)
class QueueItemRef:
    item_id: str
    cache_pt: str
    item_index: int
    label: str
    data_source: str
    quality_weight: float
    token_len: int


def _load_project_config(project_root: Path) -> dict[str, Any]:
    cfg_path = project_root / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        return yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return {}


def _build_tokenizer(project_root: Path) -> MeshTokenizer:
    cfg = _load_project_config(project_root)
    tok = cfg.get("tokenization", {}) if isinstance(cfg, dict) else {}

    vocab_size = int(tok.get("vocab_size", 8192) or 8192)
    coord_range = tok.get("coord_range", (-1.0, 1.0))
    try:
        coord_range = (float(coord_range[0]), float(coord_range[1]))
    except Exception:
        coord_range = (-1.0, 1.0)

    max_faces = int(tok.get("max_faces", 4000) or 4000)
    use_vertex_indexed = bool(tok.get("use_vertex_indexed", False))
    max_vertices = int(tok.get("max_vertices", 4096) or 4096)

    return MeshTokenizer(
        vocab_size=vocab_size,
        coord_range=coord_range,
        max_faces=max_faces,
        use_vertex_indexed=use_vertex_indexed,
        max_vertices=max_vertices,
    )


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if isinstance(v, torch.Tensor):
            return float(v.item())
        return float(v)
    except Exception:
        return float(default)


def _iter_cache_items(pt_path: Path) -> list[dict[str, Any]]:
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, list):
        return [it for it in obj if isinstance(it, dict)]
    if isinstance(obj, dict):
        return [obj]
    return []


def _pick_token_tensor(item: dict[str, Any]) -> torch.Tensor | None:
    t = item.get("mesh_tokens")
    if isinstance(t, torch.Tensor) and t.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
        return t
    # legacy keys
    t = item.get("tokens")
    if isinstance(t, torch.Tensor):
        return t
    return None


def _approx_face_count(tokens: torch.Tensor, tokenizer: MeshTokenizer) -> int:
    # This is only used for filtering. For the exported JSON we use decoded faces.
    n = int(tokens.numel())
    if n <= 2:
        return 0
    if tokenizer.use_vertex_indexed:
        # Vertex-indexed format includes variable-length vertex section.
        return max(0, n // 6)
    return max(0, (n - 2) // 9)


def export_queue(
    *,
    project_root: Path,
    cache_dir: Path,
    out_dir: Path,
    max_items: int,
    seed: int,
    min_faces: int,
    max_faces: int,
    min_quality_weight: float | None,
) -> None:
    rng = random.Random(int(seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    items_dir = out_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _build_tokenizer(project_root)

    pt_files = sorted(Path(cache_dir).glob("*.pt"))
    if not pt_files:
        raise SystemExit(f"No .pt files found in cache dir: {cache_dir}")

    refs: list[QueueItemRef] = []

    for pt in pt_files:
        try:
            cache_items = _iter_cache_items(pt)
        except Exception:
            continue

        for idx, item in enumerate(cache_items):
            tokens = _pick_token_tensor(item)
            if tokens is None:
                continue

            qw = _safe_float(item.get("quality_weight", 0.5), 0.5)
            if min_quality_weight is not None and qw < float(min_quality_weight):
                continue

            fc_approx = _approx_face_count(tokens, tokenizer)
            if fc_approx < int(min_faces) or fc_approx > int(max_faces):
                continue

            label = item.get("label")
            if not isinstance(label, str):
                label = ""

            src = item.get("data_source")
            if not isinstance(src, str):
                src = item.get("source", "unknown")
                if not isinstance(src, str):
                    src = "unknown"

            item_id = f"{pt.stem}_{idx}"
            refs.append(
                QueueItemRef(
                    item_id=item_id,
                    cache_pt=str(pt.resolve()),
                    item_index=int(idx),
                    label=label,
                    data_source=src,
                    quality_weight=qw,
                    token_len=int(tokens.numel()),
                )
            )

    if not refs:
        raise SystemExit("No eligible cache items found (check filters)")

    rng.shuffle(refs)
    refs = refs[: int(max_items)]

    index_path = out_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as index_f:
        for ref in refs:
            pt_path = Path(ref.cache_pt)
            try:
                cache_items = _iter_cache_items(pt_path)
            except Exception:
                continue
            if ref.item_index < 0 or ref.item_index >= len(cache_items):
                continue
            item = cache_items[ref.item_index]
            tokens = _pick_token_tensor(item)
            if tokens is None:
                continue

            # Decode to a simple mesh for Blender. This creates duplicated verts per face,
            # but is perfectly fine for visual validation.
            verts, faces = tokenizer.decode_tokens(tokens.tolist())

            tags = item.get("user_tags")
            if not isinstance(tags, list):
                tags = []

            payload = {
                "item_id": ref.item_id,
                "cache_pt": ref.cache_pt,
                "item_index": ref.item_index,
                "original_label": ref.label,
                "label": ref.label,
                "tags": tags,
                "data_source": ref.data_source,
                "quality_weight": ref.quality_weight,
                "token_len": ref.token_len,
                "vertex_count": len(verts),
                "face_count": len(faces),
                "vertices": verts,
                "faces": faces,
            }

            item_json = items_dir / f"{ref.item_id}.json"
            item_json.write_text(json.dumps(payload), encoding="utf-8")

            index_entry = {
                "item_id": ref.item_id,
                "item_json": str(item_json.resolve()),
                "cache_pt": ref.cache_pt,
                "item_index": ref.item_index,
                "label": ref.label,
                "data_source": ref.data_source,
                "quality_weight": ref.quality_weight,
                "token_len": ref.token_len,
                "vertex_count": len(verts),
                "face_count": len(faces),
            }
            index_f.write(json.dumps(index_entry) + "\n")

    print(f"Wrote queue: {len(refs)} items")
    print(f"  index: {index_path}")
    print(f"  items: {items_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--cache-dir", type=Path, default=Path("data/processed/.mesh_cache"))
    p.add_argument("--out", type=Path, default=Path("data/validation_queue"))
    p.add_argument("--max-items", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-faces", type=int, default=20)
    p.add_argument("--max-faces", type=int, default=4000)
    p.add_argument("--min-quality-weight", type=float, default=None)
    args = p.parse_args()

    export_queue(
        project_root=args.project_root,
        cache_dir=args.cache_dir,
        out_dir=args.out,
        max_items=args.max_items,
        seed=args.seed,
        min_faces=args.min_faces,
        max_faces=args.max_faces,
        min_quality_weight=args.min_quality_weight,
    )


if __name__ == "__main__":
    main()
