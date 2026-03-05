#!/usr/bin/env python3
"""Fix face_material_indices z-order in existing cache items.

Problem: rebuild_cache.py previously stored face_material_indices in original
face order, but mesh_tokens are z-ordered (Morton sort). The validator decodes
tokens into z-ordered faces, so FMI doesn't align — materials can't be assigned.

Fix: For each cache item with raw_vertices + raw_faces + face_material_indices,
recompute the z-order permutation and reorder FMI to match.

Usage:
    python scripts/fix_fmi_zorder.py --dry-run     # preview
    python scripts/fix_fmi_zorder.py --apply        # fix in-place
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from processing.mesh_tokenizer import MeshTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_DIR = BASE / "data" / "processed" / ".mesh_cache"


def _compute_z_order(verts_np: np.ndarray, faces_list: list) -> np.ndarray:
    """Compute Morton z-order permutation for faces, matching tokenizer logic."""
    centers = []
    for face in faces_list:
        valid = [fi for fi in face if 0 <= fi < len(verts_np)]
        if valid:
            center = verts_np[valid].mean(axis=0)
        else:
            center = np.zeros(3)
        centers.append(center)

    centers = np.array(centers, dtype=np.float64)
    if len(centers) == 0:
        return np.array([], dtype=int)

    c_min = centers.min(axis=0)
    c_max = centers.max(axis=0)
    c_range = c_max - c_min
    c_range[c_range < 1e-6] = 1.0
    norm_c = ((centers - c_min) / c_range * 1023).astype(int)
    norm_c = np.clip(norm_c, 0, 1023)

    morton_codes = np.array([
        MeshTokenizer._morton_encode_3d(int(c[0]), int(c[1]), int(c[2]))
        for c in norm_c
    ])
    return np.argsort(morton_codes)


def fix_fmi(dry_run: bool = True):
    cache_files = sorted(CACHE_DIR.glob("*.pt"))
    logger.info(f"Found {len(cache_files)} cache files")

    stats = {
        "files_checked": 0,
        "items_checked": 0,
        "items_with_fmi": 0,
        "items_already_ok": 0,
        "items_fixed": 0,
        "items_no_raw": 0,
        "items_error": 0,
        "files_modified": 0,
    }

    for fi, cache_path in enumerate(cache_files):
        if fi % 100 == 0 and fi > 0:
            logger.info(f"  Processed {fi}/{len(cache_files)} files, {stats['items_fixed']} items fixed")

        try:
            items = torch.load(cache_path, weights_only=False)
        except Exception:
            continue

        stats["files_checked"] += 1
        modified = False

        for it in items:
            stats["items_checked"] += 1
            sc = it.get("scene_context")
            if not isinstance(sc, dict):
                continue

            fmi = sc.get("face_material_indices")
            if not fmi:
                continue

            stats["items_with_fmi"] += 1

            # Get mesh_tokens to compute decoded face count
            mt = it.get("mesh_tokens")
            if mt is None:
                continue
            tl = len(mt) if hasattr(mt, "__len__") else 0
            decoded_faces = (tl - 2) // 9 if tl > 2 else 0

            if len(fmi) == decoded_faces:
                stats["items_already_ok"] += 1
                continue

            # Need raw_vertices and raw_faces to recompute z-order
            raw_verts = it.get("raw_vertices")
            raw_faces = it.get("raw_faces")
            if raw_verts is None or raw_faces is None:
                stats["items_no_raw"] += 1
                continue

            try:
                if isinstance(raw_verts, torch.Tensor):
                    verts_np = raw_verts.numpy().astype(np.float64)
                else:
                    verts_np = np.array(raw_verts, dtype=np.float64)

                if isinstance(raw_faces, torch.Tensor):
                    faces_np = raw_faces.numpy().astype(int)
                else:
                    faces_np = np.array(raw_faces, dtype=int)

                faces_list = faces_np.tolist()
                z_order = _compute_z_order(verts_np, faces_list)

                # Reorder FMI to z-order, capped to decoded face count
                # (z-order[:decoded_faces] gives the indices of faces that survived truncation)
                reordered = []
                for z_idx in z_order[:decoded_faces]:
                    if z_idx < len(fmi):
                        reordered.append(fmi[z_idx])
                    else:
                        reordered.append(0)

                sc["face_material_indices"] = reordered
                stats["items_fixed"] += 1
                modified = True

            except Exception as e:
                stats["items_error"] += 1

        if modified and not dry_run:
            torch.save(items, cache_path)
            stats["files_modified"] += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"FMI Z-ORDER FIX {'PREVIEW' if dry_run else 'COMPLETE'}")
    logger.info(f"{'='*60}")
    logger.info(f"Files checked:     {stats['files_checked']:,}")
    logger.info(f"Items checked:     {stats['items_checked']:,}")
    logger.info(f"Items with FMI:    {stats['items_with_fmi']:,}")
    logger.info(f"Already correct:   {stats['items_already_ok']:,}")
    logger.info(f"Fixed (reordered): {stats['items_fixed']:,}")
    logger.info(f"No raw data:       {stats['items_no_raw']:,}")
    logger.info(f"Errors:            {stats['items_error']:,}")
    if not dry_run:
        logger.info(f"Files modified:    {stats['files_modified']:,}")


def main():
    parser = argparse.ArgumentParser(description="Fix FMI z-order in cache")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without modifying files")
    parser.add_argument("--apply", action="store_true",
                        help="Apply fixes in-place")
    args = parser.parse_args()

    if not args.apply and not args.dry_run:
        logger.info("Neither --dry-run nor --apply specified. Running as dry-run.")
        args.dry_run = True
    if args.apply:
        args.dry_run = False

    fix_fmi(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
