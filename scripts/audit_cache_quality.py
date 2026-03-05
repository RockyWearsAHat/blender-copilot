#!/usr/bin/env python3
"""Audit the mesh cache to understand data quality landscape."""

import torch
import os
import sys
import collections

def main():
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", ".mesh_cache")
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith('.pt'))
    # Sample for speed if many files
    if len(files) > 200:
        import random
        random.seed(42)
        sampled = random.sample(files, 200)
        files_to_scan = sampled
        sample_ratio = len(files) / 200
    else:
        files_to_scan = files
        sample_ratio = 1.0
    files = files_to_scan
    
    total_items = 0
    per_file_counts = []
    face_buckets = collections.Counter()
    non_ascii_labels = 0
    scene_comp_count = 0
    dup_within_file = 0
    has_materials = 0
    no_materials = 0
    tiny_items = 0  # <50 faces
    label_samples = []

    for fn in files:
        d = torch.load(os.path.join(cache_dir, fn), map_location='cpu', weights_only=False)
        items = d if isinstance(d, list) else [d]
        per_file_counts.append(len(items))
        seen_hashes = set()
        
        for s in items:
            total_items += 1
            mt = s.get('mesh_tokens', [])
            tl = len(mt) if hasattr(mt, '__len__') else 0
            fc = (tl - 2) // 9 if tl > 2 else 0
            label = s.get('label', '')
            st = s.get('sample_type', 'object')
            sc = s.get('scene_context', {})
            mats = sc.get('materials', [])
            
            if st == 'scene_composition':
                scene_comp_count += 1
            
            if mats:
                has_materials += 1
            else:
                no_materials += 1
            
            if fc < 50:
                tiny_items += 1
            
            # Face buckets
            if fc < 20:
                face_buckets['a_<20'] += 1
            elif fc < 50:
                face_buckets['b_20-50'] += 1
            elif fc < 100:
                face_buckets['c_50-100'] += 1
            elif fc < 500:
                face_buckets['d_100-500'] += 1
            elif fc < 1000:
                face_buckets['e_500-1000'] += 1
            elif fc < 5000:
                face_buckets['f_1000-5000'] += 1
            else:
                face_buckets['g_5000+'] += 1
            
            # Non-ASCII labels
            if any(ord(c) > 127 for c in label):
                non_ascii_labels += 1
            
            # Dup check within file (first 50 tokens)
            if hasattr(mt, 'tolist'):
                tk_hash = hash(tuple(mt[:50].tolist()))
            else:
                tk_hash = hash(tuple(mt[:50]))
            if tk_hash in seen_hashes:
                dup_within_file += 1
            seen_hashes.add(tk_hash)
            
            # Sample some labels for inspection
            if total_items % 100 == 0:
                label_samples.append((fc, label[:80], len(mats)))
    
    print("=" * 60)
    print("CACHE QUALITY AUDIT")
    print("=" * 60)
    print(f"Total files:           {len(files)}")
    print(f"Total items:           {total_items}")
    print(f"Scene compositions:    {scene_comp_count}")
    print(f"Items with materials:  {has_materials}")
    print(f"Items without materials:{no_materials}")
    print(f"Non-ASCII labels:      {non_ascii_labels}")
    print(f"Within-file duplicates:{dup_within_file}")
    print(f"Tiny items (<50 faces):{tiny_items}")
    print()
    print("Per-file item counts:")
    print(f"  min={min(per_file_counts)}, max={max(per_file_counts)}, "
          f"avg={sum(per_file_counts)/len(per_file_counts):.1f}, "
          f"median={sorted(per_file_counts)[len(per_file_counts)//2]}")
    print(f"  Files with >50 items: {sum(1 for c in per_file_counts if c > 50)}")
    print(f"  Files with >20 items: {sum(1 for c in per_file_counts if c > 20)}")
    print(f"  Files with 1 item:    {sum(1 for c in per_file_counts if c == 1)}")
    print()
    print("Face count distribution:")
    for bucket in sorted(face_buckets.keys()):
        label = bucket.split('_', 1)[1]
        count = face_buckets[bucket]
        pct = count / total_items * 100
        bar = '#' * int(pct / 2)
        print(f"  {label:>12s}: {count:6d} ({pct:5.1f}%) {bar}")
    
    print()
    print("Sample labels (every 100th item):")
    for fc, lbl, nm in label_samples[:20]:
        print(f"  {fc:5d} faces, {nm} mats: {lbl}")
    
    # Quality estimate
    good_items = sum(v for k, v in face_buckets.items() if k >= 'd_100-500')
    print()
    print(f"Items with 100+ faces (reasonable quality): {good_items} ({good_items/total_items*100:.1f}%)")
    print(f"Items with <50 faces (likely trash):        {tiny_items} ({tiny_items/total_items*100:.1f}%)")

if __name__ == "__main__":
    main()
