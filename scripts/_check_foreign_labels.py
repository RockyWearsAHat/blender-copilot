"""Temporary script: check for non-English labels in training data."""
import json, os, glob, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

foreign_words = [
    'bois','verre','pierre','mur','holz','stein','glas','acier',
    'porta','pavimento','vetro','legno','czerwony','szklo','madera',
    'piedra','pneu','gorille','tubi','capacete','farol','lakier',
    'felga','oggetti','fenster','stuhl','tisch','lampe','sedia',
    'tavolo','blaetter','blatt','baum','haus','tuer','dach','wand',
    'boden','haar','haut','kopf','augen','messing','buche',
]

non_english_obj = []
non_english_mat = []

for src in ['blendswap', 'smutbase', 'open3dlab', 'github', 'blender_official']:
    jdir = BASE / 'data' / 'processed' / src
    if not jdir.exists():
        continue
    for f in sorted(jdir.glob('*.json')):
        if f.stat().st_size > 10_000_000:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for o in d.get('objects', []):
            name = (o.get('name', '') or '').lower()
            for w in foreign_words:
                if w in name:
                    non_english_obj.append(f'[{src}] {f.name} | obj: {o.get("name","")}')
                    break
            for m in o.get('materials', []):
                mname = (m.get('name', '') or '').lower()
                for w in foreign_words:
                    if w in mname:
                        non_english_mat.append(f'[{src}] {f.name} | mat: {m.get("name","")} | obj: {o.get("name","")}')
                        break

print("=== Non-English Object Names ===")
for e in non_english_obj[:20]:
    print(f'  {e}')
print(f"  Total: {len(non_english_obj)}")
print()
print("=== Non-English Material Names ===")
for e in non_english_mat[:20]:
    print(f'  {e}')
print(f"  Total: {len(non_english_mat)}")
