#!/usr/bin/env python3
"""Categorize all labels in the restructured cache."""
import torch, os, re
from collections import Counter
from pathlib import Path

CACHE = Path("data/processed/.mesh_cache")

COMMON_ENGLISH = {
    'the','a','an','of','in','on','for','with','and','or','is',
    'small','large','big','old','new','dark','light','red','blue','green','white','black',
    'metal','wood','glass','stone','plastic','concrete','steel','iron','gold','silver',
    'wall','floor','roof','door','window','table','chair','lamp','car','tree','house',
    'building','tower','bridge','road','street','fence','pipe','wire','cable','bolt',
    'screw','nail','box','cube','cylinder','sphere','cone','ring','plate','panel',
    'frame','beam','bar','rod','tube','wheel','gear','handle','knob','lever',
    'button','switch','screen','shelf','cabinet','drawer','desk','bed','sofa',
    'helmet','sword','shield','armor','weapon','gun','rifle','pistol',
    'top','bottom','left','right','front','back','side','inner','outer',
    'tall','short','flat','round','square','thin','thick','wide','narrow',
    'modern','vintage','rustic','industrial','medieval','futuristic',
    'broken','damaged','dirty','clean','rough','smooth','shiny','matte',
    'low','high','poly','detail','simple','complex','basic','ornate',
}

ASCII_FOREIGN = {
    'puertas','cocina','cemento','balcao','kolam','jalan','atap','rumah','pohon',
    'batu','tanah','pintu','dinding','lantai','meja','silla','puerta','ventana',
    'techo','suelo','pared','cocina','calle','fleur','asa','cueca','atrax',
    'ryukin','fintail','finback','giallo','verde','rojo','azul','negro','blanco',
    'marron','gris','rosa','naranja','morado','amarillo','cemento',
}

def main():
    files = sorted(CACHE.glob("*.pt"))
    categories = Counter()
    examples: dict[str, list] = {}
    total = 0

    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        items = d if isinstance(d, list) else [d]
        for s in items:
            label = s.get("label", "")
            total += 1
            clean = label.strip().lower()
            words = [w for w in re.split(r'[\s,._-]+', clean) if w]

            if re.search(r'\.\d{3}', label):
                cat = 'blender_suffix'
            elif re.search(r'^(sm|wv|trv|sd|b2)\s', clean):
                cat = 'asset_prefix'
            elif re.search(r'\b(vagina|penis|breast|nipple|genital|nsfw|nude|naked)\b', clean):
                cat = 'nsfw'
            elif set(words) & ASCII_FOREIGN:
                foreign_hits = set(words) & ASCII_FOREIGN
                if len(foreign_hits) / max(1, len(words)) >= 0.3:
                    cat = 'ascii_foreign'
                else:
                    cat = 'mixed_foreign'
            elif re.search(r'(mtl|mat)\w{2,}', clean.replace(' ', '')):
                cat = 'material_name'
            elif len(words) >= 2 and any(w in COMMON_ENGLISH for w in words):
                cat = 'decent_english'
            elif len(words) >= 2:
                cat = 'ambiguous_multi'
            elif len(words) == 1 and len(words[0]) > 3:
                cat = 'single_word'
            else:
                cat = 'other'

            categories[cat] += 1
            if cat not in examples:
                examples[cat] = []
            if len(examples[cat]) < 8:
                examples[cat].append(label[:90])

    print(f"Total: {total}")
    print()
    for cat, cnt in categories.most_common():
        pct = 100 * cnt / total
        print(f"  {cat:25s}: {cnt:5d} ({pct:.1f}%)")
        for ex in examples.get(cat, []):
            print(f"    eg: {ex}")
        print()


if __name__ == "__main__":
    main()
