#!/usr/bin/env python3
"""Debug M1911 label generation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from processing.labeler_smart import generate_smart_label, _is_low_quality_label, _clean_label_final

# Simulate M1911 inputs
label = generate_smart_label(
    obj_name="Object",
    material_names=["lambert1"],
    modifier_types=[],
    num_faces=3857,
    num_verts=2929,
    bbox_aspect=(0.2, 0.05, 0.15),
    file_label="weapon: colt m1911 ww2, 45, 3dcoat, norway",
    metadata_name="Colt M1911",
    metadata_desc="Minor flaws with the normal map, my most ambitious gun model so far.",
    metadata_tags=["ww2", "45", "3dcoat", "norway", "m1911", "pistol", "caliber", "oslo"],
    metadata_categories="weapons-military",
    sibling_names=[],
)
print(f"Final label: {label!r}")
print()

# Debug sub-checks
cleaned = _clean_label_final("Colt M1911")
print(f"_clean_label_final('Colt M1911') = {cleaned!r}")
print(f"_is_low_quality_label({cleaned!r}) = {_is_low_quality_label(cleaned)}")
alnum = ''.join(c for c in cleaned.lower() if c.isalnum())
digits = sum(c.isdigit() for c in alnum)
print(f"  alnum={alnum!r}  digits={digits}  ratio={digits/max(len(alnum),1):.3f}")

# Also check the file_label
cleaned_fl = _clean_label_final("weapon: colt m1911 ww2, 45, 3dcoat, norway")
print(f"\n_clean_label_final(file_label) = {cleaned_fl!r}")
print(f"_is_low_quality_label({cleaned_fl!r}) = {_is_low_quality_label(cleaned_fl)}")
print(f"  commas: {cleaned_fl.count(',')}")
