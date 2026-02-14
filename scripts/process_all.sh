#!/usr/bin/env bash
# process_all.sh — Full data processing pipeline
#
# Steps:
#   1. Extract mesh/material/modifier data from .blend files (Blender headless)
#   2. Quality filter (remove corrupt, too simple/complex, duplicates)
#   3. Auto-label (generate text descriptions)
#   4. Build datasets (geometry, materials, modifiers JSONL files)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════"
echo " BlenderModelTraining — Data Processing"
echo "═══════════════════════════════════════════"

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Find Blender
BLENDER="${BLENDER:-}"
if [ -z "$BLENDER" ]; then
    # Try common locations
    if command -v blender &>/dev/null; then
        BLENDER="blender"
    elif [ -f "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
        BLENDER="/Applications/Blender.app/Contents/MacOS/Blender"
    elif [ -f "/snap/bin/blender" ]; then
        BLENDER="/snap/bin/blender"
    else
        echo "❌ Blender not found. Set BLENDER=/path/to/blender"
        exit 1
    fi
fi
echo "Using Blender: $BLENDER"

# Create directories
mkdir -p data/extracted
mkdir -p data/filtered
mkdir -p data/labeled
mkdir -p data/datasets

# ─── Step 1: Extract from .blend files ────────────────────────
echo ""
echo "━━━ [1/4] Extracting data from .blend files ━━━"

BLEND_COUNT=0
for blend_dir in data/raw/blendswap data/raw/github; do
    if [ ! -d "$blend_dir" ]; then
        continue
    fi

    find "$blend_dir" -name "*.blend" -type f | while read blend_file; do
        filename=$(basename "$blend_file" .blend)
        output_file="data/extracted/${filename}.json"

        if [ -f "$output_file" ]; then
            echo "  ⏭  Already extracted: $filename"
            continue
        fi

        echo "  📦 Extracting: $blend_file"
        "$BLENDER" --background --python processing/blend_extractor.py -- \
            --input "$blend_file" \
            --output "$output_file" \
            2>/dev/null || echo "  ⚠️  Failed: $blend_file"

        BLEND_COUNT=$((BLEND_COUNT + 1))
    done
done

EXTRACTED=$(find data/extracted -name "*.json" 2>/dev/null | wc -l)
echo "  Extracted: $EXTRACTED files"

# ─── Step 2: Quality Filter ──────────────────────────────────
echo ""
echo "━━━ [2/4] Quality filtering ━━━"

python -m processing.quality_filter \
    --input data/extracted \
    --output data/filtered \
    --config config.yaml

FILTERED=$(find data/filtered -name "*.json" 2>/dev/null | wc -l)
echo "  Passed filter: $FILTERED files"

# ─── Step 3: Auto-label ──────────────────────────────────────
echo ""
echo "━━━ [3/4] Auto-labeling ━━━"

python -m processing.labeler \
    --input data/filtered \
    --output data/labeled \
    --config config.yaml

# ─── Step 4: Build datasets ──────────────────────────────────
echo ""
echo "━━━ [4/4] Building training datasets ━━━"

python -m processing.dataset_builder \
    --input data/labeled \
    --output data/datasets \
    --config config.yaml

# Summary
echo ""
echo "═══════════════════════════════════════════"
echo " Processing Complete"
echo "═══════════════════════════════════════════"

for f in data/datasets/*.jsonl; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  $(basename "$f"): $lines samples"
    fi
done

echo ""
echo " Next: python -m training.train_geometry \\"
echo "         --config config.yaml \\"
echo "         --data data/datasets/geometry_train.jsonl \\"
echo "         --val_data data/datasets/geometry_val.jsonl \\"
echo "         --output checkpoints/geometry/"
echo "═══════════════════════════════════════════"
