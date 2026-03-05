#!/bin/bash
cd /Users/alexwaldmann/blenderPlugins/blender-copilot
source .venv/bin/activate

echo "=== Starting Objaverse download (5000 models) ==="
python scripts/mass_download.py --source objaverse --max 5000 --no-extract 2>&1
echo "=== Objaverse complete ==="

echo "=== Starting BlendSwap download (200 models) ==="
python scripts/mass_download.py --source blendswap --max 200 --no-extract 2>&1
echo "=== BlendSwap complete ==="

echo "=== Starting GitHub download (200 repos) ==="
python scripts/mass_download.py --source github --max 200 --no-extract 2>&1
echo "=== GitHub complete ==="

echo "=== Starting SmutBase download (100 models) ==="
python scripts/mass_download.py --source smutbase --max 100 --no-extract 2>&1
echo "=== SmutBase complete ==="

echo "=== ALL DOWNLOADS COMPLETE ==="
