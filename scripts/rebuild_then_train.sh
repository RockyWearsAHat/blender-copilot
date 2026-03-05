#!/usr/bin/env zsh
set -euo pipefail

cd /Users/alexwaldmann/blenderPlugins/blender-copilot

ts=$(date +%Y%m%d_%H%M%S)
rebuild_log="logs/rebuild_${ts}.log"
train_log="logs/train_${ts}.log"

echo "[pipeline] rebuild log: ${rebuild_log}"
echo "[pipeline] train log:   ${train_log}"

/Users/alexwaldmann/blenderPlugins/blender-copilot/.venv/bin/python scripts/rebuild_cache.py --force-rebuild --max-per-label 100 > "${rebuild_log}" 2>&1

/Users/alexwaldmann/blenderPlugins/blender-copilot/.venv/bin/python run.py --config config.unified_m3_semantic_bootstrap.yaml train --name unified_semantic_bootstrap --resume latest > "${train_log}" 2>&1
