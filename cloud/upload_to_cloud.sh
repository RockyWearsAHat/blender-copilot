#!/bin/bash
# ============================================================================
# Upload project + data to cloud GPU instance
# ============================================================================
# Run this FROM YOUR LOCAL MAC to push code and data to the cloud.
#
# Usage:
#   bash cloud/upload_to_cloud.sh user@gpu-instance-ip
#   bash cloud/upload_to_cloud.sh user@gpu-instance-ip --data-only
#   bash cloud/upload_to_cloud.sh user@gpu-instance-ip --code-only
# ============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash cloud/upload_to_cloud.sh user@host [--data-only|--code-only]"
    exit 1
fi

REMOTE="$1"
MODE="${2:-all}"
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/home/ubuntu/BlenderGPT/blender-copilot"

cd "$PROJ_DIR"

if [ "$MODE" != "--data-only" ]; then
    echo "Uploading code..."
    rsync -avz --progress \
        --exclude '.venv' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.git' \
        --exclude 'data/raw' \
        --exclude 'data/extracted' \
        --exclude 'data/processed/*.json' \
        --exclude '.mesh_cache_backup' \
        --exclude 'node_modules' \
        --exclude '*.zip' \
        --exclude '.DS_Store' \
        --exclude 'temp_addon_build' \
        ./ "$REMOTE:$REMOTE_DIR/"
    echo "Code uploaded."
fi

if [ "$MODE" != "--code-only" ]; then
    echo "Uploading training data cache..."
    rsync -avz --progress \
        data/processed/.mesh_cache/ \
        "$REMOTE:$REMOTE_DIR/data/processed/.mesh_cache/"

    echo "Uploading BPE tokenizer..."
    rsync -avz --progress \
        data/datasets/geometry/bpe_tokenizer/ \
        "$REMOTE:$REMOTE_DIR/data/datasets/geometry/bpe_tokenizer/"

    if [ -d "checkpoints/unified" ]; then
        echo "Uploading latest checkpoint..."
        rsync -avz --progress \
            checkpoints/unified/latest.pt \
            "$REMOTE:$REMOTE_DIR/checkpoints/unified/"
    fi

    echo "Data uploaded."
fi

echo ""
echo "Done. Now SSH in and run:"
echo "  ssh $REMOTE"
echo "  cd /home/ubuntu/BlenderGPT/blender-copilot"
echo "  bash cloud/setup_instance.sh"
echo "  bash cloud/train_cloud.sh"
