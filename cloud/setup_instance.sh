#!/bin/bash
# ============================================================================
# Cloud GPU Setup for Blender Copilot Training
# ============================================================================
# This script sets up a fresh cloud GPU instance (GCP, Lambda, RunPod, etc.)
# for training the Blender Copilot mesh generation model.
#
# Usage:
#   1. SSH into your cloud instance
#   2. Run: bash cloud/setup_instance.sh
#   3. Then: bash cloud/train_cloud.sh
# ============================================================================

set -e

echo "Setting up Blender Copilot training environment..."

sudo apt-get update -qq
sudo apt-get install -y -qq git rsync tmux htop nvtop python3-pip python3-venv

if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Is CUDA installed?"
else
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

PROJ_DIR="/home/ubuntu/BlenderGPT/blender-copilot"

if [ ! -d "$PROJ_DIR" ]; then
    echo "Project directory not found at $PROJ_DIR"
    echo "Upload your project first: bash cloud/upload_to_cloud.sh user@host"
    exit 1
fi

cd "$PROJ_DIR"

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install sentencepiece

if [ -d "data/processed/.mesh_cache" ]; then
    CACHE_COUNT=$(find data/processed/.mesh_cache -name "*.pt" | wc -l)
    echo "Found $CACHE_COUNT cached training samples"
fi

if [ -d "checkpoints/unified" ]; then
    echo "Found existing checkpoints:"
    ls -lh checkpoints/unified/*.pt 2>/dev/null || echo "  (none)"
fi

echo ""
echo "Setup complete."
echo ""
echo "To start training:"
echo "  tmux new -s train"
echo "  source .venv/bin/activate"
echo "  python run.py train"
echo ""
echo "To sync checkpoints back to your Mac:"
echo "  bash cloud/sync_checkpoints.sh  (run from local machine)"
