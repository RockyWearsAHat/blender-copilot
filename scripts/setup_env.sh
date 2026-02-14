#!/usr/bin/env bash
# setup_env.sh — Set up the Python environment for BlenderModelTraining
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════"
echo " BlenderModelTraining — Environment Setup"
echo "═══════════════════════════════════════════"

# Check Python version
PYTHON="${PYTHON:-python3}"
echo "Python: $($PYTHON --version)"

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

source venv/bin/activate
echo "Virtual environment activated: $(which python)"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (detect platform)
echo ""
echo "Installing PyTorch..."
if command -v nvidia-smi &>/dev/null; then
    echo "  NVIDIA GPU detected — installing CUDA version"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif [[ "$(uname)" == "Darwin" ]]; then
    echo "  macOS detected — installing MPS-enabled version"
    pip install torch torchvision
else
    echo "  No GPU detected — installing CPU version"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/raw/blendswap
mkdir -p data/raw/github
mkdir -p data/raw/youtube
mkdir -p data/extracted
mkdir -p data/filtered
mkdir -p data/labeled
mkdir -p data/datasets
mkdir -p checkpoints/geometry
mkdir -p checkpoints/materials
mkdir -p checkpoints/modifiers

echo ""
echo "═══════════════════════════════════════════"
echo " ✅ Setup complete!"
echo ""
echo " To activate: source venv/bin/activate"
echo " Next steps:"
echo "   1. Run scrapers:  bash scripts/download_all.sh"
echo "   2. Process data:  bash scripts/process_all.sh"
echo "   3. Train:         python -m training.train_geometry --config config.yaml --data data/datasets/geometry_train.jsonl"
echo "═══════════════════════════════════════════"
