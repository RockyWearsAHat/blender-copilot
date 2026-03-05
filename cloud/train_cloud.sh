#!/bin/bash
# ============================================================================
# Train on Cloud GPU — optimized for H200/A100/H100
# ============================================================================
# Runs training with cloud-optimized settings:
#   - Larger batch size (H200 has 141GB HBM3e)
#   - bf16 mixed precision (native on H200/H100)
#   - Flash Attention if available
#   - Saves checkpoints every 1000 steps
# ============================================================================

set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

source .venv/bin/activate 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

echo "GPU: $GPU_NAME ($GPU_MEM MB)"

if [ "${GPU_MEM:-0}" -gt 100000 ]; then
    export CLOUD_BATCH_SIZE=64
    export CLOUD_GRAD_ACCUM=2
    export CLOUD_MAX_SEQ_LEN=72002
    echo "H200 (141GB) detected: batch_size=64, grad_accum=2 (effective=128) [300M model]"
elif [ "${GPU_MEM:-0}" -gt 60000 ]; then
    export CLOUD_BATCH_SIZE=16
    export CLOUD_GRAD_ACCUM=4
    export CLOUD_MAX_SEQ_LEN=72002
    echo "H100/A100-80GB detected: batch_size=16, grad_accum=4 (effective=64) [300M model]"
elif [ "${GPU_MEM:-0}" -gt 40000 ]; then
    export CLOUD_BATCH_SIZE=8
    export CLOUD_GRAD_ACCUM=8
    export CLOUD_MAX_SEQ_LEN=36002
    echo "A100-40GB/A6000 detected: batch_size=8, grad_accum=8 (effective=64) [300M model]"
elif [ "${GPU_MEM:-0}" -gt 20000 ]; then
    export CLOUD_BATCH_SIZE=8
    export CLOUD_GRAD_ACCUM=8
    export CLOUD_MAX_SEQ_LEN=36002
    echo "RTX 3090/4090 detected: batch_size=8, grad_accum=8 (effective=64) [300M model]"
else
    export CLOUD_BATCH_SIZE=16
    export CLOUD_GRAD_ACCUM=8
    export CLOUD_MAX_SEQ_LEN=16202
    echo "Smaller GPU: batch_size=16, grad_accum=8 (effective=128)"
fi

export CLOUD_MIXED_PRECISION=bf16
export CLOUD_SAVE_EVERY=1000
export CLOUD_EVAL_EVERY=2000

LOG_FILE="/tmp/train_cloud_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training... Log: $LOG_FILE"
echo "Press Ctrl+C to gracefully stop and save."
echo ""

python run.py train \
    2>&1 | tee "$LOG_FILE"
