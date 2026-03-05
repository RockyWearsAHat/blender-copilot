#!/bin/bash
# ============================================================================
# Sync checkpoints from cloud back to local Mac
# ============================================================================
# Run this FROM YOUR LOCAL MAC to pull trained checkpoints.
#
# Usage:
#   bash cloud/sync_checkpoints.sh user@gpu-instance-ip
#   bash cloud/sync_checkpoints.sh user@gpu-instance-ip --continuous
# ============================================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash cloud/sync_checkpoints.sh user@host [--continuous]"
    exit 1
fi

REMOTE="$1"
MODE="${2:-once}"
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/home/ubuntu/BlenderGPT/blender-copilot"

sync_once() {
    echo "[$(date +%H:%M:%S)] Syncing checkpoints..."

    # 0. Push RLHF feedback from local Mac to Lambda so training can use it
    FEEDBACK_DIR="$PROJ_DIR/data/feedback"
    if [ -d "$FEEDBACK_DIR" ] && ls "$FEEDBACK_DIR"/*.jsonl 1>/dev/null 2>&1; then
        echo "  Pushing RLHF feedback to Lambda..."
        rsync -avz --progress \
            "$FEEDBACK_DIR/" \
            "$REMOTE:$REMOTE_DIR/data/feedback/"
    fi

    # Also push training feedback records (approve/reject logs from Blender)
    TRAIN_FB_DIR="$PROJ_DIR/data/training_feedback"
    if [ -d "$TRAIN_FB_DIR" ] && ls "$TRAIN_FB_DIR"/*.jsonl 1>/dev/null 2>&1; then
        echo "  Pushing training feedback logs to Lambda..."
        rsync -avz --progress \
            "$TRAIN_FB_DIR/" \
            "$REMOTE:$REMOTE_DIR/data/training_feedback/"
    fi

    # 1. Always re-download latest.pt and best.pt (they get overwritten in-place)
    #    --checksum: compare by content hash (not size/mtime) since size is always ~829MB
    #    --no-times: use local write time (not remote mtime) so hot-reload detects changes
    rsync -rlvz --progress --checksum --no-times \
        --include='latest.pt' --include='best.pt' --exclude='*' \
        "$REMOTE:$REMOTE_DIR/checkpoints/unified/" \
        "$PROJ_DIR/checkpoints/unified/"

    # 2. Pull any new step_*.pt files (these are write-once, skip if exists)
    rsync -avz --progress --ignore-existing \
        --include='step_*.pt' --exclude='*' \
        "$REMOTE:$REMOTE_DIR/checkpoints/unified/" \
        "$PROJ_DIR/checkpoints/unified/"

    # 3. Pull reward model checkpoint (RLHF trains this locally too)
    rsync -avz --progress --ignore-existing \
        "$REMOTE:$REMOTE_DIR/checkpoints/reward/" \
        "$PROJ_DIR/checkpoints/reward/" 2>/dev/null || true
    
    echo "[$(date +%H:%M:%S)] Sync complete."
    
    if [ -f "$PROJ_DIR/checkpoints/unified/latest.pt" ]; then
        STEP=$(curl -s http://localhost:8420/health 2>/dev/null | grep -oE '"step":[0-9]+' | grep -oE '[0-9]+')
        if [ -n "$STEP" ]; then
            echo "Inference server running at step $STEP — will hot-reload within 30s."
        else
            echo "Inference server not running. Start with: python run.py serve"
        fi
    fi
}

if [ "$MODE" == "--continuous" ]; then
    echo "Continuous sync mode - checking every 60 seconds. Ctrl+C to stop."
    while true; do
        sync_once
        echo "---"
        sleep 60
    done
else
    sync_once
fi
