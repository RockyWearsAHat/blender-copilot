#!/bin/bash
cd ~/blender-copilot
source .venv/bin/activate
nohup bash cloud/train_cloud.sh > training.log 2>&1 &
echo "Training started with PID: $!"
sleep 2
tail -20 training.log
