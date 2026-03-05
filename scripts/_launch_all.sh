#!/bin/bash
# Restart SmutBase + Open3DLab with fixed login field
cd /Users/alexwaldmann/blenderPlugins/blender-copilot
source .venv/bin/activate

kill 39030 39031 2>/dev/null
sleep 1

nohup python scripts/rip_blendswap_smutbase.py --smutbase-only --sm-pages 500 > /tmp/rip_smutbase.log 2>&1 &
echo "SmutBase PID: $!"

nohup python scripts/rip_blendswap_smutbase.py --open3dlab-only --o3d-pages 250 > /tmp/rip_open3dlab.log 2>&1 &
echo "Open3DLab PID: $!"

echo "Done."
