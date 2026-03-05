#!/bin/bash
cd /Users/alexwaldmann/blenderPlugins/blender-copilot
exec /Users/alexwaldmann/blenderPlugins/blender-copilot/.venv/bin/python /Users/alexwaldmann/blenderPlugins/blender-copilot/scripts/rebuild_cache.py > /tmp/rebuild_dedup.txt 2>&1
