#!/bin/bash
# Build the Blender addon ZIP package

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Define the addon name
ADDON_NAME="BlenderAICopilot"
ZIP_NAME="${ADDON_NAME}.zip"
TEMP_DIR="temp_addon_build"

echo "🏗️  Building Blender AI Copilot addon..."

# Remove old ZIP if it exists
if [ -f "$ZIP_NAME" ]; then
    echo "📦 Removing old $ZIP_NAME"
    rm -f "$ZIP_NAME"
fi

# Remove old temp directory if it exists
if [ -d "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi

# Create temporary directory structure
echo "📁 Creating temporary build directory..."
mkdir -p "$TEMP_DIR/$ADDON_NAME"

# Copy addon files to the temp directory
echo "📋 Copying addon files..."
cp addon/__init__.py "$TEMP_DIR/$ADDON_NAME/"
cp addon/properties.py "$TEMP_DIR/$ADDON_NAME/"
cp addon/preferences.py "$TEMP_DIR/$ADDON_NAME/"
cp addon/panels.py "$TEMP_DIR/$ADDON_NAME/"
cp addon/operators.py "$TEMP_DIR/$ADDON_NAME/"
cp addon/ai_engine.py "$TEMP_DIR/$ADDON_NAME/"
cp addon/blender_tools.py "$TEMP_DIR/$ADDON_NAME/"
cp addon/materials.py "$TEMP_DIR/$ADDON_NAME/"
cp addon/tool_defs.py "$TEMP_DIR/$ADDON_NAME/"
cp addon/README.md "$TEMP_DIR/$ADDON_NAME/"

# Create the ZIP file from the temp directory
echo "📦 Creating $ZIP_NAME..."
cd "$TEMP_DIR"
zip -r "../$ZIP_NAME" "$ADDON_NAME/" \
    -x '*.pyc' '*__pycache__*' '*.DS_Store' '*chat_logs*' '*_backup*'
cd ..

# Clean up temp directory
echo "🧹 Cleaning up..."
rm -rf "$TEMP_DIR"

if [ -f "$ZIP_NAME" ]; then
    echo "✅ Success! Created $ZIP_NAME"
    echo ""
    echo "📋 Installation instructions:"
    echo "1. Open Blender 3.6 or newer"
    echo "2. Go to Edit → Preferences → Add-ons"
    echo "3. Click 'Install...' button"
    echo "4. Select: $SCRIPT_DIR/$ZIP_NAME"
    echo "5. Enable 'Blender AI Copilot' in the addon list"
    echo "6. Set your OpenAI API key in the addon preferences"
    echo "7. Press N in 3D viewport → AI Copilot tab to start"
    echo ""
else
    echo "❌ Failed to create ZIP file"
    exit 1
fi
