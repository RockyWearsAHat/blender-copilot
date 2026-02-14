#!/usr/bin/env bash
# download_all.sh — Run all scrapers to collect training data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════"
echo " BlenderModelTraining — Data Download"
echo "═══════════════════════════════════════════"

# Activate venv if exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Create output directories
mkdir -p data/raw/blendswap
mkdir -p data/raw/github
mkdir -p data/raw/youtube

# 1. BlendSwap scraper
echo ""
echo "━━━ [1/3] Scraping BlendSwap ━━━"
echo "  This may take a while. Rate-limited to be respectful."
python -m scrapers.blendswap_scraper \
    --config config.yaml \
    --output data/raw/blendswap \
    || echo "⚠️  BlendSwap scraper finished with warnings"

# 2. GitHub scraper
echo ""
echo "━━━ [2/3] Scraping GitHub ━━━"
echo "  Set GITHUB_TOKEN env var for higher rate limits."
python -m scrapers.github_scraper \
    --config config.yaml \
    --output data/raw/github \
    || echo "⚠️  GitHub scraper finished with warnings"

# 3. YouTube transcript scraper
echo ""
echo "━━━ [3/3] Scraping YouTube Tutorials ━━━"
python -m scrapers.youtube_scraper \
    --config config.yaml \
    --output data/raw/youtube \
    || echo "⚠️  YouTube scraper finished with warnings"

# Summary
echo ""
echo "═══════════════════════════════════════════"
echo " Download Summary"
echo "═══════════════════════════════════════════"
echo " BlendSwap files: $(find data/raw/blendswap -name '*.blend' 2>/dev/null | wc -l)"
echo " GitHub files:    $(find data/raw/github -name '*.blend' 2>/dev/null | wc -l)"
echo " YouTube files:   $(find data/raw/youtube -name '*.json' 2>/dev/null | wc -l)"
echo ""
echo " Next: bash scripts/process_all.sh"
echo "═══════════════════════════════════════════"
