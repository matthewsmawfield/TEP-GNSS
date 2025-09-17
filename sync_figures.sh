#!/bin/bash

# Automated Figure Sync Script for TEP-GNSS Website
# Syncs figures from results/figures/ to site/public/figures/

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}TEP-GNSS Figure Sync${NC}"
echo "=================================="

# Check if source directory exists
if [ ! -d "results/figures" ]; then
    echo -e "${RED}Error: results/figures/ directory not found!${NC}"
    exit 1
fi

# Check if destination directory exists
if [ ! -d "site/public/figures" ]; then
    echo -e "${YELLOW}Creating site/public/figures/ directory...${NC}"
    mkdir -p site/public/figures
fi

# Count files before sync
BEFORE_COUNT=$(ls site/public/figures/*.png 2>/dev/null | wc -l || echo "0")

echo "Source: results/figures/"
echo "Destination: site/public/figures/"
echo ""

# Show what will be updated
echo -e "${YELLOW}Files to sync:${NC}"
for file in results/figures/*.png; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        if [ -f "site/public/figures/$filename" ]; then
            # Compare modification times
            if [ "$file" -nt "site/public/figures/$filename" ]; then
                echo -e "  ${GREEN}UPDATE${NC} $filename"
            else
                echo -e "  ${YELLOW}CURRENT${NC} $filename"
            fi
        else
            echo -e "  ${GREEN}NEW${NC} $filename"
        fi
    fi
done
echo ""

# Perform the sync
echo -e "${YELLOW}Syncing figures...${NC}"
cp results/figures/*.png site/public/figures/ 2>/dev/null || {
    echo -e "${RED}Error: Failed to copy files${NC}"
    exit 1
}

# Count files after sync
AFTER_COUNT=$(ls site/public/figures/*.png 2>/dev/null | wc -l || echo "0")

echo -e "${GREEN}✓ Sync complete!${NC}"
echo "Files synced: $AFTER_COUNT PNG files"
echo ""

# Show file sizes and timestamps
echo -e "${YELLOW}Updated files:${NC}"
ls -lah site/public/figures/*.png | tail -10

echo ""
echo -e "${GREEN}Website figures are now up to date!${NC}"
