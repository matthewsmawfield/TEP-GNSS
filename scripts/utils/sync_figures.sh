#!/bin/bash

# Sync figures from results/figures to site directories
echo "🔄 Syncing figures to GNSS site..."

# Copy to site/figures/
echo "📁 Copying to site/figures/..."
cp -v results/figures/*.png site/figures/

# Copy to site/public/figures/
echo "📁 Copying to site/public/figures/..."
cp -v results/figures/*.png site/public/figures/

echo "✅ Figure sync complete!"
echo "📊 Total figures synced: $(ls results/figures/*.png | wc -l)"
