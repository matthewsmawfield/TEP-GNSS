#!/bin/bash

# Deploy script for both Paper 1 and Paper 2
# Builds both sites and combines them into a single deployment

set -e

echo "🚀 Deploying TEP-GNSS (Both Papers) to GitHub Pages..."
echo ""

# Ensure we're in the repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Build Paper 1 (Multi-Center Validation)
echo "📄 Building Paper 1 (Multi-Center Validation)..."
cd site
npm run build
cd ..
echo "✅ Paper 1 built successfully"
echo ""

# Build Paper 2 (CODE Longspan)
echo "📄 Building Paper 2 (CODE Longspan)..."
cd site-code-longspan
npm run build
cd ..
echo "✅ Paper 2 built successfully"
echo ""

# Check if both builds exist
if [ ! -d "site/dist" ] || [ -z "$(ls -A site/dist)" ]; then
    echo "❌ Paper 1 build failed - dist directory is empty or missing"
    exit 1
fi

if [ ! -d "site-code-longspan/dist" ] || [ -z "$(ls -A site-code-longspan/dist)" ]; then
    echo "❌ Paper 2 build failed - dist directory is empty or missing"
    exit 1
fi

# Create combined deployment directory
echo "📦 Combining both papers into final deployment..."
FINAL_DIST="final-dist"
rm -rf "$FINAL_DIST"
mkdir -p "$FINAL_DIST"

# Copy Paper 1 to root
echo "  • Copying Paper 1 to root..."
cp -r site/dist/* "$FINAL_DIST/"

# Copy Paper 2 to /code-longspan/ subdirectory
echo "  • Copying Paper 2 to /code-longspan/..."
mkdir -p "$FINAL_DIST/code-longspan"
cp -r site-code-longspan/dist/* "$FINAL_DIST/code-longspan/"

echo "✅ Combined deployment created"
echo ""

# Temporary directory for gh-pages branch
TEMP_DIR=$(mktemp -d)
echo "📂 Using temporary directory: $TEMP_DIR"

# Clone only the gh-pages branch to temp directory
if git ls-remote --exit-code --heads origin gh-pages >/dev/null 2>&1; then
    echo "📥 Cloning existing gh-pages branch..."
    git clone --depth 1 --branch gh-pages --single-branch "$(git remote get-url origin)" "$TEMP_DIR"
else
    echo "🆕 Creating new gh-pages branch..."
    git clone --depth 1 "$(git remote get-url origin)" "$TEMP_DIR"
    cd "$TEMP_DIR"
    git checkout --orphan gh-pages
    git rm -rf .
    cd - >/dev/null
fi

# Clear existing content (keep .git)
cd "$TEMP_DIR"
find . -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} +

# Copy combined site to temp directory
echo "📋 Copying combined site..."
cp -r "$SCRIPT_DIR/$FINAL_DIST"/* .

# Add .nojekyll to prevent Jekyll processing
touch .nojekyll

# Create a clean commit
git add -A
git config user.name "GitHub Actions Deploy"
git config user.email "deploy@github-actions.local"

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "✅ No changes to deploy"
    rm -rf "$TEMP_DIR"
    cd "$SCRIPT_DIR"
    rm -rf "$FINAL_DIST"
    exit 0
fi

# Commit with timestamp
COMMIT_MESSAGE="Deploy both papers $(date '+%Y-%m-%d %H:%M:%S UTC')"
git commit -m "$COMMIT_MESSAGE"

# Push to gh-pages
echo "📤 Pushing to gh-pages branch..."
git push origin gh-pages --force

# Cleanup
cd "$SCRIPT_DIR"
rm -rf "$TEMP_DIR"
rm -rf "$FINAL_DIST"

echo ""
echo "✅ Deployment complete!"
echo "🌐 Paper 1: https://$(git remote get-url origin | sed 's/.*github\.com[:/]\([^/]*\)\/\([^.]*\)\.git/\1.github.io\/\2/')/"
echo "🌐 Paper 2: https://$(git remote get-url origin | sed 's/.*github\.com[:/]\([^/]*\)\/\([^.]*\)\.git/\1.github.io\/\2/')/code-longspan/"
echo ""
