#!/bin/bash
# TEP-GNSS GCP Results Download Script
# ====================================
# Downloads all results and figures from GCP instance to local results/ directory

set -e

# GCP Configuration
PROJECT_ID="tvp-bbpm"
ZONE="us-central1-c"
INSTANCE_NAME="instance-20251006-195418"

echo "📥 TEP-GNSS GCP Results Download"
echo "================================="

# Check if gcloud CLI is available
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo "✅ GCP Configuration:"
echo "   Project: $PROJECT_ID"
echo "   Zone: $ZONE"
echo "   Instance: $INSTANCE_NAME"

# Set the project
gcloud config set project $PROJECT_ID

echo "🔗 Testing SSH connection..."
if ! gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="echo 'SSH OK'" >/dev/null 2>&1; then
    echo "❌ SSH connection failed!"
    echo "   Please check:"
    echo "   1. Instance is running"
    echo "   2. Firewall rules allow SSH (port 22)"
    echo "   3. You have the correct permissions"
    echo "   4. Instance name is correct: $INSTANCE_NAME"
    exit 1
fi
echo "✅ SSH connection established"

# Create local results directory structure
echo "📁 Creating local results directory structure..."
mkdir -p results/outputs
mkdir -p results/figures
mkdir -p results/tmp
mkdir -p results/exploratory

# Download all JSON result files
echo "📄 Downloading JSON result files..."
gcloud compute scp --recurse $INSTANCE_NAME:/mnt/data/results/outputs/*.json results/outputs/ --zone=$ZONE

# Download all PNG figure files
echo "🖼️  Downloading PNG figure files..."
gcloud compute scp --recurse $INSTANCE_NAME:/mnt/data/results/figures/*.png results/figures/ --zone=$ZONE

# Download logs for reference
echo "📋 Downloading latest pipeline logs..."
gcloud compute scp $INSTANCE_NAME:/mnt/data/pipeline_fixed_gravitational_final.log logs/ --zone=$ZONE 2>/dev/null || echo "   No pipeline log found"

# Download all log files from the logs directory
echo "📋 Downloading all log files..."
gcloud compute scp --recurse $INSTANCE_NAME:/mnt/data/logs/*.log logs/ --zone=$ZONE 2>/dev/null || echo "   No log files found"

# Download TID/Hilbert/Wavelet analysis files
echo "🌊 Downloading TID/Hilbert/Wavelet analysis files..."
mkdir -p results/tmp/streaming
gcloud compute scp --recurse $INSTANCE_NAME:/mnt/data/results/tmp/streaming/*.csv results/tmp/streaming/ --zone=$ZONE 2>/dev/null || echo "   No streaming files found"

# Download any additional processed data with Hilbert/wavelet analysis
echo "📊 Downloading additional processed data..."
mkdir -p data/processed
gcloud compute scp --recurse $INSTANCE_NAME:/mnt/data/data/processed/*.csv data/processed/ --zone=$ZONE 2>/dev/null || echo "   No additional processed data found"

# Download site figures if they exist
echo "🌐 Downloading site figures..."
gcloud compute scp --recurse $INSTANCE_NAME:/mnt/data/site/figures/*.png site/figures/ --zone=$ZONE 2>/dev/null || echo "   No site figures found"

# Sync key figures to site folder for website
echo "🔄 Syncing key figures to site folder..."
mkdir -p site/figures
cp results/figures/step_4_2_tep_synthesis_figure.png site/figures/ 2>/dev/null || echo "   Synthesis figure not found"
cp results/figures/step_4_4_comprehensive_gravitational_temporal_analysis.png site/figures/ 2>/dev/null || echo "   Gravitational analysis figure not found"
cp results/figures/step_4_1_binned_correlation_data.png site/figures/ 2>/dev/null || echo "   Binned correlation figure not found"
cp results/figures/figure_1_TEP_site_themed.png site/figures/ 2>/dev/null || echo "   Main TEP figure not found"

# Count downloaded files
echo ""
echo "📊 Download Summary:"
echo "   JSON files: $(find results/outputs -name "*.json" 2>/dev/null | wc -l)"
echo "   PNG files: $(find results/figures -name "*.png" 2>/dev/null | wc -l)"
echo "   Log files: $(find logs -name "*.log" 2>/dev/null | wc -l)"
echo "   TID/Streaming files: $(find results/tmp/streaming -name "*.csv" 2>/dev/null | wc -l)"
echo "   Processed data files: $(find data/processed -name "*.csv" 2>/dev/null | wc -l)"
echo "   Site figures: $(find site/figures -name "*.png" 2>/dev/null | wc -l)"
echo "   Synced to site: $(find site/figures -name "*.png" 2>/dev/null | wc -l)"

echo ""
echo "✅ GCP results download completed successfully!"
echo "   All fresh analysis results are now available locally in results/"
echo ""
echo "🔍 Key files to check:"
echo "   - results/outputs/step_4_0_advanced_analysis.json (main results)"
echo "   - results/figures/step_4_2_tep_synthesis_figure.png (synthesis figure)"
echo "   - results/figures/step_4_4_comprehensive_gravitational_temporal_analysis.png (gravitational analysis)"
echo "   - results/outputs/step_3_1_robust_block_bootstrap_*.json (bootstrap results)"
echo "   - results/outputs/step_4_6_tid_exclusion_analysis_results.json (TID/Hilbert analysis)"
echo "   - results/tmp/streaming/*.csv (TID streaming analysis files)"
echo "   - data/processed/*.csv (processed data with Hilbert/wavelet analysis)"
