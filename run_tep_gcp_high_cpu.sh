#!/bin/bash
# TEP-GNSS Google Cloud Platform (GCP) High-CPU Analysis
# ======================================================
# Optimized deployment for high-CPU GCP Compute Engine instances

set -e

# GCP Configuration - Load from environment variables
# Set these in .env.local or as environment variables:
# GCP_PROJECT_ID, GCP_ZONE, GCP_INSTANCE_NAME

# Load environment variables from .env files if they exist
if [ -f ".env.local" ]; then
    export $(grep -v '^#' .env.local | xargs)
elif [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Get configuration from environment variables
PROJECT_ID="${GCP_PROJECT_ID:-}"
ZONE="${GCP_ZONE:-}"
INSTANCE_NAME="${GCP_INSTANCE_NAME:-}"
PACKAGE_NAME="tep-gnss-gcp-optimized.tar.gz"

# High-CPU Instance recommendation:
# n2-highcpu-96 (96 vCPUs, 96 GB RAM) - Maximum performance (recommended)

echo "🚀 TEP-GNSS Google Cloud Platform Analysis"
echo "==========================================="

# Check if gcloud CLI is available
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user has set the required variables
if [ -z "$PROJECT_ID" ] || [ -z "$ZONE" ] || [ -z "$INSTANCE_NAME" ]; then
    echo "❌ GCP configuration missing!"
    echo ""
    echo "Please set the following environment variables:"
    echo "   GCP_PROJECT_ID: Your GCP project ID"
    echo "   GCP_ZONE: Your GCP zone (e.g., us-central1-a)"
    echo "   GCP_INSTANCE_NAME: Your GCP instance name"
    echo ""
    echo "Options:"
    echo "1. Create .env.local file (recommended):"
    echo "   cp env.example .env.local"
    echo "   # Edit .env.local with your GCP details"
    echo ""
    echo "2. Set environment variables directly:"
    echo "   export GCP_PROJECT_ID=your-project-id"
    echo "   export GCP_ZONE=us-central1-a"
    echo "   export GCP_INSTANCE_NAME=your-instance-name"
    echo ""
    echo "Recommended high-CPU instance type:"
    echo "  - n2-highcpu-96 (96 vCPUs, 96 GB RAM) - Maximum performance (recommended)"
    exit 1
fi

echo "✅ GCP Configuration:"
echo "   Project: $PROJECT_ID"
echo "   Zone: $ZONE"
echo "   Instance: $INSTANCE_NAME"

# Set the project
gcloud config set project $PROJECT_ID

# Get instance external IP
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME \
    --zone=$ZONE \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)')

echo "   External IP: $EXTERNAL_IP"

# Create analysis package
echo "📦 Creating TEP-GNSS analysis package..."
rm -f $PACKAGE_NAME

# Create TEP-GNSS analysis package
tar --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
    --exclude='*.log' --exclude='*.pid' --exclude='*.parquet' \
    --exclude='data/raw/*' --exclude='data/processed/*' \
    --exclude='results/*' --exclude='logs/*' \
    --exclude='venv' --exclude='site/public/figures/*' \
    --exclude='*.tar.gz' --exclude='*.zip' \
    --exclude='run_tep_paperspace_bulletproof.sh' \
    -czf $PACKAGE_NAME scripts/ requirements/ data/coordinates/ de432s.bsp 2>/dev/null || \
tar -czf $PACKAGE_NAME scripts/ requirements/

PACKAGE_SIZE=$(du -h $PACKAGE_NAME | cut -f1)
echo "   Package size: $PACKAGE_SIZE"

# Test SSH connection
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

# Transfer package
echo "📤 Transferring analysis package to GCP..."
gcloud compute scp $PACKAGE_NAME $INSTANCE_NAME:~/ --zone=$ZONE

# Create the setup script that will run on GCP
echo "🚀 Creating GCP setup and execution script..."
cat > /tmp/gcp_run_tep.sh << 'REMOTE_SCRIPT_EOF'
#!/bin/bash
set -e

echo "🔧 Setting up GCP environment for TEP-GNSS analysis..."

# Update system
sudo apt update >/dev/null 2>&1
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential gfortran libhdf5-dev libnetcdf-dev htop iotop >/dev/null 2>&1

# Get system specs
CORES=$(nproc)
MEMORY_GB=$(free -g | awk "/^Mem:/{print int(\$2*0.8)}")

echo ""
echo "⚡ GCP Analysis System Configuration:"
echo "  CPU cores: $CORES"
echo "  Memory: ${MEMORY_GB}GB"
echo "  Analysis: Full 912-day window (2023-01-01 to 2025-06-30)"

# Setup additional storage disk for high-CPU instances
echo "💾 Setting up additional storage disk..."
if ! mountpoint -q /mnt/data; then
    echo "  Setting up new disk..."
    sudo mkdir -p /mnt/data
    
    # Check for available disks and mount the first one found
    if [ -b /dev/sdb ]; then
        echo "  Found /dev/sdb, formatting and mounting..."
        # Check if disk is already formatted
        if ! sudo blkid /dev/sdb >/dev/null 2>&1; then
            echo "  Formatting /dev/sdb with ext4..."
            sudo mkfs.ext4 -F /dev/sdb
        else
            echo "  /dev/sdb already formatted, mounting..."
        fi
        sudo mount /dev/sdb /mnt/data
        echo "  ✅ /dev/sdb mounted to /mnt/data"
    elif [ -b /dev/sdc ]; then
        echo "  Found /dev/sdc, formatting and mounting..."
        if ! sudo blkid /dev/sdc >/dev/null 2>&1; then
            echo "  Formatting /dev/sdc with ext4..."
            sudo mkfs.ext4 -F /dev/sdc
        else
            echo "  /dev/sdc already formatted, mounting..."
        fi
        sudo mount /dev/sdc /mnt/data
        echo "  ✅ /dev/sdc mounted to /mnt/data"
    else
        echo "  ⚠️  No additional disk found, using root disk"
        sudo ln -sf /home/$USER /mnt/data
    fi
    
    # Set proper ownership
    sudo chown -R $USER:$USER /mnt/data
    echo "  ✅ Storage setup complete"
else
    echo "  ✅ Storage already configured"
fi

# Check disk space
echo "📊 Disk space status:"
df -h /mnt/data | tail -1 | awk "{print \"  Data disk: \" \$2 \" total, \" \$3 \" used, \" \$4 \" available (\" \$5 \" used)\"}"
df -h / | tail -1 | awk "{print \"  Root disk: \" \$2 \" total, \" \$3 \" used, \" \$4 \" available (\" \$5 \" used)\"}"

# Move to data disk for all operations
cd /mnt/data
echo "  🎯 Working directory: $(pwd)"

# Copy package to data disk
echo "📦 Moving analysis package to data disk..."
cp ~/tep-gnss-gcp-optimized.tar.gz ./
echo "  Extracting analysis package..."
tar -xzf tep-gnss-gcp-optimized.tar.gz 2>/dev/null || {
    echo "  ⚠️  Standard extraction failed, trying with ignore-unknown-options..."
    tar --ignore-unknown-options -xzf tep-gnss-gcp-optimized.tar.gz
}

# Verify extraction
if [ -d "scripts" ] && [ -d "requirements" ]; then
    echo "  ✅ Package extracted successfully"
else
    echo "  ❌ Package extraction failed!"
    exit 1
fi

# Create required directories on data disk
mkdir -p data/raw/{igs_combined,code,esa_final} data/processed data/coordinates
mkdir -p logs results/{outputs,figures,tmp}

# Install Python dependencies in virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "  Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
echo "  Installing from requirements.txt..."
pip install -r requirements/requirements.txt
echo "  Installing additional specialized GNSS packages..."
pip install pyIGRF jplephem skyfield
echo "  ✅ All dependencies installed successfully"

# Set optimal environment for high-CPU GCP instance
export TEP_WORKERS=$CORES
export TEP_MAX_PARALLEL_DOWNLOADS=$CORES
export TEP_RESUME=1
export TEP_MEMORY_LIMIT_GB=$MEMORY_GB
export TEP_BOOTSTRAP_ITER=5000  # Higher iterations for high-CPU
export TEP_NULL_ITERATIONS=500  # Higher iterations for high-CPU
export TEP_DATE_START=2023-01-01
export TEP_DATE_END=2025-06-30
export TEP_FULL_FRESH_RUN=1  # Enable complete fresh analysis
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES
export NUMEXPR_NUM_THREADS=$CORES

echo ""
echo "🚀 Starting TEP-GNSS analysis on GCP (COMPLETE DEPLOYMENT)..."
echo "   This includes:"
echo "     - COMPLETE DATA CLEANUP (fresh start)"
echo "     - Data acquisition (Step 1) - Full date range"
echo "     - Core analysis (Step 2)" 
echo "     - Validation suite (Step 3)"
echo "     - Advanced analysis & visualization (Step 4)"
echo ""
echo "   Date Range: 2023-01-01 to 2025-06-30 (911 days)"
echo "   High-CPU Configuration:"
echo "     - Workers: $TEP_WORKERS (all CPU cores)"
echo "     - Parallel downloads: $TEP_MAX_PARALLEL_DOWNLOADS"
echo "     - Bootstrap iterations: $TEP_BOOTSTRAP_ITER"
echo "     - Null test iterations: $TEP_NULL_ITERATIONS"
echo "     - Fresh deployment: ENABLED"
echo "     - Working directory: $(pwd)"
echo ""
echo "   Starting full pipeline execution..."

# Run the complete TEP-GNSS analysis pipeline with error handling
echo "🚀 Starting TEP-GNSS FRESH 912-day analysis pipeline..."
echo "   Start time: $(date)"
echo "   Working directory: $(pwd)"
echo "   Available disk space:"
df -h /mnt/data | tail -1

# Run with comprehensive logging using virtual environment
echo "  Activating virtual environment..."
source venv/bin/activate
echo "  Starting TEP-GNSS analysis in background..."
echo "  Log file: /mnt/data/full_pipeline.log"
nohup python scripts/clean_run_full_pipeline.py > full_pipeline.log 2>&1 &
ANALYSIS_PID=$!
echo "  ✅ Analysis started with PID: $ANALYSIS_PID"
echo "  Monitor progress with: tail -f /mnt/data/full_pipeline.log"
echo "  Check status with: ps aux | grep $ANALYSIS_PID"

echo ""
echo "✅ TEP-GNSS analysis is now running on GCP!"
echo "   The analysis will continue in the background."
echo "   You can disconnect safely - the process will keep running."
REMOTE_SCRIPT_EOF

# Transfer the script to GCP
echo "📤 Transferring setup script to GCP..."
gcloud compute scp /tmp/gcp_run_tep.sh $INSTANCE_NAME:~/ --zone=$ZONE

# Execute the script on GCP in the background
echo "▶️  Executing setup and analysis script on GCP..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="chmod +x ~/gcp_run_tep.sh && nohup ~/gcp_run_tep.sh > ~/gcp_setup.log 2>&1 &"

echo ""
echo "✅ GCP analysis setup complete!"
echo ""
echo "📊 Monitor progress:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='cd /mnt/data && tail -f full_pipeline.log'"
echo ""
echo "📊 Connect to instance:"
echo "gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "📥 Download results when complete:"
echo "gcloud compute scp --recurse $INSTANCE_NAME:/mnt/data/results/ ./gcp_results/ --zone=$ZONE"
echo ""
echo "💰 To stop instance:"
echo "gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "🛑 To delete instance:"
echo "gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "Instance: $INSTANCE_NAME"
echo "External IP: $EXTERNAL_IP"
echo ""
echo "💡 High-CPU Instance Recommendation:"
echo "   - n2-highcpu-96: 96 vCPUs, 96 GB RAM (Maximum performance - recommended)"
