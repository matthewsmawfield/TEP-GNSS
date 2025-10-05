#!/bin/bash
# Create High-CPU GCP Instance for TEP-GNSS Analysis
# ===================================================
# This script creates a high-CPU GCP instance optimized for TEP-GNSS analysis

set -e

# Configuration
PROJECT_ID="tvp-bbpm"
ZONE="us-central1-f"
REGION="us-central1"

echo "🚀 Creating High-CPU GCP Instance for TEP-GNSS Analysis"
echo "========================================================"

# Set the project
gcloud config set project $PROJECT_ID

# Function to create instance
create_instance() {
    local INSTANCE_NAME=$1
    local MACHINE_TYPE=$2
    local DESCRIPTION=$3
    
    echo "📋 Creating instance: $INSTANCE_NAME"
    echo "   Machine type: $MACHINE_TYPE"
    echo "   Description: $DESCRIPTION"
    
    # Check if instance already exists
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE >/dev/null 2>&1; then
        echo "   ⚠️  Instance $INSTANCE_NAME already exists"
        echo "   Current status:"
        gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(status)"
        echo ""
        return 0
    fi
    
    # Create the instance
    gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --network-interface=network-tier=PREMIUM,subnet=default \
        --maintenance-policy=MIGRATE \
        --provisioning-model=STANDARD \
        --service-account=default \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --create-disk=auto-delete=yes,boot=yes,device-name=$INSTANCE_NAME,image=projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20241219,mode=rw,size=20,type=projects/$PROJECT_ID/zones/$ZONE/diskTypes/pd-balanced \
        --create-disk=auto-delete=yes,device-name=data-disk,size=200,type=projects/$PROJECT_ID/zones/$ZONE/diskTypes/pd-ssd \
        --no-shielded-secure-boot \
        --shielded-vtpm \
        --shielded-integrity-monitoring \
        --labels=purpose=tep-gnss-analysis,type=high-cpu \
        --reservation-affinity=any
    
    echo "   ✅ Instance $INSTANCE_NAME created successfully"
    echo ""
}

# Display options
echo "Available High-CPU Instance Types:"
echo ""
echo "1. c2-standard-60 (60 vCPUs, 240 GB RAM) - Maximum Performance"
echo "2. c2-standard-30 (30 vCPUs, 120 GB RAM) - High Performance" 
echo "3. c2-standard-16 (16 vCPUs, 64 GB RAM) - Good Performance"
echo "4. n2-highcpu-80 (80 vCPUs, 80 GB RAM) - Maximum CPU Cores"
echo "5. n2-highcpu-32 (32 vCPUs, 32 GB RAM) - High CPU Cores"
echo "6. n2-highcpu-16 (16 vCPUs, 16 GB RAM) - Moderate CPU Cores"
echo "7. Custom - Enter your own machine type"
echo ""

# Get user choice
read -p "Select instance type (1-7): " choice

case $choice in
    1)
        INSTANCE_NAME="tep-gnss-c2-60"
        MACHINE_TYPE="c2-standard-60"
        DESCRIPTION="Maximum Performance - 60 vCPUs, 240 GB RAM"
        ;;
    2)
        INSTANCE_NAME="tep-gnss-c2-30"
        MACHINE_TYPE="c2-standard-30"
        DESCRIPTION="High Performance - 30 vCPUs, 120 GB RAM"
        ;;
    3)
        INSTANCE_NAME="tep-gnss-c2-16"
        MACHINE_TYPE="c2-standard-16"
        DESCRIPTION="Good Performance - 16 vCPUs, 64 GB RAM"
        ;;
    4)
        INSTANCE_NAME="tep-gnss-n2-80"
        MACHINE_TYPE="n2-highcpu-80"
        DESCRIPTION="Maximum CPU Cores - 80 vCPUs, 80 GB RAM"
        ;;
    5)
        INSTANCE_NAME="tep-gnss-n2-32"
        MACHINE_TYPE="n2-highcpu-32"
        DESCRIPTION="High CPU Cores - 32 vCPUs, 32 GB RAM"
        ;;
    6)
        INSTANCE_NAME="tep-gnss-n2-16"
        MACHINE_TYPE="n2-highcpu-16"
        DESCRIPTION="Moderate CPU Cores - 16 vCPUs, 16 GB RAM"
        ;;
    7)
        read -p "Enter instance name: " INSTANCE_NAME
        read -p "Enter machine type: " MACHINE_TYPE
        DESCRIPTION="Custom - $MACHINE_TYPE"
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "🎯 Selected Configuration:"
echo "   Instance name: $INSTANCE_NAME"
echo "   Machine type: $MACHINE_TYPE"
echo "   Description: $DESCRIPTION"
echo ""

# Confirm creation
read -p "Create this instance? (y/N): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "❌ Instance creation cancelled."
    exit 0
fi

# Create the instance
create_instance $INSTANCE_NAME $MACHINE_TYPE "$DESCRIPTION"

# Get instance details
echo "📊 Instance Details:"
EXTERNAL_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "   Name: $INSTANCE_NAME"
echo "   Zone: $ZONE"
echo "   External IP: $EXTERNAL_IP"
echo "   Machine type: $MACHINE_TYPE"

# Wait for instance to be ready
echo ""
echo "⏳ Waiting for instance to be ready..."
sleep 30

# Test SSH connection
echo "🔗 Testing SSH connection..."
for i in {1..10}; do
    if gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="echo 'SSH OK'" >/dev/null 2>&1; then
        echo "   ✅ SSH connection established"
        break
    else
        echo "   ⏳ Attempt $i/10: SSH not ready yet..."
        sleep 10
    fi
done

echo ""
echo "✅ High-CPU instance created successfully!"
echo ""
echo "🚀 Next steps:"
echo "1. Update the INSTANCE_NAME in run_tep_gcp_high_cpu.sh to: $INSTANCE_NAME"
echo "2. Run the analysis: ./run_tep_gcp_high_cpu.sh"
echo ""
echo "📊 Monitor instance:"
echo "gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "💰 Estimated costs (approximate):"
case $MACHINE_TYPE in
    c2-standard-60)
        echo "   ~$15-20/hour (60 vCPUs, 240 GB RAM)"
        ;;
    c2-standard-30)
        echo "   ~$8-12/hour (30 vCPUs, 120 GB RAM)"
        ;;
    c2-standard-16)
        echo "   ~$4-6/hour (16 vCPUs, 64 GB RAM)"
        ;;
    n2-highcpu-80)
        echo "   ~$6-8/hour (80 vCPUs, 80 GB RAM)"
        ;;
    n2-highcpu-32)
        echo "   ~$2.5-4/hour (32 vCPUs, 32 GB RAM)"
        ;;
    n2-highcpu-16)
        echo "   ~$1.2-2/hour (16 vCPUs, 16 GB RAM)"
        ;;
esac
echo ""
echo "🛑 To stop instance:"
echo "gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "🗑️  To delete instance:"
echo "gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"

