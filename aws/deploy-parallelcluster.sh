#!/bin/bash
# Deploy AWS ParallelCluster for TEP-GNSS analysis
# Alternative to AWS Batch for more control over HPC environment

set -e

# Configuration
CLUSTER_NAME="tep-gnss-cluster"
REGION="us-east-1"

echo "Deploying AWS ParallelCluster for TEP-GNSS analysis..."

# Step 1: Install AWS ParallelCluster (if not already installed)
if ! command -v pcluster &> /dev/null; then
    echo "Installing AWS ParallelCluster..."
    pip install aws-parallelcluster
fi

# Step 2: Create cluster
echo "Creating ParallelCluster: ${CLUSTER_NAME}"
pcluster create-cluster \
    --cluster-name ${CLUSTER_NAME} \
    --cluster-configuration parallelcluster-config.yaml \
    --region ${REGION}

# Step 3: Wait for cluster to be ready
echo "Waiting for cluster to be ready..."
pcluster describe-cluster \
    --cluster-name ${CLUSTER_NAME} \
    --region ${REGION} \
    --query 'clusterStatus'

echo "ParallelCluster deployed successfully!"
echo ""
echo "To connect to the cluster:"
echo "pcluster ssh --cluster-name ${CLUSTER_NAME} --region ${REGION}"
echo ""
echo "To submit jobs:"
echo "pcluster ssh --cluster-name ${CLUSTER_NAME} --region ${REGION} -t 'sbatch /shared/tep-gnss-job.sh'"
echo ""
echo "To delete the cluster when done:"
echo "pcluster delete-cluster --cluster-name ${CLUSTER_NAME} --region ${REGION}"
