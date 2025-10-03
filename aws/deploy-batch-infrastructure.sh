#!/bin/bash
# Deploy AWS Batch infrastructure
# Run this after building and pushing Docker image

set -e

# Configuration
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PROJECT_NAME="tep-gnss"

echo "Deploying AWS Batch infrastructure..."

# Step 1: Create compute environment
echo "Creating compute environment..."
aws batch create-compute-environment \
    --cli-input-json file://batch-compute-environment.json \
    --region ${REGION} || echo "Compute environment may already exist"

# Wait for compute environment to be ready
echo "Waiting for compute environment to be ready..."
aws batch describe-compute-environments \
    --compute-environments tep-gnss-hpc-environment \
    --region ${REGION} \
    --query 'computeEnvironments[0].status' \
    --output text

# Step 2: Create job queue
echo "Creating job queue..."
aws batch create-job-queue \
    --cli-input-json file://batch-job-queue.json \
    --region ${REGION} || echo "Job queue may already exist"

# Step 3: Register job definition
echo "Registering job definition..."
aws batch register-job-definition \
    --cli-input-json file://batch-job-definition.json \
    --region ${REGION}

echo "AWS Batch infrastructure deployed successfully!"
echo ""
echo "You can now submit jobs using: ./submit-batch-job.sh"
