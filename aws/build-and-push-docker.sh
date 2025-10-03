#!/bin/bash
# Build and push Docker image to AWS ECR
# Run this after setting up AWS infrastructure

set -e

# Configuration
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PROJECT_NAME="tep-gnss"
ECR_REPOSITORY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT_NAME}"

echo "Building and pushing Docker image for TEP-GNSS..."
echo "ECR Repository: ${ECR_REPOSITORY}"

# Step 1: Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Step 2: Build Docker image
echo "Building Docker image..."
docker build -t ${PROJECT_NAME}:latest .

# Step 3: Tag image for ECR
echo "Tagging image for ECR..."
docker tag ${PROJECT_NAME}:latest ${ECR_REPOSITORY}:latest

# Step 4: Push image to ECR
echo "Pushing image to ECR..."
docker push ${ECR_REPOSITORY}:latest

echo "Docker image successfully pushed to ECR!"
echo "Image URI: ${ECR_REPOSITORY}:latest"
