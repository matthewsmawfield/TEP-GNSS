#!/bin/bash
# AWS Infrastructure Setup Script for TEP-GNSS HPC Deployment
# Run this script to set up all necessary AWS resources

set -e

# Configuration
REGION="us-east-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PROJECT_NAME="tep-gnss"
BUCKET_NAME="${PROJECT_NAME}-data-bucket-${ACCOUNT_ID}"

echo "Setting up AWS infrastructure for TEP-GNSS HPC deployment..."
echo "Account ID: ${ACCOUNT_ID}"
echo "Region: ${REGION}"

# Step 1: Create S3 bucket for data storage
echo "Creating S3 bucket: ${BUCKET_NAME}"
aws s3 mb s3://${BUCKET_NAME} --region ${REGION} || echo "Bucket may already exist"

# Step 2: Create IAM roles for Batch
echo "Creating IAM roles for AWS Batch..."

# Batch service role
aws iam create-role \
    --role-name ${PROJECT_NAME}-batch-service-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "batch.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }' || echo "Service role may already exist"

aws iam attach-role-policy \
    --role-name ${PROJECT_NAME}-batch-service-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole || echo "Policy may already be attached"

# Batch instance role
aws iam create-role \
    --role-name ${PROJECT_NAME}-batch-instance-role \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "ec2.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }' || echo "Instance role may already exist"

aws iam attach-role-policy \
    --role-name ${PROJECT_NAME}-batch-instance-role \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role || echo "Policy may already be attached"

# Create instance profile
aws iam create-instance-profile \
    --instance-profile-name ${PROJECT_NAME}-batch-instance-role || echo "Instance profile may already exist"

aws iam add-role-to-instance-profile \
    --instance-profile-name ${PROJECT_NAME}-batch-instance-role \
    --role-name ${PROJECT_NAME}-batch-instance-role || echo "Role may already be in instance profile"

# Step 3: Create EFS file system
echo "Creating EFS file system..."
EFS_ID=$(aws efs create-file-system \
    --creation-token ${PROJECT_NAME}-$(date +%s) \
    --performance-mode generalPurpose \
    --throughput-mode provisioned \
    --provisioned-throughput-in-mibps 1000 \
    --encrypted \
    --tags Key=Name,Value=${PROJECT_NAME}-efs Key=Project,Value=${PROJECT_NAME} \
    --query 'FileSystemId' --output text 2>/dev/null || echo "EFS may already exist")

if [ -z "$EFS_ID" ]; then
    EFS_ID=$(aws efs describe-file-systems \
        --query 'FileSystems[?CreationToken==`'${PROJECT_NAME}'-*`].FileSystemId' \
        --output text | head -1)
fi

echo "EFS ID: ${EFS_ID}"

# Step 4: Create VPC and networking (if needed)
echo "Setting up VPC and networking..."
VPC_ID=$(aws ec2 describe-vpcs \
    --filters "Name=tag:Name,Values=default" \
    --query 'Vpcs[0].VpcId' --output text)

if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
    echo "Creating VPC..."
    VPC_ID=$(aws ec2 create-vpc \
        --cidr-block 10.0.0.0/16 \
        --query 'Vpc.VpcId' --output text)
    
    aws ec2 create-tags \
        --resources ${VPC_ID} \
        --tags Key=Name,Value=${PROJECT_NAME}-vpc
fi

# Get subnet
SUBNET_ID=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=${VPC_ID}" \
    --query 'Subnets[0].SubnetId' --output text)

# Get security group
SECURITY_GROUP_ID=$(aws ec2 describe-security-groups \
    --filters "Name=vpc-id,Values=${VPC_ID}" "Name=group-name,Values=default" \
    --query 'SecurityGroups[0].GroupId' --output text)

# Step 5: Create ECR repository
echo "Creating ECR repository..."
aws ecr create-repository \
    --repository-name ${PROJECT_NAME} \
    --region ${REGION} || echo "ECR repository may already exist"

# Get ECR login token
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo "Infrastructure setup complete!"
echo ""
echo "Next steps:"
echo "1. Update configuration files with the following values:"
echo "   - ACCOUNT_ID: ${ACCOUNT_ID}"
echo "   - REGION: ${REGION}"
echo "   - VPC_ID: ${VPC_ID}"
echo "   - SUBNET_ID: ${SUBNET_ID}"
echo "   - SECURITY_GROUP_ID: ${SECURITY_GROUP_ID}"
echo "   - EFS_ID: ${EFS_ID}"
echo "   - BUCKET_NAME: ${BUCKET_NAME}"
echo ""
echo "2. Run: ./build-and-push-docker.sh"
echo "3. Run: ./deploy-batch-infrastructure.sh"
echo "4. Run: ./submit-batch-job.sh"
