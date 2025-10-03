#!/bin/bash
# Submit TEP-GNSS analysis job to AWS Batch
# Run this to execute your analysis on HPC infrastructure

set -e

# Configuration
REGION="us-east-1"
PROJECT_NAME="tep-gnss"
JOB_NAME="tep-gnss-analysis-$(date +%Y%m%d-%H%M%S)"

echo "Submitting TEP-GNSS analysis job to AWS Batch..."
echo "Job Name: ${JOB_NAME}"

# Submit job
JOB_ID=$(aws batch submit-job \
    --job-name ${JOB_NAME} \
    --job-queue tep-gnss-hpc-queue \
    --job-definition tep-gnss-analysis \
    --region ${REGION} \
    --query 'jobId' --output text)

echo "Job submitted successfully!"
echo "Job ID: ${JOB_ID}"
echo ""
echo "Monitor job status with:"
echo "aws batch describe-jobs --jobs ${JOB_ID} --region ${REGION}"
echo ""
echo "View job logs with:"
echo "aws logs describe-log-groups --log-group-name-prefix /aws/batch/job --region ${REGION}"
echo ""
echo "To cancel the job:"
echo "aws batch cancel-job --job-id ${JOB_ID} --reason 'User requested cancellation' --region ${REGION}"
