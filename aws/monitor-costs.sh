#!/bin/bash
# Cost monitoring script for TEP-GNSS AWS HPC deployment
# Run this to monitor and optimize costs

set -e

# Configuration
REGION="us-east-1"
PROJECT_NAME="tep-gnss"

echo "TEP-GNSS AWS Cost Monitoring"
echo "============================"

# Function to get current costs
get_current_costs() {
    echo "Current AWS costs for TEP-GNSS resources:"
    echo ""
    
    # EC2 costs (HPC instances)
    echo "EC2 HPC Instance Costs:"
    aws ec2 describe-instances \
        --filters "Name=tag:Project,Values=${PROJECT_NAME}" \
        --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name,LaunchTime]' \
        --output table --region ${REGION}
    
    # Batch job costs
    echo ""
    echo "AWS Batch Job Status:"
    aws batch list-jobs \
        --job-queue tep-gnss-hpc-queue \
        --job-status RUNNING \
        --query 'jobSummaryList[*].[jobId,jobName,createdAt,status]' \
        --output table --region ${REGION}
    
    # EFS costs
    echo ""
    echo "EFS File System Usage:"
    aws efs describe-file-systems \
        --query 'FileSystems[*].[FileSystemId,SizeInBytes.Value,PerformanceMode]' \
        --output table --region ${REGION}
    
    # S3 costs
    echo ""
    echo "S3 Bucket Usage:"
    aws s3 ls s3://tep-gnss-data-bucket-* --region ${REGION} --summarize --human-readable
}

# Function to estimate costs
estimate_costs() {
    echo ""
    echo "Cost Estimates for TEP-GNSS Analysis:"
    echo "====================================="
    
    # HPC6a instance costs (approximate)
    HPC6A_HOURLY_RATE=3.50  # Approximate rate per hour
    ESTIMATED_HOURS=4        # Estimated runtime
    
    echo "HPC6a Instance (96 cores, 384GB RAM):"
    echo "  Hourly rate: \$${HPC6A_HOURLY_RATE}"
    echo "  Estimated runtime: ${ESTIMATED_HOURS} hours"
    echo "  Estimated cost: \$$(echo "$HPC6A_HOURLY_RATE * $ESTIMATED_HOURS" | bc)"
    
    # EFS costs
    EFS_GB_MONTHLY=0.30
    EFS_SIZE_GB=100
    
    echo ""
    echo "EFS Storage:"
    echo "  Size: ${EFS_SIZE_GB} GB"
    echo "  Monthly rate: \$${EFS_GB_MONTHLY} per GB"
    echo "  Monthly cost: \$$(echo "$EFS_GB_MONTHLY * $EFS_SIZE_GB" | bc)"
    
    # S3 costs
    S3_GB_MONTHLY=0.023
    S3_SIZE_GB=50
    
    echo ""
    echo "S3 Storage:"
    echo "  Size: ${S3_SIZE_GB} GB"
    echo "  Monthly rate: \$${S3_GB_MONTHLY} per GB"
    echo "  Monthly cost: \$$(echo "$S3_GB_MONTHLY * $S3_SIZE_GB" | bc)"
    
    echo ""
    echo "Total estimated cost per analysis: \$$(echo "$HPC6A_HOURLY_RATE * $ESTIMATED_HOURS" | bc)"
}

# Function to optimize costs
optimize_costs() {
    echo ""
    echo "Cost Optimization Recommendations:"
    echo "=================================="
    
    echo "1. Use Spot Instances:"
    echo "   - Can reduce costs by 50-70%"
    echo "   - Configure in batch-compute-environment.json"
    echo "   - Set bidPercentage to 50-70"
    
    echo ""
    echo "2. Auto-scaling:"
    echo "   - Set minvCpus to 0 in compute environment"
    echo "   - Instances only run when jobs are submitted"
    
    echo ""
    echo "3. Right-sizing:"
    echo "   - Monitor actual resource usage"
    echo "   - Adjust instance types based on needs"
    
    echo ""
    echo "4. Data lifecycle:"
    echo "   - Move old results to S3 Infrequent Access"
    echo "   - Delete temporary files regularly"
    
    echo ""
    echo "5. Reserved Instances (if running frequently):"
    echo "   - Consider 1-year or 3-year reservations"
    echo "   - Can save 30-60% on compute costs"
}

# Main execution
case "${1:-all}" in
    "costs")
        get_current_costs
        ;;
    "estimate")
        estimate_costs
        ;;
    "optimize")
        optimize_costs
        ;;
    "all")
        get_current_costs
        estimate_costs
        optimize_costs
        ;;
    *)
        echo "Usage: $0 [costs|estimate|optimize|all]"
        echo "  costs    - Show current AWS costs"
        echo "  estimate - Show cost estimates"
        echo "  optimize - Show optimization recommendations"
        echo "  all      - Show all information (default)"
        ;;
esac
