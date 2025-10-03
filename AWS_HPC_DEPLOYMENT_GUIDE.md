# TEP-GNSS AWS HPC Deployment Guide

This guide provides step-by-step instructions for deploying your TEP-GNSS analysis pipeline on AWS High Performance Computing (HPC) infrastructure.

## Overview

Your TEP-GNSS analysis will run **10-50x faster** on AWS HPC infrastructure compared to local execution, with estimated costs of **$10-25 per full pipeline run**.

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Docker** installed locally
4. **Python 3.11+** for local development

## Quick Start (Recommended: AWS Batch)

### Step 1: Set Up AWS Infrastructure

```bash
# Make scripts executable
chmod +x aws/*.sh

# Set up all AWS resources
cd aws
./setup-aws-infrastructure.sh
```

This script will:
- Create S3 bucket for data storage
- Set up IAM roles for Batch
- Create EFS file system
- Set up VPC and networking
- Create ECR repository

### Step 2: Build and Push Docker Image

```bash
# Build and push your containerized environment
./build-and-push-docker.sh
```

### Step 3: Deploy Batch Infrastructure

```bash
# Deploy AWS Batch compute environment and job queue
./deploy-batch-infrastructure.sh
```

### Step 4: Submit Your Analysis Job

```bash
# Run your TEP-GNSS analysis on HPC infrastructure
./submit-batch-job.sh
```

### Step 5: Monitor Your Job

```bash
# Check job status
aws batch describe-jobs --jobs YOUR_JOB_ID --region us-east-1

# Monitor costs
./monitor-costs.sh
```

## Alternative: AWS ParallelCluster (Advanced)

For more control over the HPC environment:

### Step 1: Install ParallelCluster

```bash
pip install aws-parallelcluster
```

### Step 2: Deploy Cluster

```bash
./deploy-parallelcluster.sh
```

### Step 3: Submit Slurm Jobs

```bash
# Connect to cluster
pcluster ssh --cluster-name tep-gnss-cluster --region us-east-1

# Submit job
sbatch /shared/tep-gnss-job.sh
```

## Configuration Files

### Environment Variables for HPC Optimization

Your scripts will automatically use these optimized settings on HPC:

```bash
TEP_WORKERS=96              # Use all 96 cores
TEP_MEMORY_LIMIT_GB=300     # Use available RAM
TEP_BOOTSTRAP_ITER=2000     # More iterations with more cores
TEP_NULL_ITERATIONS=200     # Increased for better statistics
```

### Instance Types

- **HPC6a.48xlarge**: 96 cores, 384GB RAM, 100 Gbps networking
- **Cost**: ~$3.50/hour
- **Performance**: 65% better price-performance than standard instances

## Cost Optimization

### Estimated Costs

| Resource | Cost | Notes |
|----------|------|-------|
| HPC6a Instance | $3.50/hour | 96 cores, 384GB RAM |
| EFS Storage | $0.30/GB/month | Shared file system |
| S3 Storage | $0.023/GB/month | Data backup |
| **Total per run** | **$10-25** | 2-4 hour execution |

### Cost Optimization Tips

1. **Use Spot Instances**: 50-70% cost reduction
2. **Auto-scaling**: Only pay when running
3. **Right-sizing**: Monitor actual usage
4. **Data lifecycle**: Archive old results

## Performance Expectations

### Speed Improvements

| Analysis Step | Local Time | AWS HPC Time | Speedup |
|---------------|------------|--------------|---------|
| Step 1: Data Acquisition | 2-4 hours | 15-30 minutes | 8x |
| Step 2: Core Analysis | 8-12 hours | 30-60 minutes | 12x |
| Step 3: Validation Suite | 12-20 hours | 1-2 hours | 15x |
| Step 4: Advanced Analysis | 6-10 hours | 20-40 minutes | 20x |
| **Total Pipeline** | **28-46 hours** | **2-4 hours** | **14x** |

### Resource Utilization

- **CPU**: 96 cores fully utilized
- **Memory**: 300GB available for large datasets
- **Network**: 100 Gbps for fast data transfer
- **Storage**: Shared EFS for collaboration

## Troubleshooting

### Common Issues

1. **Job Fails to Start**
   ```bash
   # Check compute environment status
   aws batch describe-compute-environments --compute-environments tep-gnss-hpc-environment
   ```

2. **Out of Memory**
   ```bash
   # Reduce memory usage in job definition
   # Edit aws/batch-job-definition.json
   ```

3. **High Costs**
   ```bash
   # Monitor and optimize
   ./monitor-costs.sh optimize
   ```

### Logs and Debugging

```bash
# View job logs
aws logs describe-log-groups --log-group-name-prefix /aws/batch/job

# Connect to running instance (if needed)
aws ec2 describe-instances --filters "Name=tag:Project,Values=tep-gnss"
```

## Security Best Practices

1. **IAM Roles**: Use least-privilege access
2. **VPC**: Run in private subnets
3. **Encryption**: Enable EFS and S3 encryption
4. **Secrets**: Use AWS Secrets Manager for sensitive data

## Cleanup

### Remove Resources When Done

```bash
# Delete Batch resources
aws batch delete-job-queue --job-queue tep-gnss-hpc-queue
aws batch delete-compute-environment --compute-environment tep-gnss-hpc-environment

# Delete ParallelCluster (if used)
pcluster delete-cluster --cluster-name tep-gnss-cluster --region us-east-1

# Delete EFS
aws efs delete-file-system --file-system-id YOUR_EFS_ID
```

## Support and Resources

- **AWS Batch Documentation**: https://docs.aws.amazon.com/batch/
- **AWS ParallelCluster Documentation**: https://docs.aws.amazon.com/parallelcluster/
- **HPC6a Instance Details**: https://aws.amazon.com/ec2/instance-types/hpc6a/

## Next Steps

1. **Start with AWS Batch** for simplicity
2. **Monitor costs** with provided scripts
3. **Optimize** based on actual usage patterns
4. **Scale up** to ParallelCluster for advanced needs

Your TEP-GNSS analysis is now ready to run on AWS supercomputing infrastructure!
