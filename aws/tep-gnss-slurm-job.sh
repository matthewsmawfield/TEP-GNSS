#!/bin/bash
#SBATCH --job-name=tep-gnss-analysis
#SBATCH --partition=hpc-queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/logs/tep-gnss-%j.out
#SBATCH --error=/shared/logs/tep-gnss-%j.err

# TEP-GNSS Analysis Job Script for Slurm
# Optimized for HPC6a instances (96 cores, 384GB RAM)

echo "Starting TEP-GNSS analysis on HPC infrastructure..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"

# Set environment variables for HPC optimization
export TEP_WORKERS=96
export TEP_MEMORY_LIMIT_GB=300
export TEP_BOOTSTRAP_ITER=2000
export TEP_NULL_ITERATIONS=200
export PYTHONUNBUFFERED=1

# Create logs directory
mkdir -p /shared/logs

# Change to project directory
cd /shared/tep-gnss

# Run the full pipeline
echo "Executing TEP-GNSS full pipeline..."
python scripts/clean_run_full_pipeline.py

# Check exit status
if [ $? -eq 0 ]; then
    echo "TEP-GNSS analysis completed successfully!"
    echo "End time: $(date)"
else
    echo "TEP-GNSS analysis failed!"
    echo "End time: $(date)"
    exit 1
fi

# Copy results to S3 (if configured)
if [ ! -z "$S3_BUCKET" ]; then
    echo "Uploading results to S3..."
    aws s3 sync results/ s3://$S3_BUCKET/results/$(date +%Y%m%d-%H%M%S)/
fi

echo "Job completed at: $(date)"
