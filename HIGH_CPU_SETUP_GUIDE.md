# TEP-GNSS High-CPU GCP Setup Guide

## Overview
This guide helps you set up and run TEP-GNSS analysis on high-CPU GCP instances for maximum performance.

## Quick Start

### 1. Create High-CPU Instance
```bash
./create_high_cpu_instance.sh
```

### 2. Run Analysis
```bash
./run_tep_gcp_high_cpu.sh
```

## Instance Type Recommendations

| Instance Type | vCPUs | RAM | Use Case | Est. Cost/Hour |
|---------------|-------|-----|----------|----------------|
| **c2-standard-60** | 60 | 240 GB | Maximum Performance | $15-20 |
| **c2-standard-30** | 30 | 120 GB | High Performance | $8-12 |
| **c2-standard-16** | 16 | 64 GB | Good Performance | $4-6 |
| **n2-highcpu-80** | 80 | 80 GB | Maximum CPU Cores | $6-8 |
| **n2-highcpu-32** | 32 | 32 GB | High CPU Cores | $2.5-4 |
| **n2-highcpu-16** | 16 | 16 GB | Moderate CPU Cores | $1.2-2 |

## High-CPU Optimizations

### Automatic Optimizations
- **All CPU cores utilized**: `TEP_WORKERS` = number of vCPUs
- **Maximum parallel downloads**: `TEP_MAX_PARALLEL_DOWNLOADS` = number of vCPUs
- **Higher iteration counts**: 
  - Bootstrap iterations: 5000 (vs 1000 default)
  - Null test iterations: 500 (vs 100 default)
- **Checkpointing enabled**: `TEP_RESUME=1`
- **Thread optimization**: OMP, MKL, NUMEXPR threads set to vCPU count
- **Additional storage**: 200GB SSD disk mounted at `/mnt/data`

### Performance Benefits
- **2-5x faster** data acquisition with maximum parallel downloads
- **2-3x faster** analysis with all CPU cores
- **Higher statistical power** with increased bootstrap iterations
- **Resumable** with checkpointing
- **No disk space issues** with dedicated 200GB storage

## Configuration Files

### `create_high_cpu_instance.sh`
- Interactive instance creation
- Multiple instance type options
- Automatic disk setup (20GB boot + 200GB data)
- Cost estimates
- SSH connection testing

### `run_tep_gcp_high_cpu.sh`
- Optimized for high-CPU instances
- Automatic resource detection
- Environment variable optimization
- Real-time monitoring commands

## Usage Examples

### Create Maximum Performance Instance
```bash
./create_high_cpu_instance.sh
# Select option 1: c2-standard-60
```

### Create High CPU Core Instance
```bash
./create_high_cpu_instance.sh
# Select option 4: n2-highcpu-80
```

### Run Analysis
```bash
# Update INSTANCE_NAME in run_tep_gcp_high_cpu.sh first
./run_tep_gcp_high_cpu.sh
```

## Monitoring

### Real-time Progress
```bash
gcloud compute ssh tep-gnss-c2-60 --zone=us-central1-f --command='cd /mnt/data && tail -f full_pipeline.log'
```

### System Resources
```bash
gcloud compute ssh tep-gnss-c2-60 --zone=us-central1-f --command='htop'
```

### Disk Usage
```bash
gcloud compute ssh tep-gnss-c2-60 --zone=us-central1-f --command='df -h'
```

## Cost Management

### Stop Instance (Preserve Data)
```bash
gcloud compute instances stop tep-gnss-c2-60 --zone=us-central1-f
```

### Start Instance
```bash
gcloud compute instances start tep-gnss-c2-60 --zone=us-central1-f
```

### Delete Instance (Clean Up)
```bash
gcloud compute instances delete tep-gnss-c2-60 --zone=us-central1-f
```

## Expected Performance

### c2-standard-60 (60 vCPUs)
- **Data acquisition**: ~5-10 minutes (vs 30+ minutes on 32-core)
- **Analysis**: ~15-30 minutes (vs 60+ minutes on 32-core)
- **Total runtime**: ~30-60 minutes (vs 2+ hours on 32-core)

### n2-highcpu-80 (80 vCPUs)
- **Data acquisition**: ~3-8 minutes
- **Analysis**: ~10-25 minutes
- **Total runtime**: ~20-45 minutes

## Troubleshooting

### Instance Won't Start
```bash
# Check instance status
gcloud compute instances describe tep-gnss-c2-60 --zone=us-central1-f

# Check quotas
gcloud compute project-info describe --format="value(quotas[].limit,quotas[].usage)"
```

### SSH Connection Issues
```bash
# Test SSH
gcloud compute ssh tep-gnss-c2-60 --zone=us-central1-f --command="echo 'SSH OK'"

# Check firewall
gcloud compute firewall-rules list --filter="name~ssh"
```

### Disk Space Issues
```bash
# Check disk usage
gcloud compute ssh tep-gnss-c2-60 --zone=us-central1-f --command='df -h'

# Check mounted disks
gcloud compute ssh tep-gnss-c2-60 --zone=us-central1-f --command='lsblk'
```

## Best Practices

1. **Start with n2-highcpu-32** for cost-effective high performance
2. **Use c2-standard-60** for maximum performance when time is critical
3. **Stop instances** when not in use to save costs
4. **Monitor progress** with real-time logs
5. **Download results** immediately after completion
6. **Delete instances** when analysis is complete

## Comparison with Standard Instance

| Metric | Standard (32-core) | High-CPU (60-core) | High-CPU (80-core) |
|--------|-------------------|-------------------|-------------------|
| **Data Acquisition** | 30+ minutes | 5-10 minutes | 3-8 minutes |
| **Analysis** | 60+ minutes | 15-30 minutes | 10-25 minutes |
| **Total Runtime** | 2+ hours | 30-60 minutes | 20-45 minutes |
| **Cost** | ~$2-4/hour | ~$15-20/hour | ~$6-8/hour |
| **Value** | Good | Excellent | Best |

