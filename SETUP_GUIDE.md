# TEP-GNSS Setup Guide

## Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/matthewsmawfield/TEP-GNSS.git
cd TEP-GNSS
pip install -r requirements/requirements.txt
```

### 2. Configure Environment (Required for GCP deployment)

#### Option A: Environment File (Recommended)
```bash
# Copy the template
cp env.example .env.local

# Edit with your GCP details
nano .env.local
```

Add your GCP configuration:
```bash
GCP_PROJECT_ID=your-project-id-here
GCP_ZONE=us-central1-a
GCP_INSTANCE_NAME=your-instance-name-here
```

#### Option B: Environment Variables
```bash
export GCP_PROJECT_ID=your-project-id-here
export GCP_ZONE=us-central1-a
export GCP_INSTANCE_NAME=your-instance-name-here
```

### 3. Create GCP Instance (For GCP deployment)

Before running the GCP deployment script, you need to create a high-CPU instance through the GCP Console:

#### GCP Console Instance Creation Steps:

1. **Open GCP Console**: Go to [Google Cloud Console](https://console.cloud.google.com)

2. **Navigate to Compute Engine**: Go to Compute Engine → VM instances

3. **Create Instance**: Click "CREATE INSTANCE"

4. **Configure Instance**:
   - **Name**: Enter your instance name (same as `GCP_INSTANCE_NAME` in your .env.local)
   - **Region/Zone**: Select your preferred zone (same as `GCP_ZONE` in your .env.local)
   - **Machine Configuration**:
     - **Machine family**: General-purpose
     - **Series**: N2
     - **Machine type**: `n2-highcpu-96` (96 vCPUs, 96 GB memory)

5. **Configure Boot Disk**:
   - **Operating System**: Ubuntu
   - **Version**: Ubuntu 20.04 LTS
   - **Boot disk type**: Standard persistent disk
   - **Size (GB)**: 100 GB ⚠️ **Important: Change from default 10 GB to 100 GB**

6. **Configure Firewall**:
   - ✅ Allow HTTP traffic
   - ✅ Allow HTTPS traffic

7. **Click "CREATE"** to create the instance

8. **Wait for instance to start** (Status should show green checkmark)

### 4. Run Analysis

#### Local Analysis (No GCP required)
```bash
# Quick start - core analysis
python scripts/clean_run_step1_2.py

# Full pipeline
python scripts/clean_run_full_pipeline.py
```

#### GCP Deployment (High-performance)
```bash
# Deploy to GCP (requires configuration from step 2)
./run_tep_gcp_high_cpu.sh
```

## 5. Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT_ID` | - | Your GCP project ID (required for GCP deployment) |
| `GCP_ZONE` | - | Your GCP zone (required for GCP deployment) |
| `GCP_INSTANCE_NAME` | - | Your GCP instance name (required for GCP deployment) |
| `TEP_WORKERS` | 14 | Number of parallel workers |
| `TEP_MEMORY_LIMIT_GB` | 8.0 | Memory limit in GB |
| `TEP_BOOTSTRAP_ITER` | 1000 | Bootstrap iterations |
| `TEP_DATE_START` | 2023-01-01 | Analysis start date |
| `TEP_DATE_END` | 2025-06-30 | Analysis end date |

### Analysis Parameters

All analysis parameters can be overridden via environment variables. See `scripts/utils/config.py` for complete documentation.

## 6. Security Notes

- `.env.local` files are gitignored and never committed
- GCP credentials are loaded securely from environment variables
- No hardcoded credentials in the codebase
- Use `.env.local` for personal development, `.env` for shared team settings

## Troubleshooting

### GCP Configuration Issues
```bash
# Check your configuration
python scripts/utils/gcp_config.py

# Create template file
python -c "from scripts.utils.gcp_config import create_env_file_template; create_env_file_template()"
```

### Missing Dependencies
```bash
pip install -r requirements/requirements.txt
```

### Permission Issues
```bash
# Make scripts executable
chmod +x run_tep_gcp_high_cpu.sh
chmod +x gcp_*.sh
```
