# Google Cloud Platform (GCP) Setup Instructions

## Step 1: Get Your GCP Instance Details

1. Go to **Google Cloud Console** → **Compute Engine** → **VM instances**
2. Find your instance and note:
   - **Project ID** (shown in the top bar)
   - **Zone** (e.g., `us-central1-a`)
   - **Instance name** (e.g., `tep-gnss-instance`)

## Step 2: Install Google Cloud CLI (if not already installed)

```bash
# macOS
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

## Step 3: Authenticate with GCP

```bash
# Login to your Google account
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

## Step 4: Configure the Deployment Script

Edit the `run_tep_gcp.sh` file and set these variables:

```bash
PROJECT_ID="your-project-id"        # Your GCP project ID
ZONE="us-central1-a"                # Your instance zone
INSTANCE_NAME="your-instance-name"  # Your instance name
```

## Step 5: Run the Analysis

```bash
# Execute the GCP deployment
./run_tep_gcp.sh
```

## Step 6: Monitor Progress

```bash
# Monitor the analysis in real-time
gcloud compute ssh YOUR_INSTANCE_NAME --zone=YOUR_ZONE --command='tail -f full_pipeline.log'

# Connect to the instance
gcloud compute ssh YOUR_INSTANCE_NAME --zone=YOUR_ZONE
```

## Step 7: Download Results

```bash
# Download results when analysis is complete
gcloud compute scp --recurse YOUR_INSTANCE_NAME:~/results/ ./gcp_results/ --zone=YOUR_ZONE
```

## Step 8: Clean Up

```bash
# Stop the instance (to avoid charges)
gcloud compute instances stop YOUR_INSTANCE_NAME --zone=YOUR_ZONE

# Delete the instance (permanent)
gcloud compute instances delete YOUR_INSTANCE_NAME --zone=YOUR_ZONE
```

## Expected Performance

- **Execution time**: 5-7 hours (vs 8-12 hours on Paperspace)
- **Performance gain**: 25-35% faster
- **Cost**: ~$6-10 total (depending on instance type)

## Troubleshooting

### SSH Connection Issues
```bash
# Check if instance is running
gcloud compute instances list

# Check firewall rules
gcloud compute firewall-rules list
```

### Permission Issues
```bash
# Check your permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID
```

### Instance Not Found
```bash
# List all instances
gcloud compute instances list --project=YOUR_PROJECT_ID
```


