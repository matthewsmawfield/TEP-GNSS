# Simple Cloud Setup for TEP-GNSS Analysis

## Option 1: Cursor Remote Development (Easiest)

### Step 1: Create a Cloud VM
```bash
# Using Google Cloud (free tier available)
gcloud compute instances create tep-gnss-vm \
    --machine-type=e2-standard-32 \
    --zone=us-central1-a \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB

# Or using AWS EC2 (simpler than Batch)
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type c6a.8xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-your-security-group
```

### Step 2: Connect with Cursor Remote
1. Open Cursor
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows)
3. Type "Remote-SSH: Connect to Host"
4. Add your VM's IP address
5. Cursor will connect and you can work exactly like local!

### Step 3: Run Your Scripts
```bash
# On the remote VM, just run your existing scripts
python scripts/clean_run_full_pipeline.py
```

**Benefits:**
- ✅ Works exactly like your local setup
- ✅ No Docker or complex AWS setup needed
- ✅ Cursor's full features work remotely
- ✅ 32 cores, 128GB RAM for ~$1-2/hour
- ✅ Can stop/start as needed

## Option 2: Google Colab Pro (Even Simpler)

### Step 1: Upload to Google Drive
```bash
# Zip your project
zip -r tep-gnss-project.zip . -x "venv/*" "data/raw/*" "logs/*" "results/*"
# Upload to Google Drive
```

### Step 2: Use Colab Pro
- $10/month for Colab Pro
- 32GB RAM, high-end GPUs
- Direct file access from Drive
- No setup needed!

## Option 3: GitHub Codespaces (Simplest)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Add TEP-GNSS analysis"
git push origin main
```

### Step 2: Create Codespace
1. Go to your GitHub repo
2. Click "Code" → "Codespaces" → "Create codespace"
3. Choose 32-core machine ($0.18/hour)
4. Cursor can connect to Codespaces!

**Benefits:**
- ✅ No setup at all
- ✅ Pre-configured environment
- ✅ Pay only when using
- ✅ Works with Cursor remote development
