# 🚀 Quick Start: Run TEP-GNSS in the Cloud (5 Minutes)

## Option 1: GitHub Codespaces (Recommended - Simplest)

### Step 1: Push to GitHub (2 minutes)
```bash
# If you haven't already
git init
git add .
git commit -m "Initial TEP-GNSS analysis"
git remote add origin https://github.com/YOUR_USERNAME/TEP-GNSS.git
git push -u origin main
```

### Step 2: Create Codespace (1 minute)
1. Go to your GitHub repo: `https://github.com/YOUR_USERNAME/TEP-GNSS`
2. Click the green **"Code"** button
3. Click **"Codespaces"** tab
4. Click **"Create codespace on main"**
5. Choose **32-core machine** ($0.18/hour)

### Step 3: Connect Cursor (1 minute)
1. Open Cursor
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows)
3. Type "Remote-SSH: Connect to Host"
4. Enter your Codespace SSH connection string
5. Cursor connects automatically!

### Step 4: Run Your Analysis (1 minute)
```bash
# In the Codespace terminal
python scripts/clean_run_full_pipeline.py
```

**That's it!** Your analysis now runs on 32 cores with 64GB RAM.

---

## Option 2: Google Cloud VM (Alternative)

### Step 1: Create VM (3 minutes)
```bash
# Install Google Cloud CLI first: https://cloud.google.com/sdk/docs/install
gcloud compute instances create tep-gnss-vm \
    --machine-type=e2-standard-32 \
    --zone=us-central1-a \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --preemptible  # 80% cheaper!
```

### Step 2: Connect with Cursor
1. Get VM IP: `gcloud compute instances describe tep-gnss-vm --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)'`
2. In Cursor: `Cmd+Shift+P` → "Remote-SSH: Connect to Host"
3. Enter: `ssh -i ~/.ssh/your-key username@VM_IP`

### Step 3: Setup Environment
```bash
# On the VM
sudo apt update
sudo apt install python3-pip git
git clone https://github.com/YOUR_USERNAME/TEP-GNSS.git
cd TEP-GNSS
pip3 install -r requirements/requirements.txt
```

### Step 4: Run Analysis
```bash
python3 scripts/clean_run_full_pipeline.py
```

---

## Cost Comparison

| Solution | Setup Time | Cost/Hour | Cores | RAM | Complexity |
|----------|------------|-----------|-------|-----|------------|
| **GitHub Codespaces** | 5 min | $0.18 | 32 | 64GB | ⭐ (Easiest) |
| **Google Cloud VM** | 10 min | $0.50 | 32 | 128GB | ⭐⭐ |
| **AWS Batch** | 60+ min | $3.50 | 96 | 384GB | ⭐⭐⭐⭐⭐ |

---

## Why These Are Better Than AWS Batch

✅ **No Docker setup needed**  
✅ **No complex AWS configuration**  
✅ **Cursor works exactly like local**  
✅ **Start/stop instantly**  
✅ **Pay only when running**  
✅ **Familiar development environment**  

---

## Quick Commands

### GitHub Codespaces
```bash
# Start codespace
gh codespace create --repo YOUR_USERNAME/TEP-GNSS

# Connect Cursor
# Use Remote-SSH with codespace connection string
```

### Google Cloud
```bash
# Start VM
gcloud compute instances start tep-gnss-vm --zone=us-central1-a

# Stop VM (saves money)
gcloud compute instances stop tep-gnss-vm --zone=us-central1-a
```

**Recommendation: Start with GitHub Codespaces - it's the simplest and works perfectly with Cursor!**
