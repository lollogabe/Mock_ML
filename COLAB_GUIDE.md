# Using Google Colab with Git for Training

Complete guide to use Google Colab for GPU training while keeping preprocessing and evaluation on your local machine.

---

## Overview: Recommended Workflow

```
Local Machine
├── 1. Preprocess data (one-time or periodic)
│   └── python scripts/preprocess.py --group 37
│   └── data/ directory created locally
│
└── 2. Initialize Git repository
    └── git push to GitHub/GitLab

                    ↓ PUSH CODE

Google Colab
├── 1. Clone repository
├── 2. Setup environment
├── 3. Download data (or mount from Drive)
├── 4. Train model
│   └── Generate checkpoints in checkpoints/
│   └── Generate logs in logs/
│
└── 5. Push results back to Git

                    ↓ PULL RESULTS

Local Machine
└── 3. Evaluate results
    └── python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt
    └── Generate plots/
```

---

## Prerequisites

### Local Machine
- [ ] Project set up: `bash setup.sh`
- [ ] Git installed: `git --version`
- [ ] GitHub account (or GitLab, Gitea, etc.)
- [ ] Repository created and initialized

### Google Colab
- [ ] Google account
- [ ] Access to colab.research.google.com
- [ ] Optional: Google Drive for data backup

---

## Step 1: Setup Git Repository (Local)

### 1.1 Initialize Repository

```bash
cd /Users/lollogabe/Desktop/Mock_ML

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit initial snapshot
git commit -m "Initial commit: ML project setup"

# Add remote (choose one)
# GitHub:
git remote add origin https://github.com/YOUR_USERNAME/Mock_ML.git

# GitLab:
git remote add origin https://gitlab.com/YOUR_USERNAME/Mock_ML.git

# Push to remote
git branch -M main
git push -u origin main
```

### 1.2 Create .gitignore

Create `.gitignore` to exclude large files and temporary data:

```bash
cat > .gitignore << 'EOF'
# Virtual environments
venv/
env/
.venv

# Data (large files - download in Colab)
data/raw/*.npz
data/processed/

# Checkpoints (will be downloaded separately)
checkpoints/
logs/slurm_*.out

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Optional: exclude cached data but keep structure
data/.gitkeep
logs/.gitkeep
checkpoints/.gitkeep
plots/.gitkeep
EOF

git add .gitignore
git commit -m "Add gitignore for large files and environments"
git push
```

### 1.3 Optional: Add GitHub Workflow for Colab

Create `.github/workflows/sync.yml` to auto-pull latest code in Colab (optional):

```bash
mkdir -p .github/workflows

cat > .github/workflows/sync.yml << 'EOF'
name: Sync Colab

on:
  push:
    branches: [main, develop]

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Notify Colab
        run: echo "Changes pushed - Pull latest in Colab"
EOF

git add .github/workflows/sync.yml
git commit -m "Add GitHub Actions workflow"
git push
```

---

## Step 2: Preprocess Data (Local)

Run preprocessing once locally to generate data files:

```bash
# Activate environment
source venv/bin/activate

# Download and preprocess datasets
python scripts/preprocess.py --group 37

# Verify data was downloaded
ls -lh data/raw/
# Output: Normal_data.npz, Test_data_low.npz, Test_data_high.npz
```

**Option A: Keep Data Local (Recommended)**
- Don't commit data to Git (it's in .gitignore)
- Download fresh in Colab each time (takes ~2 min)
- Saves space and simplifies workflow

**Option B: Backup Data to Google Drive**
- If you want to reuse data across multiple Colab runs:
  ```bash
  # Upload locally:
  cp -r data/raw/ ~/Google\ Drive/Mock_ML_data/
  
  # In Colab, mount and link:
  from google.colab import drive
  drive.mount('/content/drive')
  !ln -s /content/drive/MyDrive/Mock_ML_data data/raw
  ```

---

## Step 3: Create Colab Notebook for Training

### 3.1 Create `training_colab.ipynb`

Create a new Jupyter notebook file locally, then push to Git:

```bash
cat > training_colab.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CERN Jet Anomaly Detection - Training in Colab\n",
    "\n",
    "This notebook trains the autoencoder on GPU."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 1: Mount Google Drive and Clone Repository"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mount Google Drive if needed for data/outputs\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "print('✓ Google Drive mounted')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Clone or pull latest code from Git\n",
    "import os\n",
    "os.chdir('/content')\n",
    "\n",
    "# Clone if not already cloned\n",
    "if not os.path.exists('Mock_ML'):\n",
    "    !git clone https://github.com/YOUR_USERNAME/Mock_ML.git\n",
    "    print('✓ Repository cloned')\n",
    "else:\n",
    "    os.chdir('/content/Mock_ML')\n",
    "    !git pull origin main\n",
    "    print('✓ Repository updated')\n",
    "\n",
    "os.chdir('/content/Mock_ML')\n",
    "!pwd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 2: Setup Environment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install dependencies\n",
    "!pip install -q -r requirements.txt\n",
    "print('✓ Dependencies installed')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Verify CUDA is available\n",
    "import torch\n",
    "print(f'CUDA available: {torch.cuda.is_available()}')\n",
    "if torch.cuda.is_available():\n",
    "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
    "    print(f'CUDA Version: {torch.version.cuda}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 3: Download Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download datasets\n",
    "!python scripts/preprocess.py --group 37\n",
    "print('✓ Data downloaded')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 4: Train Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Start training\n",
    "!python scripts/train.py --config configs/config.yaml --device cuda\n",
    "print('✓ Training complete')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Verify checkpoint was saved\n",
    "import os\n",
    "if os.path.exists('checkpoints/ae_best.pt'):\n",
    "    size_mb = os.path.getsize('checkpoints/ae_best.pt') / (1024**2)\n",
    "    print(f'✓ Checkpoint saved: ae_best.pt ({size_mb:.1f} MB)')\n",
    "    \n",
    "# Show training logs\n",
    "!tail -5 logs/train_loss.csv"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 5: Upload Results to Google Drive (Optional)\n",
    "\n",
    "Only needed if you want to backup results. For Git push, skip this."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Option A: Copy to Drive for backup\n",
    "!cp -r checkpoints /content/drive/MyDrive/Mock_ML_checkpoints_colab_run/\n",
    "!cp -r logs /content/drive/MyDrive/Mock_ML_logs_colab_run/\n",
    "print('✓ Results backed up to Google Drive')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 6: Push Results to Git"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configure Git (first time only)\n",
    "!git config user.email \"your_email@example.com\"\n",
    "!git config user.name \"Your Name\"\n",
    "\n",
    "# Add results\n",
    "!git add logs/train_loss.csv plots/training_loss.png\n",
    "!git commit -m \"Training results from Colab - GPU run\"\n",
    "\n",
    "# Push back to repository\n",
    "# Note: You'll need to provide credentials via GitHub Personal Access Token\n",
    "!git push origin main\n",
    "print('✓ Results pushed to Git')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Step 7: Download Checkpoint Locally\n",
    "\n",
    "Show instructions for downloading to local machine"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"\"\"\\n✓ Training Complete!\\n\n",
    "Next steps on your local machine:\\n\n",
    "1. Pull latest code and logs:\\n\n",
    "   git pull origin main\\n\n",
    "2. Download checkpoint from Colab:\\n\n",
    "   scp colab:/content/Mock_ML/checkpoints/ae_best.pt ./checkpoints/\\n\n",
    "   OR check Google Drive for backup\\n\n",
    "3. Evaluate locally:\\n\n",
    "   python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt\\n\n",
    "4. View training loss plot:\\n\n",
    "   open plots/training_loss.png\\n\"\"\")"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": [],
   "gpuType": "T4"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
EOF

git add training_colab.ipynb
git commit -m "Add Colab training notebook with Git integration"
git push
```

### 3.2 Or Use the Notebook directly in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File** → **Open notebook** → **GitHub**
3. Paste: `https://github.com/YOUR_USERNAME/Mock_ML`
4. Select `training_colab.ipynb`
5. Make sure GPU is enabled: **Runtime** → **Change runtime type** → **GPU**

---

## Step 4: Run Training in Colab

### 4.1 Execute Notebook in Colab

Open the notebook from your GitHub and run cells in order:

```
1. ✓ Mount Google Drive
2. ✓ Clone/pull repository
3. ✓ Install dependencies (~2-3 min)
4. ✓ Verify CUDA available
5. ✓ Download data (~2 min)
6. ⏳ Train model (~15-30 min for 20 epochs on T4 GPU)
7. ✓ Save checkpoint
8. ✓ Push results to Git
```

### 4.2 Expected Output

```
==> CUDA target: cu121
==> Creating virtual environment...
==> Installing PyTorch...
==> Installing project requirements...

  torch      : 2.0.0  (CUDA: True)
  GPU        : Tesla T4

✓ Data downloaded
Epoch 001/020  train_loss=0.452354  val_loss=0.433287  time=12.3s
Epoch 002/020  train_loss=0.387654  val_loss=0.391548  time=12.1s
...
Epoch 020/020  train_loss=0.201234  val_loss=0.204567  time=11.9s

Training complete — best loss: 0.201234
Training loss plot saved → plots/training_loss.png
```

---

## Step 5: Push Results Back to Git

### 5.1 Authenticate with GitHub

In Colab, you need a Personal Access Token for authentication:

**Generate Token on GitHub:**
1. GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. Click **Generate new token**
3. Name it: `Colab-Training`
4. Select scopes: `repo` (full control)
5. Generate and copy token

**Use Token in Colab:**
```python
# In Colab notebook, before first git push:
import os
os.environ['GIT_AUTHOR_NAME'] = 'Colab-Agent'
os.environ['GIT_AUTHOR_EMAIL'] = 'your_email@example.com'
os.environ['GIT_COMMITTER_NAME'] = 'Colab-Agent'
os.environ['GIT_COMMITTER_EMAIL'] = 'your_email@example.com'

# Create .netrc for authentication
netrc_content = f"""machine github.com
login YOUR_USERNAME
password YOUR_PERSONAL_ACCESS_TOKEN
"""

with open(os.path.expanduser('~/.netrc'), 'w') as f:
    f.write(netrc_content)
os.chmod(os.path.expanduser('~/.netrc'), 0o600)

# Now git operations will authenticate automatically
!git push origin main
```

### 5.2 What to Push Back

**Important data to push:**
```bash
# Files to include in git push:
git add logs/train_loss.csv          # Training history
git add plots/training_loss.png      # Loss visualization
git add configs/config.yaml          # Config used

# Files to NOT push (too large):
# - checkpoints/*.pt (push separately)
# - data/raw/*.npz (redownload in Colab)

git commit -m "Training results from Colab - GPU run completed

- Trained for 20 epochs on Tesla T4
- Best validation loss: 0.204567
- Training loss plot saved to plots/training_loss.png
- Checkpoint: ae_best.pt (download separately)
"

git push origin main
```

---

## Step 6: Download Checkpoint Locally

### 6.1 Direct Download from Colab

**Option A: During Notebook Execution**

In the last cell of the Colab notebook:
```python
# Download checkpoint to local via browser
from google.colab import files
files.download('checkpoints/ae_best.pt')
```

Then Colab will prompt you to save to Downloads folder.

**Option B: From Google Drive**

If you backed up to Drive:
1. Go to [drive.google.com](https://drive.google.com)
2. Find folder: `Mock_ML_checkpoints_colab_run`
3. Right-click `ae_best.pt` → Download
4. Move to local: `checkpoints/ae_best.pt`

**Option C: Via SSH (if you have an instance)**

```bash
# From local machine
scp username@instance_ip:/content/Mock_ML/checkpoints/ae_best.pt ./checkpoints/
```

---

## Step 7: Evaluate Locally

### 7.1 Pull Latest Results from Git

```bash
cd /Users/lollogabe/Desktop/Mock_ML

# Pull training results (logs, plots)
git pull origin main

# View training loss plot
open plots/training_loss.png

# Check training history
cat logs/train_loss.csv | tail -5
```

### 7.2 Run Evaluation

```bash
# Activate environment
source venv/bin/activate

# Full evaluation with all plots
python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt

# Or skip slow UMAP
python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt --no-umap

# Or just metrics (no plots)
python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt \
  --no-dimensionality --no-gmm --no-plot
```

### 7.3 View Results

```bash
# Evaluation plots generated
ls -lh plots/

# Open main plots
open plots/anomaly_scores_mse.png
open plots/anomaly_scores_mahalanobis.png
open plots/pca_projection.png  # or with UMAP
```

---

## Complete Workflow: Step-by-Step

### Workflow Diagram

```
Day 1: Setup
├── Local: bash setup.sh
├── Local: python scripts/preprocess.py --group 37
├── Local: git init && git push
└── Colab: Fork GitHub repo

Day 2-N: Iterations
├── Local (5 min)
│   ├── Make code changes
│   ├── Test locally: python -m pytest tests/ -v
│   └── git commit && git push
│
├── Colab (20-30 min)
│   ├── Run training_colab.ipynb
│   ├── Monitor training loss
│   └── Download ae_best.pt
│
└── Local (10-15 min)
    ├── git pull (get logs/plots)
    ├── Place ae_best.pt in checkpoints/
    └── python scripts/evaluate.py
```

### Example: Full Training Cycle

**Time breakdown for 20 epochs on Google Colab T4:**
- Setup environment: ~3 min
- Download data: ~2 min
- Training: ~4 min/epoch × 20 = ~80 min
- **Total: ~85-90 min (~1.5 hours)**

**Local evaluation:**
- Download checkpoint: ~2 min
- Evaluate: ~5-10 min (depending on UMAP)

---

## Troubleshooting

### Issue: Git Authentication Fails

**Solution:**
```python
# In Colab, set up authentication:
!git config --global credential.helper store

# Paste your token when prompted, or use:
import subprocess
token = "ghp_YOUR_TOKEN_HERE"
subprocess.run(f"git remote set-url origin https://{token}@github.com/USERNAME/Mock_ML.git", shell=True)
```

### Issue: Out of Memory During Training

**Solution:**
```bash
# In Colab, reduce batch size
!python scripts/train.py --config configs/config.yaml \
  --device cuda --batch_size 32

# Or reduce number of epochs for testing
!python scripts/train.py --config configs/config.yaml \
  --device cuda --epochs 5  # Quick test run
```

### Issue: CUDA Not Available

**Solution:**
1. Check runtime: **Runtime** → **Change runtime type**
2. Select **GPU** (T4, A100, or V100)
3. Click **Save**
4. Restart runtime

### Issue: Checkpoint Too Large to Download

```python
# In Colab, compress checkpoint before download
!zip -j checkpoints.zip checkpoints/ae_best.pt
from google.colab import files
files.download('checkpoints.zip')

# Locally:
unzip checkpoints.zip -d checkpoints/
```

### Issue: Data Download Fails

```python
# In Colab, increase timeout:
import socket
socket.setdefaulttimeout(300)  # 5 minutes

# Then retry
!python scripts/preprocess.py --group 37
```

---

## Advanced: Automate Training Pipeline

### Create Workflow Script for Colab

Create `colab_train.py` to run everything non-interactively:

```python
#!/usr/bin/env python3
"""
Automated training script for Google Colab.
Run this from Colab terminal or directly call it.
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Execute shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Error: {description} failed")
        sys.exit(1)
    print(f"✓ {description} complete")

def main():
    # 1. Setup
    run_command("pip install -q -r requirements.txt", "Installing dependencies")
    
    # 2. Download data
    run_command("python scripts/preprocess.py --group 37", "Downloading datasets")
    
    # 3. Train
    run_command(
        "python scripts/train.py --config configs/config.yaml --device cuda",
        "Training model (20 epochs)"
    )
    
    # 4. Verify checkpoint
    if os.path.exists("checkpoints/ae_best.pt"):
        size_mb = os.path.getsize("checkpoints/ae_best.pt") / (1024**2)
        print(f"\n✓ Checkpoint saved: ae_best.pt ({size_mb:.1f} MB)")
    else:
        print("\n❌ Checkpoint not found!")
        sys.exit(1)
    
    # 5. Git push (optional)
    print("\n" + "="*60)
    print("  Pushing results to Git")
    print("="*60)
    os.system("git config user.email 'colab@example.com'")
    os.system("git config user.name 'Colab-Agent'")
    os.system("git add logs/train_loss.csv plots/training_loss.png")
    os.system("git commit -m 'Training results from Colab GPU run'")
    os.system("git push origin main 2>/dev/null || echo 'Push requires authentication token'")
    
    print("\n" + "="*60)
    print("  ✅ Training Complete!")
    print("="*60)
    print("\nNext: Download ae_best.pt and evaluate locally")

if __name__ == "__main__":
    main()
```

Then in Colab:
```python
!python colab_train.py
```

---

## Best Practices

### ✅ Do This

- **Test locally first** before running in Colab:
  ```bash
  python -m pytest tests/ -v
  python scripts/train.py --epochs 2 --device cpu
  ```

- **Use Git to track experiments:**
  ```bash
  git checkout -b exp/new_architecture
  # Make changes
  git commit -m "Try new encoder design"
  git push origin exp/new_architecture
  ```

- **Save logs and metrics:**
  ```bash
  git add logs/train_loss.csv
  git add plots/training_loss.png
  git commit -m "Training metrics from run #5"
  ```

- **Keep secrets safe:**
  ```bash
  # Add to .gitignore:
  .env
  secrets.json
  colab_token.txt
  
  # Use secrets in Colab:
  from google.colab import userdata
  token = userdata.get('GITHUB_TOKEN')
  ```

### ❌ Don't Do This

- ❌ Push large checkpoints to Git (use Drive or direct download instead)
- ❌ Keep sensitive credentials in Colab notebooks
- ❌ Run large data downloads repeatedly (download once, cache)
- ❌ Train for long hours without monitoring (set up notifications)
- ❌ Forget to pull latest code before each Colab run

---

## Summary Table

| Task | Where | Command |
|------|-------|---------|
| Setup | Local | `bash setup.sh` |
| Preprocess | Local (once) | `python scripts/preprocess.py` |
| Version control | Local + Git | `git commit && git push` |
| **Training** | **Colab GPU** | **`training_colab.ipynb`** |
| Get logs | Local | `git pull && cat logs/train_loss.csv` |
| Get checkpoint | Colab → Local | Download via browser or Drive |
| Evaluate | Local | `python scripts/evaluate.py` |
| Save results | Git | `git add && git push` |

---

## Quick Reference: Command Cheatsheet

### Local Machine

```bash
# First time setup
bash setup.sh && source venv/bin/activate

# Initialize Git
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/Mock_ML.git
git push -u origin main

# After training in Colab
git pull origin main             # Get logs and plots
python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt
```

### Google Colab Notebook

```python
# 1. Clone repo
!git clone https://github.com/USERNAME/Mock_ML.git
%cd Mock_ML

# 2. Install & verify GPU
!pip install -q -r requirements.txt
import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')

# 3. Get data (downloads fresh each time, ~2 min)
!python scripts/preprocess.py --group 37

# 4. Train
!python scripts/train.py --config configs/config.yaml --device cuda

# 5. Download checkpoint
from google.colab import files
files.download('checkpoints/ae_best.pt')

# 6. Push logs back (optional)
!git config user.email "you@example.com"
!git config user.name "Your Name"
!git add logs/ plots/
!git commit -m "Results from Colab"
!git push origin main  # Requires token setup
```

---

## For Multiple Runs / Experiments

If you want to train multiple times with different hyperparameters:

```bash
# Local: Branch for each experiment
git checkout -b exp/batch_size_32
# Edit configs/config.yaml or scripts/train.py
git push origin exp/batch_size_32

# Colab: Pull specific branch
!git clone -b exp/batch_size_32 https://github.com/USERNAME/Mock_ML.git
!python scripts/train.py

# Local: Compare results
git checkout main
git checkout exp/batch_size_32
# Review logs and plots for each
```

---

## Support & Next Steps

For more details, see:
- [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) — Full workflow guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) — Code changes
- Requirements: [requirements.txt](requirements.txt)

Happy training! 🚀
