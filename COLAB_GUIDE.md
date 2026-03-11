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
- [ ] **SSH key set up** (see Step 0 below — REQUIRED for team collaboration)
- [ ] Repository created and initialized

### Google Colab
- [ ] Google account
- [ ] Access to colab.research.google.com
- [ ] Optional: Google Drive for data backup

---

## Step 0: Setup Git SSH Authentication (REQUIRED FOR TEAM)

**Why SSH instead of tokens?**
- Each collaborator uses their own SSH key (no shared credentials)
- More secure than Personal Access Tokens
- Works seamlessly in both local and Colab
- Standard in professional teams

### 0.1 Generate SSH Key (Each Team Member)

**On your local machine:**

```bash
# Check if you already have a key
ls -la ~/.ssh/id_ed25519

# If not, generate one (each team member does this)
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter for all prompts (or set custom passphrase)

# Verify it was created
cat ~/.ssh/id_ed25519.pub
```

### 0.2 Add SSH Key to GitHub (Each Team Member)

1. Copy your public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

2. Go to [github.com/settings/keys](https://github.com/settings/keys)

3. Click **New SSH key**
   - Title: `MacBook` (or your machine name)
   - Key type: Authentication Key
   - Paste your public key
   - Click **Add SSH key**

4. Test connection:
   ```bash
   ssh -T git@github.com
   # Should output: Hi USERNAME! You've successfully authenticated...
   ```

**Each collaborator repeats steps 0.1-0.2 with their own SSH key.**

### 0.3 Configure Git (Each Team Member)

```bash
# Set your identity (use your real name/email)
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"

# Verify
git config --global --list | grep user
```

---

## Step 1: Setup Git Repository (Local)

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

# Add remote using SSH (not HTTPS)
git remote add origin git@github.com:YOUR_USERNAME/Mock_ML.git

# Push to remote
git branch -M main
git push -u origin main
```

✅ **Note:** Using SSH (`git@github.com:...`) instead of HTTPS means no credentials are stored in URLs. Each team member authenticates automatically with their own SSH key.

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
    "# Clone using SSH (no credentials needed if SSH key is set up in GitHub)\n",
    "import os\n",
    "os.chdir('/content')\n",
    "\n",
    "# Clone if not already cloned\n",
    "if not os.path.exists('Mock_ML'):\n",
    "    !git clone git@github.com:YOUR_USERNAME/Mock_ML.git\n",
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
    "# Configure Git identity for this Colab session\n",
    "!git config user.email \"your_email@example.com\"\n",
    "!git config user.name \"Your Name\"\n",
    "\n",
    "# Add training results\n",
    "!git add logs/train_loss.csv plots/training_loss.png\n",
    "\n",
    "# Commit\n",
    "!git commit -m \"Training results from Colab - GPU run\"\n",
    "\n",
    "# Push back using SSH (automatic authentication via SSH key)\n",
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

### 5.1 SSH Authentication in Colab

**Good news:** SSH keys work seamlessly in Colab without any extra configuration!

When you run `!git push origin main` in Colab, it automatically uses your SSH key from GitHub. No tokens needed.

**However** — SSH keys aren't automatically available in Colab's isolated environment. To push from Colab, you have **two options:**

**Option A: Use Your SSH Key (Recommended for Privacy)**

Create a Colab Secret with your SSH private key:
1. Local machine: Copy your private key
   ```bash
   cat ~/.ssh/id_ed25519
   ```

2. In Colab: Click **🔑 Secrets** (left sidebar) → Add `GITHUB_SSH_KEY`

3. In Colab notebook, before `git push`:
   ```python
   from google.colab import userdata
   import os

   # Get SSH key from secrets
   ssh_key = userdata.get('GITHUB_SSH_KEY')
   
   # Write to Colab's ssh directory
   os.makedirs('/root/.ssh', exist_ok=True)
   with open('/root/.ssh/id_ed25519', 'w') as f:
       f.write(ssh_key)
   os.chmod('/root/.ssh/id_ed25519', 0o600)
   
   # Add GitHub to known_hosts to skip fingerprint prompt
   !mkdir -p /root/.ssh
   !ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null
   
   # Now git operations will work
   !git push origin main
   ```

**Option B: Use GitHub Token as Fallback**

If you prefer not to handle SSH keys in Colab:
1. Create a Personal Access Token at [github.com/settings/tokens](https://github.com/settings/tokens)
   - Scopes: `repo` (full control)
   - This is a temporary, revocable credential

2. In Colab, add it as a secret: **🔑 Secrets** → Add `GITHUB_TOKEN`

3. Before pushing:
   ```python
   from google.colab import userdata
   
   token = userdata.get('GITHUB_TOKEN')
   !git remote set-url origin https://lollogabe:{token}@github.com/lollogabe/Mock_ML.git
   !git push origin main
   ```

**⚠️ Security Note:** Option A (SSH) is more secure because your private key is only exposed within your Colab environment. Option B exposes a token in Colab logs.

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
# 1. Setup SSH for git push (if using Option A from Step 5)
from google.colab import userdata
import os

ssh_key = userdata.get('GITHUB_SSH_KEY')
os.makedirs('/root/.ssh', exist_ok=True)
with open('/root/.ssh/id_ed25519', 'w') as f:
    f.write(ssh_key)
os.chmod('/root/.ssh/id_ed25519', 0o600)
!ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null

# 2. Clone repo using SSH
!git clone git@github.com:USERNAME/Mock_ML.git
%cd Mock_ML

# 3. Install & verify GPU
!pip install -q -r requirements.txt
import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')

# 4. Get data (downloads fresh each time, ~2 min)
!python scripts/preprocess.py --group 37

# 5. Train
!python scripts/train.py --config configs/config.yaml --device cuda

# 6. Download checkpoint
from google.colab import files
files.download('checkpoints/ae_best.pt')

# 7. Push logs back
!git config user.email "you@example.com"
!git config user.name "Your Name"
!git add logs/ plots/
!git commit -m "Results from Colab"
!git push origin main  # Works with SSH setup from above
```

---

## Team Collaboration Guide (For Your 3 Collaborators)

Since you're working with 2 other collaborators, here's how to coordinate smoothly:

### Setup (Do This Once Per Team Member)

**Each team member (you + 2 collaborators) should:**

1. **Generate SSH key** (Step 0.1)
   ```bash
   ssh-keygen -t ed25519 -C "their_email@example.com"
   ```

2. **Add SSH key to GitHub** (Step 0.2)
   - Go to [github.com/settings/keys](https://github.com/settings/keys)
   - Paste their public key (`cat ~/.ssh/id_ed25519.pub`)

3. **Configure Git identity** (Step 0.3)
   ```bash
   git config --global user.name "Their Name"
   git config --global user.email "their_email@example.com"
   ```

4. **Clone the repository**
   ```bash
   git clone git@github.com:lollogabe/Mock_ML.git
   cd Mock_ML
   ```

### Workflow: Three-Person Team

```
Person 1 (You)          Person 2               Person 3
├─ Local: Preprocess    ├─ Colab: Train        ├─ Colab: Train
├─ git push             ├─ git pull (code)     ├─ git pull (code)
│                       ├─ git push (logs)     ├─ git push (logs)
│                       │                      │
└─ git pull (results)───┼──────────────────────┴─ get results
   ├─ Local: Evaluate
   ├─ git push (eval plots)
   └─ share results
```

### Practical Example: First Training Run

**Day 1 - Setup (All Members)**
```bash
# Each person runs Step 0 (SSH authentication)
# Each person: git clone git@github.com:lollogabe/Mock_ML.git
```

**Day 2 - Prep (Person 1 - You)**
```bash
cd Mock_ML
bash setup.sh
source venv/bin/activate
python scripts/preprocess.py --group 37

git add data/.gitkeep  # Just tracking structure, not actual data
git commit -m "Data structure ready, actual files generated"
git push origin main
```

**Day 2-3 - Training (Person 2 or 3)**
```bash
# Pull latest code
git pull origin main

# In Colab: training_colab.ipynb
# - Clones from git@github.com:lollogabe/Mock_ML.git
# - Downloads data fresh (2 min)
# - Trains model (80 min)
# - Pushes logs and plots back: git push origin main

# After training, Person 1-3 can:
git pull origin main
# Get training logs and plots
```

**Day 3-4 - Evaluation (Person 1 - You)**
```bash
git pull origin main
# Get latest checkpoint (from Person 2's training) and plots

source venv/bin/activate
python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt

# Push evaluation results
git add plots/evaluation_*.png logs/evaluate.log
git commit -m "Evaluation complete - metrics and plots"
git push origin main

# Share with team:
echo "Results ready! Check plots/evaluation_*.png"
```

### Handling Conflicts (If Multiple People Push Together)

If two people push at the same time:

```bash
git pull origin main  # Fetch latest
# Git will merge automatically if changes don't conflict

# If conflict occurs:
git status  # See which files have conflicts
# Edit conflicting files manually
git add .
git commit -m "Merged training results from Person 2 and Person 3"
git push origin main
```

### Coordination Tips

1. **Use branches for experiments:**
   ```bash
   # Person 2 trying new architecture
   git checkout -b exp/new_encoder
   # Make changes
   git push origin exp/new_encoder
   
   # Others can try it:
   git checkout exp/new_encoder
   ```

2. **Label your commits clearly:**
   ```bash
   git commit -m "Training run #5 - batch_size=32 on Tesla A100
   
   - Trained for 20 epochs
   - Best validation loss: 0.187654
   - Training time: 45 min
   - Person: [Your Name]"
   ```

3. **Check who did what:**
   ```bash
   git log --oneline --author="Person Name"
   git log --graph --all --oneline --decorate
   ```

4. **Stay synced:**
   ```bash
   # Before starting work
   git pull origin main
   # Before pushing
   git pull origin main
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
!git clone -b exp/batch_size_32 git@github.com:lollogabe/Mock_ML.git
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
