# Running on Google Colab — Complete Guide

> Updated workflow: cleaner, tested against actual Colab environment restrictions

## TL;DR — Quick Start

```python
# Cell 1: Clone (HTTPS — simplest, no auth needed for public repos)
import os
os.chdir('/content')
!git clone https://github.com/lollogabe/Mock_ML.git
%cd Mock_ML

# Cell 2: Setup
!python colab_setup.py --setup

# Cell 3: Download Data
!python scripts/preprocess.py --group 37

# Cell 4: Train (35-45 min on T4 GPU)
!python scripts/train.py --config configs/config.yaml --device cuda

# Cell 5: Evaluate
!python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt --device cuda --no-plot

# Cell 6: (OPTIONAL) Push results back to Git
# Only needed if you want to save training logs to repo
# See "Pushing Results Back" section below
```

---

## Why Colab Is Different (And How We Handle It)

| Aspect | Local / HPC | Colab | Solution |
|--------|-------------|-------|----------|
| **Shell Scripts** | ✅ Works | ❌ Limited | Use Python helper (`colab_setup.py`) |
| **Package Installation** | Manual | ✅ Automatic Python env | Use `pip` directly |
| **Git SSH Keys** | System-wide | ⚠️ Isolated sandbox | Store key in Secrets, load in Python |
| **File System** | Persistent | 🔄 Ephemeral | Download results before session ends |
| **GPU** | Optional | ✅ Always available | Auto-detect and use |

---

## Prerequisites

### **Before You Start**

1. **Repository on GitHub:** Push your project to GitHub/GitLab (required for cloning in Colab)
   ```bash
   # On your local machine
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/Mock_ML.git
   git push -u origin main
   ```

2. **SSH Key Set Up (for git@github.com authentication):**
   - Generate key locally: `ssh-keygen -t ed25519 -C "your_email@example.com"`
   - Add to GitHub: [github.com/settings/keys](https://github.com/settings/keys)
   - OR use token-based auth (see Step 1.2 below)

---

## Step 1: Setup Colab Environment

### **Usually You Need Nothing!**

For **cloning** the repo:
- ✅ Public repo: Just use HTTPS (no authentication)
- ✅ Private repo: Use token (see below)

```python
import os
os.chdir('/content')

# Public repo — no auth needed
!git clone https://github.com/lollogabe/Mock_ML.git
%cd Mock_ML

print("✓ Repository cloned")
```

---

### **Option A: GitHub Token (For Pushing Results Back)**

Use a **Personal Access Token** if you want to **push training logs back to Git**.

#### **Step 1a: Create Token on GitHub**

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Set:
   - **Token name:** `Colab-Training`
   - **Expiration:** 90 days (or longer)
   - **Scopes:** Check only ✅ `repo` (full control of private repositories)
4. Click **"Generate token"**
5. **Copy the token** (it only shows once!)

#### **Step 1b: Store Token in Colab Secrets**

1. In your Colab notebook, click **🔑 Secrets** (left sidebar)
2. Click **+ Add new secret**
3. **Name:** `GITHUB_TOKEN`
4. **Value:** Paste your token
5. Click **"Add secret"**

#### **Step 1c: Clone with Token**

```python
from google.colab import userdata
import os

os.chdir('/content')

# Get token from Secrets
token = userdata.get('GITHUB_TOKEN')

# Clone using token (works for both public and private repos)
!git clone https://{token}@github.com/lollogabe/Mock_ML.git
%cd Mock_ML

print("✓ Repository cloned with token authentication")
```

Then continue to Step 2 (setup).

**⚠️ Security Note:** The token is stored in Colab Secrets (encrypted). Colab doesn't log shell commands, so the token won't be exposed. Best practice: Delete the token on GitHub after you're done training.

---

### **Option B: Public Repo (No Auth Needed)**

If your repo is public, just clone directly:

```python
import os
os.chdir('/content')

!git clone https://github.com/lollogabe/Mock_ML.git
%cd Mock_ML

print("✓ Repository cloned (no authentication needed)")
```

---

## Step 2: Install Dependencies

Use the provided **Colab setup helper**:

```python
!python colab_setup.py --setup
```

This:
- ✅ Detects CUDA version
- ✅ Installs PyTorch with correct CUDA support
- ✅ Installs all project dependencies
- ✅ Verifies installation
- ✅ Shows GPU info

**Expected output:**
```
═══════════════════════════════════════════════════════════════
  Google Colab Setup for Mock_ML Project
═══════════════════════════════════════════════════════════════
==> Detecting CUDA version...
  CUDA available but version unclear, using cu121 (safe default)
==> Installing PyTorch for cu121
==> Installing project requirements (excluding torch/torchvision)
==> Verifying installation
  torch      : 2.3.0  (CUDA: True)
  numpy      : 1.24.3
  scikit-learn: 1.3.2
  matplotlib : 3.8.2
  yaml       : 6.0.1
  GPU Device: Tesla T4
  GPU Memory: 15.0 GB
```

---

## Step 3: Download Data

```python
!python scripts/preprocess.py --group 37
```

**Time:** ~2 minutes
**Output:** `data/raw/Normal_data.npz`, `Test_data_low.npz`, `Test_data_high.npz`

---

## Step 4: Train Model

```python
# Full training (20 epochs on T4 GPU: ~40 min)
!python scripts/train.py --config configs/config.yaml --device cuda

# Quick test (2 epochs for debugging: ~4 min)
# !python scripts/train.py --config configs/config.yaml --device cuda --epochs 2
```

**Expected output:**
```
==> CUDA target: cu121
==> Training on device: cuda (Tesla T4)

Epoch 001/020  train_loss=0.512345  val_loss=0.498765  time=1.92s
Epoch 002/020  train_loss=0.456789  val_loss=0.442134  time=1.90s
...
Epoch 020/020  train_loss=0.201234  val_loss=0.204567  time=1.89s

✓ Training complete
✓ Best checkpoint: checkpoints/ae_best.pt
✓ Training logs: logs/train_loss.csv
```

---

## Step 5: Evaluate Model

```python
# Evaluate (skip slow UMAP in Colab)
!python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt --device cuda --no-plot --no-umap
```

**Output:**
- Anomaly scores (MSE, Mahalanobis distance)
- PCA visualization
- GMM clustering results
- Evaluation metrics (AUC, precision, recall)

---

## Step 6: Download Results

### **Option A: Direct Browser Download**

```python
from google.colab import files

# Download checkpoint
files.download('checkpoints/ae_best.pt')

# Download training logs
files.download('logs/train_loss.csv')

# Download plots
files.download('plots/training_loss.png')
files.download('plots/anomaly_scores_mse.png')
```

Files appear in your Downloads folder.

### **Option B: Save to Google Drive (for large files)**

```python
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Copy results to Drive
!mkdir -p /content/drive/MyDrive/Mock_ML_results
!cp -r checkpoints /content/drive/MyDrive/Mock_ML_results/
!cp -r logs /content/drive/MyDrive/Mock_ML_results/
!cp -r plots /content/drive/MyDrive/Mock_ML_results/

print("✓ Results saved to Google Drive")
```

---

## Step 7: Push Results Back to Git (Optional)

Only do this if you want to save **training logs** to your repository.

**⚠️ Important:** Don't push large checkpoint files (.pt) — they're too big for Git. Download them separately instead.

### **Using GitHub Token (Recommended)**

```python
from google.colab import userdata
import subprocess
import os

# Get token from Colab Secrets (set up in Step 1)
token = userdata.get('GITHUB_TOKEN')

# Configure Git with your info
subprocess.run('git config user.email "your_email@example.com"', shell=True)
subprocess.run('git config user.name "Your Name"', shell=True)

# Add training results (NOT the checkpoint)
!git add logs/train_loss.csv plots/training_loss.png

# Commit
!git commit -m "Training results from Colab - GPU run

- Trained for 20 epochs on Tesla T4
- Best validation loss: 0.204567
- Training time: ~40 min"

# Push using token (automatically embedded in URL)
url = f"https://{token}@github.com/lollogabe/Mock_ML.git"
result = subprocess.run(f'git remote set-url origin {url}', shell=True, capture_output=True)
!git push origin main

print("✓ Results pushed to Git")
```

**That's it!** No SSH key management needed.

### **Cleanup After Training (Security)**

After training is complete, delete the token on GitHub:

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Find `Colab-Training`
3. Click **Delete**

The token becomes inactive immediately. Safe!

---

### **If You Don't Want to Push Results**

Just download the checkpoint and logs locally (Step 6) — no Git push needed!

```python
from google.colab import files

files.download('checkpoints/ae_best.pt')
files.download('logs/train_loss.csv')
files.download('plots/training_loss.png')

print("✓ Results downloaded locally")
# Then commit these to Git on your Mac if you want
```

---

## Complete Colab Notebook Template

Create a Colab notebook with these cells in order:

**Option A: Just Download Results (Simplest — no token needed)**

```python
# === CELL 1: Clone ===
import os
os.chdir('/content')
!git clone https://github.com/lollogabe/Mock_ML.git
%cd Mock_ML
print("✓ Cloned")

# === CELL 2: Setup ===
!python colab_setup.py --setup

# === CELL 3: Data ===
!python scripts/preprocess.py --group 37

# === CELL 4: Train ===
!python scripts/train.py --config configs/config.yaml --device cuda

# === CELL 5: Evaluate ===
!python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt --device cuda --no-plot

# === CELL 6: Download Results ===
from google.colab import files
files.download('checkpoints/ae_best.pt')
files.download('logs/train_loss.csv')
files.download('plots/training_loss.png')
print("✓ Downloaded!")
```

---

**Option B: Push Results to Git (With Token)**

```python
# === CELL 1: Clone with Token ===
from google.colab import userdata
import os

os.chdir('/content')
token = userdata.get('GITHUB_TOKEN')  # Stored in Secrets
!git clone https://{token}@github.com/lollogabe/Mock_ML.git
%cd Mock_ML
print("✓ Cloned with token")

# === CELL 2: Setup ===
!python colab_setup.py --setup

# === CELL 3: Data ===
!python scripts/preprocess.py --group 37

# === CELL 4: Train ===
!python scripts/train.py --config configs/config.yaml --device cuda

# === CELL 5: Evaluate ===
!python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt --device cuda --no-plot

# === CELL 6: Push Results ===
import subprocess

token = userdata.get('GITHUB_TOKEN')
subprocess.run('git config user.email "you@example.com"', shell=True)
subprocess.run('git config user.name "Your Name"', shell=True)

!git add logs/train_loss.csv plots/training_loss.png
!git commit -m "Training results from Colab"

url = f"https://{token}@github.com/lollogabe/Mock_ML.git"
subprocess.run(f'git remote set-url origin {url}', shell=True)
!git push origin main
print("✓ Pushed to Git")

# === CELL 7: Download Checkpoint ===
from google.colab import files
files.download('checkpoints/ae_best.pt')
print("✓ Checkpoint downloaded")

# === CELL 8: Cleanup ===
print("Remember to delete token from github.com/settings/tokens for security!")
```


---

## Common Issues & Solutions

### **Issue: `ModuleNotFoundError: No module named 'google.colab'`**

**Solution:** You're not in Colab. Use a different approach:
- For local testing: `python -c "import sys; print(hasattr(sys, 'ps1'))"`
- For Colab: This error shouldn't happen in Colab notebook cells

### **Issue: `Permission denied (publickey)` when cloning with SSH**

**Root Cause:** SSH key corrupted when pasted into Colab Secrets (line breaks lost or formatting broken)

**Quick Fix — Use HTTPS instead:**
```python
!git clone https://github.com/USERNAME/Mock_ML.git
%cd Mock_ML
```
(No authentication needed, works immediately)

**Better SSH Fix — Generate key directly in Colab:**
```python
import subprocess
import os

os.makedirs('/root/.ssh', exist_ok=True)

# Generate new key in Colab
subprocess.run(
    'ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519 -N "" -C "colab@example.com"',
    shell=True, capture_output=True
)

# Show public key (copy to GitHub Settings → SSH Keys)
!cat /root/.ssh/id_ed25519.pub

# Add GitHub to known hosts
!ssh-keyscan -H github.com >> /root/.ssh/known_hosts 2>/dev/null

# Now SSH will work
!git clone git@github.com:USERNAME/Mock_ML.git
```

**Why this works:** Generating the key directly in Colab avoids copy-paste corruption issues.

**Verification:**
```python
!ssh -T git@github.com  # Should show "Hi USERNAME! You've successfully authenticated"
```

### **Issue: Out of Memory During Training**

**Solution:** Reduce batch size:
```python
!python scripts/train.py --config configs/config.yaml --device cuda --batch_size 32
```

### **Issue: `CUDA out of memory` error**

**Solution:** Run garbage collection and reduce model:
```python
import torch
torch.cuda.empty_cache()

# Then retry with smaller batch size
!python scripts/train.py --config configs/config.yaml --device cuda --batch_size 16
```

### **Issue: Runtime timeout (12 hours max in Colab)**

**Solution:** For long training runs:
1. Use Colab Pro ($10/month) for longer sessions
2. Save checkpoints frequently and resume from them
3. Train with fewer epochs first, then continue

---

## Pro Tips

### **Tip 1: Monitor Training in Real-Time**

```python
import subprocess
import time

# Start training in background
process = subprocess.Popen(
    ["python", "scripts/train.py", "--config", "configs/config.yaml", "--device", "cuda"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Monitor logs
while process.poll() is None:
    try:
        with open('logs/train_loss.csv', 'r') as f:
            lines = f.readlines()
            if lines:
                print(lines[-1])  # Last line = latest epoch
    except:
        pass
    time.sleep(5)
```

### **Tip 2: Keep Colab Session Alive During Long Training**

```javascript
// Paste into browser console (F12 → Console) to prevent disconnection
function keepAlive() {
    var buttons = document.querySelectorAll("colab-dialog button");
    buttons.forEach(btn => {
        if (btn.textContent === "CANCEL") btn.click();
    });
}
setInterval(keepAlive, 60000);  // Check every minute
```

### **Tip 3: Use Branches for Experiments**

```python
!git checkout -b exp/batch_size_32
# Make changes to configs/config.yaml
!git add configs/config.yaml
!git commit -m "Test batch size 32"
!git push origin exp/batch_size_32

# Later: compare with main
!git checkout main
!git diff exp/batch_size_32
```

---

## Files Modified for Colab Support

- ✅ `colab_setup.py` — Python helper for environment setup (NEW)
- ✅ `scripts/utils.sh` — Fixed shell script bugs (FIXED)
- ✅ `COLAB_GUIDE.md` — This comprehensive guide (NEW)

---

## Next Steps After Colab Training

**On your local machine:**

```bash
# 1. Pull latest code and results from Colab
git pull origin main

# 2. Get the checkpoint (download from Colab browser)
# Save to: checkpoints/ae_best.pt

# 3. Evaluate locally (better interactive plots)
source venv/bin/activate
python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt

# 4. Commit your evaluation results
git add plots/
git commit -m "Local evaluation of Colab-trained model"
git push origin main
```

---

## FAQ

**Q: Can I use Colab for preprocessing?**
A: Yes, but unnecessary — preprocessing is fast locally. Download data once locally, then download fresh copies in Colab each time.

**Q: Should I commit the checkpoint (.pt file)?**
A: No, it's too large (~100 MB). Download it instead, or store on Google Drive.

**Q: Can multiple team members train simultaneously?**
A: Yes, but use different branches to avoid conflicts:
```python
!git checkout -b colab/person1_run1
!git push origin colab/person1_run1
```

**Q: How do I resume training if the session disconnects?**
A: Colab Pro provides longer sessions. Alternatively:
```python
!python scripts/train.py --config configs/config.yaml --device cuda --resume checkpoints/ae_last.pt
```
(Requires support in train.py for `--resume` flag)

**Q: Is Colab free?**
A: Yes! Up to 12 hours per session. Colab Pro ($10/month) offers longer sessions and better GPUs.

---

## Summary

| Task | Time | Command |
|------|------|---------|
| Clone + Setup | ~3 min | `python colab_setup.py --setup` |
| Download Data | ~2 min | `python scripts/preprocess.py --group 37` |
| **Train** | **~40 min** | `!python scripts/train.py --device cuda` |
| Evaluate | ~5 min | `!python scripts/evaluate.py --checkpoint ae_best.pt --device cuda --no-plot` |
| **Total** | **~50 min** | End-to-end ML pipeline |

---

**Questions?** Check the [main README](README.md) or examine `scripts/train.py` for command-line options.

Happy training! 🚀
