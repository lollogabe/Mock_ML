# Colab Workflow: Token-Based Authentication (Updated)

> **Simpler, more reliable than SSH**

---

## Why Token-Based Auth?

| Aspect | SSH | Token |
|--------|-----|-------|
| **Setup Effort** | Complex (copy-paste private key) | Easy (15-second token flow) |
| **Failure Rate in Colab** | High (key corruption common) | Low (just string in Secrets) |
| **Can It Break?** | Yes (formatting issues) | Rarely (just alphanumeric) |
| **Team Collaboration** | Each person needs SSH key | Everyone uses same repo + tokens |
| **Security** | Most secure | Sufficient (revocable tokens) |
| **Recommended?** | ❌ Not for Colab | ✅ YES |

---

## Quick Setup (5 minutes)

### **1. Generate Token (GitHub)**

```
1. Go: https://github.com/settings/tokens
2. Click: "Generate new token" → "Generate new token (classic)"
3. Name: Colab-Training
4. Expiration: 90 days
5. Scopes: ✅ repo (just this one checkbox)
6. Generate → Copy token
```

### **2. Store in Colab Secrets**

```
1. Open Colab notebook
2. Click: 🔑 Secrets (left sidebar)
3. Add new secret:
   - Name: GITHUB_TOKEN
   - Value: (paste your token)
4. Save
```

### **3. Use in Colab**

```python
from google.colab import userdata
import os

os.chdir('/content')
token = userdata.get('GITHUB_TOKEN')
!git clone https://{token}@github.com/lollogabe/Mock_ML.git
%cd Mock_ML
```

That's it!

---

## Full Colab Workflow (Token-Based)

### **Cell 1: Clone**
```python
from google.colab import userdata
import os

os.chdir('/content')
token = userdata.get('GITHUB_TOKEN')
!git clone https://{token}@github.com/lollogabe/Mock_ML.git
%cd Mock_ML
print("✓ Cloned")
```

### **Cell 2: Setup**
```python
!python colab_setup.py --setup
```

### **Cell 3: Download Data**
```python
!python scripts/preprocess.py --group 37
```

### **Cell 4: Train (40 min)**
```python
!python scripts/train.py --config configs/config.yaml --device cuda
```

### **Cell 5: Evaluate**
```python
!python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt --device cuda --no-plot
```

### **Cell 6: Push Results (Optional)**
```python
from google.colab import userdata
import subprocess

token = userdata.get('GITHUB_TOKEN')

# Configure git
subprocess.run('git config user.email "you@example.com"', shell=True)
subprocess.run('git config user.name "Your Name"', shell=True)

# Add results
!git add logs/train_loss.csv plots/training_loss.png
!git commit -m "Training from Colab"

# Push with token
url = f"https://{token}@github.com/lollogabe/Mock_ML.git"
subprocess.run(f'git remote set-url origin {url}', shell=True)
!git push origin main

print("✓ Pushed to Git")
```

### **Cell 7: Download Files**
```python
from google.colab import files
files.download('checkpoints/ae_best.pt')
print("✓ Downloaded")
```

---

## No SSH Needed Anymore ✨

### **Remove SSH (Optional Cleanup)**

```bash
# On your Mac: these are optional (SSH keys are safe to keep)
# Just won't use them for this project

# GitHub: delete SSH key from Settings → SSH and GPG keys
# (optional, you can keep it for other projects)

# Colab: delete GITHUB_SSH_KEY from Secrets 🔑
# (just delete it, no longer needed)
```

### **Tokens > SSH for Colab**
- ❌ SSH in Colab: fragile, copy-paste issues, key corruption
- ✅ Tokens: simple, revocable, no key management

---

## Token Security

**Yes, it's safe:**

1. **Token is encrypted in Colab Secrets** — not logged in shell output
2. **Token is temporary** — expires in 90 days
3. **Token is revocable** — delete anytime from github.com/settings/tokens
4. **Scope limited** — only gives `repo` access, not profile/admin/security

**After training:**
```
1. Go: https://github.com/settings/tokens
2. Find: Colab-Training
3. Delete (immediate ✓)
```

Token is now inactive forever.

---

## What Changed in the Project

### **`COLAB_GUIDE.md`**
- ❌ Removed all SSH instructions
- ✅ Added token-based setup (Step 1)
- ✅ Updated troubleshooting (removed SSH issues)
- ✅ 2 notebook templates: download-only + push-results

### **`colab_setup.py`**
- ❌ Removed `--with-ssh` flag
- ✅ Simplified to just: detect CUDA → install → verify
- ✅ Help text recommends token-based push

### **What Stayed the Same**
- ✅ `setup.sh` — local setup (unchanged)
- ✅ `scripts/utils.sh` — fixed versions (unchanged, working)
- ✅ All training/eval scripts (unchanged)

---

## Comparison: Before vs After

### **Before (SSH Problems)**
```
Problem  : SSH key copy-paste → corruption → "Permission denied"
User     : Frustrated, tries SSH fix, still fails
Solution : Give up, use HTTPS read-only (can't push)
```

### **After (Token Easy)**
```
Setup    : 2 clicks on GitHub → 1 paste into Colab
Push     : 3 lines of Python code
Failure  : Token expired? Delete and create new in 30 seconds
User     : Happy, full access, no key management
```

---

## For Team Collaboration

**Multiple team members training:**

```
Person 1 (Local)
├─ Preprocess data
├─ Push to Git
└─ Pull results

Person 2 (Colab)
├─ Clone with GITHUB_TOKEN
├─ Train model
├─ Push logs with token
└─ Download checkpoint

Person 3 (Colab)
├─ Clone with GITHUB_TOKEN (same token)
├─ Train variant
├─ Push logs with token
└─ Download checkpoint

All use same GITHUB_TOKEN from shared Secrets
   OR
Each person creates their own token (same access)
```

**Easy:** Just share: "Create token, add to your Colab Secrets as GITHUB_TOKEN"

---

## FAQ

**Q: Do I need to remove SSH keys from my Mac?**
A: No, safe to keep. Just won't use them for this project.

**Q: Can multiple people use the same token?**
A: Yes, but it's better practice for each person to create their own token. Easier to revoke one person later if needed.

**Q: What if I lose the token?**
A: Create a new one! Go to github.com/settings/tokens → Generate new token. Old one still works if kept, but can be deleted.

**Q: Is the token exposed in my Colab code?**
A: No! Colab Secrets are encrypted and excluded from logs. The token is only in memory during execution

**Q: Can someone steal my token from Colab?**
A: Only if they access your Colab notebook. Keep it private. Revoke token afterward for extra safety.

**Q: Should I remove all SSH keys from GitHub?**
A: Only if you don't use SSH elsewhere. Keep SSH keys for local development if you like them. Just use tokens for Colab.

---

## Next: Commit Everything

```bash
git add scripts/utils.sh colab_setup.py COLAB_GUIDE.md docs/COLAB.md COLAB_IMPROVEMENTS.md

git commit -m "Simplify Colab workflow: token-based auth (no SSH)

- Recommend HTTPS for cloning (no auth needed for public repos)
- Add token-based auth for pushing results (simpler, more reliable)
- Remove SSH from Colab setup (fragile, prone to copy-paste errors)
- Simplify colab_setup.py (just detect → install → verify)
- Update COLAB_GUIDE.md with token-focused instructions
- Fix 3 shell script bugs for portability (macOS/Linux)"

git push origin main
```

---

## Summary

| Task | Before | After |
|------|--------|-------|
| Clone repo | ❌ SSH setup (fragile) | ✅ HTTPS (no auth) |
| Push results | ❌ SSH key management | ✅ Token (revocable) |
| Troubleshooting | ❌ SSH corruption issues | ✅ Simple token refresh |
| Team collab | ❌ Each needs SSH key | ✅ All use same token |
| Documentation | ❌ Conflicting guides | ✅ Single clear guide |

**Result:**  Colab setup now takes **5 minutes** (not 30+ trying SSHfixes) ✨
