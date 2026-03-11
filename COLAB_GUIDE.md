# Google Colab Setup Guide — Complete Workflow

Esegui il progetto CERN Jet Anomaly Detection su Google Colab gratuitamente con GPU T4/V100.

⏱️ **Setup**: 3 minuti | **Training**: 35-45 minuti | **Evaluation**: 3-5 minuti

---

## 🎯 Cosa Troverai

- Setup completo automatico (Python dependencies)
- Download datasets da CERN
- Training Autoencoder su GPU Colab
- Evaluation con visualizzazioni (PCA, UMAP)
- Salvataggio modelli e risultati

---

## 📌 Prerequisiti

1. ✅ Account Google (gratuito)
2. ✅ Acceso a [Google Colab](https://colab.research.google.com)
3. ✅ ~2 GB spazio storage temporaneo (ephemeral, wiped after session)

---

## 🚀 Quick Start (Copy-Paste)

Apri [Google Colab](https://colab.research.google.com), crea nuovo notebook, e esegui questi comandi in ordine:

### Cell 1: Clone Repository + Setup
```python
import os
os.chdir('/content')

# Clone repo (HTTPS, nessuna auth richiesta)
!git clone https://github.com/lollogabe/Mock_ML.git
%cd Mock_ML

# Setup l'ambiente
!python colab_setup.py --setup
```

**Output atteso:**
```
Cloning into 'Mock_ML'...
remote: Counting objects: ...
Installing dependencies...
✓ All dependencies installed successfully
✓ Colab environment ready
```

---

### Cell 2: Download Data
```python
!python scripts/preprocess.py --group 37 --data-dir data/raw
```

**Output atteso:**
```
Downloading Normal_data.npz from ... (12000 samples)
Downloading Test_data_low.npz from ... (3000 samples)
Downloading Test_data_high.npz from ... (3000 samples)
All files downloaded successfully
```

⏱️ **Tempo**: 2-10 minuti (dipende da connessione internet)

---

### Cell 3: Verifica GPU
```python
import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Output atteso:**
```
✓ PyTorch version: 2.0.0
✓ CUDA available: True
✓ GPU: Tesla T4
  Memory: 15.0 GB
```

---

### Cell 4: Training (35-45 min)
```python
!python scripts/train.py --config configs/config.yaml --device cuda
```

**Output atteso:**
```
2024-03-11 10:30:45 - Training epoch 1/20
[████████████████] 100% - loss: 0.4567
2024-03-11 10:32:15 - Training epoch 2/20
...
2024-03-11 11:45:30 - Saved best checkpoint: checkpoints/ae_best.pt
Training completed in 75.4 seconds
```

**Output files:**
- `checkpoints/ae_best.pt` — trained model
- `logs/train.log` — training log
- `logs/train_loss.csv` — loss curve (epoch, train_loss, val_loss, time)

---

### Cell 5: Evaluation (3-5 min)
```python
!python scripts/evaluate.py \
    --checkpoint checkpoints/ae_best.pt \
    --config configs/config.yaml \
    --device cuda
```

**Output atteso:**
```
Computing latent embeddings...
Computing anomaly scores...
TPR (at FPR=10%): 0.87
AUPRC: 0.92
Generating plots...
Saved plots to plots/
```

**Output plots:**
- `plots/latent_pca.png` — 2D PCA projection
- `plots/latent_umap.png` — 2D UMAP projection
- `plots/anomaly_scores_histogram.png` — anomaly score distributions
- `plots/gmm_clusters.png` — GMM clustering results

---

### Cell 6: Visualizza Loss Curve (Opzionale)
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logs/train_loss.csv')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_loss'], marker='o', label='Train Loss')
plt.plot(df['epoch'], df['val_loss'], marker='s', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')
plt.legend()
plt.grid()
plt.title('Training Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['time_s'], marker='o', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.grid()
plt.title('Epoch Duration')

plt.tight_layout()
plt.show()
```

---

## 📥 Download Risultati

Dopo training/evaluation, scarica i risultati **prima che la sessione finisca** (Colab cancella tutto dopo 12 ore di inattività).

### Metodo 1: Zip Archive (Consigliato)
```python
import shutil

# Crea archive con risultati
shutil.make_archive('results', 'zip', root_dir='.', base_dir='.')

# Download
from google.colab import files
files.download('results.zip')
```

### Metodo 2: File individuali
```python
from google.colab import files

files.download('checkpoints/ae_best.pt')
files.download('logs/train_loss.csv')
files.download('plots/latent_pca.png')
```

---

## ⚙️ Configurazione Avanzata

### Modifica Hyperparameters

Per training custom, modificare `configs/config.yaml` **dentro Colab**:

```python
# Edit config direttamente
config_content = """
batch_size: 32          # Riduci se out-of-memory
hidden_channels: 32     # Model capacity
latent_dim: 4           # Bottleneck dimensionality
epochs: 30              # Più epochs
lr: 5.0e-4              # Learning rate
weight_decay: 1.0e-5
device: cuda
"""

with open('configs/config.yaml', 'w') as f:
    f.write(config_content)

# Esegui training con config modificato
!python scripts/train.py --config configs/config.yaml --device cuda
```

---

### Ridimensiona Batch Size

Se ottieni **CUDA out of memory**:

```python
# Riduci batch size
import yaml

with open('configs/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

cfg['batch_size'] = 32  # Da 64 a 32

with open('configs/config.yaml', 'w') as f:
    yaml.dump(cfg, f)

# Riavvia training
!python scripts/train.py --config configs/config.yaml
```

---

### Training Rapido (Test)

Per testare il pipeline (2 epochs, 5 min):

```python
!python scripts/train.py \
    --config configs/config.yaml \
    --epochs 2 \
    --device cuda
```

---

## 🔄 Workflow Iterativo

Per esperimenti ripetuti (modifiche modello, hyperparameter tuning):

```python
import subprocess
import yaml

def train_experiment(config_mods: dict):
    """Run experiment con custom config"""
    # Leggi config base
    with open('configs/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Modifica
    cfg.update(config_mods)
    
    # Salva
    with open('configs/config.yaml', 'w') as f:
        yaml.dump(cfg, f)
    
    # Train
    subprocess.run(['python', 'scripts/train.py', 
                   '--config', 'configs/config.yaml'], 
                   check=True)

# Esperimento 1: Standard
train_experiment({'epochs': 20, 'batch_size': 64})

# Esperimento 2: Smaller batch (se OOM)
train_experiment({'epochs': 20, 'batch_size': 32})

# Esperimento 3: Longer training
train_experiment({'epochs': 30, 'batch_size': 32})
```

---

## 🆘 Troubleshooting

### Problema: "Failed to clone repository"

**Causa**: Repository non esiste or non è public  
**Soluzione**: Assicurati che il repo GitHub sia accessibile pubblicamente

```python
# Verifica URL
!curl -I https://github.com/lollogabe/Mock_ML.git
```

Dovrebbe ritornare **200 OK**, non 404.

---

### Problema: "ModuleNotFoundError" dopo setup

**Causa**: Dependencies installation fallita  
**Soluzione**: Esegui setup manualmente:

```python
!pip install --upgrade pip
!pip install torch torchvision scipy sklearn umap-learn pyyaml python-dotenv
!python scripts/preprocess.py --group 37
```

---

### Problema: "CUDA out of memory"

**Causa**: Batch size troppo grande per GPU T4 (15 GB VRAM)  
**Soluzione**:

```python
# Riduci batch size
import yaml

cfg = yaml.safe_load(open('configs/config.yaml'))
cfg['batch_size'] = 16  # Smaller batch
yaml.dump(cfg, open('configs/config.yaml', 'w'))

# Riprova
!python scripts/train.py --config configs/config.yaml
```

---

### Problema: Download data timeout

**Causa**: CERN server lento o connessione internet instabile  
**Soluzione**:

```python
# Riprova download con timeout
import time

for attempt in range(3):
    try:
        !python scripts/preprocess.py --group 37 --data-dir data/raw
        break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed, retrying...")
        time.sleep(10)
```

---

### Problema: "RuntimeError: Expected 4D input, got 3D instead"

**Causa**: Bug nel data loading  
**Soluzione**: Aggiungi cell di debug:

```python
import torch
from src.data_loader import load_tensors

try:
    normal_t, low_t, high_t = load_tensors(
        data_dir='data/raw',
        normal_file='Normal_data.npz'
    )
    print(f"Normal shape: {normal_t.shape}")  # Deve essere (N, 1, 100, 100)
    print(f"Low shape: {low_t.shape}")
    print(f"High shape: {high_t.shape}")
except Exception as e:
    print(f"Error: {e}")
    # Se shape è (N, 100, 100), aggiungi dimensione mancante
    normal_t = normal_t.unsqueeze(1)
```

---

## 📝 Note Importanti

### Ephemeral Storage
- Colab cancella tutto dopo 12 ore di inattività
- **Sempre download risultati** prima di fine sessione
- Usa Google Drive per storage persistente (opzionale)

### GPU Limits
- **Idle timeout**: 12 ore inattività → session kills
- **Data limits**: ~100 GB free tier (non è problema per questo progetto)
- **Peak hours**: Potrebbe avere T4, non V100 in ore di picco

### Best Practices
1. ✅ **Salva checkpoint** ogni epoch (già fatto in `train.py`)
2. ✅ **Download risultati regolarmente** (non aspettare fine training)
3. ✅ **Test con --epochs 2** prima di 20 epochs
4. ✅ **Mantieni una copia locale** del config e results

---

## 🔗 Link Utili

- [Google Colab](https://colab.research.google.com)
- [GitHub Repo](https://github.com/lollogabe/Mock_ML)
- [CERN Dataset](http://giagu.web.cern.ch/giagu/CERN/P2025)

---

## 📚 Prossimi Steps

1. ✅ Training completato? → Vedi evaluation results
2. ✅ Pronti per production? → [CINECA_GUIDE.md](CINECA_GUIDE.md)
3. ✅ Sviluppo locale? → [LOCAL_SETUP.md](LOCAL_SETUP.md)
4. ✅ Modifica modello? → Vedi `src/model.py`, `src/train.py`

Buon training! 🚀

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
