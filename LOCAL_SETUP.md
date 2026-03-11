# Local Development Setup Guide

Guida completa per configurare e far girare il progetto on your workstation (macOS, Linux, Windows).

---

## 📋 Requisiti Preliminari

- **Python 3.9+** (verifica: `python3 --version`)
- **pip** pacchetto manager (incluso in Python ≥ 3.4)
- **Optional**: Git per clonare il repo
- **Disco rigido**: ~2 GB per datasets + virtual environment

### Verifica Sistema

```bash
python3 --version      # Python 3.9+
pip --version          # pip 20.0+
python3 -c "import venv; print('✓ venv available')"
```

---

## 🚀 Fase 1: Clone Repo e Setup Ambiente

### Opzione A: Git (Raccomandato)
```bash
git clone https://github.com/lollogabe/Mock_ML.git
cd Mock_ML
```

### Opzione B: Download ZIP
- Scarica ZIP dal repo GitHub
- `unzip Mock_ML-main.zip && cd Mock_ML`

---

## 🔧 Fase 2: Scegli Metodo Installazione

### **Metodo 1: Script Automatico (Raccomandato)**

```bash
bash setup.sh
```

Questo automaticamente:
- Rileva CUDA version (nvidia-smi)
- Crea virtual environment Python
- Installa PyTorch con GPU support
- Installa dependencies

**Opzioni avanzate:**
```bash
CUDA=cu121 bash setup.sh      # Forza CUDA 12.1
CUDA=cpu   bash setup.sh      # CPU-only (no GPU)
PYTHON=python3.11 bash setup.sh  # Usa Python 3.11 specifico
VENV_DIR=.venv bash setup.sh  # Diverso venv path
```

---

### **Metodo 2: Manual Setup con Conda**

Se preferisci Conda:

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate jet-anomaly

# Verifica installazione
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

### **Metodo 3: Manual venv + pip**

Per controllo massimo:

```bash
# 1. Crea venv
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# o su Windows: venv\Scripts\activate

# 2. Upgrade pip
pip install --upgrade pip wheel setuptools

# 3. Installa PyTorch (scegli comando basato su tuo CUDA)
# Per CPU:
pip install torch>=2.0.0 torchvision>=0.15.0

# Per CUDA 11.8:
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Per CUDA 12.1:
pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Installa dependencies
pip install -r requirements.txt
```

---

## ✅ Fase 3: Verifica Installazione

```bash
# Attiva environment se necessario
source venv/bin/activate  # (or conda activate jet-anomaly)

# Test import
python -c "
import torch
import numpy as np
from src.model import build_model

print('✓ PyTorch version:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  → Device:', torch.cuda.get_device_name(0))
print('✓ Build model OK')
"

# Test unit tests (opzionale)
python -m pytest tests/ -v --tb=short
```

### Output atteso:
```
✓ PyTorch version: 2.0.0
✓ CUDA available: True
  → Device: NVIDIA GeForce RTX 3090
✓ Build model OK
```

---

## 📥 Fase 4: Download Data

```bash
# Attiva ambiente
source venv/bin/activate

# Download datasets (una sola volta)
python scripts/preprocess.py --group 37 --data-dir data/raw
```

**Cosa accade:**
- Scarica 3 file NPZ da CERN (~500 MB total)
- 12K immagini normali `Normal_data.npz`
- 3K test low-anomaly `Test_data_low.npz`
- 3K test high-anomaly `Test_data_high.npz`
- Salva in `data/raw/`

⏱️ **Tempo**: 2-10 min (dipende da connessione internet)

---

## 🏋️ Fase 5: Training

### Train modello completo (20 epoch)

```bash
python scripts/train.py --config configs/config.yaml
```

**Output**:
- `checkpoints/ae_best.pt` — best model (based on validation loss)
- `logs/train.log` — training log file
- `logs/train_loss.csv` — loss curve (epoch, train_loss, val_loss, time_s)

⏱️ **Tempo**: 30-45 minuti su GPU standard (RTX 3090: ~30 min, CPU: 2-3 ore)

### Training rapido (test)

```bash
# Solo 2 epoch, CPU, batch_size piccolo (testing only)
python scripts/train.py \
    --config configs/config.yaml \
    --device cpu \
    --epochs 2
```

⏱️ **Tempo**: 5-10 minuti su CPU (verifica pipeline)

---

## 📊 Fase 6: Evaluation

### Evaluation completo

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/ae_best.pt \
    --config configs/config.yaml
```

**Output**:
- Latent embeddings (PCA 2D, UMAP 2D visualization)
- Anomaly scores (reconstruction loss + Mahalanobis distance)
- GMM clustering results
- Plots in `plots/` directory
- Metrics: TPR, FPR, AUPRC, etc.

⏱️ **Tempo**: 2-5 minuti

### Evaluation senza plot (veloce)

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/ae_best.pt \
    --config configs/config.yaml \
    --no-plot
```

---

## 📈 Development Workflow

### Modifica Hyperparameteri

Modifica direttamente `configs/config.yaml`:

```yaml
batch_size:    64       # Riduci se GPU out-of-memory
hidden_channels: 32     # Model capacity
latent_dim: 4           # Bottleneck dimensionality
epochs: 20              # Training length
lr: 1.0e-3              # Adam learning rate
weight_decay: 1.0e-5    # L2 regularization
```

Oppure override da CLI:
```bash
python scripts/train.py --epochs 30 --device cuda:0
```

---

### Test Veloce

```bash
# Unit tests (< 1 min)
python -m pytest tests/ -v

# Smoke test (model build + 1 forward pass)
python -c "
from src.model import build_model
import torch
m = build_model(hidden_channels=32, latent_dim=4)
x = torch.randn(2, 1, 100, 100)
print('Output shape:', m(x).shape)
"
```

---

### Visualizza Loss Curve

```bash
python -c "
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logs/train_loss.csv')
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train_loss'], label='train', marker='o')
plt.plot(df['epoch'], df['val_loss'], label='val', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('loss_curve.png')
print('Saved to loss_curve.png')
"
```

---

## 🆘 Troubleshooting

### Problema: `ModuleNotFoundError: torch`

**Causa**: Virtual environment non attivo  
**Soluzione**:
```bash
source venv/bin/activate  # Linux/macOS
# o
venv\Scripts\activate     # Windows
```

---

### Problema: `RuntimeError: CUDA out of memory`

**Causa**: Batch size troppo grande per GPU  
**Soluzione**:
1. Riduci `batch_size` in `configs/config.yaml`:
   ```yaml
   batch_size: 32  # (di default è 64)
   ```
2. O esegui con CPU:
   ```bash
   python scripts/train.py --device cpu
   ```

---

### Problema: Download data segue "Failed to download"

**Cause possibili**:
1. **Connessione internet lenta** → riprova più tardi
2. **CERN server down** → verifica manualmente: `curl http://giagu.web.cern.ch/giagu/CERN/P2025/Normal_data.npz`
3. **Certificato SSL** → macOS Python spesso ha problemi. Soluzione:
   ```bash
   /Applications/Python\ 3.x/Install\ Certificates.command
   ```

---

### Problema: Setup script fallisce su macOS ("command not found: grep")

**Causa**: GNU grep non installato  
**Soluzione**:
```bash
# Installa GNU grep con Homebrew
brew install gnu-sed
brew install grep

# Oppure manual setup (vedi Metodo 3 sopra)
```

---

### Problema: Unit tests falliscono

**Check**:
```bash
# Verifica import librerie
python -c "import torch, numpy, sklearn, umap; print('✓ All imports OK')"

# Esegui test con output verboso
python -m pytest tests/ -vv --tb=long
```

---

## 🔄 Riproducibilità

Tutto è controllato dal `seed: 42` in `configs/config.yaml`:
- Weights iniziali
- Data shuffling
- Dropout/batch norm randomness

Modifica seed per diversi run:
```yaml
seed: 99  # Diversi risultati ogni volta
```

---

## 📝 Fase Successiva

Dopo training locale, puoi:

1. **Push su HPC per training production** → [CINECA_GUIDE.md](CINECA_GUIDE.md)
2. **Esperimenti rapidi su Colab** → [COLAB_GUIDE.md](COLAB_GUIDE.md)
3. **Modifica architettura modello** → vedi `src/model.py`
4. **Custom evaluation metrics** → vedi `src/evaluate.py`

---

**Tutti gli script supportano `--help`:**
```bash
python scripts/train.py --help
python scripts/evaluate.py --help
python scripts/preprocess.py --help
```

Buon training! 🚀
