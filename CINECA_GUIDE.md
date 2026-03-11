# CINECA HPC Cluster Setup Guide

Guida completa per eseguire il progetto CERN Jet Anomaly Detection su supercomputer CINECA (Marconi, Leonardo, etc.).

**Ambiente**: CINECA Marconi con GPU Tesla A100  
**Training time**: ~15-20 minuti (vs 35-45 min Colab T4)  
**Throughput**: 100-200x più veloce di CPU locale

---

## 🎯 Cosa Troverai

- Setup environment su CINECA con moduli disponibili
- Download dataset da CERN
- SLURM job submission per training scalabile
- GPU A100 support + multi-GPU training
- Monitoring job execution in real-time
- Output consolidation e backup

---

## 📋 Prerequisiti

1. ✅ Account CINECA con accesso a Marconi
2. ✅ SSH configurato (`ssh hpc4ai@login.marconi.cineca.it`)
3. ✅ Quota storage sufficiente:
   - `$HOME`: ~100 MB (venv)
   - `$CINECA_SCRATCH`: ~2 GB (datasets + results)

Controlla quota:
```bash
ssh hpc4ai@login.marconi.cineca.it
quota -s
# CINECA_SCRATCH: dovrebbe avere almeno 2 GB disponibili
```

---

## 🚀 Quick Start

### Step 1: Login a CINECA
```bash
ssh hpc4ai@login.marconi.cineca.it
cd $CINECA_SCRATCH
```

---

### Step 2: Clone Repo
```bash
git clone https://github.com/lollogabe/Mock_ML.git
cd Mock_ML
```

---

### Step 3: Setup Ambiente (una sola volta)

```bash
# Load required modules
module load profile/deeplrn
module load cuda/11.8
module load python/3.10.8-gcc11.3

# Setup virtual environment in scratch (NOT in $HOME)
bash setup_hpc.sh
```

**Output atteso:**
```
═══════════════════════════════════════════════════════════════
  HPC Python Environment Setup
═══════════════════════════════════════════════════════════════
==> Using CINECA_SCRATCH: /cineca/prod/hpc4ai/scratch
==> Creating venv at: /cineca/prod/hpc4ai/scratch/jet_anomaly_venv
==> Installing PyTorch for cu118...
==> Installation successful
✓ venv ready at: /cineca/prod/hpc4ai/scratch/jet_anomaly_venv
```

---

### Step 4: Download Data (una sola volta)

```bash
# Attiva venv
source /cineca/prod/hpc4ai/scratch/jet_anomaly_venv/bin/activate

# Download datasets
python scripts/preprocess.py --group 37 --data-dir data/raw
```

⏱️ **Tempo**: 2-5 minuti (dipende da connessione)

---

### Step 5: Submit Training Job

```bash
# Rimani in project directory
# (venv non deve essere attivo per SLURM)

sbatch submit_job.sh
```

**Output:**
```
Submitted batch job 12345678
```

---

### Step 6: Monitor Job

```bash
# Check job status
squeue -u $USER

# Watch output (real-time)
tail -f logs/training_job_12345678.log

# Cancel job (se necessario)
scancel 12345678
```

---

## 📁 Project Layout su CINECA

```
$CINECA_SCRATCH/
├── Mock_ML/                          # Project clone
│   ├── scripts/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── preprocess.py
│   ├── src/
│   ├── configs/
│   │   └── config.yaml
│   ├── data/
│   │   └── raw/                      # Datasets (downloaded)
│   │       ├── Normal_data.npz       # ~200 MB
│   │       ├── Test_data_low.npz     # ~50 MB
│   │       └── Test_data_high.npz    # ~50 MB
│   │
│   ├── checkpoints/                  # Best model (after training)
│   │   └── ae_best.pt                # ~2 MB
│   │
│   ├── logs/                         # Training outputs
│   │   ├── train.log                 # Per-epoch metrics
│   │   ├── train_loss.csv            # CSV logs
│   │   └── training_job_*.log        # SLURM job output
│   │
│   ├── plots/                        # Evaluation visualizations
│   │   ├── latent_pca.png
│   │   ├── latent_umap.png
│   │   └── ...
│   │
│   └── setup_hpc.sh                  # Setup script (already run)
│
└── jet_anomaly_venv/                 # Python virtual environment (~500 MB)
    ├── bin/
    │   ├── python
    │   ├── pip
    │   └── activate
    └── lib/
        └── python3.10/site-packages/
```

**Storage summary:**
- Venv: ~500 MB
- Datasets: ~300 MB
- Checkpoints: ~2 MB
- Logs + plots: ~50 MB
- **Total**: ~1 GB

---

## 🔧 Configurazione Dettagliata

### A. Modifica SLURM Job Parameters

Apri `submit_job.sh` e modifica questi parametri:

```bash
#SBATCH --job-name=jet_anomaly      # Job name
#SBATCH --time=00:30:00             # Max time (30 min)
#SBATCH --ntasks=1                  # 1 task
#SBATCH --cpus-per-task=8           # 8 CPU per task
#SBATCH --partition=gpu             # GPU partition
#SBATCH --gpus=1                    # 1 GPU (A100)
#SBATCH --mem=32GB                  # RAM allocation
```

**Profili comuni:**

**🚀 Fast Single GPU (15 min training)**
```bash
#SBATCH --time=00:20:00
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --partition=gpu_large
```

**⚡ Multi-GPU (4x A100, 5-10 min)**
```bash
#SBATCH --time=00:15:00
#SBATCH --gpus=4
#SBATCH --mem=128GB
#SBATCH --partition=gpu_large
```

**💪 Ultra-Fast (8x A100, 3-5 min)**
```bash
#SBATCH --time=00:10:00
#SBATCH --gpus=8
#SBATCH --mem=256GB
#SBATCH --partition=gpu_huge
```

---

### B. Modifica Hyperparameters

Apri `configs/config.yaml`:

```yaml
# ── Training ───────────────────────────────────────────────────────────────────
batch_size:    128      # Più alto con GPU A100 (vs 64 in Colab)
test_frac:     0.2
val_frac:      0.1
seed:          42
lr:            1.0e-3
weight_decay:  1.0e-5
epochs:        20       # Aumenta per risultati migliori (Colab usa 20)

# ── Hardware ────────────────────────────────────────────────────────────────────
device: cuda            # Auto-select first GPU
```

---

### C. Enable Multi-GPU Training

Se usi 2+ GPU, modifica `submit_job.sh`:

```bash
# Dopo module load:
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Per 4 GPU

# Training script automatico usa tutti i GPU disponibili
python scripts/train.py --config configs/config.yaml
```

Il codice usa `torch.nn.DataParallel` automaticamente se rilevati multipli GPU.

---

## 📜 SLURM Job Scripts Detail

### Standard Job (Single GPU A100)

`submit_job.sh` template:
```bash
#!/bin/bash
#SBATCH --job-name=jet_anomaly
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32GB
#SBATCH --output=logs/training_job_%j.log

# Load modules
module load profile/deeplrn
module load cuda/11.8
module load python/3.10.8-gcc11.3

# Find venv (auto-detect CINECA_SCRATCH or WORK)
if [[ -d "$CINECA_SCRATCH/jet_anomaly_venv" ]]; then
    VENV_PATH="$CINECA_SCRATCH/jet_anomaly_venv"
elif [[ -d "$WORK/jet_anomaly_venv" ]]; then
    VENV_PATH="$WORK/jet_anomaly_venv"
else
    echo "ERROR: venv not found"
    exit 1
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Training
echo "Starting training at $(date)"
python scripts/train.py --config configs/config.yaml --device cuda
echo "Training completed at $(date)"

# Evaluation
echo "Starting evaluation at $(date)"
python scripts/evaluate.py \
    --checkpoint checkpoints/ae_best.pt \
    --config configs/config.yaml \
    --device cuda
echo "Evaluation completed at $(date)"
```

---

## 📊 Monitoraggio Job

### Check Job Status

```bash
# Ancora in queue / running?
squeue -u $USER -l

# Detailed job info
scontrol show job 12345678

# Cancel job
scancel 12345678
```

### Watch Training Output

```bash
# Real-time log
tail -f logs/training_job_*.log

# Last 50 lines
tail -50 logs/training_job_*.log

# Search for errors
grep -i "error\|cuda\|memory" logs/training_job_*.log
```

### Dopo Training

```bash
# Check final status
sacct -j 12345678 --format=JobID,Status,Elapsed,NodeList

# Output atteso:
# Status: COMPLETED (success)
# Status: FAILED (error - check logs)
# Status: TIMEOUT (tempo insufficiente)
```

---

## 🆘 Troubleshooting

### Problema: "Module not found"

**Causa**: Moduli non caricati  
**Soluzione**:
```bash
module load profile/deeplrn
module load cuda/11.8
module load python/3.10.8-gcc11.3
python --version  # Verifica Python 3.10+
```

---

### Problema: "CUDA out of memory"

**Causa**: Batch size troppo grande per GPU  
**Soluzione**:

```yaml
# configs/config.yaml
batch_size: 32  # Riduci (di default 128 per A100)
```

Oppure:
```bash
# Override da command line
python scripts/train.py --config configs/config.yaml --batch_size 32
```

---

### Problema: "Job cancelled because exceeding time limit"

**Causa**: `--time` insufficient  
**Soluzione**:

```bash
# Aumenta time limit in submit_job.sh
#SBATCH --time=00:45:00  # Da 30 min a 45 min

# Oppure submit con override
sbatch --time=00:45:00 submit_job.sh
```

---

### Problema: "Venv not found" durante SLURM job

**Causa**: Path $CINECA_SCRATCH non trovato  
**Soluzione**:

```bash
# Login a CINECA e verifica
ssh hpc4ai@login.marconi.cineca.it
echo $CINECA_SCRATCH  # Stampa path

# Se vuoto, usa WORK
mkdir -p $WORK/jet_anomaly_venv
python -m venv $WORK/jet_anomaly_venv
```

Poi modifica `submit_job.sh` per usare `$WORK`.

---

### Problema: "Dataset download timeout"

**Causa**: CERN server lento  
**Soluzione**:

```bash
# Cancella download parziale e riprova
rm -f data/raw/*.npz
python scripts/preprocess.py --group 37 --data-dir data/raw

# Se continua a fallire, scarica manualmente:
# wget http://giagu.web.cern.ch/giagu/CERN/P2025/Normal_data.npz
```

---

## 📤 Download Risultati

Dopo job completato:

### Opzione 1: sftp (veloce)
```bash
# Da tuo Mac
sftp hpc4ai@login.marconi.cineca.it
cd Mock_ML
get checkpoints/ae_best.pt ./
get logs/train_loss.csv ./
get plots/* ./plots/
bye
```

### Opzione 2: scp
```bash
# Da tuo Mac
scp -r hpc4ai@login.marconi.cineca.it:$CINECA_SCRATCH/Mock_ML/checkpoints ./
scp -r hpc4ai@login.marconi.cineca.it:$CINECA_SCRATCH/Mock_ML/logs ./
scp -r hpc4ai@login.marconi.cineca.it:$CINECA_SCRATCH/Mock_ML/plots ./
```

### Opzione 3: Zip & Download
```bash
# Su CINECA
cd $CINECA_SCRATCH/Mock_ML
zip -r results.zip checkpoints/ logs/ plots/

# Poi download via sftp o HTTP (se disponibile)
```

---

## 🔄 Workflow Completo (Quick Reference)

```bash
# === STEP 1: Primo login ===
ssh hpc4ai@login.marconi.cineca.it
cd $CINECA_SCRATCH
git clone https://github.com/lollogabe/Mock_ML.git
cd Mock_ML

# === STEP 2: Setup (una sola volta) ===
module load profile/deeplrn cuda/11.8 python/3.10.8-gcc11.3
bash setup_hpc.sh

# === STEP 3: Download data (una sola volta) ===
source /cineca/prod/hpc4ai/scratch/jet_anomaly_venv/bin/activate
python scripts/preprocess.py --group 37 --data-dir data/raw

# === STEP 4: Submit job ===
sbatch submit_job.sh
# Job 12345678 submitted

# === STEP 5: Monitor (login in nuovo terminal) ===
ssh hpc4ai@login.marconi.cineca.it
cd $CINECA_SCRATCH/Mock_ML
squeue -u $USER
tail -f logs/training_job_*.log

# === STEP 6: Dopo completamento ===
# Download risultati via sftp
# Nuovo training? Basta: sbatch submit_job.sh (datasets già presenti)
```

---

## 💡 Tips & Best Practices

### ✅ Do's
1. **Metti venv in scratch**, non in $HOME (quota limit)
2. **Usa profile/deeplrn** — carica CUDA stack automaticamente
3. **Test local first** — bash setup.sh + 2 epochs before submission
4. **Monitor tail -f** — watch per problemi durante execution
5. **Download regolarmente** — non aspettare fine session

### ❌ Don'ts
1. **Non mettere datasets in $HOME** — quota exceeded
2. **Non cancellare $CINECA_SCRATCH** — perde datasets e results
3. **Non usare python (system)** — usa module-loaded Python
4. **Non submitta job ogni 1 minuto** — queue congestion
5. **Non disattivare venv during SLURM job** — job script gestisce

---

## 🎓 Scaling a Multi-Node (Avanzato)

Per training su multipli nodi con Distributed Data Parallel:

```bash
# submit_job.sh modification:
#SBATCH --ntasks=4              # 4 nodi
#SBATCH --nodes=4               # 1 GPU per nodo
#SBATCH --ntasks-per-node=1
```

Poi modifica `scripts/train.py` per usare `torch.distributed`:
```python
# Pseudo-code (requires implementation):
if torch.distributed.is_available():
    torch.distributed.init_process_group(backend='nccl')
    # Wraps model con DistributedDataParallel
```

(Out of scope per questa guida, ma suddividere batch across nodi è possibile)

---

## 🔗 Link Utili

- [CINECA HPC Documentation](https://wiki.u-gov.it/confluence/display/SCAI)
- [SLURM Job Scheduling](https://slurm.schedmd.com/sbatch.html)
- [GitHub Repo](https://github.com/lollogabe/Mock_ML)

---

## 📚 Prossimi Steps

1. ✅ Training su CINECA completato?
   - Download checkpoints localmente
   - [Evaluation locale](LOCAL_SETUP.md#fase-6-evaluation)
   - Modifica model / hyperparameters e resubmit

2. ✅ Esperimenti rapidi?
   - [Google Colab](COLAB_GUIDE.md) per development
   - [Local CPU](LOCAL_SETUP.md#fase-2-scegli-metodo-installazione) per debugging

3. ✅ Scalare a più dataset groups?
   - `--group 1` / `--group 2` / etc.
   - Modifica `sbatch --array=1-10` per loop jobs

Buon training! 🚀
