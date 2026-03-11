# Execution Guide: Local vs HPC Workflow

This guide explains how to run the CERN Jet Anomaly Detection project locally on your workstation or on HPC clusters.

**TL;DR Quick Start:**
- **Local**: `bash setup.sh && python scripts/train.py`
- **HPC**: `bash setup_hpc.sh` → `sbatch submit_job.sh`

---

## Recommended Workflow

### Phase 1: Development & Testing (Local)

```bash
# 1. Setup environment once
bash setup.sh
source venv/bin/activate

# 2. Download datasets (one-time)
python scripts/preprocess.py --group 37

# 3. Validate code before submitting to HPC
python -m pytest tests/ -v              # Unit tests
python scripts/train.py --epochs 2 --device cpu  # CPU smoke test

# If all passes ✓, proceed to HPC
```

### Phase 2: Training (HPC)

```bash
# On login node:
module load profile/deeplrn
module load cuda/11.8
module load python/3.10.8-gcc11.3
bash setup_hpc.sh        # Setup once per cluster

# Submit training (data already downloaded locally)
sbatch --export=ALL,SKIP_PREPROCESS=1 submit_job.sh

# Monitor:
squeue -u $USER          # Check job status
tail -f logs/slurm_<jobid>.out    # Watch output
```

### Phase 3: Evaluation (Local)

```bash
# After training completes, download checkpoint:
scp cluster:Mock_ML/checkpoints/ae_best.pt ./checkpoints/

# Evaluate with plots
python scripts/evaluate.py \
  --checkpoint checkpoints/ae_best.pt \
  --device cpu
  
# Or skip slow UMAP (faster):
python scripts/evaluate.py \
  --checkpoint checkpoints/ae_best.pt \
  --device cpu \
  --no-umap
```

---

## Detailed Workflows

### Local Development Workflow

**Use this for:**
- Code changes and debugging
- Algorithm prototyping
- Hyperparameter tuning (short runs)
- Model evaluation and analysis

**Steps:**

```bash
# 1. ONE-TIME: Setup environment
bash setup.sh
source venv/bin/activate

# 2. ONE-TIME: Download datasets
python scripts/preprocess.py --group 37

# 3. Develop & test
# Edit code, then run tests
python -m pytest tests/ -v

# 4. Quick smoke test (CPU, 1-2 epochs)
python scripts/train.py \
  --config configs/config.yaml \
  --device cpu \
  --epochs 2

# 5. If satisfied, proceed to full training on HPC
```

**Commands Reference:**

| Task | Command |
|------|---------|
| Download data | `python scripts/preprocess.py --group 37` |
| Run all tests | `python -m pytest tests/ -v` |
| Run specific test | `python -m pytest tests/test_model.py::TestEncoder::test_output_shape -v` |
| Train locally (CPU) | `python scripts/train.py --device cpu --epochs 5` |
| Evaluate locally | `python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt --device cpu` |
| Evaluate (no plots) | `python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt --no-plot` |

### HPC Workflow (CINECA Leonardo)

**Use this for:**
- Full training runs (20+ epochs)
- Intensive model evaluation
- Large-scale hyperparameter searches  

**Setup (ONE-TIME):**

```bash
# Login to login node
ssh username@leonardo.cineca.it

# Navigate to project
cd Mock_ML

# Load necessary modules
module load profile/deeplrn    # Deep learning environment
module load cuda/11.8          # (or cuda/12.1 if available)
module load python/3.10.8-gcc11.3

# Verify modules loaded
module list

# Setup environment
bash setup_hpc.sh

# Verify
source activate_env.sh         # Should activate venv
```

**Preprocessing (if needed):**

```bash
# Option A: Run locally, then skip on HPC
# (recommended - save HPC queue time)
python scripts/preprocess.py --group 37

# Option B: Run first time on HPC
srun python scripts/preprocess.py --group 37
```

**Submit Training Job:**

```bash
# Standard submission
sbatch submit_job.sh

# With data skip (if already preprocessed)
sbatch --export=ALL,SKIP_PREPROCESS=1 submit_job.sh

# With custom job name
sbatch --job-name=exp_001 submit_job.sh

# Check job status
squeue -u $USER

# Monitor output in real-time
tail -f logs/slurm_<JOBID>.out

# Cancel job if needed
scancel <JOBID>
```

**Customize Hardware (Edit `submit_job.sh` before submitting):**

```bash
# For longer training on Leonardo A100s
#SBATCH --time=04:00:00         # 4 hours (default 2h)
#SBATCH --mem=64G               # More memory if needed
#SBATCH --cpus-per-task=16      # More CPU cores for data loading

# For Galileo100 (V100 GPUs instead)
#SBATCH --partition=g100_usr_prod
#SBATCH --gres=gpu:1            # V100 instead of A100
```

**Download Results:**

```bash
# From local machine:
scp -r username@leonardo.cineca.it:Mock_ML/checkpoints ./

# Or specific checkpoint
scp username@leonardo.cineca.it:Mock_ML/checkpoints/ae_best.pt ./checkpoints/

# Download logs
scp username@leonardo.cineca.it:Mock_ML/logs/slurm_*.out ./logs/
```

---

## Evaluation Pipeline Options

The evaluation script has flexible options for different scenarios:

### Run Everything (Default)
```bash
python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt
```
Runs: Metrics → Dimensionality Reduction (PCA + UMAP) → GMM Clustering

### Skip Slow Operations
```bash
python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt --no-umap
```
Skips UMAP (which can be slow for large latent spaces)

### Run Only Metrics (Fast)
```bash
python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt \
  --no-dimensionality --no-gmm
```
Quick anomaly detection scores without visualization

### Run Only Visualizations (Requires Prior Metrics)
```bash
python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt \
  --no-metrics --no-gmm
```
Only dimensionality reduction (PCA/UMAP)

### Suppress All Plots (For HPC)
```bash
python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt \
  --no-plot --no-umap
```
Compute metrics without generating figures

---

## Troubleshooting

### Environment Issues

**Problem**: `ModuleNotFoundError: No module named 'torch'`
```bash
# Solution: Activate environment
source venv/bin/activate           # Local
source activate_env.sh             # HPC
```

**Problem**: CUDA not detected on HPC
```bash
# Check loaded modules
module list

# Load CUDA module explicitly
module load cuda/11.8

# Verify
nvidia-smi

# Then run setup
bash setup_hpc.sh
```

### Data Download Issues

**Problem**: Download fails with SSL error
```bash
# Current code handles this, but if needed, predownload locally:
python scripts/preprocess.py --group 37

# Then transfer to HPC
scp -r data/raw/* username@cluster:Mock_ML/data/raw/
```

### Training Issues

**Problem**: OOM (Out of Memory)
```bash
# Reduce batch size
python scripts/train.py --config configs/config.yaml \
  --batch_size 32     # Default is 64
```

**Problem**: Training too slow on CPU
```bash
# Use GPU (local machine if available, or HPC)
python scripts/train.py --device cuda
# or
sbatch submit_job.sh
```

### HPC Job Queue

**Problem**: Long queue times
```bash
# Solution: Only submit training (GPU-intensive work)
# Do preprocessing and evaluation locally
sbatch --export=ALL,SKIP_PREPROCESS=1 submit_job.sh
```

**Problem**: Job timeout
```bash
# Increase wall time in submit_job.sh:
#SBATCH --time=04:00:00      # Increase from 02:00:00

# Or reduce epochs
sbatch --export=ALL,EPOCHS=10 submit_job.sh
```

---

## File Organization

```
Mock_ML/
├── setup.sh              # Local setup (run once)
├── setup_hpc.sh          # HPC setup (run once on login node)
├── activate_env.sh       # Auto-generated by setup_hpc.sh
├── scripts/
│   ├── utils.sh         # Shared shell utilities
│   ├── train.py         # Entry point: python scripts/train.py
│   ├── evaluate.py      # Entry point: python scripts/evaluate.py
│   └── preprocess.py    # Entry point: python scripts/preprocess.py
├── src/
│   ├── train.py         # Training logic
│   ├── evaluate.py      # Evaluation logic
│   ├── model.py         # Model definition
│   └── data_loader.py   # Data loading (now with validation set)
├── configs/
│   └── config.yaml      # Centralized configuration
├── tests/
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_utils.py
├── data/
│   ├── raw/             # Downloaded .npz files
│   └── processed/       # (if needed)
├── checkpoints/         # Trained model weights
├── logs/                # Training logs + job outputs
└── plots/               # Generated evaluation plots
```

---

## Configuration

All hyperparameters are centralized in `configs/config.yaml`:

```yaml
# Dataset
group:    37             # CERN dataset group
data_dir: data/raw

# Model
hidden_channels: 32
latent_dim:      4

# Training
batch_size:    64
test_frac:     0.2       # 20% held-out test set
val_frac:      0.1       # 10% validation set (NEW)
seed:          42
lr:            1.0e-3
weight_decay:  1.0e-5
epochs:        20

# Evaluation
fpr_threshold: 0.10

# Hardware
device: auto
```

Override via command line:
```bash
python scripts/train.py --config configs/config.yaml --epochs 30
```

---

## Best Practices

1. **Always run tests locally first:**
   ```bash
   python -m pytest tests/ -v
   ```

2. **Test with small data before HPC:**
   ```bash
   python scripts/train.py --epochs 1-2 --device cpu
   ```

3. **Use `--skip-preprocess` to save HPC queue time:**
   ```bash
   sbatch --export=ALL,SKIP_PREPROCESS=1 submit_job.sh
   ```

4. **Monitor HPC jobs:**
   ```bash
   watch -n 10 squeue -u $USER
   ```

5. **Save evaluation plots locally:**
   ```bash
   # Download plots after evaluation
   scp -r cluster:Mock_ML/plots ./
   ```

6. **Keep logs for debugging:**
   ```bash
   # Logs from training
   cat logs/train.log
   
   # Job output from HPC
   cat logs/slurm_<jobid>.out
   ```

---

## FAQ

**Q: Can I train locally instead of HPC?**
A: Yes, but for 20 epochs on CPU it will take ~hours. Use HPC for faster training.

**Q: Do I need to redownload data for each HPC job?**
A: No. Download once locally, transfer via `scp`, then use `SKIP_PREPROCESS=1`.

**Q: Can I modify the model architecture?**
A: Yes, edit `src/model.py` and run tests locally first to validate.

**Q: How do I use a different dataset group?**
A: `python scripts/preprocess.py --group 120` (or edit `config.yaml`)

**Q: Can I continue training from a checkpoint?**
A: Currently no (would require checkpoint loading changes). Submit new training runs.

**Q: How do I share this project?**
A: Use `setup.sh` or `setup_hpc.sh` — all dependencies are pinned in `requirements.txt` and `environment.yml`.

---

## Support

For issues or questions:
1. Check [STRATEGIC_PLAN.md](STRATEGIC_PLAN.md) for project context
2. Review shell script comments in `setup.sh` and `setup_hpc.sh`
3. Check logs:
   - Local: `logs/train.log`
   - HPC: `logs/slurm_<jobid>.out`
