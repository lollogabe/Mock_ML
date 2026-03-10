# CERN Jet Anomaly Detection

> Unsupervised anomaly detection on 100×100 jet-image histograms using a Convolutional Autoencoder (AE), Mahalanobis distance in latent space, and GMM clustering.

---

## Project Structure

```
Mock_ML/
├── notebooks/                    # Original Jupyter notebook
│   └── DiProfio_Franco_Gabellini_37.ipynb
├── src/                          # Reusable library modules
│   ├── __init__.py
│   ├── data_loader.py            # Download, load, DataLoader construction
│   ├── model.py                  # Encoder, Decoder, AE, build_model()
│   ├── train.py                  # Training loop, optimizer factory
│   ├── evaluate.py               # Anomaly scoring, PCA/UMAP, GMM
│   └── utils.py                  # Seed, device, logging, purity score
├── scripts/                      # CLI entry-points
│   ├── preprocess.py             # Download & verify datasets
│   ├── train.py                  # Run training
│   └── evaluate.py               # Run full evaluation pipeline
├── configs/
│   └── config.yaml               # All hyperparameters in one place
├── data/
│   ├── raw/                      # Downloaded .npz files (git-ignored)
│   └── processed/                # Derived data (git-ignored)
├── logs/                         # Training logs & CSV loss curves
├── checkpoints/                  # Saved model weights
├── tests/                        # Pytest unit tests (no GPU/data needed)
│   ├── test_utils.py
│   ├── test_model.py
│   └── test_data_loader.py
├── docs/
├── .env                          # Local environment overrides
├── .gitignore
├── README.md
├── requirements.txt
├── setup.sh                      # Local venv setup
├── environment.yml               # Conda environment
└── submit_job.sh                 # Slurm HPC submission script
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA 11.8+ recommended for GPU)
- See `requirements.txt` for the full list

---

## Quick Start (local)

### 1. Setup

```bash
bash setup.sh
source venv/bin/activate
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate jet-anomaly
```

### 2. Download data

```bash
python scripts/preprocess.py --group 37 --data-dir data/raw
```

### 3. Train

```bash
python scripts/train.py --config configs/config.yaml
```

Checkpoints are saved to `checkpoints/`; loss curve to `logs/train_loss.csv`.

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/ae_best.pt
```

---

## Reproducing the experiment

All randomness is controlled by `seed: 42` in `configs/config.yaml`.  
Key hyperparameters:

| Parameter | Value |
|---|---|
| `hidden_channels` | 32 |
| `latent_dim` | 4 |
| `batch_size` | 64 |
| `epochs` | 20 |
| `lr` | 1e-3 |
| `fpr_threshold` | 0.10 |

---

## Running on HPC (Slurm / CINECA)

```bash
sbatch submit_job.sh
```

Edit `submit_job.sh` to set partition, account, and module names for your cluster.

---

## Running tests

```bash
pytest tests/ -v
```

Tests use mock data; no GPU or network access is required.

---

## Method overview

1. **Autoencoder training** — BCE loss on normal (background) jet images only.
2. **Anomaly scoring**
   - *MSE reconstruction loss* — high for anomalous jets the AE cannot reconstruct well.
   - *Mahalanobis distance* in latent space — distance from the centroid of the normal embedding distribution.
3. **Threshold** — 90th percentile of the normal-train score → FPR ≤ 10 %.
4. **Latent-space visualisation** — PCA and UMAP scatter plots (coloured by dataset).
5. **GMM clustering**
   - Strategy 1: 1-component GMM on normal train → log-likelihood threshold.
   - Strategy 2: 2-component GMM on all data → minority cluster = anomalies.

---

## Citation / Credits

Dataset from: [http://giagu.web.cern.ch/giagu/CERN/P2025/](http://giagu.web.cern.ch/giagu/CERN/P2025/)  
Authors: DiProfio, Franco, Gabellini (Group 37)
