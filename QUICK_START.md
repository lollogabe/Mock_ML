# Quick Start Guide — CERN Jet Anomaly Detection

Benvenuto! Questa è una guida rapida per scegliere l'ambiente di lavoro e iniziare subito.

---

## 🚀 Scegli il tuo ambiente

| Ambiente | Velocità Configurazione | GPU | Supporto Interattivo | Durata Training |
|----------|---|---|---|---|
| **Local** | 5 min | ✅ (opcional) | ✅ Jupyter | 30-45 min |
| **Google Colab** | 3 min | ✅ T4/V100 | ✅ Notebook | 35-45 min |
| **CINECA HPC** | 10 min | ✅ A100 | ❌ (batch only) | 15-20 min |

---

## ⚡ Setup ultra-veloce

### Local (Raccomandato per sviluppo)
```bash
bash setup.sh
python scripts/preprocess.py --group 37
python scripts/train.py --config configs/config.yaml
python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt
```
👉 **Guida completa**: [LOCAL_SETUP.md](LOCAL_SETUP.md)

---

### Google Colab (Raccomandato per esperimenti rapidi)
1. Apri [Colab](https://colab.research.google.com)
2. Paste questo codice nella prima cella:
```python
import os
os.chdir('/content')
!git clone https://github.com/lollogabe/Mock_ML.git
%cd Mock_ML
!python colab_setup.py --setup
!python scripts/preprocess.py --group 37
!python scripts/train.py --config configs/config.yaml
!python scripts/evaluate.py --checkpoint checkpoints/ae_best.pt
```
👉 **Guida completa**: [COLAB_GUIDE.md](COLAB_GUIDE.md)

---

### CINECA HPC (Raccomandato per training production)
```bash
# Su login node:
module load profile/deeplrn cuda/11.8 python/3.10.8-gcc11.3
bash setup_hpc.sh
sbatch submit_job.sh
```
👉 **Guida completa**: [CINECA_GUIDE.md](CINECA_GUIDE.md)

---

## 📊 Cosa fa il progetto?

**Task**: Rilevare anomalie in immagini di jet CERN (100×100 pixel)

**Pipeline**:
1. **Download data** → 12K immagini normali + test with anomalies
2. **Train Autoencoder** → imparare a ricostruire immagini normali
3. **Evaluate** → calcolare anomaly score per ogni immagine
4. **Analyze** → PCA/UMAP visualization + GMM clustering

**Output**:
- `checkpoints/ae_best.pt` — modello addestrato
- `logs/train_loss.csv` — loss curve
- `plots/` — visualizzazioni anomaly scores

---

## 🔧 Requisiti Minimi

- Python ≥ 3.9
- PyTorch ≥ 2.0
- 4 GB RAM (local) / GPU consigliata (T4+ con 15GB VRAM su Colab)

---

## 🆘 Troubleshooting Rapido

| Problema | Soluzione |
|----------|-----------|
| `ModuleNotFoundError: torch` | Esegui lo script di setup (setup.sh / setup_hpc.sh) |
| Download data fallisce | Controlla connessione; i file sono grandi (~500MB) |
| CUDA out of memory | Riduci `batch_size` in `configs/config.yaml` |
| Su Colab: "Permission denied" | Assicurati di avere accesso in lettura al repo |

Vedi [LOCAL_SETUP.md](LOCAL_SETUP.md) per problemi più dettagliati.

---

## 📚 Documentazione Completa

- **[LOCAL_SETUP.md](LOCAL_SETUP.md)** — Setup locale e development workflow
- **[COLAB_GUIDE.md](COLAB_GUIDE.md)** — Guida Google Colab con sezione pushing results
- **[CINECA_GUIDE.md](CINECA_GUIDE.md)** — Guida CINECA HPC con SLURM job submission
- **[README.md](README.md)** — Overview progetto e architettura modello
- **[configs/config.yaml](configs/config.yaml)** — Tutti gli hyperparameteri
- **[tests/](tests/)** — Unit tests (esegui: `python -m pytest tests/ -v`)

---

**Domande?** Vedi le guide environment-specific sopra elencate. ✨
