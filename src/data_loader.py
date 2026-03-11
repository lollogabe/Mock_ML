"""
data_loader.py — Data downloading, loading, and DataLoader construction.

Datasets (from CERN):
  - Normal_data.npz      → 12 000 normal jet images (100×100)
  - Test_data_low.npz    → 3 000 jets, fraction of anomalies ≤ 45 %
  - Test_data_high.npz   → 3 000 jets, fraction of anomalies ≥ 55 %

Each file contains a single array stored under the key
  'normal_data' (for the normal file) or 'test_data' (for the test files).
Images are reshaped to (N, 1, 100, 100) for use with Conv2d.
"""

import logging
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Remote URLs
# ──────────────────────────────────────────────────────────────────────────────
_BASE_URL = "http://giagu.web.cern.ch/giagu/CERN/P2025"

_URLS = {
    "Normal_data.npz": f"{_BASE_URL}/Normal_data.npz",
    "Test_data_low.npz": "{base}/G{group}/Test_data_low.npz",
    "Test_data_high.npz": "{base}/G{group}/Test_data_high.npz",
}


# ──────────────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────────────

def download_data(group: int = 37, data_dir: str = "data/raw") -> None:
    """Download the three dataset files from CERN if they are not already present.

    Args:
        group:    Dataset group identifier (used to build the URL).
        data_dir: Local directory in which to save the files.
    """
    import urllib.request

    os.makedirs(data_dir, exist_ok=True)
    urls = {
        "Normal_data.npz": f"{_BASE_URL}/Normal_data.npz",
        "Test_data_low.npz": f"{_BASE_URL}/G{group}/Test_data_low.npz",
        "Test_data_high.npz": f"{_BASE_URL}/G{group}/Test_data_high.npz",
    }
    for filename, url in urls.items():
        dest = os.path.join(data_dir, filename)
        if os.path.exists(dest):
            logger.info(f"Already present: {dest}")
            continue
        
        logger.info(f"Downloading {filename} from {url} ...")
        try:
            import ssl
            # Bypass SSL certificate verification for macOS Python installations
            # and handle CERN's potential self-signed certs or redirects
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=context) as response, open(dest, 'wb') as out_file:
                out_file.write(response.read())
            logger.info(f"Saved → {dest}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            if os.path.exists(dest):
                os.remove(dest)
            raise IOError(f"Failed to download {filename} from {url}") from e



# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_tensors(
    data_dir: str = "data/raw",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load .npz files and return float32 tensors shaped (N, 1, 100, 100).

    Returns:
        (normal_t, low_t, high_t)
    """
    def _load(path: str, key: str) -> torch.Tensor:
        arr = np.load(path)[key]          # (N, 100, 100)
        arr = arr[:, np.newaxis, :, :]    # (N, 1, 100, 100)
        return torch.as_tensor(arr, dtype=torch.float32)

    normal_t = _load(os.path.join(data_dir, "Normal_data.npz"), "normal_data")
    low_t    = _load(os.path.join(data_dir, "Test_data_low.npz"), "test_data")
    high_t   = _load(os.path.join(data_dir, "Test_data_high.npz"), "test_data")

    logger.info(
        f"Loaded tensors — normal: {tuple(normal_t.shape)}, "
        f"low: {tuple(low_t.shape)}, high: {tuple(high_t.shape)}"
    )
    return normal_t, low_t, high_t


# ──────────────────────────────────────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    normal_t: torch.Tensor,
    low_t: torch.Tensor,
    high_t: torch.Tensor,
    batch_size: int = 64,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
    """Split the normal dataset and build five DataLoaders.

    The normal dataset is split into train, validation, and test subsets:
      • train:  (1 - test_frac - val_frac) of normal data
      • val:    val_frac of normal data
      • test:   test_frac of normal data (held-out, not used during training)
    
    Low and high anomaly datasets are used for evaluation only.

    Args:
        normal_t:   Normal jet tensor, shape (N, 1, 100, 100).
        low_t:      Low-anomaly tensor.
        high_t:     High-anomaly tensor.
        batch_size: Mini-batch size for the training DataLoader.
        test_frac:  Fraction of normal data to keep as held-out test set (default 0.2).
        val_frac:   Fraction of normal data to use as validation set (default 0.1).
        seed:       Manual seed for the train/val/test split.

    Returns:
        (dl_train, dl_val, dl_n_test, dl_low, dl_high)
    """
    dataset_n = TensorDataset(normal_t, normal_t)

    n_total = len(dataset_n)
    n_test  = int(test_frac * n_total)
    n_val   = int(val_frac * n_total)
    n_train = n_total - n_test - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset_n, [n_train, n_val, n_test], generator=generator
    )

    dl_train  = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    dl_val    = DataLoader(val_ds,   batch_size=1,          shuffle=False)
    dl_n_test = DataLoader(test_ds,  batch_size=1,          shuffle=False)
    dl_low    = DataLoader(TensorDataset(low_t,  low_t),  batch_size=1, shuffle=False)
    dl_high   = DataLoader(TensorDataset(high_t, high_t), batch_size=1, shuffle=False)

    logger.info(
        f"DataLoaders — train: {len(train_ds)} samples, "
        f"val: {len(val_ds)}, test: {len(test_ds)}, low: {len(low_t)}, high: {len(high_t)}"
    )
    return dl_train, dl_val, dl_n_test, dl_low, dl_high
