"""
utils.py — Helper functions for reproducibility and device management.
"""

import random
import logging
import os
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility across NumPy, Python and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")


def get_device(device_str: str = "auto") -> torch.device:
    """
    Return a torch.device.

    Args:
        device_str: 'auto' (prefer CUDA if available), 'cpu', 'cuda', or 'cuda:N'.
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    return device


def setup_logging(log_dir: str = "logs", log_file: str = "run.log") -> None:
    """Configure root logger to write to both console and a log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )
    logger.info(f"Logging initialised. Log file: {log_path}")


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute clustering purity: fraction of samples in the majority true-class
    within each predicted cluster.

    Args:
        y_true: ground-truth binary labels (0/1).
        y_pred: predicted cluster labels (0/1).

    Returns:
        Purity in [0, 1].
    """
    from scipy.stats import mode

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.zeros_like(y_pred)
    for cluster in np.unique(y_pred):
        mask = y_pred == cluster
        labels[mask] = mode(y_true[mask], keepdims=True).mode[0]
    return float(np.mean(labels == y_true))


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Save model state dict to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    """Load model state dict from *path* and return the model."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    logger.info(f"Checkpoint loaded ← {path}")
    return model
