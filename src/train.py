"""
train.py — Training loop for the Convolutional Autoencoder.

Training strategy (from notebook):
  • Loss:      BCE (Binary Cross-Entropy), appropriate because the Decoder
               uses a Sigmoid activation, making outputs interpretable as
               probabilities in [0, 1].
  • Optimiser: Adam with weight_decay=1e-5 for light L2 regularisation.
  • Epochs:    20 — loss stabilises; more would risk overfitting.
"""

import logging
import os
import time
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils import save_checkpoint

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Single epoch
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the mean batch loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    from tqdm.auto import tqdm
    
    for xb, _ in tqdm(dataloader, desc="Training", leave=False):
        xb = xb.to(device)
        xhat = model(xb)
        loss = loss_fn(xhat, xb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ──────────────────────────────────────────────────────────────────────────────

def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one validation epoch and return the mean batch loss (no gradients)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            xhat = model(xb)
            loss = loss_fn(xhat, xb)
            
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int = 20,
    val_dataloader: Optional[DataLoader] = None,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
) -> tuple:
    """Train the model for *epochs* epochs with optional validation.

    Saves:
      • A checkpoint after every epoch:  checkpoints/ae_epoch_{N:03d}.pt
      • The best-loss checkpoint:        checkpoints/ae_best.pt
      • Loss history CSV:                logs/train_loss.csv

    Args:
        model:           The AE model.
        dataloader:      Training DataLoader (normal images only).
        optimizer:       Configured optimiser.
        loss_fn:         Loss criterion (BCE recommended).
        device:          Compute device.
        epochs:          Number of training epochs.
        val_dataloader:  Optional validation DataLoader. If provided, validation
                         loss is computed each epoch and saved to CSV.
        checkpoint_dir:  Directory to save model weights.
        log_dir:         Directory for the loss log CSV.

    Returns:
        (train_losses, val_losses) tuple of lists. val_losses is empty list if
        val_dataloader is None.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model.to(device)

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    best_loss = float("inf")

    # CSV log
    log_path = os.path.join(log_dir, "train_loss.csv")
    with open(log_path, "w") as f:
        if val_dataloader is not None:
            f.write("epoch,train_loss,val_loss,time_s\n")
        else:
            f.write("epoch,train_loss,time_s\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        epoch_train_loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device)
        train_loss_history.append(epoch_train_loss)
        
        # Compute validation loss if dataloader provided
        epoch_val_loss = None
        if val_dataloader is not None:
            epoch_val_loss = validate_one_epoch(model, val_dataloader, loss_fn, device)
            val_loss_history.append(epoch_val_loss)
        
        elapsed = time.time() - t0

        if epoch_val_loss is not None:
            logger.info(
                f"Epoch {epoch:03d}/{epochs}  "
                f"train_loss={epoch_train_loss:.6f}  "
                f"val_loss={epoch_val_loss:.6f}  "
                f"time={elapsed:.1f}s"
            )
        else:
            logger.info(
                f"Epoch {epoch:03d}/{epochs}  "
                f"train_loss={epoch_train_loss:.6f}  "
                f"time={elapsed:.1f}s"
            )

        # Append to CSV
        with open(log_path, "a") as f:
            if epoch_val_loss is not None:
                f.write(f"{epoch},{epoch_train_loss:.8f},{epoch_val_loss:.8f},{elapsed:.2f}\n")
            else:
                f.write(f"{epoch},{epoch_train_loss:.8f},{elapsed:.2f}\n")

        # Save checkpoint every epoch
        ckpt_path = os.path.join(checkpoint_dir, f"ae_epoch_{epoch:03d}.pt")
        save_checkpoint(model, ckpt_path)

        # Save best checkpoint (based on validation loss if available, else training loss)
        loss_for_best = epoch_val_loss if epoch_val_loss is not None else epoch_train_loss
        if loss_for_best < best_loss:
            best_loss = loss_for_best
            save_checkpoint(model, os.path.join(checkpoint_dir, "ae_best.pt"))

    logger.info(f"Training complete — best loss: {best_loss:.6f}")
    return train_loss_history, val_loss_history


# ──────────────────────────────────────────────────────────────────────────────
# Optimiser factory
# ──────────────────────────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> torch.optim.Adam:
    """Return an Adam optimiser configured for the given model."""
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
