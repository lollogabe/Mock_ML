"""
scripts/train.py — Entry-point for training the Convolutional Autoencoder.

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --device cuda:0
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_device, set_seed, setup_logging
from src.data_loader import download_data, load_tensors, build_dataloaders
from src.model import build_model
from src.train import train, build_optimizer

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the CERN jet AE.")
    p.add_argument("--config",  type=str, default="configs/config.yaml")
    p.add_argument("--device",  type=str, default=None, help="Override device (e.g. cuda:0, cpu)")
    p.add_argument("--epochs",  type=int, default=None, help="Override number of epochs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    log_dir        = cfg.get("log_dir",        os.getenv("LOG_DIR",        "logs"))
    checkpoint_dir = cfg.get("checkpoint_dir", os.getenv("CHECKPOINT_DIR", "checkpoints"))
    data_dir       = cfg.get("data_dir",       os.getenv("DATA_DIR",       "data/raw"))

    setup_logging(log_dir=log_dir, log_file="train.log")
    logger = logging.getLogger(__name__)

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Seed: {seed}")

    # ── Device ────────────────────────────────────────────────────────────────
    device_str = args.device or cfg.get("device", os.getenv("DEVICE", "auto"))
    device = get_device(device_str)

    # ── Data ──────────────────────────────────────────────────────────────────
    group = cfg.get("group", int(os.getenv("GROUP", "37")))
    download_data(group=group, data_dir=data_dir)
    normal_t, low_t, high_t = load_tensors(data_dir=data_dir)

    dl_train, dl_n_test, dl_low, dl_high = build_dataloaders(
        normal_t, low_t, high_t,
        batch_size=cfg.get("batch_size", 64),
        test_frac=cfg.get("test_frac", 0.2),
        seed=seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        hidden_channels=cfg.get("hidden_channels", 32),
        latent_dim=cfg.get("latent_dim", 4),
    )

    # ── Optimiser & Loss ──────────────────────────────────────────────────────
    optimizer = build_optimizer(
        model,
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-5),
    )
    loss_fn = nn.BCELoss()

    # ── Train ─────────────────────────────────────────────────────────────────
    epochs = args.epochs or cfg.get("epochs", 20)
    logger.info(f"Starting training — {epochs} epochs on {device}")

    loss_history = train(
        model=model,
        dataloader=dl_train,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    logger.info(
        f"Training complete. Final loss: {loss_history[-1]:.6f}  "
        f"Best loss: {min(loss_history):.6f}"
    )


if __name__ == "__main__":
    main()
