"""
scripts/evaluate.py — Run full evaluation pipeline on a trained AE checkpoint.

Usage:
    python scripts/evaluate.py --config configs/config.yaml --checkpoint checkpoints/ae_best.pt
    python scripts/evaluate.py --config configs/config.yaml --checkpoint checkpoints/ae_best.pt --no-plot
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_device, set_seed, setup_logging, load_checkpoint
from src.data_loader import download_data, load_tensors, build_dataloaders
from src.model import build_model
from src.evaluate import (
    compute_latent_embeddings,
    compute_reconstruction_losses,
    compute_normal_statistics,
    compute_mahalanobis,
    find_anomalies,
    run_pca_umap,
    run_gmm,
)

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the trained CERN jet AE.")
    p.add_argument("--config",     type=str, default="configs/config.yaml")
    p.add_argument("--checkpoint", type=str, default="checkpoints/ae_best.pt")
    p.add_argument("--device",     type=str, default=None)
    p.add_argument("--no-plot",    action="store_true", help="Suppress figures")
    p.add_argument("--no-umap",    action="store_true", help="Skip UMAP (slow)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    log_dir  = cfg.get("log_dir",  os.getenv("LOG_DIR",  "logs"))
    data_dir = cfg.get("data_dir", os.getenv("DATA_DIR", "data/raw"))

    setup_logging(log_dir=log_dir, log_file="evaluate.log")
    logger = logging.getLogger(__name__)

    set_seed(cfg.get("seed", 42))
    device = get_device(args.device or cfg.get("device", os.getenv("DEVICE", "auto")))

    # ── Data ──────────────────────────────────────────────────────────────────
    group = cfg.get("group", int(os.getenv("GROUP", "37")))
    download_data(group=group, data_dir=data_dir)
    normal_t, low_t, high_t = load_tensors(data_dir=data_dir)

    dl_train, dl_n_test, dl_low, dl_high = build_dataloaders(
        normal_t, low_t, high_t,
        batch_size=cfg.get("batch_size", 64),
        test_frac=cfg.get("test_frac", 0.2),
        seed=cfg.get("seed", 42),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        hidden_channels=cfg.get("hidden_channels", 32),
        latent_dim=cfg.get("latent_dim", 4),
    )
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    plot = not args.no_plot

    # ── Latent embeddings ─────────────────────────────────────────────────────
    logger.info("Computing latent embeddings …")
    Z_train   = compute_latent_embeddings(model, dl_train,  device)
    Z_n_test  = compute_latent_embeddings(model, dl_n_test, device)
    Z_low     = compute_latent_embeddings(model, dl_low,    device)
    Z_high    = compute_latent_embeddings(model, dl_high,   device)
    Z_all     = torch.cat([Z_high, Z_train, Z_n_test, Z_low], dim=0)

    # ── Normal statistics ─────────────────────────────────────────────────────
    centroid, precision = compute_normal_statistics(Z_train)

    # ── Anomaly scores ────────────────────────────────────────────────────────
    logger.info("Computing reconstruction losses …")
    loss_n_train = compute_reconstruction_losses(model, dl_train,  device)
    loss_n_test  = compute_reconstruction_losses(model, dl_n_test, device)
    loss_h       = compute_reconstruction_losses(model, dl_high,   device)
    loss_l       = compute_reconstruction_losses(model, dl_low,    device)

    logger.info("Computing Mahalanobis distances …")
    dist_n_train = compute_mahalanobis(Z_train,  centroid, precision).numpy()
    dist_n_test  = compute_mahalanobis(Z_n_test, centroid, precision).numpy()
    dist_h       = compute_mahalanobis(Z_high,   centroid, precision).numpy()
    dist_l       = compute_mahalanobis(Z_low,    centroid, precision).numpy()

    fpr = cfg.get("fpr_threshold", 0.10)

    # ── Thresholding ──────────────────────────────────────────────────────────
    logger.info("Finding anomalies via MSE loss …")
    loss_lbl_train, loss_lbl_test, loss_lbl_low, loss_lbl_high = find_anomalies(
        loss_n_train, loss_n_test, loss_h, loss_l,
        fpr_threshold=fpr, score_type="Loss (MSE)", plot=plot,
    )

    logger.info("Finding anomalies via Mahalanobis distance …")
    dist_lbl_train, dist_lbl_test, dist_lbl_low, dist_lbl_high = find_anomalies(
        dist_n_train, dist_n_test, dist_h, dist_l,
        fpr_threshold=fpr, score_type="Mahalanobis distance", plot=plot,
    )

    # ── Dimensionality reduction ───────────────────────────────────────────────
    labels = np.concatenate([
        np.full(len(Z_high),   3),
        np.full(len(Z_train),  1),
        np.full(len(Z_n_test), 0),
        np.full(len(Z_low),    2),
    ])
    run_pca_umap(Z_all, labels, plot=plot, use_umap=not args.no_umap)

    # ── GMM ───────────────────────────────────────────────────────────────────
    logger.info("Running GMM clustering …")
    gmm_results = run_gmm(
        Z_train, Z_all,
        loss_lbl_low, loss_lbl_high,
        dist_lbl_low, dist_lbl_high,
        fpr_threshold=fpr,
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
