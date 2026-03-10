"""
evaluate.py — Evaluation routines: latent embeddings, anomaly scores,
              dimensionality reduction, and GMM clustering.

Anomaly scoring methods:
  1. MSE reconstruction loss — per-sample pixel error between input & output.
  2. Mahalanobis distance — distance of a latent embedding from the centroid
     of the normal training distribution (weighted by the precision matrix).

Both methods use the 90th-percentile of the normal-train score as threshold
so that FPR ≤ 10 % on the training set.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Latent embeddings
# ──────────────────────────────────────────────────────────────────────────────

def compute_latent_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Forward all batches through the encoder and return stacked latent vectors.

    Args:
        model:      AE model (must have a .encoder attribute).
        dataloader: Any DataLoader whose first element is the image batch.
        device:     Compute device.

    Returns:
        FloatTensor of shape (N, latent_dim) on CPU.
    """
    model.eval()
    model.to(device)
    Z: List[torch.Tensor] = []
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            z = model.encoder(xb)
            Z.append(z.cpu())
    return torch.cat(Z, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruction loss (MSE, per sample)
# ──────────────────────────────────────────────────────────────────────────────

def compute_reconstruction_losses(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Compute per-sample MSE reconstruction loss.

    MSE is preferred over BCE for anomaly *scoring* because it provides better
    class separability when a threshold is applied.

    Returns:
        Float numpy array of shape (N,).
    """
    mse = nn.MSELoss(reduction="mean")
    model.eval()
    model.to(device)
    losses: List[float] = []
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            xhat = model(xb)
            losses.append(mse(xhat, xb).item())
    return np.array(losses)


# ──────────────────────────────────────────────────────────────────────────────
# Normal statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_normal_statistics(
    Z_normal_train: torch.Tensor,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the centroid and regularised precision matrix of normal embeddings.

    Args:
        Z_normal_train: Latent embeddings of the normal training set, (N, D).
        epsilon:        Regularisation term added to the diagonal of the
                        covariance matrix for numerical stability.

    Returns:
        (centroid, precision_matrix) — both FloatTensors on CPU.
    """
    centroid = Z_normal_train.mean(dim=0)
    cov = torch.cov(Z_normal_train.T)
    cov = cov + torch.eye(cov.shape[0]) * epsilon
    precision = torch.linalg.inv(cov)
    logger.info(
        f"Normal statistics computed — centroid: {centroid.shape}, "
        f"precision: {precision.shape}"
    )
    return centroid, precision


# ──────────────────────────────────────────────────────────────────────────────
# Mahalanobis distance
# ──────────────────────────────────────────────────────────────────────────────

def compute_mahalanobis(
    Z_embeddings: torch.Tensor,
    centroid: torch.Tensor,
    precision_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute Mahalanobis distance for each embedding.

    d_M(z) = sqrt( (z - mu)^T  *  P  *  (z - mu) )

    Args:
        Z_embeddings:    (N, D) latent embeddings.
        centroid:        (D,) mean of the normal training distribution.
        precision_matrix: (D, D) inverse covariance matrix.

    Returns:
        (N,) FloatTensor of distances.
    """
    diff = Z_embeddings - centroid                          # (N, D)
    term = torch.matmul(diff, precision_matrix)            # (N, D)
    dist_sq = torch.sum(term * diff, dim=1)                # (N,)
    return torch.sqrt(dist_sq.clamp(min=0))


# ──────────────────────────────────────────────────────────────────────────────
# Anomaly detection: threshold + fractions
# ──────────────────────────────────────────────────────────────────────────────

def find_anomalies(
    n_train_scores: np.ndarray,
    n_test_scores: np.ndarray,
    h_scores: np.ndarray,
    l_scores: np.ndarray,
    fpr_threshold: float = 0.10,
    score_type: str = "Loss (MSE)",
    plot: bool = True,
    y_max: float = 1.0,
    ylog: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a percentile threshold and compute anomaly fractions.

    The threshold is chosen as the (1 - fpr_threshold) quantile of the
    normal training scores, ensuring FPR ≤ fpr_threshold on the train set.

    Args:
        n_train_scores: Anomaly scores for normal training samples.
        n_test_scores:  Anomaly scores for normal test samples.
        h_scores:       Anomaly scores for high-anomaly samples.
        l_scores:       Anomaly scores for low-anomaly samples.
        fpr_threshold:  Target FPR upper bound (default 10 %).
        score_type:     X-axis label for the plot.
        plot:           If True, show a histogram of the scores.
        y_max:          Upper Y-axis limit for histogram.
        ylog:           If True, use log-scale Y axis.

    Returns:
        (labels_n_train, labels_n_test, labels_l, labels_h) — binary arrays.
    """
    percentile = (1.0 - fpr_threshold) * 100         # e.g. 90th percentile
    threshold = np.percentile(n_train_scores, percentile)

    labels_n_train = (n_train_scores >= threshold).astype(int)
    labels_n_test  = (n_test_scores  >= threshold).astype(int)
    labels_l       = (l_scores       >= threshold).astype(int)
    labels_h       = (h_scores       >= threshold).astype(int)

    f_train = labels_n_train.mean()
    f_test  = labels_n_test.mean()
    f_low   = labels_l.mean()
    f_high  = labels_h.mean()

    logger.info(
        f"[{score_type}] threshold={threshold:.6f} | "
        f"f_train={f_train:.3f}  f_test={f_test:.3f}  "
        f"f_low={f_low:.3f}  f_high={f_high:.3f}"
    )

    if plot:
        all_vals = np.concatenate([n_train_scores, n_test_scores, h_scores, l_scores])
        vmin, vmax = all_vals.min(), all_vals.max()
        plt.figure(figsize=(9, 5))
        plt.title(
            f"FPR: $f_{{train}}={f_train:.2f}$ | $f_{{test}}={f_test:.2f}$ | "
            f"$f_{{low}}={f_low:.2f}$ | $f_{{high}}={f_high:.2f}$"
        )
        plt.vlines(threshold, 0, y_max, color="red", ls="-.", label="Threshold")
        for arr, label in [
            (n_train_scores, "Train"),
            (n_test_scores,  "Test"),
            (h_scores,       "High"),
            (l_scores,       "Low"),
        ]:
            plt.hist(arr, bins=100, alpha=0.5, range=[vmin, vmax],
                     density=True, label=label)
        plt.xlabel(score_type)
        plt.ylabel("Normalised entries")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if ylog:
            plt.yscale("log")
        plt.tight_layout()
        plt.show()

    return labels_n_train, labels_n_test, labels_l, labels_h


# ──────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ──────────────────────────────────────────────────────────────────────────────

def run_pca_umap(
    Z_all: torch.Tensor,
    labels: np.ndarray,
    plot: bool = True,
    use_umap: bool = True,
) -> Dict[str, np.ndarray]:
    """Apply PCA (and optionally UMAP) to the full latent space and plot.

    Args:
        Z_all:    (N, D) all latent embeddings concatenated.
        labels:   (N,) integer dataset labels
                  (0=n_test, 1=n_train, 2=low, 3=high).
        plot:     If True, generate scatter plots.
        use_umap: If True, also run UMAP.

    Returns:
        Dict with keys 'pca' and optionally 'umap', each (N, 2) numpy arrays.
    """
    from sklearn.decomposition import PCA

    Z_np = Z_all.numpy() if isinstance(Z_all, torch.Tensor) else Z_all
    result: Dict[str, np.ndarray] = {}

    # PCA
    pca = PCA(n_components=2, random_state=42)
    Z_pca = pca.fit_transform(Z_np)
    result["pca"] = Z_pca
    logger.info(
        f"PCA explained variance: "
        f"{pca.explained_variance_ratio_[0]*100:.1f}% + "
        f"{pca.explained_variance_ratio_[1]*100:.1f}%"
    )

    if plot:
        from matplotlib.colors import ListedColormap
        colors = plt.get_cmap("Set1").colors[:4]
        cmap = ListedColormap(colors)
        plt.figure(figsize=(9, 7))
        sc = plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=labels,
                         cmap=cmap, s=10, alpha=0.6)
        plt.title("Latent Space — PCA")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.colorbar(sc, ticks=[0, 1, 2, 3],
                     label="0=n_test, 1=n_train, 2=low, 3=high")
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # UMAP
    if use_umap:
        try:
            import umap as umap_lib
        except ImportError:
            logger.warning("umap-learn not installed. Skipping UMAP.")
            return result
        reducer = umap_lib.UMAP(n_components=2, random_state=42)
        Z_umap = reducer.fit_transform(Z_np)
        result["umap"] = Z_umap
        if plot:
            from matplotlib.colors import ListedColormap
            colors = plt.get_cmap("Set1").colors[:4]
            cmap = ListedColormap(colors)
            plt.figure(figsize=(9, 7))
            sc = plt.scatter(Z_umap[:, 0], Z_umap[:, 1], c=labels,
                             cmap=cmap, s=10, alpha=0.6)
            plt.title("Latent Space — UMAP")
            plt.xlabel("UMAP1")
            plt.ylabel("UMAP2")
            plt.colorbar(sc, ticks=[0, 1, 2, 3],
                         label="0=n_test, 1=n_train, 2=low, 3=high")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# GMM clustering
# ──────────────────────────────────────────────────────────────────────────────

def run_gmm(
    Z_normal_train: torch.Tensor,
    Z_all: torch.Tensor,
    loss_labels_low: np.ndarray,
    loss_labels_high: np.ndarray,
    dist_labels_low: np.ndarray,
    dist_labels_high: np.ndarray,
    fpr_threshold: float = 0.10,
) -> Dict[str, object]:
    """Run GMM-based anomaly classification (2 strategies).

    Strategy 1: Fit GMM(1 component) on Z_normal_train; threshold on
                log-likelihood at (fpr_threshold * 100)th percentile.
    Strategy 2: Fit GMM(2 components) on Z_all; assign anomaly cluster as
                the smaller cluster.

    Args:
        Z_normal_train: (N_train, D) normal training embeddings.
        Z_all:          (N_all,  D) all embeddings (high + train + test + low).
        *_labels_*:     Binary anomaly labels from anomaly-score thresholding.
        fpr_threshold:  FPR upper bound used for strategy-1 threshold.

    Returns:
        Dict with fractions, purity scores, and cluster label arrays.
    """
    from sklearn.mixture import GaussianMixture
    from src.utils import purity_score

    Z_train_np = Z_normal_train.numpy()
    Z_all_np   = Z_all.numpy() if isinstance(Z_all, torch.Tensor) else Z_all

    results: Dict[str, object] = {}

    # ── Strategy 1: 1-component GMM on normal train ──────────────────────────
    gmm1 = GaussianMixture(n_components=1, random_state=42)
    gmm1.fit(Z_train_np)

    ll_train = gmm1.score_samples(Z_train_np)
    threshold_ll = np.percentile(ll_train, fpr_threshold * 100)

    # Low log-likelihood ⟹ anomalous
    def _ll_label(scores):
        return (scores < threshold_ll).astype(int)

    ll_n_train = ll_train
    ll_n_test  = gmm1.score_samples(Z_all_np[len(Z_all_np) - len(ll_train):])   # placeholder
    ll_labels  = {k: _ll_label(gmm1.score_samples(Z_all_np)) for k in ["all"]}

    results["gmm1"] = {
        "model": gmm1,
        "threshold_ll": threshold_ll,
        "purity_loss_low":  purity_score(loss_labels_low,  _ll_label(gmm1.score_samples(Z_all_np[-len(loss_labels_low):]))),
        "purity_loss_high": purity_score(loss_labels_high, _ll_label(gmm1.score_samples(Z_all_np[:len(loss_labels_high)]))),
        "purity_dist_low":  purity_score(dist_labels_low,  _ll_label(gmm1.score_samples(Z_all_np[-len(dist_labels_low):]))),
        "purity_dist_high": purity_score(dist_labels_high, _ll_label(gmm1.score_samples(Z_all_np[:len(dist_labels_high)]))),
    }

    logger.info(
        "GMM Strategy 1 — purity (loss) low={:.3f} high={:.3f}".format(
            results["gmm1"]["purity_loss_low"],  # type: ignore
            results["gmm1"]["purity_loss_high"],  # type: ignore
        )
    )

    # ── Strategy 2: 2-component GMM on all data ───────────────────────────────
    gmm2 = GaussianMixture(n_components=2, random_state=42)
    gmm2.fit(Z_all_np)
    cl_all = gmm2.predict(Z_all_np)

    # The anomaly cluster is the minority cluster
    anom_cluster = int(cl_all.mean() < 0.5)
    labels_cl_all = (cl_all == anom_cluster).astype(int)

    results["gmm2"] = {
        "model":            gmm2,
        "anom_cluster":     anom_cluster,
        "labels_all":       labels_cl_all,
        "purity_loss_low":  purity_score(loss_labels_low,  (gmm2.predict(Z_all_np[-len(loss_labels_low):]) == anom_cluster).astype(int)),
        "purity_loss_high": purity_score(loss_labels_high, (gmm2.predict(Z_all_np[:len(loss_labels_high)]) == anom_cluster).astype(int)),
        "purity_dist_low":  purity_score(dist_labels_low,  (gmm2.predict(Z_all_np[-len(dist_labels_low):]) == anom_cluster).astype(int)),
        "purity_dist_high": purity_score(dist_labels_high, (gmm2.predict(Z_all_np[:len(dist_labels_high)]) == anom_cluster).astype(int)),
    }

    logger.info(
        "GMM Strategy 2 — purity (loss) low={:.3f} high={:.3f}".format(
            results["gmm2"]["purity_loss_low"],  # type: ignore
            results["gmm2"]["purity_loss_high"],  # type: ignore
        )
    )

    return results
