"""
model.py — Convolutional Autoencoder architecture for jet-image anomaly detection.

Architecture (matching the notebook):
  Encoder: Conv2d (k=6,s=2) → Conv2d (k=4,s=2) → flatten → Linear → latent z
  Decoder: Linear → reshape → ConvTranspose2d × 2 → Sigmoid output (100×100)

BatchNorm and LeakyReLU are used throughout to prevent gradient vanishing and
model collapse on sparse jet images.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Encoder
# ──────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Convolutional encoder: maps (B, 1, 100, 100) → (B, latent_dim).

    Layer design rationale (from notebook):
      • Large kernel (k=6) in the first layer to capture global patterns on
        sparse images where local information density is low.
      • BatchNorm prevents model collapse; LeakyReLU prevents dying neurons.
    """

    def __init__(self, hidden_channels: int = 32, latent_dim: int = 4) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            # Layer 1: (B,1,100,100) → (B,hidden,49,49)
            nn.Conv2d(1, hidden_channels, kernel_size=6, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2: (B,hidden,49,49) → (B,hidden*2,23,23)
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Compute flattened size with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 100, 100)
            flat_size = self.conv(dummy).numel()

        self.fc = nn.Linear(flat_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ──────────────────────────────────────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """Symmetric transposed-convolutional decoder: (B, latent_dim) → (B, 1, 100, 100).

    Sigmoid output constrains pixel values to [0, 1] — required by BCE loss.
    """

    def __init__(self, hidden_channels: int = 32, latent_dim: int = 4) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim

        # We need to reverse-engineer the spatial dimensions from the encoder.
        # Encoder: 100 → (100+2*1-6)//2+1 = 49 → (49+2*1-4)//2+1 = 24
        # We project back to (hidden*2, 24, 24) and upsample.
        self.fc = nn.Linear(latent_dim, hidden_channels * 2 * 24 * 24)

        self.deconv = nn.Sequential(
            # (B, hidden*2, 24, 24) → (B, hidden, 49, 49)
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, hidden, 49, 49) → (B, 1, 100, 100)
            nn.ConvTranspose2d(hidden_channels, 1,
                               kernel_size=6, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self.hidden_channels * 2, 24, 24)
        x = self.deconv(x)
        # Ensure exact output size matches input (100×100)
        x = F.interpolate(x, size=(100, 100), mode="bilinear", align_corners=False)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Autoencoder
# ──────────────────────────────────────────────────────────────────────────────

class AE(nn.Module):
    """Convolutional Autoencoder wrapping an Encoder and a Decoder."""

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(hidden_channels: int = 32, latent_dim: int = 4) -> AE:
    """Construct and return the Autoencoder model.

    Args:
        hidden_channels: Base channel count (notebook default: 32).
        latent_dim:      Dimensionality of the bottleneck (notebook default: 4).

    Returns:
        Assembled AE model in training mode.
    """
    enc = Encoder(hidden_channels=hidden_channels, latent_dim=latent_dim)
    dec = Decoder(hidden_channels=hidden_channels, latent_dim=latent_dim)
    model = AE(enc, dec)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model built — {n_params:,} trainable parameters")
    return model
