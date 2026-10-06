"""Unsupervised 1D convolutional autoencoder; fallback if the LSTM autoencoder fails (M2-alt).

Same interface as the LSTM autoencoder (score = mean squared reconstruction error; decision threshold
calibrated on validation normals, src/calibration.py). Candidate 3 of the autoencoder feasibility
protocol (docs/DECISIONS.md K-030, K-033).
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.lstm_autoencoder import reconstruction_score  # noqa: F401  (shared interface)


class ConvAutoencoder(nn.Module):
    """Window length must be divisible by 4 (100 -> 50 -> 25)."""

    def __init__(self, n_features: int = 10, channels=(32, 64), latent_channels: int = 16):
        super().__init__()
        c1, c2 = channels
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, c1, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(c1, c2, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(c2, latent_channels, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, c2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(c2, c1, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(c1, n_features, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] % 4:
            raise ValueError("window length must be divisible by 4")
        return torch.sigmoid(self.decoder(self.encoder(x.transpose(1, 2)))).transpose(1, 2)


def build(cfg: dict, n_features: int = 10) -> ConvAutoencoder:
    return ConvAutoencoder(n_features, tuple(cfg["channels"]), cfg["latent_channels"])
