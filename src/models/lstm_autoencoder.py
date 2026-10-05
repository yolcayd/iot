"""Unsupervised LSTM autoencoder trained on normal traffic only (M2).

Encoder LSTM -> last hidden state -> linear latent -> repeated over the window -> decoder LSTM -> linear
+ sigmoid (features lie in [0, 1]). Anomaly score = mean squared reconstruction error of the window
(all steps and features). The threshold is a percentile of the TRAINING (normal subset) score
distribution only (docs/DECISIONS.md K-022, K-030); ``fit_threshold`` refuses scores from any other split.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int = 10, hidden: int = 64, latent: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.decoder = nn.LSTM(latent, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])                          # (B, latent)
        z = z.unsqueeze(1).expand(-1, x.shape[1], -1)      # repeat over time
        y, _ = self.decoder(z)
        return torch.sigmoid(self.out(y))


def reconstruction_score(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Per-window mean squared reconstruction error."""
    return ((model(x) - x) ** 2).mean(dim=(1, 2))


def fit_threshold(scores: np.ndarray, split: str, percentile: float) -> dict:
    """Anomaly threshold from training scores only (CLAUDE.md rule 3)."""
    if split != "train":
        raise ValueError(f"the anomaly threshold may only be fitted on training scores, got split={split!r}")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("need a non-empty, finite score array")
    return {"threshold": float(np.percentile(scores, percentile)), "percentile": float(percentile),
            "fitted_on_split": split, "n_scores": int(scores.size)}


def build(cfg: dict, n_features: int = 10) -> LSTMAutoencoder:
    return LSTMAutoencoder(n_features, cfg["hidden"], cfg["latent"])
