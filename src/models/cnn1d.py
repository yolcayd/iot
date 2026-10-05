"""Supervised 1D-CNN window classifier (M1).

Input: (batch, window, 10) feature windows in [0, 1]; output: one logit per window (attack = 1).
Conv + BatchNorm + ReLU blocks fold cleanly for post-training quantisation (F3); every op exports to ONNX.
"""

from __future__ import annotations

import torch
from torch import nn


class CNN1D(nn.Module):
    def __init__(self, n_features: int = 10, channels=(32, 64, 64), kernels=(5, 5, 3), hidden: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        if len(channels) != len(kernels):
            raise ValueError("channels and kernels must have the same length")
        layers, c_in = [], n_features
        for i, (c, k) in enumerate(zip(channels, kernels)):
            layers += [nn.Conv1d(c_in, c, k, padding=k // 2, bias=False), nn.BatchNorm1d(c), nn.ReLU()]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool1d(2))
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(c_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x.transpose(1, 2))     # (B, C, T)
        return self.head(h.mean(dim=2)).squeeze(1)


def build(cfg: dict, n_features: int = 10) -> CNN1D:
    return CNN1D(n_features, tuple(cfg["channels"]), tuple(cfg["kernels"]), cfg["hidden"], cfg["dropout"])
