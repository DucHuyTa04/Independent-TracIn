"""Tiny conv U-Net with linear bottleneck for CIFAR denoising (diffusion-style proxy)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyDenoiseUNet(nn.Module):
    """Encoder–conv decoder with a two-layer linear bottleneck (ghost-hookable)."""

    def __init__(self) -> None:
        super().__init__()
        self.e1 = nn.Conv2d(3, 32, 3, padding=1)
        self.e2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.e3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        flat = 64 * 8 * 8
        self.b_fc1 = nn.Linear(flat, 256)
        self.b_fc2 = nn.Linear(256, flat)
        self.d1 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.d2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.d3 = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = F.relu(self.e1(x))
        e2 = F.relu(self.e2(e1))
        e3 = F.relu(self.e3(e2))
        b, c, h, w = e3.shape
        z = e3.flatten(1)
        z = F.relu(self.b_fc1(z))
        z = self.b_fc2(z)
        z = z.view(b, c, h, w)
        u1 = F.relu(self.d1(z))
        u2 = F.relu(self.d2(u1))
        out = self.d3(u2 + e1)
        return out
