"""Small MNIST autoencoder for reconstruction-based influence benchmarks."""

import torch
import torch.nn as nn


class MnistAutoencoder(nn.Module):
    """Encoder–decoder MLP that reconstructs flattened 784-dim MNIST images.

    Architecture::

        Encoder:  fc1(784, 64) → ReLU → fc2(64, 8) → ReLU
        Decoder:  fc3(8, 8)    → ReLU → fc_out(8, 784)

    Ghost layer (last ``nn.Linear``):  ``fc_out``
        H = 8  (input to fc_out)
        C = 784 (error signal dimension = output dim for MSE)
        ghost_dim = 8 × 784 = 6 272
        total_params = 57 888  →  coverage ≈ 10.8 %
    """

    def __init__(self, latent_dim: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, 784)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # (B, 784)
        # Encoder
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        # Decoder
        h = self.relu(self.fc3(h))
        return self.fc_out(h)  # (B, 784) — raw logits, no sigmoid
