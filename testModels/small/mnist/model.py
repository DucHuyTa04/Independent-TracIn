"""MNIST MLP model definition."""

import torch
import torch.nn as nn


class MnistMLP(nn.Module):
    """Simple MLP: 784 → hidden_dim (ReLU) → 10 classes."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))
