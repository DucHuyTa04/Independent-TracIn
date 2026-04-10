"""Tiny MLP for synthetic regression."""

import torch.nn as nn


class SynthRegressionMLP(nn.Module):
    """5 -> 32 -> 1 with ReLU."""

    def __init__(self, in_dim: int = 5, hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))
