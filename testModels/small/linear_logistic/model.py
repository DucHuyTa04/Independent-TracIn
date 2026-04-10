"""Single linear layer on flattened MNIST (convex multiclass logistic regression)."""

import torch
import torch.nn as nn


class MnistLinear(nn.Module):
    """784 → 10 logits (no hidden layers)."""

    def __init__(self, input_dim: int = 784, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.view(x.size(0), -1))
