"""Multi-task MLP: shared backbone with two heads on MNIST (~135K params).

Task A: digit classification (10 classes)
Task B: odd/even classification (2 classes)

This tests ghost TracIn on models with multiple output heads and a
combined loss, where the single backbone must serve both objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskMLP(nn.Module):
    """Shared backbone → two classification heads.

    Architecture::

        backbone: fc1(784→256) → ReLU → fc2(256→128) → ReLU
        digit_head:  Linear(128→10)   — which digit (0–9)
        parity_head: Linear(128→2)    — even (0) or odd (1)

    Ghost layers (all Linear): fc1, fc2, digit_head, parity_head.
    Total: ~135K params.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.digit_head = nn.Linear(128, 10)
        self.parity_head = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor):
        """Returns ``(digit_logits, parity_logits)``."""
        flat = x.view(x.size(0), -1)
        h = F.relu(self.fc1(flat))
        h = F.relu(self.fc2(h))
        return self.digit_head(h), self.parity_head(h)
