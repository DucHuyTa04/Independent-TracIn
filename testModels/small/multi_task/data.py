"""MNIST dataset for multi-task benchmark (digit class + odd/even)."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class MnistMultiTaskDataset(Dataset):
    """MNIST yielding ``(image, digit_label, idx)``.

    Parity labels (odd/even) are derived from the digit label inside the
    loss function so the data loader interface stays ``(x, y, idx)``
    compatible with the ghost TracIn pipeline.
    """

    def __init__(self, train: bool, root: str = "data") -> None:
        try:
            from torchvision import datasets
        except ImportError:
            raise ImportError("torchvision required for MNIST. pip install torchvision")

        self.ds = datasets.MNIST(root=root, train=train, download=True)
        self.data = self.ds.data.float() / 255.0
        self.targets = self.ds.targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        x = self.data[idx].unsqueeze(0)  # (1, 28, 28)
        y = self.targets[idx]
        return x, y, idx
