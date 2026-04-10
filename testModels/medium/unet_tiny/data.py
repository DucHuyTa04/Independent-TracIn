"""CIFAR-10 subset for denoising: returns (noisy_image, clean_image, sample_id)."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class CifarDenoiseDataset(Dataset):
    """Normalized CIFAR-10; target is clean image (same tensor as input without noise)."""

    def __init__(self, train: bool, root: str = "data", noise_std: float = 0.15) -> None:
        try:
            from torchvision import datasets, transforms
        except ImportError as e:
            raise ImportError("torchvision required. pip install torchvision") from e

        self.noise_std = noise_std
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.ds = datasets.CIFAR10(root=root, train=train, download=True, transform=tfm)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        clean, _y = self.ds[idx]
        noise = self.noise_std * torch.randn_like(clean)
        noisy = (clean + noise).clamp(-1.0, 1.0)
        return noisy, clean, idx
