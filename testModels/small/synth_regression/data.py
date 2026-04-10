"""Synthetic regression dataset y = X @ w + noise."""

import torch
from torch.utils.data import DataLoader, Dataset


class SynthDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        in_dim: int = 5,
        seed: int = 0,
        noise_std: float = 0.1,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n_samples, in_dim, generator=g)
        w = torch.randn(in_dim, 1, generator=g)
        noise = noise_std * torch.randn(n_samples, 1, generator=g)
        self.Y = self.X @ w + noise
        self.w = w

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        y = self.Y[idx].squeeze()
        return self.X[idx], y, idx


def make_loaders(
    n_train: int = 200,
    n_test: int = 50,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, dict[int, str]]:
    train_ds = SynthDataset(n_train, seed=seed)
    test_ds = SynthDataset(n_test, seed=seed + 1)
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    te = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    meta = {}
    for i in range(n_train):
        y = float(train_ds.Y[i].item())
        meta[i] = "positive" if y >= 0 else "negative"
    return tr, te, meta
