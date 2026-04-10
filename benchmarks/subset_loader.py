"""Helpers for capping full-gradient TracIn baseline size on large training sets."""

from __future__ import annotations

from typing import List

import torch
from torch.utils.data import DataLoader, Dataset


def baseline_subset_indices(n_train: int, max_baseline_n: int, seed: int) -> List[int]:
    """Return sorted indices ``0..n_train-1`` or a random subset of size ``max_baseline_n``."""
    if n_train <= max_baseline_n:
        return list(range(n_train))
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n_train, generator=g)
    return sorted(int(perm[i]) for i in range(max_baseline_n))


class SubsetByOriginalId(Dataset):
    """Wraps a dataset indexed by ``0..N-1`` whose ``__getitem__(i)`` returns ``(..., i)`` as last element."""

    def __init__(self, base: Dataset, original_ids: List[int]) -> None:
        self.base = base
        self.original_ids = list(original_ids)

    def __len__(self) -> int:
        return len(self.original_ids)

    def __getitem__(self, idx: int):
        oid = self.original_ids[idx]
        return self.base[oid]


def make_baseline_loader(
    base_ds: Dataset,
    full_loader: DataLoader,
    n_train: int,
    max_baseline_n: int,
    seed: int,
    batch_size: int,
) -> tuple[DataLoader, List[int], int | None]:
    """Return loader for full-gradient baseline, ids used for comparison, and subset size or None."""
    if n_train <= max_baseline_n:
        return full_loader, list(range(n_train)), None
    ids = baseline_subset_indices(n_train, max_baseline_n, seed)
    sub = SubsetByOriginalId(base_ds, ids)
    return (
        DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=0),
        ids,
        len(ids),
    )
