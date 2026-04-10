"""TinyShakespeare character-level dataset for language modelling."""

from __future__ import annotations

import os
import urllib.request

import torch
from torch.utils.data import Dataset

_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _download(root: str) -> str:
    """Download TinyShakespeare if not cached; return path to text file."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "tinyshakespeare.txt")
    if not os.path.isfile(path):
        print(f"Downloading TinyShakespeare → {path} …")
        urllib.request.urlretrieve(_SHAKESPEARE_URL, path)  # noqa: S310 — trusted URL
    return path


class CharLMDataset(Dataset):
    """Fixed-length character sequences from TinyShakespeare.

    Each sample: ``(input_ids[0:T], target_ids[1:T+1], idx)``
    where ``target = input shifted right by 1``.
    """

    def __init__(
        self,
        root: str = "data",
        ctx_len: int = 64,
        train: bool = True,
        train_frac: float = 0.9,
    ) -> None:
        text_path = _download(root)
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.ctx_len = ctx_len

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = len(data) - ctx_len  # number of valid start positions
        split = int(n * train_frac)
        if train:
            self.data = data[:split + ctx_len]
            self.n_seqs = split
        else:
            self.data = data[split:]
            self.n_seqs = len(self.data) - ctx_len

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.ctx_len]
        y = self.data[idx + 1: idx + self.ctx_len + 1]
        return x, y, idx
