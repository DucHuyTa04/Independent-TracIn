"""TinyShakespeare binary classification dataset for encoder transformer.

Task: Given a 64-char sequence, predict whether the character immediately
after the sequence is uppercase (1) or not (0).  This is a natural signal
in Shakespeare text (sentence beginnings, speaker names, act headings).
"""

from __future__ import annotations

import os
import urllib.request

import torch
from torch.utils.data import Dataset

_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _download(root: str) -> str:
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "tinyshakespeare.txt")
    if not os.path.isfile(path):
        print(f"Downloading TinyShakespeare → {path} …")
        urllib.request.urlretrieve(_SHAKESPEARE_URL, path)  # noqa: S310 — trusted URL
    return path


class CharClassifyDataset(Dataset):
    """Binary classification: is the next character uppercase?

    Each sample: ``(input_ids[ctx_len], label ∈ {0,1}, idx)``
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
        # Binary labels: 1 if the character at position ctx_len is uppercase
        labels = torch.tensor(
            [1 if text[i + ctx_len].isupper() else 0 for i in range(len(text) - ctx_len)],
            dtype=torch.long,
        )

        n = len(labels)
        split = int(n * train_frac)
        if train:
            self.data = data
            self.labels = labels[:split]
            self.n_seqs = split
        else:
            self.offset = split
            self.data = data
            self.labels = labels[split:]
            self.n_seqs = len(self.labels)

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int):
        start = idx if not hasattr(self, "offset") else idx + self.offset
        x = self.data[start : start + self.ctx_len]
        y = self.labels[idx]
        return x, y, idx
