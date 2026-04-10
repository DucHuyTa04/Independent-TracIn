"""Tiny GRU language model with explicit Linear projections (ghost-hookable)."""

from __future__ import annotations

import torch.nn as nn


class TinyGRULM(nn.Module):
    """Embedding -> Linear -> GRU -> Linear logits (B, T, vocab_size)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.proj_in = nn.Linear(embed_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.proj_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.embed(x)
        h = self.proj_in(e)
        o, _ = self.gru(h)
        return self.proj_out(o)
