"""Tiny GPT-style character-level language model (~100K params)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyGPT(nn.Module):
    """Minimal causal transformer for character-level language modelling.

    Architecture::

        token_emb(vocab → embed) + pos_emb(ctx → embed)
        → 2 × TransformerBlock(embed=128, heads=4, mlp=256)
        → LayerNorm → output_proj(128 → vocab)

    Ghost layer: ``output_proj`` (Linear(128 → vocab_size)).
    Total: ~120K params.  Ghost coverage: 128*vocab / total ≈ 10%.
    """

    def __init__(
        self,
        vocab_size: int = 96,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ctx_len: int = 64,
        mlp_dim: int = 256,
    ) -> None:
        super().__init__()
        self.ctx_len = ctx_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(ctx_len, embed_dim)
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, n_heads, mlp_dim) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) long ids → (B, T, vocab) logits."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        h = self.token_emb(x) + self.pos_emb(pos)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.output_proj(h)  # (B, T, vocab)


class _Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _CausalSelfAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, T, nh, hd)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale  # (B, nh, T, T)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)
