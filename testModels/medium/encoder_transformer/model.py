"""Tiny BERT-style bidirectional encoder for sequence classification (~120K params)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyBERT(nn.Module):
    """Small bidirectional transformer encoder for sequence classification.

    Architecture::

        token_emb(vocab → embed) + pos_emb(ctx+1 → embed) + [CLS] token
        → 2 × EncoderBlock(embed=128, heads=4, mlp=256)
        → LayerNorm → CLS output → head(128 → num_classes)

    Unlike a causal (GPT) model, attention is **bidirectional** — every
    token can attend to every other token.  Classification uses the
    learned [CLS] token at position 0.
    """

    def __init__(
        self,
        vocab_size: int = 96,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ctx_len: int = 64,
        mlp_dim: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.ctx_len = ctx_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(ctx_len + 1, embed_dim)  # +1 for CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_EncoderBlock(embed_dim, n_heads, mlp_dim) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) long ids → (B, num_classes) logits."""
        B, T = x.shape
        tok = self.token_emb(x)  # (B, T, D)
        cls = self.cls_token.expand(B, 1, -1)  # (B, 1, D)
        h = torch.cat([cls, tok], dim=1)  # (B, T+1, D)
        pos = torch.arange(T + 1, device=x.device).unsqueeze(0)
        h = h + self.pos_emb(pos)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        cls_out = h[:, 0]  # (B, D) — CLS representation
        return self.head(cls_out)


class _EncoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _BidirectionalSelfAttention(dim, n_heads)
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


class _BidirectionalSelfAttention(nn.Module):
    """Full (non-causal) multi-head self-attention."""

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
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        # NO causal mask — bidirectional attention
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)
