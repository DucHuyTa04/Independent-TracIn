"""Tiny Vision Transformer for CIFAR-10 (~40K params)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ViTMicro(nn.Module):
    """Minimal ViT: patch_size=4, embed_dim=64, 2 layers, 2 heads, CLS→FC(64→10).

    Architecture::

        PatchEmbed(3, embed=64, patch=4)  → (B, 64+1, 64)
        + CLS token + learnable positional embedding
        → 2 × TransformerEncoderBlock(64, 2 heads, mlp=128)
        → CLS output → head(64 → 10)

    Ghost layer: ``head`` (Linear(64 → 10)).
    Total: ~40K params.  Ghost coverage ≈ 1.6%.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        mlp_dim: int = 128,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0
        n_patches = (img_size // patch_size) ** 2  # 64 for 32/4

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [_EncoderBlock(embed_dim, n_heads, mlp_dim) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Patch embed: (B, C, H, W) → (B, n_patches, embed_dim)
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, p], dim=1) + self.pos_embed
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        cls_out = h[:, 0]  # CLS token
        return self.head(cls_out)


class _EncoderBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _SelfAttention(dim, n_heads)
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


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)
