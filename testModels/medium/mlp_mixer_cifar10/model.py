"""Tiny MLP-Mixer for CIFAR-10 (all-linear mixer blocks + patch embed as Conv)."""

from __future__ import annotations

import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    """Token-mixing MLP + channel-mixing MLP (Tolstikhin et al., 2021)."""

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.token_mix = nn.Sequential(
            nn.Linear(num_patches, tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, num_patches),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(embed_dim, channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        y = self.norm1(x)
        y = y.transpose(1, 2)  # (B, C, N)
        y = self.token_mix(y)
        y = y.transpose(1, 2)
        x = x + y
        x = x + self.channel_mix(self.norm2(x))
        return x


class MixerMicro(nn.Module):
    """Minimal MLP-Mixer: patch stem + mixer blocks + GAP + classifier."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 2,
        tokens_mlp_dim: int = 64,
        channels_mlp_dim: int = 128,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    self.num_patches,
                    embed_dim,
                    tokens_mlp_dim,
                    channels_mlp_dim,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, N, embed_dim)
        p = self.patch_embed(x).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            p = blk(p)
        p = self.norm(p)
        p = p.mean(dim=1)
        return self.head(p)
