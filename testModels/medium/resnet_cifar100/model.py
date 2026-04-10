"""Small ResNet for CIFAR-100 (benchmark-friendly, ~170K params)."""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Pre-activation BasicBlock (no BN for deterministic ghost vectors)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + self.shortcut(x))


class ResNetSmall(nn.Module):
    """Tiny ResNet: [2,2,2] blocks, channels [16,32,64], two-layer FC head.

    Architecture::

        conv1(3→16) → [Block×2: 16→16] → [Block×2: 16→32, stride=2]
        → [Block×2: 32→64, stride=2] → AdaptiveAvgPool(1)
        → fc1(64→256) → ReLU → fc2(256→num_classes)

    Ghost layers: ``fc1`` and ``fc2`` (64×256 + 256×100 = 42,240 weight params).
    Total params: ~217K.  Ghost coverage ≈ 19%.
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.layer1 = nn.Sequential(BasicBlock(16, 16), BasicBlock(16, 16))
        self.layer2 = nn.Sequential(BasicBlock(16, 32, stride=2), BasicBlock(32, 32))
        self.layer3 = nn.Sequential(BasicBlock(32, 64, stride=2), BasicBlock(64, 64))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
