"""Small CNN for CIFAR-10 (benchmark-friendly)."""

import torch.nn as nn


class CifarSmallCNN(nn.Module):
    """Conv -> ReLU -> adaptive pool -> linear classifier."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
