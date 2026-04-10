"""ResNet-50 for CIFAR-100 (~23.7M params, conv-heavy).

Uses torchvision's ResNet50 backbone with a modified first conv layer
(3×3 kernel, no maxpool) to fit 32×32 CIFAR images.  The FC head is
``fc(2048 → 100)``.

Ghost coverage: ``fc`` = 2048*100 + 100 = 204,900 / 23.7M ≈ 0.86%.
This is the realistic worst-case for ghost vectors: nearly all mass is
in Conv2d layers, unreachable by the current ghost implementation.
"""

import torch.nn as nn

try:
    from torchvision.models import resnet50
except ImportError:
    resnet50 = None  # type: ignore[assignment,misc]


def build_resnet50_cifar100(num_classes: int = 100) -> nn.Module:
    """Return a ResNet-50 adapted for CIFAR-100 (32×32 inputs)."""
    if resnet50 is None:
        raise ImportError("torchvision required. pip install torchvision")
    model = resnet50(weights=None, num_classes=num_classes)
    # Replace the ImageNet-oriented stem (7×7 conv + maxpool) with a CIFAR-friendly 3×3.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # type: ignore[assignment]
    return model
