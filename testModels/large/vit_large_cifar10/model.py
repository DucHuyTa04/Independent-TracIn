"""Large Vision Transformer for CIFAR-10 (~22M params).

Scales ``ViTMicro`` to production-relevant size:
  - 12 encoder blocks, embed_dim=384, 6 heads, mlp_dim=1536, patch_size=4

Ghost coverage (hooking all nn.Linear): ~95% of total params (nearly all-linear).
This model demonstrates that transformers at scale are ideal for ghost vectors.
"""

from testModels.medium.vit_cifar10.model import ViTMicro


def build_large_vit(num_classes: int = 10, img_size: int = 32) -> ViTMicro:
    """Return a ~22M param ViT for CIFAR-10."""
    return ViTMicro(
        img_size=img_size,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=384,
        n_heads=6,
        n_layers=12,
        mlp_dim=1536,
    )
