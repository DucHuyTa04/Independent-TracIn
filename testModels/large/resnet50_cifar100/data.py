"""CIFAR-100 dataset wrapper (reuses resnet_cifar100 data module)."""

from testModels.medium.resnet_cifar100.data import Cifar100Dataset

__all__ = ["Cifar100Dataset"]
