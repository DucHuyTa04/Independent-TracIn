"""CIFAR-10 dataset for ViT benchmark (reuses same data as cifar10_cnn)."""

from torch.utils.data import Dataset


class CifarDataset(Dataset):
    """CIFAR-10 yielding (image, label, sample_id) — normalized for ViT."""

    def __init__(self, train: bool, root: str = "data") -> None:
        try:
            from torchvision import datasets, transforms
        except ImportError as e:
            raise ImportError("torchvision required. pip install torchvision") from e

        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.ds = datasets.CIFAR10(root=root, train=train, download=True, transform=tfm)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        x, y = self.ds[idx]
        return x, y, idx
