"""CIFAR-100 dataset and loaders."""

from torch.utils.data import DataLoader, Dataset


class Cifar100Dataset(Dataset):
    """CIFAR-100 yielding (image, label, sample_id)."""

    def __init__(self, train: bool, root: str = "data") -> None:
        try:
            from torchvision import datasets, transforms
        except ImportError as e:
            raise ImportError("torchvision required. pip install torchvision") from e

        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        self.ds = datasets.CIFAR100(root=root, train=train, download=True, transform=tfm)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        x, y = self.ds[idx]
        return x, y, idx
