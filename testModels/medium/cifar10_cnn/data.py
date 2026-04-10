"""CIFAR-10 dataset and loaders."""

import torch
from torch.utils.data import DataLoader, Dataset


class CifarDataset(Dataset):
    """CIFAR-10 yielding (image, label, sample_id)."""

    def __init__(self, train: bool, root: str = "data") -> None:
        try:
            from torchvision import datasets, transforms
        except ImportError as e:
            raise ImportError("torchvision required. pip install torchvision") from e

        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        )
        self.ds = datasets.CIFAR10(root=root, train=train, download=True, transform=tfm)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        x, y = self.ds[idx]
        return x, y, idx


def make_loaders(
    batch_size: int = 64,
    data_root: str = "data",
) -> tuple[DataLoader, DataLoader, dict[int, str]]:
    train_ds = CifarDataset(train=True, root=data_root)
    test_ds = CifarDataset(train=False, root=data_root)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    tgs = train_ds.ds.targets
    sample_meta = {i: classes[int(tgs[i])] for i in range(len(train_ds))}
    return train_loader, test_loader, sample_meta
