"""MNIST dataset and data loader factory."""

import torch
from torch.utils.data import DataLoader, Dataset


class MnistDataset(Dataset):
    """MNIST dataset yielding (input, label, sample_id) 3-tuples."""

    def __init__(self, train: bool, root: str = "../../data") -> None:
        try:
            from torchvision import datasets
        except ImportError:
            raise ImportError("torchvision required for MNIST. pip install torchvision")

        self.ds = datasets.MNIST(root=root, train=train, download=True)
        self.data = self.ds.data.float() / 255.0
        self.targets = self.ds.targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        x = self.data[idx].unsqueeze(0)  # (1, 28, 28)
        y = self.targets[idx]
        return x, y, idx


def make_loaders(
    batch_size: int = 64,
    data_root: str = "../../data",
) -> tuple[DataLoader, DataLoader, dict[int, str]]:
    """Build train/test loaders and sample-to-rights-holder metadata.

    Returns:
        (train_loader, test_loader, sample_id_to_rights_holder)
    """
    train_ds = MnistDataset(train=True, root=data_root)
    test_ds = MnistDataset(train=False, root=data_root)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Use digit class as placeholder rights holder
    sample_meta = {
        i: f"digit_{train_ds.targets[i].item()}" for i in range(len(train_ds))
    }

    return train_loader, test_loader, sample_meta
