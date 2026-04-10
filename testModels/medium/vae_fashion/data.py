"""Fashion-MNIST dataset for VAE reconstruction benchmark."""

from torch.utils.data import Dataset


class FashionMnistDataset(Dataset):
    """Fashion-MNIST yielding (image, flattened_target, sample_id)."""

    def __init__(self, train: bool, root: str = "data") -> None:
        try:
            from torchvision import datasets, transforms
        except ImportError as e:
            raise ImportError("torchvision required. pip install torchvision") from e

        tfm = transforms.ToTensor()  # [0,1] range, shape (1,28,28)
        self.ds = datasets.FashionMNIST(root=root, train=train, download=True, transform=tfm)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        x, _ = self.ds[idx]
        # target = flattened image (784,) for reconstruction loss
        return x, x.view(-1), idx
