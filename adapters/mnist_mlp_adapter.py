import torch
import torch.nn as nn

from adapters.model_adapter_base import BaseModelAdapter


class MnistMLP(nn.Module):
    """Single-hidden-layer MLP: 784 -> 128 -> 10."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        hidden = self.relu(self.fc1(x))
        return self.fc2(hidden)

    def hidden_activation(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.relu(self.fc1(x))


class MnistMLPAdapter(BaseModelAdapter):
    """Model adapter for MNIST MLP training and extraction."""

    def build_model(self, config: dict, device: str) -> torch.nn.Module:
        model_cfg = config["model"]
        model = MnistMLP(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_classes=int(model_cfg["num_classes"]),
        )
        return model.to(device)

    def hidden_activation(self, model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
        return model.hidden_activation(images)
