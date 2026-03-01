import torch


class BaseModelAdapter:
    """Interface for model-specific logic used by trainer and extractors."""

    def build_model(self, config: dict, device: str) -> torch.nn.Module:
        raise NotImplementedError("Subclasses must implement build_model().")

    def forward_logits(self, model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
        return model(images)

    def hidden_activation(self, model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement hidden_activation().")
