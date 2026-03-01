import torch


class BaseTaskAdapter:
    """Task adapter interface for loss and ghost error-signal definitions."""

    def prepare_targets(self, targets: torch.Tensor, device: str) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement prepare_targets().")

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement compute_loss().")

    def error_signal(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement error_signal().")
