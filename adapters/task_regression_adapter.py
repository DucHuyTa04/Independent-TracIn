import torch
import torch.nn as nn

from adapters.task_adapter_base import BaseTaskAdapter


class RegressionTaskAdapter(BaseTaskAdapter):
    """Regression task behavior for training and ghost extraction."""

    def __init__(self):
        self.criterion = nn.MSELoss()

    def prepare_targets(self, targets: torch.Tensor, device: str) -> torch.Tensor:
        target = targets.to(device).float()
        if target.dim() == 1:
            target = target.unsqueeze(1)
        return target

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.shape != logits.shape:
            targets = targets.expand_as(logits)
        return self.criterion(logits, targets)

    def error_signal(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.shape != logits.shape:
            targets = targets.expand_as(logits)
        return logits - targets
