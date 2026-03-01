import torch
import torch.nn as nn

from adapters.task_adapter_base import BaseTaskAdapter


class ClassificationTaskAdapter(BaseTaskAdapter):
    """Classification task behavior for training and ghost extraction."""

    def __init__(self):
        self.criterion = nn.CrossEntropyLoss()

    def prepare_targets(self, targets: torch.Tensor, device: str) -> torch.Tensor:
        return targets.to(device).long()

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)

    def error_signal(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        error = probabilities.clone()
        for row_idx in range(logits.shape[0]):
            error[row_idx, int(targets[row_idx].item())] -= 1.0
        return error
