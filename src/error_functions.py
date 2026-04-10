"""Built-in error signals E for TracIn ghost vectors.

These must match the training loss used for the model:
- Cross-entropy (classification): E = softmax(logits) - one_hot(targets)
- MSE (regression): E = logits - targets (up to constant factor)
"""

from __future__ import annotations

import torch


def classification_error(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """E = softmax(logits) - one_hot(targets). Matches cross-entropy gradient w.r.t. logits.

    Args:
        logits: (batch, num_classes)
        targets: (batch,) integer class indices

    Returns:
        (batch, num_classes) error tensor.
    """
    probs = torch.softmax(logits, dim=-1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, targets.unsqueeze(1).long(), 1.0)
    return probs - one_hot


def regression_error(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """E = logits - targets for MSE-style loss.

    Omits the 2/N factor from MSELoss(reduction='mean'). Rankings are unchanged;
    raw magnitudes differ by a constant from full-parameter gradients.
    """
    t = targets.float()
    if t.dim() < logits.dim():
        t = t.view_as(logits) if t.numel() == logits.numel() else t.unsqueeze(-1)
    if t.shape != logits.shape:
        t = t.view_as(logits)
    return logits - t


def get_error_fn(loss_type: str):
    """Return error_fn for ``loss_type``.

    Args:
        loss_type: ``"classification"`` or ``"regression"``.

    Returns:
        Callable (logits, targets) -> E.

    Raises:
        ValueError: if ``loss_type`` is unknown.
    """
    lt = loss_type.strip().lower()
    if lt in ("classification", "class", "ce", "cross_entropy"):
        return classification_error
    if lt in ("regression", "reg", "mse", "l2"):
        return regression_error
    raise ValueError(
        f"Unknown loss_type {loss_type!r}. Use 'classification' or 'regression'."
    )
