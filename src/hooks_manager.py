"""Non-destructive hook manager for capturing activations and gradients.

Attaches PyTorch forward/backward hooks to a target layer, capturing:
    - A (activation): input to the target layer, shape [Batch, Hidden]
    - grad_output: gradient of loss w.r.t. layer output (backward mode only)

All tensors are flattened to 2D [Batch, Hidden] via mean pooling over
spatial (CNN) or sequence (Transformer) dimensions before storage.

Usage (forward-only, classification/regression):
    with HookManager(model, model.fc2) as hm:
        logits = model(inputs)
        A = hm.activation                       # (B, H)
        E = error_fn(logits, targets)            # user-provided

Usage (forward+backward, generative models):
    with HookManager(model, model.cross_attn, backward=True) as hm:
        logits = model(inputs)
        loss = surrogate_loss(logits, targets)
        loss.backward()
        A = hm.activation                        # (B, H)
        E = hm.grad_output                       # (B, C)
"""

from __future__ import annotations

from types import TracebackType
from typing import Optional, Type

import torch
import torch.nn as nn


def _flatten_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten any tensor to [Batch, Hidden] via mean pooling.

    Dimensionality standardization (Rule 3):
        - 2D [B, H]         -> pass through
        - 3D [B, Seq, H]    -> mean over dim 1 -> [B, H]
        - 4D [B, C, H, W]   -> mean over dims 2,3 -> [B, C]

    Args:
        tensor: Input tensor of any supported shape.

    Returns:
        2D tensor of shape [Batch, Hidden].
    """
    if tensor.dim() == 2:
        return tensor
    elif tensor.dim() == 3:
        # Transformer: [Batch, Seq, Hidden] -> mean over sequence
        return tensor.mean(dim=1)
    elif tensor.dim() == 4:
        # CNN: [Batch, Channels, H, W] -> mean over spatial dims
        return tensor.mean(dim=(2, 3))
    else:
        raise ValueError(
            f"Unsupported tensor shape {tensor.shape}. "
            "Expected 2D [B,H], 3D [B,Seq,H], or 4D [B,C,H,W]."
        )


class HookManager:
    """Context manager for non-destructive tensor interception.

    Attaches forward (and optionally backward) hooks to a target layer.
    Hooks are guaranteed to be removed on exit, even if an exception occurs.

    Attributes:
        activation: Captured input activation, shape [Batch, Hidden].
        grad_output: Captured gradient of loss w.r.t. layer output (backward mode).
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        backward: bool = False,
    ) -> None:
        """Initialize the hook manager.

        Args:
            model: The host model (never modified).
            target_layer: The nn.Module layer to hook (e.g., model.fc2).
            backward: If True, also register a full backward hook to capture
                      grad_output for generative models.
        """
        self._model = model
        self._target_layer = target_layer
        self._backward = backward
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._activation: Optional[torch.Tensor] = None
        self._logits: Optional[torch.Tensor] = None
        self._grad_output: Optional[torch.Tensor] = None

    def __enter__(self) -> HookManager:
        """Register hooks and return self."""
        self._attach_hooks()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Remove all hooks unconditionally."""
        self.remove_hooks()

    def _attach_hooks(self) -> None:
        """Register forward hook (always) and backward hook (if requested)."""

        def _forward_hook(
            module: nn.Module,
            inp: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            self._activation = _flatten_to_2d(inp[0].detach())
            self._logits = output.detach()

        handle = self._target_layer.register_forward_hook(_forward_hook)
        self._handles.append(handle)

        if self._backward:

            def _backward_hook(
                module: nn.Module,
                grad_input: tuple[torch.Tensor, ...],
                grad_output: tuple[torch.Tensor, ...],
            ) -> None:
                self._grad_output = _flatten_to_2d(grad_output[0].detach())

            handle = self._target_layer.register_full_backward_hook(_backward_hook)
            self._handles.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks from the target layer."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def activation(self) -> torch.Tensor:
        """Captured input activation, shape [Batch, Hidden].

        Raises:
            RuntimeError: If no forward pass has been executed yet.
        """
        if self._activation is None:
            raise RuntimeError(
                "No activation captured. Run a forward pass first."
            )
        return self._activation

    @property
    def logits(self) -> torch.Tensor:
        """Captured layer output (logits), shape [Batch, C].

        Raises:
            RuntimeError: If no forward pass has been executed yet.
        """
        if self._logits is None:
            raise RuntimeError(
                "No logits captured. Run a forward pass first."
            )
        return self._logits

    @property
    def grad_output(self) -> torch.Tensor:
        """Captured gradient of loss w.r.t. layer output (backward mode).

        Raises:
            RuntimeError: If backward=False or no backward pass executed.
        """
        if not self._backward:
            raise RuntimeError(
                "HookManager not in backward mode. Set backward=True."
            )
        if self._grad_output is None:
            raise RuntimeError(
                "No grad_output captured. Run loss.backward() first."
            )
        return self._grad_output
