"""Non-destructive hook manager for capturing activations and gradients.

HookManager: attach forward/backward hooks to a single target layer.
MultiLayerBackwardGhostManager: hook multiple layers for per-layer ghost vectors.

Supports nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.Embedding,
nn.LayerNorm, nn.BatchNorm2d, and nn.RNNBase (GRU/LSTM).

Usage (forward-only):
    with HookManager(model, model.fc2) as hm:
        logits = model(inputs)
        A = hm.activation        # (B, H)
        E = error_fn(logits, targets)

Usage (backward):
    with HookManager(model, model.fc2, backward=True) as hm:
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        A = hm.activation        # (B, H)
        E = hm.grad_output       # (B, C)
"""

from __future__ import annotations

import logging
import warnings
from types import TracebackType
from typing import Any, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _conv2d_unfold_patches(
    activation: torch.Tensor,
    layer: nn.Conv2d,
) -> torch.Tensor:
    """``F.unfold`` output ``[B, C*kH*kW, L]`` (single im2col; group-aware dots in ghost_faiss)."""
    return F.unfold(
        activation,
        kernel_size=layer.kernel_size,
        dilation=layer.dilation,
        padding=layer.padding,
        stride=layer.stride,
    )


def _flatten_to_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten any tensor to [Batch, Hidden] via mean pooling.

    - 2D [B, H]      -> pass through
    - 3D [B, Seq, H]  -> mean over dim 1
    - 4D [B, C, H, W] -> mean over dims 2, 3
    """
    if tensor.dim() == 2:
        return tensor
    elif tensor.dim() == 3:
        return tensor.mean(dim=1)
    elif tensor.dim() == 4:
        return tensor.mean(dim=(2, 3))
    else:
        raise ValueError(
            f"Unsupported tensor shape {tensor.shape}. "
            "Expected 2D [B,H], 3D [B,Seq,H], or 4D [B,C,H,W]."
        )


def _unfold_conv2d_input(
    activation: torch.Tensor,
    layer: nn.Conv2d,
) -> torch.Tensor:
    """Im2col-unfold + mean-pool over spatial locations → [B, C_in*kH*kW].

    Approximate ghost for Conv2d when only 2D blocks are available.
    For exact ghosts, use keep_raw=True and sum outer products over spatial sites.
    """
    unfolded = _conv2d_unfold_patches(activation, layer)
    return unfolded.mean(dim=2)


def _zero_insert_spatial(x: torch.Tensor, stride_h: int, stride_w: int) -> torch.Tensor:
    """Insert (stride-1) zeros between spatial samples (fractionally-strided view)."""
    b, c, h, w = x.shape
    out = x.new_zeros(b, c, (h - 1) * stride_h + 1, (w - 1) * stride_w + 1)
    out[:, :, ::stride_h, ::stride_w] = x
    return out


def _unfold_conv_transpose2d_input_nosplit(
    activation: torch.Tensor,
    layer: nn.ConvTranspose2d,
) -> torch.Tensor:
    """Im2col-unfold for one input channel group (``groups==1`` slice)."""
    sh, sw = layer.stride[0], layer.stride[1]
    kh, kw = layer.kernel_size[0], layer.kernel_size[1]
    ph, pw = layer.padding[0], layer.padding[1]
    oph, opw = layer.output_padding[0], layer.output_padding[1]
    dil_h, dil_w = layer.dilation[0], layer.dilation[1]
    x_up = _zero_insert_spatial(activation, sh, sw)
    # Asymmetric padding matching PyTorch's weight-gradient computation.
    # output_padding adds extra rows/columns on one side only, so
    # F.unfold's symmetric padding is insufficient.
    pad_top = dil_h * (kh - 1) - ph
    pad_bottom = dil_h * (kh - 1) - ph + oph
    pad_left = dil_w * (kw - 1) - pw
    pad_right = dil_w * (kw - 1) - pw + opw
    x_padded = F.pad(x_up, (pad_left, pad_right, pad_top, pad_bottom))
    unfolded = F.unfold(
        x_padded,
        kernel_size=(kh, kw),
        dilation=(dil_h, dil_w),
        padding=0,
        stride=1,
    )
    return unfolded.permute(0, 2, 1)


def _unfold_conv_transpose2d_input(
    activation: torch.Tensor,
    layer: nn.ConvTranspose2d,
) -> torch.Tensor:
    """Im2col-unfold for ConvTranspose2d ghost (exact per-layer weight grad SOP)."""
    return _unfold_conv_transpose2d_input_nosplit(activation, layer)


def _maybe_append_bias_ones(act: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    """Append a trailing dimension of 1s so outer products include bias gradients."""
    if getattr(layer, "bias", None) is None:
        return act
    if act.dim() == 2:
        ones = act.new_ones(act.shape[0], 1)
        return torch.cat([act, ones], dim=-1)
    if act.dim() == 3:
        ones = act.new_ones(act.shape[0], act.shape[1], 1)
        return torch.cat([act, ones], dim=-1)
    return act


def _layernorm_x_normalized(x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
    """Match ``nn.LayerNorm`` normalized input (affine applied separately)."""
    dims = tuple(range(-len(ln.normalized_shape), 0))
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + ln.eps)


def _batchnorm2d_x_normalized(x: torch.Tensor, bn: nn.BatchNorm2d) -> torch.Tensor:
    """Normalized activations using running stats (eval-mode BatchNorm ghost)."""
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError(
            "Ghost TracIn BatchNorm2d hook requires track_running_stats=True "
            "(running_mean / running_var must be present).",
        )
    rm = bn.running_mean.detach().view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    rv = bn.running_var.detach().view(1, -1, 1, 1).to(device=x.device, dtype=x.dtype)
    return (x - rm) / torch.sqrt(rv + bn.eps)


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
        self._model = model
        self._target_layer = target_layer
        self._backward = backward
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._activation: Optional[torch.Tensor] = None
        self._logits: Optional[torch.Tensor] = None
        self._grad_output: Optional[torch.Tensor] = None

    def __enter__(self) -> HookManager:
        self._attach_hooks()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.remove_hooks()

    def _attach_hooks(self) -> None:
        layer = self._target_layer
        is_conv2d = isinstance(layer, nn.Conv2d)
        is_ct = isinstance(layer, nn.ConvTranspose2d)

        def _forward_hook(
            module: nn.Module,
            inp: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            raw = inp[0].detach()
            if is_conv2d and raw.dim() == 4:
                act = _unfold_conv2d_input(raw, layer)
            elif is_ct and raw.dim() == 4:
                act = _unfold_conv_transpose2d_input(raw, layer).mean(dim=1)
            else:
                act = _flatten_to_2d(raw)
            self._activation = _maybe_append_bias_ones(act, layer)
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


class MultiLayerBackwardGhostManager:
    """Hook multiple layers; after ``forward`` + ``loss.backward()`` get per-layer A and ∂L/∂output.

    Use with the same training loss as optimization (e.g. ``CrossEntropyLoss``, ``MSELoss``).
    Concatenate per-layer ghost vectors with ``math_utils.form_multi_layer_ghost_vectors``.
    """

    def __init__(
        self,
        target_layers: list[nn.Module],
        *,
        keep_raw: bool = False,
        max_spatial_positions: Optional[int] = None,
    ) -> None:
        if not target_layers:
            raise ValueError("target_layers must be non-empty")
        self.target_layers = list(target_layers)
        self._keep_raw = keep_raw
        self._max_spatial_positions = max_spatial_positions
        self._handles: list[Any] = []
        self._act: dict[int, Optional[torch.Tensor]] = {id(m): None for m in self.target_layers}
        self._grad: dict[int, Optional[torch.Tensor]] = {id(m): None for m in self.target_layers}
        self._raw_act: dict[int, Optional[torch.Tensor]] = {id(m): None for m in self.target_layers}
        self._raw_grad: dict[int, Optional[torch.Tensor]] = {id(m): None for m in self.target_layers}
        self._ln_xnorm: dict[int, torch.Tensor] = {}
        self._bn_xnorm: dict[int, torch.Tensor] = {}

    def __enter__(self) -> MultiLayerBackwardGhostManager:
        for layer in self.target_layers:
            lid = id(layer)
            if isinstance(layer, nn.Conv2d):
                self._register_conv2d_hooks(layer, lid)
            elif isinstance(layer, nn.ConvTranspose2d):
                self._register_conv_transpose2d_hooks(layer, lid)
            elif isinstance(layer, nn.Embedding):
                self._register_embedding_hooks(layer, lid)
            elif isinstance(layer, nn.LayerNorm):
                self._register_layernorm_hooks(layer, lid)
            elif isinstance(layer, nn.BatchNorm2d):
                self._register_batchnorm2d_hooks(layer, lid)
            elif isinstance(layer, nn.RNNBase):
                self._register_rnn_hooks(layer, lid)
            else:
                self._register_linear_like_hooks(layer, lid)
        return self

    def _register_linear_like_hooks(self, layer: nn.Module, lid: int) -> None:
        def _forward_hook(
            module: nn.Module,
            inp: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            raw = inp[0].detach()
            if not self._keep_raw:
                act = _flatten_to_2d(raw)
                self._act[lid] = _maybe_append_bias_ones(act, layer)
            if self._keep_raw:
                ra = raw
                if ra.dim() in (2, 3):
                    self._raw_act[lid] = _maybe_append_bias_ones(ra, layer)
                else:
                    self._raw_act[lid] = ra

        def _backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            raw = grad_output[0].detach()
            if not self._keep_raw:
                self._grad[lid] = _flatten_to_2d(raw)
            if self._keep_raw:
                self._raw_grad[lid] = raw

        self._handles.append(layer.register_forward_hook(_forward_hook))
        self._handles.append(layer.register_full_backward_hook(_backward_hook))

    def _register_conv2d_hooks(self, layer: nn.Conv2d, lid: int) -> None:
        def _forward_hook(
            module: nn.Module,
            inp: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            raw = inp[0].detach()
            if raw.dim() != 4:
                raise ValueError(f"Conv2d hook expected 4D input, got {raw.shape}")
            unfolded = _conv2d_unfold_patches(raw, layer)
            if not self._keep_raw:
                act = unfolded.mean(dim=2)
                self._act[lid] = _maybe_append_bias_ones(act, layer)
            if self._keep_raw:
                _b, _hk, l_spatial = unfolded.shape
                max_l = self._max_spatial_positions
                if max_l is not None and l_spatial > max_l:
                    logger.warning(
                        "Conv2d raw ghost: L=%d spatial positions exceeds "
                        "max_spatial_positions=%d; using mean-pooled fallback "
                        "for layer (approximate ghost).",
                        l_spatial,
                        max_l,
                    )
                    mp = unfolded.mean(dim=2)
                    self._raw_act[lid] = _maybe_append_bias_ones(mp, layer)
                else:
                    self._raw_act[lid] = _maybe_append_bias_ones(
                        unfolded.permute(0, 2, 1), layer,
                    )

        def _backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            raw = grad_output[0].detach()
            if not self._keep_raw:
                self._grad[lid] = _flatten_to_2d(raw)
            if self._keep_raw:
                if raw.dim() == 4:
                    _b, _c_out, hp, wp = raw.shape
                    l_spatial = hp * wp
                    max_l = self._max_spatial_positions
                    if max_l is not None and l_spatial > max_l:
                        self._grad[lid] = _flatten_to_2d(raw)
                        self._raw_grad[lid] = self._grad[lid]
                    else:
                        self._raw_grad[lid] = raw.flatten(2).permute(0, 2, 1)
                else:
                    self._raw_grad[lid] = raw

        self._handles.append(layer.register_forward_hook(_forward_hook))
        self._handles.append(layer.register_full_backward_hook(_backward_hook))

    def _register_conv_transpose2d_hooks(self, layer: nn.ConvTranspose2d, lid: int) -> None:
        def _forward_hook(
            module: nn.Module,
            inp: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            raw = inp[0].detach()
            if raw.dim() != 4:
                raise ValueError(f"ConvTranspose2d hook expected 4D input, got {raw.shape}")
            unfolded = _unfold_conv_transpose2d_input(raw, layer)
            act = unfolded.mean(dim=1)
            if not self._keep_raw:
                self._act[lid] = _maybe_append_bias_ones(act, layer)
            if self._keep_raw:
                _b, l_spatial, _hk = unfolded.shape
                max_l = self._max_spatial_positions
                if max_l is not None and l_spatial > max_l:
                    logger.warning(
                        "ConvTranspose2d raw ghost: L=%d exceeds max_spatial_positions=%d; "
                        "using mean-pooled fallback (approximate).",
                        l_spatial,
                        max_l,
                    )
                    self._raw_act[lid] = _maybe_append_bias_ones(act, layer)
                else:
                    self._raw_act[lid] = _maybe_append_bias_ones(unfolded, layer)

            def _tensor_grad_hook(grad: torch.Tensor) -> None:
                g_raw = grad.detach()
                if not self._keep_raw:
                    self._grad[lid] = _flatten_to_2d(g_raw)
                if self._keep_raw:
                    if g_raw.dim() == 4:
                        _b, _c_out, hp, wp = g_raw.shape
                        l_spatial = hp * wp
                        max_l = self._max_spatial_positions
                        if max_l is not None and l_spatial > max_l:
                            self._grad[lid] = _flatten_to_2d(g_raw)
                            self._raw_grad[lid] = self._grad[lid]
                        else:
                            self._raw_grad[lid] = g_raw.flatten(2).permute(0, 2, 1)
                    else:
                        self._raw_grad[lid] = g_raw

            h = output.register_hook(_tensor_grad_hook)
            self._handles.append(h)

        self._handles.append(layer.register_forward_hook(_forward_hook))

    def _register_batchnorm2d_hooks(self, layer: nn.BatchNorm2d, lid: int) -> None:
        """Ghost for affine BN in eval mode.

        Uses tensor-level output.register_hook to avoid BackwardHookFunction wrapper
        that breaks downstream inplace ops (e.g. residual +=, relu inplace).
        """

        def _forward_hook(
            module: nn.Module,
            inp: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if layer.training:
                warnings.warn(
                    "BatchNorm2d ghost uses running stats; in training mode the "
                    "live forward uses batch stats — ghost may not match autograd.",
                    UserWarning,
                    stacklevel=3,
                )
            x = inp[0].detach()
            xn = _batchnorm2d_x_normalized(x, layer)
            self._bn_xnorm[lid] = xn
            self._act[lid] = _flatten_to_2d(xn)

            # Capture grad_output via a tensor hook — no BackwardHookFunction
            # wrapper, so downstream inplace ops (+=, relu_) are safe.
            def _tensor_grad_hook(grad: torch.Tensor) -> None:
                g = grad.detach().float()
                xn_saved = self._bn_xnorm.pop(lid)
                h_gamma = (xn_saved.float() * g).sum(dim=(2, 3))
                h_beta = g.sum(dim=(2, 3))
                self._grad[lid] = h_beta
                if self._keep_raw:
                    self._raw_act[lid] = h_gamma
                    self._raw_grad[lid] = h_beta

            h = output.register_hook(_tensor_grad_hook)
            self._handles.append(h)

        self._handles.append(layer.register_forward_hook(_forward_hook))

    def _register_embedding_hooks(self, layer: nn.Embedding, lid: int) -> None:
        def _forward_hook(
            module: nn.Module,
            inp: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            toks = inp[0].detach().long()
            self._act[lid] = toks.float()  # placeholder for 2D path (invalid for ghost)
            if self._keep_raw:
                self._raw_act[lid] = toks

        def _backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            g = grad_output[0].detach()
            if not self._keep_raw:
                self._grad[lid] = _flatten_to_2d(g.float())
            if self._keep_raw:
                self._raw_grad[lid] = g

        self._handles.append(layer.register_forward_hook(_forward_hook))
        self._handles.append(layer.register_full_backward_hook(_backward_hook))

    def _register_layernorm_hooks(self, layer: nn.LayerNorm, lid: int) -> None:
        def _forward_hook(
            module: nn.Module,
            inp: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            x = inp[0].detach()
            xn = _layernorm_x_normalized(x, layer)
            self._ln_xnorm[lid] = xn
            self._act[lid] = xn.mean(dim=tuple(range(1, xn.dim() - 1))) if xn.dim() > 2 else xn

        def _backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            g = grad_output[0].detach().float()
            xn = self._ln_xnorm.pop(lid)
            if g.dim() <= 2:
                h_gamma = xn * g
                h_beta = g
            else:
                lead = tuple(range(1, g.dim() - 1))
                h_gamma = (xn * g).sum(dim=lead)
                h_beta = g.sum(dim=lead)
            self._grad[lid] = h_beta
            if self._keep_raw:
                self._raw_act[lid] = h_gamma
                self._raw_grad[lid] = h_beta

        self._handles.append(layer.register_forward_hook(_forward_hook))
        self._handles.append(layer.register_full_backward_hook(_backward_hook))

    def _register_rnn_hooks(self, layer: nn.RNNBase, lid: int) -> None:
        """Hook RNN/GRU/LSTM: capture input sequence and grad of output sequence."""
        batch_first = getattr(layer, "batch_first", False)

        def _forward_hook(
            module: nn.Module,
            inp: tuple[torch.Tensor, ...],
            output: object,
        ) -> None:
            raw = inp[0].detach()
            if not batch_first and raw.dim() == 3:
                raw = raw.transpose(0, 1)  # (T, B, H) -> (B, T, H)
            if not self._keep_raw:
                act = _flatten_to_2d(raw)
                self._act[lid] = act
            if self._keep_raw:
                self._raw_act[lid] = raw

        def _backward_hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            raw = grad_output[0].detach()
            if not batch_first and raw.dim() == 3:
                raw = raw.transpose(0, 1)  # (T, B, H) -> (B, T, H)
            if not self._keep_raw:
                self._grad[lid] = _flatten_to_2d(raw)
            if self._keep_raw:
                self._raw_grad[lid] = raw

        self._handles.append(layer.register_forward_hook(_forward_hook))
        self._handles.append(layer.register_full_backward_hook(_backward_hook))

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._ln_xnorm.clear()
        self._bn_xnorm.clear()

    def numpy_blocks(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Per-layer activations and error signals as float32 numpy (batch-aligned)."""
        A_list: list[np.ndarray] = []
        E_list: list[np.ndarray] = []
        for layer in self.target_layers:
            if isinstance(layer, (nn.Embedding, nn.LayerNorm, nn.BatchNorm2d, nn.RNNBase)):
                raise RuntimeError(
                    "numpy_blocks() does not support nn.Embedding / nn.LayerNorm / "
                    "nn.BatchNorm2d / nn.RNNBase; use keep_raw=True and raw_torch_blocks().",
                )
            lid = id(layer)
            a = self._act.get(lid)
            e = self._grad.get(lid)
            if a is None or e is None:
                raise RuntimeError(
                    "Missing activation or grad_output for a hooked layer; "
                    "run forward + backward inside the context."
                )
            A_list.append(a.cpu().numpy().astype(np.float32))
            E_list.append(e.cpu().numpy().astype(np.float32))
        return A_list, E_list

    def torch_blocks(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Per-layer activations and ∂L/∂output as float32 tensors (same device as capture)."""
        A_list: list[torch.Tensor] = []
        E_list: list[torch.Tensor] = []
        for layer in self.target_layers:
            if isinstance(layer, (nn.Embedding, nn.LayerNorm, nn.BatchNorm2d, nn.RNNBase)):
                raise RuntimeError(
                    "torch_blocks() does not support nn.Embedding / nn.LayerNorm / "
                    "nn.BatchNorm2d / nn.RNNBase; use keep_raw=True and raw_torch_blocks().",
                )
            lid = id(layer)
            a = self._act.get(lid)
            e = self._grad.get(lid)
            if a is None or e is None:
                raise RuntimeError(
                    "Missing activation or grad_output for a hooked layer; "
                    "run forward + backward inside the context."
                )
            A_list.append(a.detach().float())
            E_list.append(e.detach().float())
        return A_list, E_list

    def raw_torch_blocks(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Per-layer activations and ∂L/∂output as un-flattened float32 tensors.

        Requires ``keep_raw=True`` at construction time.  Returns shapes suitable
        for sum-of-outer-products: ``[B, T, H]`` for sequence models,
        ``[B, L, C_in*kH*kW]`` and ``[B, L, C_out]`` for ``nn.Conv2d`` (L = H'*W'),
        or 4D/2D tensors for other layers.  If ``max_spatial_positions`` was exceeded
        for a Conv2d layer, that layer falls back to 2D mean-pooled blocks.
        """
        if not self._keep_raw:
            raise RuntimeError(
                "raw_torch_blocks() requires keep_raw=True at construction."
            )
        A_list: list[torch.Tensor] = []
        E_list: list[torch.Tensor] = []
        for layer in self.target_layers:
            lid = id(layer)
            a = self._raw_act.get(lid)
            e = self._raw_grad.get(lid)
            if a is None or e is None:
                raise RuntimeError(
                    "Missing raw activation or grad_output for a hooked layer; "
                    "run forward + backward inside the context."
                )
            if isinstance(layer, nn.Embedding):
                A_list.append(a.detach().long())
            else:
                A_list.append(a.detach().float())
            E_list.append(e.detach().float())
        return A_list, E_list
