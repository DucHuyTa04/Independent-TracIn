"""Full-parameter TracIn (no ghost trick) for small models — benchmark baseline."""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _flatten_grads(model: nn.Module) -> np.ndarray:
    """Concatenate all parameter gradients into one float64 vector."""
    parts: list[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is not None:
            parts.append(p.grad.detach().reshape(-1).double())
        else:
            parts.append(torch.zeros(p.numel(), dtype=torch.float64, device=p.device))
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return torch.cat(parts).cpu().numpy()


def _per_sample_gradient(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    device: str,
) -> np.ndarray:
    """One forward-backward; return flattened full gradient (float64 numpy)."""
    model.zero_grad(set_to_none=True)
    x_b = x.to(device).unsqueeze(0)  # always add batch dim (single sample)
    y_b = y.to(device)
    if y_b.dim() == 0:
        y_b = y_b.unsqueeze(0)
    # MSE / regression: ensure target shape matches typical (N, out_dim)
    if isinstance(loss_fn, nn.MSELoss) and y_b.dim() == 1:
        y_b = y_b.unsqueeze(0)

    # Eval mode for deterministic gradients; RNN stays train for cuDNN backward.
    was_training = model.training
    rnn_prev = [
        (m, m.training)
        for m in model.modules()
        if isinstance(m, nn.RNNBase)
    ]
    model.eval()
    for m, _ in rnn_prev:
        m.train()
    try:
        logits = model(x_b)
        loss = loss_fn(logits, y_b)
        if loss.dim() > 0:
            loss = loss.sum()
        loss.backward()
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
        for m, prev in rnn_prev:
            m.train(prev)
    return _flatten_grads(model)


def compute_full_gradient_tracin_scores(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    query_inputs: torch.Tensor,
    query_targets: torch.Tensor,
    checkpoints: list[dict],
    device: str = "cpu",
    load_weights_fn: Optional[Callable[[nn.Module, str, str], None]] = None,
) -> dict[int, float]:
    """Full-parameter TracIn: ``sum_t lr_t <grad_query_t, grad_train_i_t>``.

    No ghost trick, projection, or FAISS. Intended for small models (benchmarks).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _default_load_weights(m: nn.Module, path: str, dev: str) -> None:
        state = torch.load(path, map_location=dev, weights_only=True)
        m.load_state_dict(state)

    load_fn = load_weights_fn or _default_load_weights

    all_sample_ids: list[int] = []
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for inputs, targets, batch_ids in data_loader:
        for i in range(inputs.shape[0]):
            all_sample_ids.append(int(batch_ids[i]))
            xs.append(inputs[i].detach().cpu())
            ys.append(targets[i].detach().cpu())
    n_train = len(all_sample_ids)
    if n_train == 0:
        return {}

    if query_inputs.dim() == 1:
        qx = query_inputs.unsqueeze(0).detach().cpu()
    else:
        qx = query_inputs.detach().cpu()
    if query_targets.dim() == 0:
        qy = query_targets.unsqueeze(0).detach().cpu()
    elif query_targets.dim() == 1 and query_targets.shape[0] == qx.shape[0]:
        qy = query_targets.detach().cpu()
    else:
        qy = query_targets.detach().cpu()
    n_q = int(qx.shape[0])
    if n_q == 0:
        return {}

    scores = np.zeros(n_train, dtype=np.float64)
    model.to(device)

    for ckpt in checkpoints:
        lr = float(ckpt["learning_rate"])
        load_fn(model, ckpt["weights_path"], device)

        g_queries = [
            _per_sample_gradient(model, loss_fn, qx[k], qy[k], device)
            for k in range(n_q)
        ]

        for i in range(n_train):
            g_train = _per_sample_gradient(model, loss_fn, xs[i], ys[i], device)
            dot_mean = sum(float(np.dot(gq, g_train)) for gq in g_queries) / n_q
            scores[i] += lr * dot_mean

    return {sid: float(scores[j]) for j, sid in enumerate(all_sample_ids)}
