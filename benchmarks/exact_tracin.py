"""Exact last-layer TracIn scores (full ghost vectors, no projection / FAISS)."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.hooks_manager import HookManager
from src.math_utils import (
    apply_adam_correction,
    form_ghost_vectors,
    load_adam_second_moment,
)

logger = logging.getLogger(__name__)


def _weight_shape_for_adam(layer: nn.Module) -> Optional[tuple[int, int]]:
    w = getattr(layer, "weight", None)
    if w is not None and w.dim() == 2:
        return (int(w.shape[0]), int(w.shape[1]))
    return None


def compute_exact_tracin_scores(
    model: nn.Module,
    target_layer: nn.Module,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    query_inputs: torch.Tensor,
    query_targets: torch.Tensor,
    checkpoints: list[dict],
    adam_param_key: Union[int, str] = 2,
    optimizer_state_path: Optional[str] = None,
    use_adam: bool = True,
    device: str = "cpu",
    load_weights_fn: Optional[Callable[[nn.Module, str, str], None]] = None,
) -> dict[int, float]:
    """Brute-force TracIn using full ghost vectors (no SJLT, no FAISS).

    Accumulates ``sum_t lr_t * g_{i,t}`` per training sample. Query ghost is
    built from the last checkpoint only. Returns ``{sample_id: score}``.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _default_load_weights(m: nn.Module, path: str, dev: str) -> None:
        state = torch.load(path, map_location=dev, weights_only=True)
        m.load_state_dict(state)

    load_fn = load_weights_fn or _default_load_weights

    all_sample_ids: list[int] = []
    for _, _, batch_ids in data_loader:
        all_sample_ids.extend(int(x) for x in batch_ids)
    n_train = len(all_sample_ids)
    if n_train == 0:
        return {}

    # Probe ghost dim
    model.to(device)
    load_fn(model, checkpoints[0]["weights_path"], device)
    model.eval()
    inputs0, targets0, _ = next(iter(data_loader))
    inputs0 = inputs0.to(device)
    targets0 = targets0.to(device)
    with HookManager(model, target_layer) as hm:
        with torch.no_grad():
            logits0 = model(inputs0)
        A0 = hm.activation
        E0 = error_fn(logits0, targets0)
    H = A0.shape[1]
    C = E0.shape[1] if E0.dim() > 1 else 1
    ghost_dim = H * C

    accumulated = np.zeros((n_train, ghost_dim), dtype=np.float32)
    w_shape = _weight_shape_for_adam(target_layer)

    for ckpt in checkpoints:
        weights_path = ckpt["weights_path"]
        opt_path = ckpt.get("optimizer_state_path")
        lr = float(ckpt["learning_rate"])
        load_fn(model, weights_path, device)
        model.eval()

        adam_v = None
        if use_adam and opt_path:
            try:
                adam_v = load_adam_second_moment(
                    opt_path, adam_param_key, weight_shape=w_shape,
                )
            except Exception as e:
                logger.warning("Exact TracIn: could not load Adam state: %s", e)

        offset = 0
        for inputs, targets, _ in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with HookManager(model, target_layer) as hm:
                with torch.no_grad():
                    logits = model(inputs)
                A = hm.activation.cpu().numpy().astype(np.float32)
                E_tensor = error_fn(logits, targets)
                E = E_tensor.detach().cpu().numpy().astype(np.float32)
            g = form_ghost_vectors(A, E)
            if adam_v is not None:
                g = apply_adam_correction(g, adam_v)
            bs = g.shape[0]
            accumulated[offset : offset + bs] += lr * g
            offset += bs

    last = checkpoints[-1]
    load_fn(model, last["weights_path"], device)
    model.eval()
    opt_for_query = optimizer_state_path or last.get("optimizer_state_path")

    if query_inputs.dim() == 1:
        query_inputs = query_inputs.unsqueeze(0)
    if query_targets.dim() == 0:
        query_targets = query_targets.unsqueeze(0)
    n_q = int(query_inputs.shape[0])
    dots_sum = np.zeros(n_train, dtype=np.float64)

    for qi in range(n_q):
        q_in = query_inputs[qi : qi + 1].to(device)
        q_tg = query_targets[qi : qi + 1].to(device)
        if q_tg.dim() == 0:
            q_tg = q_tg.unsqueeze(0)

        with HookManager(model, target_layer) as hm:
            with torch.no_grad():
                q_logits = model(q_in)
            Aq = hm.activation.cpu().numpy().astype(np.float32)
            Eq = error_fn(q_logits, q_tg).detach().cpu().numpy().astype(np.float32)
        g_q = form_ghost_vectors(Aq, Eq)
        if use_adam and opt_for_query:
            try:
                av = load_adam_second_moment(
                    opt_for_query, adam_param_key, weight_shape=w_shape,
                )
                g_q = apply_adam_correction(g_q, av)
            except Exception as e:
                logger.warning("Exact TracIn query: Adam correction skipped: %s", e)

        dots_sum += (g_q @ accumulated.T).ravel() / n_q

    return {sid: float(dots_sum[j]) for j, sid in enumerate(all_sample_ids)}


def compute_true_tracin_ghost_scores(
    model: nn.Module,
    target_layer: nn.Module,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    query_inputs: torch.Tensor,
    query_targets: torch.Tensor,
    checkpoints: list[dict],
    adam_param_key: Union[int, str] = 2,
    use_adam: bool = True,
    device: str = "cpu",
    load_weights_fn: Optional[Callable[[nn.Module, str, str], None]] = None,
) -> dict[int, float]:
    """True last-layer ghost TracIn: ``sum_t lr_t <g_query_t, g_train_i_t>``.

    Recomputes the query ghost at **each** checkpoint (not TracIn-last-on-query).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _default_load_weights(m: nn.Module, path: str, dev: str) -> None:
        state = torch.load(path, map_location=dev, weights_only=True)
        m.load_state_dict(state)

    load_fn = load_weights_fn or _default_load_weights

    all_sample_ids: list[int] = []
    for _, _, batch_ids in data_loader:
        all_sample_ids.extend(int(x) for x in batch_ids)
    n_train = len(all_sample_ids)
    if n_train == 0:
        return {}

    w_shape = _weight_shape_for_adam(target_layer)
    scores = np.zeros(n_train, dtype=np.float64)

    if query_inputs.dim() == 1:
        query_inputs = query_inputs.unsqueeze(0)
    if query_targets.dim() == 0:
        query_targets = query_targets.unsqueeze(0)
    n_q = int(query_inputs.shape[0])

    for ckpt in checkpoints:
        weights_path = ckpt["weights_path"]
        opt_path = ckpt.get("optimizer_state_path")
        lr = float(ckpt["learning_rate"])
        load_fn(model, weights_path, device)
        model.eval()

        adam_v = None
        if use_adam and opt_path:
            try:
                adam_v = load_adam_second_moment(
                    opt_path, adam_param_key, weight_shape=w_shape,
                )
            except Exception as e:
                logger.warning("True TracIn ghost: could not load Adam state: %s", e)

        g_query_acc: np.ndarray | None = None
        for qi in range(n_q):
            q_in = query_inputs[qi : qi + 1].to(device)
            q_tg = query_targets[qi : qi + 1].to(device)
            if q_tg.dim() == 0:
                q_tg = q_tg.unsqueeze(0)
            with HookManager(model, target_layer) as hm:
                with torch.no_grad():
                    q_logits = model(q_in)
                Aq = hm.activation.cpu().numpy().astype(np.float32)
                Eq = error_fn(q_logits, q_tg).detach().cpu().numpy().astype(np.float32)
            g_q = form_ghost_vectors(Aq, Eq)
            if adam_v is not None:
                g_q = apply_adam_correction(g_q, adam_v)
            g_query_acc = g_q if g_query_acc is None else g_query_acc + g_q
        assert g_query_acc is not None
        g_query = g_query_acc / float(n_q)

        offset = 0
        for inputs, targets, _ in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with HookManager(model, target_layer) as hm:
                with torch.no_grad():
                    logits = model(inputs)
                A = hm.activation.cpu().numpy().astype(np.float32)
                E_tensor = error_fn(logits, targets)
                E = E_tensor.detach().cpu().numpy().astype(np.float32)
            g_train = form_ghost_vectors(A, E)
            if adam_v is not None:
                g_train = apply_adam_correction(g_train, adam_v)
            dots = (g_query @ g_train.T).ravel()
            bs = int(dots.shape[0])
            scores[offset : offset + bs] += lr * dots
            offset += bs
            del A, E, E_tensor, g_train, dots

    return {sid: float(scores[j]) for j, sid in enumerate(all_sample_ids)}
