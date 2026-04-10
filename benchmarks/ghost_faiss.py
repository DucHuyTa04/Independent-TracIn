"""Raw inner-product scores from the projected Ghost + FAISS pipeline."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.faiss_store import FAISSStore
from src.hooks_manager import HookManager, MultiLayerBackwardGhostManager
from src.indexer import _probe_ghost_dim, build_index
from src.math_utils import (
    apply_adam_correction,
    build_dense_projection,
    build_sjlt_matrix,
    concatenate_adam_second_moments,
    form_ghost_vectors,
    form_multi_layer_ghost_vectors,
    load_adam_inverse_sqrt_scale_matrix_ghost_layout,
    load_adam_second_moment,
    project,
)

logger = logging.getLogger(__name__)

# Explicitly hookable layer types (2D CNN + common dense/norm).  Conv1d/Conv3d,
# BatchNorm1d/3d, GroupNorm, etc. are not supported by the ghost hooks; use
# ``auto_fallback=True`` in ``compute_ghost_tracin_scores`` for exact grads.
HOOKABLE_GHOST_LAYER_TYPES = (
    nn.Linear,
    nn.Embedding,
    nn.LayerNorm,
    nn.BatchNorm2d,
)


# ---------------------------------------------------------------------------
# Automatic ghost layer selection
# ---------------------------------------------------------------------------


def _is_dead_layer(layer: nn.Module, dead_std: float = 1e-7) -> bool:
    """True if the layer's weight looks collapsed (near-constant / uninitialized)."""
    w = getattr(layer, "weight", None)
    if w is None:
        return False
    std = float(torch.std(w.detach()).item())
    if std != std:  # NaN
        return False
    return std < dead_std


def _ghost_dim_for_layer(layer: nn.Module) -> int:
    """Return the ghost vector dimensionality for a hookable layer.

    Linear(in, out)       → in * out (+ out if bias)
    Conv2d / ConvTranspose2d  → (C_in * kH * kW) * C_out (+ C_out if bias)
    Embedding(V, D)      → V * D
    LayerNorm(D)         → 2 * D (gamma and beta)
    """
    if isinstance(layer, nn.Linear):
        d = int(layer.in_features) * int(layer.out_features)
        if layer.bias is not None:
            d += int(layer.out_features)
        return d
    if isinstance(layer, nn.Conv2d):
        kH, kW = layer.kernel_size
        cin_g = int(layer.in_channels) // int(layer.groups)
        d = cin_g * kH * kW * int(layer.out_channels)
        if layer.bias is not None:
            d += int(layer.out_channels)
        return d
    if isinstance(layer, nn.ConvTranspose2d):
        kH, kW = layer.kernel_size
        cin_g = int(layer.in_channels) // int(layer.groups)
        d = cin_g * kH * kW * int(layer.out_channels)
        if layer.bias is not None:
            d += int(layer.out_channels)
        return d
    if isinstance(layer, nn.Embedding):
        return int(layer.num_embeddings) * int(layer.embedding_dim)
    if isinstance(layer, nn.LayerNorm):
        n = int(torch.tensor(layer.normalized_shape, dtype=torch.int64).prod().item())
        return 2 * n
    if isinstance(layer, nn.BatchNorm2d):
        return 2 * int(layer.num_features)
    if isinstance(layer, nn.RNNBase):
        return _rnn_ih_ghost_param_count(layer)
    return 0


def _rnn_ih_ghost_param_count(layer: nn.RNNBase) -> int:
    """Parameters whose gradients are approximated by the RNN hook (input-to-hidden only)."""
    return sum(
        int(p.numel())
        for name, p in layer.named_parameters()
        if "weight_ih" in name or "bias_ih" in name
    )


def _adam_scale_matrix_for_layer(
    layer: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Identity Adam scale (H, C) when optimizer state is missing."""
    w = getattr(layer, "weight", None)
    if w is None:
        raise ValueError("adam factored path expects layer.weight")
    if w.dim() == 2:
        c_out, h_in = int(w.shape[0]), int(w.shape[1])
        return torch.ones(h_in, c_out, device=device, dtype=dtype)
    if w.dim() == 4:
        c_out = int(w.shape[0])
        h_in = int(w.shape[1] * w.shape[2] * w.shape[3])
        return torch.ones(h_in, c_out, device=device, dtype=dtype)
    raise ValueError(
        f"adam factored fallback scale: unsupported weight dim {w.dim()}",
    )


def auto_ghost_layers(
    model: nn.Module,
    target_coverage: float = 0.5,
    strategy: str = "last",
    *,
    max_layers: int = 64,
    dead_weight_std: float = 1e-7,
    include_conv: bool = True,
    include_rnn: bool = False,
) -> list[nn.Module]:
    """Automatically select hookable layers for ghost vectors.

    Collects ``nn.Linear``, ``nn.Embedding``, ``nn.LayerNorm``, ``nn.BatchNorm2d``
    (and optionally ``nn.Conv2d`` / ``nn.ConvTranspose2d``; optionally ``nn.RNNBase``).
    Skips ``BatchNorm2d`` without ``affine`` or ``track_running_stats``.
    When ``target_coverage >= 1.0``, ``max_layers`` is raised to include all candidates.
    Greedy coverage until ``target_coverage`` or ``max_layers``.

    Strategies:
        ``"last"`` (default): Prefer layers closer to the model output
        (reverse ``named_modules()`` order — typical PyTorch registration order).
        ``"largest"``: Prefer layers with the most parameters.

    Args:
        model: The PyTorch model to inspect.
        target_coverage: Fraction of total model parameters to cover (0.0–1.0).
        strategy: ``"last"`` or ``"largest"``.
        max_layers: Maximum number of layers to select (default 64).
        dead_weight_std: Skip layers whose weight ``std()`` is below this.
        include_conv: If True (default), also consider ``nn.Conv2d`` /
            ``nn.ConvTranspose2d``.
        include_rnn: If True, also consider ``nn.RNNBase``. Default False: RNN
            ghost only captures input-to-hidden; including it often hurts ranking.

    Returns:
        List of modules to use as ``ghost_layers``.
    """
    total_params = sum(int(p.numel()) for p in model.parameters())
    if total_params == 0:
        return []

    conv_t = (nn.Conv2d, nn.ConvTranspose2d) if include_conv else tuple()
    hookable_types = HOOKABLE_GHOST_LAYER_TYPES + conv_t
    if include_rnn:
        hookable_types = hookable_types + (nn.RNNBase,)

    candidate_layers: list[tuple[str, nn.Module, int]] = []
    for name, m in model.named_modules():
        if isinstance(m, hookable_types):
            if isinstance(m, nn.LayerNorm) and not m.elementwise_affine:
                continue
            if isinstance(m, nn.BatchNorm2d) and (
                not m.affine or not m.track_running_stats
            ):
                continue
            if _is_dead_layer(m, dead_std=dead_weight_std):
                logger.info(
                    "auto_ghost_layers: skip dead layer %s (weight std < %g)",
                    name,
                    dead_weight_std,
                )
                continue
            if isinstance(m, nn.RNNBase):
                n_params = _rnn_ih_ghost_param_count(m)
            else:
                n_params = sum(int(p.numel()) for p in m.parameters())
            candidate_layers.append((name, m, n_params))

    if not candidate_layers:
        logger.warning("auto_ghost_layers: no hookable modules found")
        return []

    if strategy == "last":
        candidate_layers.reverse()
    elif strategy == "largest":
        candidate_layers.sort(key=lambda t: (-t[2], t[0]))
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}; use 'last' or 'largest'",
        )

    if target_coverage >= 1.0:
        max_layers = len(candidate_layers)

    selected = []
    covered = 0
    target = target_coverage * total_params
    for name, m, n_params in candidate_layers:
        if len(selected) >= max_layers:
            break
        selected.append(m)
        covered += n_params
        layer_type = type(m).__name__
        logger.info(
            "auto_ghost_layers [%s]: added %s (%s, %d params, %.1f%% cumulative)",
            strategy,
            name,
            layer_type,
            n_params,
            100.0 * covered / total_params,
        )
        if covered >= target:
            break

    return selected


def _weight_shape_for_adam(layer: nn.Module) -> Optional[tuple[int, int]]:
    """Return (out_features, in_features) for adam correction, or None.

    For Conv2d, the ghost vector layout is [C_out, C_in*kH*kW], matching
    the im2col formulation where the weight is reshaped to a 2D matrix.
    """
    w = getattr(layer, "weight", None)
    if w is None:
        return None
    if w.dim() == 2:
        return (int(w.shape[0]), int(w.shape[1]))
    elif w.dim() == 4:
        # Conv2d / ConvTranspose2d: [C_out, C_in, kH, kW] → [C_out, C_in*kH*kW]
        return (int(w.shape[0]), int(w.shape[1]) * int(w.shape[2]) * int(w.shape[3]))
    return None


def compute_ghost_faiss_scores(
    model: nn.Module,
    target_layer: nn.Module,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader,
    checkpoints: list[dict],
    sample_metadata: dict[int, str],
    query_inputs: torch.Tensor,
    query_targets: torch.Tensor,
    checkpoint_path: str,
    optimizer_state_path: Optional[str],
    index_dir: str,
    index_filename: str = "bench_faiss_index",
    metadata_filename: str = "bench_faiss_meta.json",
    projection_dim: Optional[int] = 1280,
    projection_type: str = "sjlt",
    projection_seed: int = 42,
    adam_param_key: Union[int, str] = 2,
    device: str = "cpu",
) -> dict[int, float]:
    """Build FAISS index like production, then return raw IP scores for all training ids.

    Query ghost uses ``checkpoint_path`` and ``optimizer_state_path`` (TracIn-last).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_weights(m: nn.Module, path: str, dev: str) -> None:
        state = torch.load(path, map_location=dev, weights_only=True)
        m.load_state_dict(state)

    model.to(device)
    _load_weights(model, checkpoints[0]["weights_path"], device)
    model.eval()
    ghost_dim = _probe_ghost_dim(model, target_layer, error_fn, data_loader, device)
    proj_arg = projection_dim if projection_dim is not None else ghost_dim

    build_index(
        model=model,
        target_layer=target_layer,
        error_fn=error_fn,
        data_loader=data_loader,
        checkpoints=checkpoints,
        sample_metadata=sample_metadata,
        projection_dim=proj_arg,
        projection_type=projection_type,
        projection_seed=projection_seed,
        adam_param_key=adam_param_key,
        output_dir=index_dir,
        index_filename=index_filename,
        metadata_filename=metadata_filename,
        device=device,
    )

    idx_path = f"{index_dir.rstrip('/')}/{index_filename}"
    meta_path = f"{index_dir.rstrip('/')}/{metadata_filename}"

    if query_inputs.dim() == 1:
        query_inputs = query_inputs.unsqueeze(0)
    if query_targets.dim() == 0:
        query_targets = query_targets.unsqueeze(0)

    model.to(device)
    _load_weights(model, checkpoint_path, device)
    model.eval()
    n_q = int(query_inputs.shape[0])

    store = FAISSStore(index_type="flat")
    store.load(idx_path, meta_path)
    n_idx = len(store.metadata["sample_ids"])

    proj_matrix: object | None = None
    all_g_proj: list[np.ndarray] = []

    for qi in range(n_q):
        q_in = query_inputs[qi : qi + 1].to(device)
        q_tg = query_targets[qi : qi + 1].to(device)
        if q_tg.dim() == 0:
            q_tg = q_tg.unsqueeze(0)

        with HookManager(model, target_layer) as hm:
            with torch.no_grad():
                logits = model(q_in)
            A = hm.activation.cpu().numpy().astype(np.float32)
            E = error_fn(logits, q_tg).detach().cpu().numpy().astype(np.float32)
        g = form_ghost_vectors(A, E)
        if optimizer_state_path:
            try:
                av = load_adam_second_moment(
                    optimizer_state_path,
                    adam_param_key,
                    weight_shape=_weight_shape_for_adam(target_layer),
                )
                g = apply_adam_correction(g, av)
            except Exception:
                pass

        gd = g.shape[1]
        if projection_dim is not None and projection_dim < gd:
            if proj_matrix is None:
                if projection_type == "sjlt":
                    proj_matrix = build_sjlt_matrix(gd, projection_dim, seed=projection_seed)
                else:
                    proj_matrix = build_dense_projection(gd, projection_dim, seed=projection_seed)
            g_proj = project(g, proj_matrix)  # type: ignore[arg-type]
        else:
            g_proj = g

        all_g_proj.append(g_proj)

    stacked = np.vstack(all_g_proj)
    scores_all, _, ids_all = store.query(stacked, top_k=n_idx)

    acc: dict[int, float] = {}
    for qi in range(n_q):
        for j, sid in enumerate(ids_all[qi]):
            k = int(sid)
            acc[k] = acc.get(k, 0.0) + float(scores_all[qi, j]) / n_q

    return acc


# ---------------------------------------------------------------------------
# Model-agnostic Ghost TracIn (per-checkpoint, multi-layer, backward hooks)
# ---------------------------------------------------------------------------


def _weight_shapes_for_layers(layers: list[nn.Module]) -> list[Optional[tuple[int, int]]]:
    """(out_features, in_features) for each layer's weight, or None.

    For Conv2d, returns (C_out, C_in*kH*kW) matching the im2col ghost layout.
    """
    shapes: list[Optional[tuple[int, int]]] = []
    for layer in layers:
        w = getattr(layer, "weight", None)
        if w is not None and w.dim() == 2:
            shapes.append((int(w.shape[0]), int(w.shape[1])))
        elif w is not None and w.dim() == 4:
            shapes.append((int(w.shape[0]), int(w.shape[1]) * int(w.shape[2]) * int(w.shape[3])))
        else:
            shapes.append(None)
    return shapes


def _run_forward_backward(
    model: nn.Module,
    ghost_layers: list[nn.Module],
    training_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: str,
    *,
    as_torch: bool = False,
    raw: bool = False,
    max_spatial_positions: Optional[int] = None,
    rnn_modules: Optional[list[nn.Module]] = None,
    inplace_modules: Optional[list[nn.Module]] = None,
) -> Union[
    tuple[list[np.ndarray], list[np.ndarray]],
    tuple[list[torch.Tensor], list[torch.Tensor]],
]:
    """Forward + backward, return per-layer (A_list, E_list) raw blocks.

    Uses sum-reduced loss so that grad_output at each hooked layer is the
    *per-sample* error signal (not divided by batch_size).

    If ``as_torch`` is True, returns tensors on the same device as ``inputs``;
    otherwise float32 numpy arrays on CPU.

    If ``raw`` is True (implies ``as_torch=True``), returns un-flattened
    tensors (e.g. [B,T,H] for 3D layers) via ``raw_torch_blocks()``.
    """
    inputs = inputs.to(device)
    targets = targets.to(device)
    batch_n = inputs.shape[0]
    keep_raw = raw
    # Eval mode for deterministic gradients (VAE reparam, Dropout, BN running stats).
    # RNN modules stay in train() for cuDNN backward compatibility.
    was_training = model.training
    if rnn_modules is not None:
        rnn_prev = [(m, m.training) for m in rnn_modules]
    else:
        rnn_prev = [
            (m, m.training)
            for m in model.modules()
            if isinstance(m, nn.RNNBase)
        ]
    model.eval()
    for m, _ in rnn_prev:
        m.train()
    # Temporarily disable inplace on ReLU (and similar) modules.
    # register_full_backward_hook wraps hooked-layer outputs in a
    # BackwardHookFunction view; a subsequent inplace op on that view
    # causes "view + inplace" RuntimeError.  Disabling inplace avoids
    # this while producing identical numerics.
    _inplace_prev: list[tuple[nn.Module, bool]] = []
    if inplace_modules is not None:
        for m in inplace_modules:
            if m.inplace:
                _inplace_prev.append((m, True))
                m.inplace = False
    else:
        for m in model.modules():
            if hasattr(m, "inplace") and m.inplace:
                _inplace_prev.append((m, True))
                m.inplace = False
    try:
        with MultiLayerBackwardGhostManager(
            ghost_layers,
            keep_raw=keep_raw,
            max_spatial_positions=max_spatial_positions if keep_raw else None,
        ) as hm:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = model(inputs)
                loss = training_loss_fn(logits, targets)
                reduction = getattr(training_loss_fn, "reduction", "mean")
                if loss.dim() > 0:
                    loss = loss.sum()
                elif reduction == "mean" and batch_n > 1:
                    loss = loss * float(batch_n)
                loss.backward()
    finally:
        for m, prev in _inplace_prev:
            m.inplace = prev
        if was_training:
            model.train()
        else:
            model.eval()
        for m, prev in rnn_prev:
            m.train(prev)
    if raw:
        return hm.raw_torch_blocks()
    if as_torch:
        return hm.torch_blocks()
    return hm.numpy_blocks()


def _embedding_ghost_matrix(
    tokens: torch.Tensor,
    grad_out: torch.Tensor,
    num_embeddings: int,
) -> torch.Tensor:
    """Materialise vec(dL/dW_emb) with shape [B, V*D] (small V only)."""
    b, t_len, d_dim = grad_out.shape
    tokens = tokens.long().clamp(0, num_embeddings - 1)
    g = grad_out.new_zeros(b, num_embeddings * d_dim)
    ar = torch.arange(d_dim, device=grad_out.device, dtype=torch.long).view(1, 1, -1)
    idx = tokens.unsqueeze(-1) * d_dim + ar
    g.scatter_add_(1, idx.reshape(b, -1), grad_out.reshape(b, -1))
    return g


def _grouped_conv_ghost_matrix_from_raw_3d(
    layer: nn.Module,
    a: torch.Tensor,
    e: torch.Tensor,
) -> torch.Tensor:
    """Materialize ``[B, D]`` ghost rows for 3D conv/transpose raw blocks (``groups``-aware)."""
    a = a.float()
    e = e.float()
    bsz = int(a.shape[0])
    if not isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        gmat = torch.einsum("bth,btv->bhv", a, e)
        return gmat.reshape(bsz, -1)
    g = int(layer.groups)
    if g <= 1:
        gmat = torch.einsum("bth,btv->bhv", a, e)
        return gmat.reshape(bsz, -1)
    cin_g = int(layer.in_channels) // g
    cout_g = int(layer.out_channels) // g
    kh, kw = layer.kernel_size
    patch_w = cin_g * int(kh) * int(kw)
    pieces: list[torch.Tensor] = []
    oa = 0
    oe = 0
    for _ in range(g):
        ag = a[:, :, oa : oa + patch_w]
        eg = e[:, :, oe : oe + cout_g]
        oa += patch_w
        oe += cout_g
        pieces.append(torch.einsum("bth,btv->bhv", ag, eg).reshape(bsz, -1))
    # Bias ghost vector: dL/db[c] = sum_l E[c,l] (not group-split).
    if getattr(layer, "bias", None) is not None:
        pieces.append(e.sum(dim=1))  # [B, C_out]
    return torch.cat(pieces, dim=1)


def _ghost_matrix_torch_from_raw_blocks(
    raw_A: list[torch.Tensor],
    raw_E: list[torch.Tensor],
    *,
    ghost_layers: Optional[list[nn.Module]] = None,
) -> torch.Tensor:
    """Materialise concatenated ghost vectors [B, D] as float32 torch (same device as activations)."""
    # Determine expected batch size from the first non-Embedding E tensor.
    batch_size: Optional[int] = None
    for li, e in enumerate(raw_E):
        lm = ghost_layers[li] if ghost_layers is not None else None
        if not isinstance(lm, nn.Embedding):
            batch_size = int(e.shape[0])
            break
    if batch_size is None:
        batch_size = int(raw_E[0].shape[0])

    parts: list[torch.Tensor] = []
    for li, (a, e) in enumerate(zip(raw_A, raw_E)):
        layer_mod = ghost_layers[li] if ghost_layers is not None else None
        if isinstance(layer_mod, nn.Embedding):
            # Skip Embedding layers with non-batched input (e.g. pos_emb).
            if int(e.shape[0]) != batch_size:
                logger.info(
                    "Skipping Embedding layer %d in ghost matrix: non-batched "
                    "(grad batch %d != expected %d)",
                    li, int(e.shape[0]), batch_size,
                )
                continue
            toks = a.long()
            g = _embedding_ghost_matrix(toks, e.float(), int(layer_mod.num_embeddings))
            parts.append(g)
        elif isinstance(layer_mod, (nn.LayerNorm, nn.BatchNorm2d)):
            parts.append(torch.cat([a.float(), e.float()], dim=-1))
        else:
            a = a.float()
            e = e.float()
            if a.dim() == 3:
                if e.dim() != 3:
                    raise ValueError(
                        f"Expected 3D grad_output for 3D activation; got a.dim={a.dim()}, e.dim={e.dim()}"
                    )
                if a.shape[0] != e.shape[0] or a.shape[1] != e.shape[1]:
                    raise ValueError(
                        f"Conv/seq ghost L mismatch: A.shape={tuple(a.shape)}, E.shape={tuple(e.shape)}"
                    )
                if isinstance(layer_mod, (nn.Conv2d, nn.ConvTranspose2d)):
                    parts.append(
                        _grouped_conv_ghost_matrix_from_raw_3d(layer_mod, a, e),
                    )
                else:
                    g = torch.einsum("bth,btv->bhv", a, e)
                    parts.append(g.reshape(g.shape[0], -1))
            else:
                g = torch.einsum("bh,bv->bhv", a, e)
                parts.append(g.reshape(g.shape[0], -1))
    return torch.cat(parts, dim=1)


def _ghost_matrix_from_raw_blocks(
    raw_A: list[torch.Tensor],
    raw_E: list[torch.Tensor],
    *,
    ghost_layers: Optional[list[nn.Module]] = None,
) -> np.ndarray:
    """Numpy ghost matrix for Adam / projection path."""
    return (
        _ghost_matrix_torch_from_raw_blocks(raw_A, raw_E, ghost_layers=ghost_layers)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def _embedding_ghost_dots(
    toks_q: torch.Tensor,
    eq: torch.Tensor,
    toks_t: torch.Tensor,
    et: torch.Tensor,
    num_embeddings: int,
) -> torch.Tensor:
    """Exact inner product of embedding weight gradients (token-matching SDP).

    Aggregates per-vocab-id over sequence, then sums dot products over active
    vocab only (O(|active|) memory vs O(V)).
    """
    toks_q = toks_q.long().clamp(0, num_embeddings - 1)
    toks_t = toks_t.long().clamp(0, num_embeddings - 1)
    dev = eq.device
    dt = torch.float32
    eq = eq.to(dtype=dt)
    et = et.to(dtype=dt)
    active = torch.unique(torch.cat([toks_q.reshape(-1), toks_t.reshape(-1)]))
    na = int(active.numel())
    if na == 0:
        return torch.zeros(
            eq.shape[0], et.shape[0], device=dev, dtype=dt,
        )
    d = int(eq.shape[-1])
    col_map = torch.full(
        (num_embeddings,),
        -1,
        device=dev,
        dtype=torch.long,
    )
    col_map[active] = torch.arange(na, device=dev, dtype=torch.long)
    idx_q = col_map[toks_q].unsqueeze(-1).expand(-1, -1, d)
    idx_t = col_map[toks_t].unsqueeze(-1).expand(-1, -1, d)
    hq = eq.new_zeros(eq.shape[0], na, d)
    ht = et.new_zeros(et.shape[0], na, d)
    hq.scatter_add_(1, idx_q, eq)
    ht.scatter_add_(1, idx_t, et)
    return torch.einsum("qad,bad->qb", hq, ht)


def _find_uncovered_params(
    model: nn.Module,
    ghost_layers: list[nn.Module],
) -> list[tuple[str, nn.Parameter]]:
    """Parameters not belonging to any ``ghost_layers`` module (by object id)."""
    covered_ids: set[int] = set()
    for layer in ghost_layers:
        for p in layer.parameters():
            covered_ids.add(id(p))
    return [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad and id(p) not in covered_ids
    ]


def _per_sample_fallback_grad_matrix(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    uncovered_params: list[tuple[str, nn.Parameter]],
    device: str,
    *,
    rnn_modules: Optional[list[nn.Module]] = None,
    inplace_modules: Optional[list[nn.Module]] = None,
) -> torch.Tensor:
    """Stack per-sample flattened grads for uncovered params → ``[N, D]`` float32."""
    if not uncovered_params:
        return torch.empty(
            int(inputs.shape[0]), 0, device=device, dtype=torch.float32,
        )
    param_list = [p for _, p in uncovered_params]
    n = int(inputs.shape[0])
    was_training = model.training
    if rnn_modules is not None:
        rnn_prev = [(m, m.training) for m in rnn_modules]
    else:
        rnn_prev = [
            (m, m.training)
            for m in model.modules()
            if isinstance(m, nn.RNNBase)
        ]
    model.eval()
    for m, _ in rnn_prev:
        m.train()
    _inplace_prev: list[tuple[nn.Module, bool]] = []
    if inplace_modules is not None:
        for m in inplace_modules:
            if getattr(m, "inplace", False):
                _inplace_prev.append((m, True))
                m.inplace = False
    else:
        for m in model.modules():
            if hasattr(m, "inplace") and m.inplace:
                _inplace_prev.append((m, True))
                m.inplace = False
    rows: list[torch.Tensor] = []
    try:
        for i in range(n):
            model.zero_grad(set_to_none=True)
            x_i = inputs[i : i + 1].to(device)
            y_i = targets[i : i + 1].to(device)
            if isinstance(loss_fn, nn.MSELoss) and y_i.dim() == 1:
                y_i = y_i.unsqueeze(0)
            with torch.enable_grad():
                out = model(x_i)
                loss = loss_fn(out, y_i)
                if loss.dim() > 0:
                    loss = loss.sum()
                loss.backward()
            g = torch.cat(
                [
                    (p.grad.detach().flatten().float() if p.grad is not None
                     else torch.zeros(p.numel(), device=device, dtype=torch.float32))
                    for p in param_list
                ],
            )
            rows.append(g)
    finally:
        for m, prev in _inplace_prev:
            m.inplace = prev
        if was_training:
            model.train()
        else:
            model.eval()
        for m, prev in rnn_prev:
            m.train(prev)
    return torch.stack(rows, dim=0)


def _three_d_layer_ghost_dots(
    aq: torch.Tensor,
    eq: torch.Tensor,
    at: torch.Tensor,
    et: torch.Tensor,
    *,
    n_q: int,
    n_b: int,
) -> torch.Tensor:
    """SDP vs materialisation for one 3D conv/seq ghost block (``groups==1`` layout)."""
    dev = aq.device
    dt = torch.float32
    if aq.dtype != dt:
        aq = aq.to(dtype=dt)
    if at.dtype != dt:
        at = at.to(dtype=dt)
    if eq.dtype != dt:
        eq = eq.to(dtype=dt)
    if et.dtype != dt:
        et = et.to(dtype=dt)
    if at.device != dev:
        at = at.to(device=dev)
    if eq.device != dev:
        eq = eq.to(device=dev)
    if et.device != dev:
        et = et.to(device=dev)
    l_spatial = int(aq.shape[1])
    h_f = int(aq.shape[2])
    v_f = int(eq.shape[2])
    ghost_dim = h_f * v_f
    if l_spatial * l_spatial < ghost_dim // 2:
        s_aa = torch.einsum("qlh,bmh->qblm", aq, at)
        s_ee = torch.einsum("qlv,bmv->qblm", eq, et)
        layer_score = (s_aa * s_ee).sum(dim=(-2, -1))
    else:
        gq = torch.einsum("qth,qtv->qhv", aq, eq).reshape(n_q, -1)
        gt = torch.einsum("bth,btv->bhv", at, et).reshape(n_b, -1)
        layer_score = torch.mm(gq, gt.T)
    return layer_score


def _grouped_conv_ghost_dots_from_raw(
    layer: nn.Module,
    aq: torch.Tensor,
    eq: torch.Tensor,
    at: torch.Tensor,
    et: torch.Tensor,
    *,
    n_q: int,
    n_b: int,
) -> torch.Tensor:
    """Sum per-group 3D ghost dots (correct when ``groups > 1``)."""
    if not isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        raise TypeError("expected Conv2d or ConvTranspose2d")
    g = int(layer.groups)
    if g <= 1:
        return _three_d_layer_ghost_dots(
            aq, eq, at, et,
            n_q=n_q, n_b=n_b,
        )
    cin_g = int(layer.in_channels) // g
    cout_g = int(layer.out_channels) // g
    kh, kw = layer.kernel_size
    patch_w = cin_g * int(kh) * int(kw)
    dev = aq.device
    dt = torch.float32
    aq = aq.to(device=dev, dtype=dt)
    at = at.to(device=dev, dtype=dt)
    eq = eq.to(device=dev, dtype=dt)
    et = et.to(device=dev, dtype=dt)
    layer_score = torch.zeros(n_q, n_b, device=dev, dtype=dt)
    oa = 0
    oe = 0
    for _ in range(g):
        aq_g = aq[:, :, oa : oa + patch_w]
        at_g = at[:, :, oa : oa + patch_w]
        eq_g = eq[:, :, oe : oe + cout_g]
        et_g = et[:, :, oe : oe + cout_g]
        oa += patch_w
        oe += cout_g
        layer_score.add_(
            _three_d_layer_ghost_dots(
                aq_g, eq_g, at_g, et_g,
                n_q=n_q, n_b=n_b,
            ),
        )
    # Bias gradient: dL/db[c] = sum_l E[c,l].  The ones column appended by
    # _maybe_append_bias_ones is at position ``oa`` and is not group-split.
    if getattr(layer, "bias", None) is not None:
        eq_sum = eq.sum(dim=1)   # [n_q, C_out]
        et_sum = et.sum(dim=1)   # [n_b, C_out]
        layer_score.add_(torch.mm(eq_sum, et_sum.T))
    return layer_score


def debug_per_layer_ghost_accuracy(
    model: nn.Module,
    ghost_layers: list[nn.Module],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    device: str,
    *,
    max_spatial_positions: Optional[int] = None,
) -> None:
    """Log per-layer ghost vs autograd inner product (first sample of each batch)."""
    if x1.dim() == 1:
        x1 = x1.unsqueeze(0)
    if y1.dim() == 0:
        y1 = y1.unsqueeze(0)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(0)
    if y2.dim() == 0:
        y2 = y2.unsqueeze(0)
    x1 = x1[:1].to(device)
    y1 = y1[:1].to(device)
    x2 = x2[:1].to(device)
    y2 = y2[:1].to(device)
    if isinstance(loss_fn, nn.MSELoss):
        if y1.dim() == 1:
            y1 = y1.unsqueeze(0)
        if y2.dim() == 1:
            y2 = y2.unsqueeze(0)
    was_training = model.training
    rnn_prev = [
        (m, m.training)
        for m in model.modules()
        if isinstance(m, nn.RNNBase)
    ]
    try:
        model.to(device)
        model.eval()
        for m, _ in rnn_prev:
            m.train()
        raw_a1, raw_e1 = _run_forward_backward(
            model, ghost_layers, loss_fn, x1, y1, device,
            raw=True, max_spatial_positions=max_spatial_positions,
        )
        raw_a2, raw_e2 = _run_forward_backward(
            model, ghost_layers, loss_fn, x2, y2, device,
            raw=True, max_spatial_positions=max_spatial_positions,
        )
        up = _find_uncovered_params(model, ghost_layers)
        logger.info(
            "debug_per_layer_ghost_accuracy: uncovered param tensors=%d (not in table)",
            len(up),
        )
        for li, layer in enumerate(ghost_layers):
            g_ghost = float(
                _layer_ghost_dots_from_raw_blocks(
                    [raw_a1[li]],
                    [raw_e1[li]],
                    [raw_a2[li]],
                    [raw_e2[li]],
                    ghost_layers=[layer],
                ).sum().item(),
            )
            model.zero_grad(set_to_none=True)
            l1 = loss_fn(model(x1), y1)
            if l1.dim() > 0:
                l1 = l1.sum()
            l1.backward()
            g1 = {
                id(p): p.grad.detach().flatten().float().clone()
                for p in layer.parameters()
                if p.grad is not None
            }
            model.zero_grad(set_to_none=True)
            l2 = loss_fn(model(x2), y2)
            if l2.dim() > 0:
                l2 = l2.sum()
            l2.backward()
            g_auto = 0.0
            for p in layer.parameters():
                if p.grad is None:
                    continue
                t = id(p)
                if t in g1:
                    g_auto += float(
                        (g1[t] @ p.grad.detach().flatten().float()).item(),
                    )
            logger.info(
                "  layer[%d] %s ghost_ip=%.8g autograd_ip=%.8g",
                li,
                type(layer).__name__,
                g_ghost,
                g_auto,
            )
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
        for m, prev in rnn_prev:
            m.train(prev)


def _ghost_layers_need_per_layer_path(
    ghost_layers: list[nn.Module],
    raw_A: list[torch.Tensor],
) -> bool:
    """True if factored CUDA/CPU path cannot be used (3D, Embedding, LayerNorm)."""
    if any(a.dim() == 3 for a in raw_A):
        return True
    if any(isinstance(m, (nn.Embedding, nn.LayerNorm, nn.BatchNorm2d)) for m in ghost_layers):
        return True
    if any(a.dtype in (torch.int64, torch.int32) for a in raw_A):
        return True
    return False


def _factored_2d_linear_indices(
    ghost_layers: list[nn.Module],
    raw_A: list[torch.Tensor],
) -> list[int]:
    """Indices where ghost inner product is exact 2D factored ``mm*mm``."""
    out: list[int] = []
    for i, (m, a) in enumerate(zip(ghost_layers, raw_A)):
        if isinstance(m, (nn.Embedding, nn.LayerNorm, nn.BatchNorm2d)):
            continue
        if a.dim() != 2:
            continue
        if a.dtype in (torch.int64, torch.int32):
            continue
        out.append(i)
    return out


def _layer_ghost_dots_from_raw_blocks(
    raw_q_A: list[torch.Tensor],
    raw_q_E: list[torch.Tensor],
    raw_t_A: list[torch.Tensor],
    raw_t_E: list[torch.Tensor],
    *,
    ghost_layers: Optional[list[nn.Module]] = None,
    normalize_layer_dots: bool = False,
) -> torch.Tensor:
    """Per-layer ghost inner products between query batch Q and train batch B.

    For 2D layers: factored identity ``(A_q @ A_t^T) * (E_q @ E_t^T)``.
    For 3D (Conv/seq): exact ``<vec(Σ_l E_l A_l^T)_q, vec(Σ_m E_m A_m^T)_t>``
    via spatial dot-product (SDP) when ``L^2 < H*V``, else per-layer ghost
    materialisation and ``mm``.
    """
    if len(raw_q_A) != len(raw_q_E) or len(raw_t_A) != len(raw_t_E):
        raise ValueError("raw A/E lists must be paired")
    if len(raw_q_A) != len(raw_t_A):
        raise ValueError("query and train raw block lists must have same length")

    n_layers = len(raw_q_A)
    if n_layers == 0:
        raise ValueError("empty ghost layer list")

    dev = raw_q_E[0].device
    dt = torch.float32
    # Determine batch sizes from the first non-Embedding layer (Embedding
    # layers with shared inputs may have batch dim == 1).
    n_q: Optional[int] = None
    n_b: Optional[int] = None
    for _i in range(n_layers):
        _lm = ghost_layers[_i] if ghost_layers is not None else None
        if not isinstance(_lm, nn.Embedding):
            n_q = int(raw_q_E[_i].shape[0])
            n_b = int(raw_t_E[_i].shape[0])
            break
    if n_q is None:
        n_q = int(raw_q_E[0].shape[0])
        n_b = int(raw_t_E[0].shape[0])

    dots = torch.zeros(n_q, n_b, device=dev, dtype=dt)

    for l_idx in range(n_layers):
        aq = raw_q_A[l_idx]
        eq = raw_q_E[l_idx].to(device=dev, dtype=dt)
        at = raw_t_A[l_idx]
        et = raw_t_E[l_idx].to(device=dev, dtype=dt)

        layer_mod = ghost_layers[l_idx] if ghost_layers is not None else None

        if isinstance(layer_mod, nn.Embedding):
            # Skip Embedding layers whose input is not per-sample batched
            # (e.g. positional embeddings with shared indices).  Their
            # grad_output is summed over the batch, so per-sample ghost
            # inner products cannot be computed.
            if int(eq.shape[0]) != n_q or int(et.shape[0]) != n_b:
                logger.info(
                    "Skipping Embedding layer %d: non-batched input "
                    "(grad batch %d != expected %d)",
                    l_idx, int(eq.shape[0]), n_q,
                )
                continue
            vocab = int(layer_mod.num_embeddings)
            layer_score = _embedding_ghost_dots(
                aq.to(dev), eq, at.to(dev), et, vocab,
            )
        elif isinstance(layer_mod, nn.LayerNorm):
            aq2 = aq.to(device=dev, dtype=dt)
            at2 = at.to(device=dev, dtype=dt)
            layer_score = torch.mm(aq2, at2.T) + torch.mm(eq, et.T)
        elif isinstance(layer_mod, nn.BatchNorm2d):
            aq2 = aq.to(device=dev, dtype=dt)
            at2 = at.to(device=dev, dtype=dt)
            layer_score = torch.mm(aq2, at2.T) + torch.mm(eq, et.T)
        elif aq.dim() == 2:
            aq = aq.to(device=dev, dtype=dt)
            at = at.to(device=dev, dtype=dt)
            layer_score = torch.mm(aq, at.T) * torch.mm(eq, et.T)
        elif aq.dim() == 3:
            if eq.dim() != 3 or at.dim() != 3 or et.dim() != 3:
                raise ValueError(
                    f"Layer {l_idx}: expected 3D query/train tensors; "
                    f"got aq.dim={aq.dim()}, eq.dim={eq.dim()}, "
                    f"at.dim={at.dim()}, et.dim={et.dim()}"
                )
            if isinstance(layer_mod, (nn.Conv2d, nn.ConvTranspose2d)):
                layer_score = _grouped_conv_ghost_dots_from_raw(
                    layer_mod,
                    aq,
                    eq,
                    at,
                    et,
                    n_q=n_q,
                    n_b=n_b,
                )
            else:
                layer_score = _three_d_layer_ghost_dots(
                    aq,
                    eq,
                    at,
                    et,
                    n_q=n_q,
                    n_b=n_b,
                )
        else:
            raise ValueError(
                f"Layer {l_idx}: unsupported activation dim {aq.dim()} "
                "(expected 2D or 3D raw blocks)"
            )

        if normalize_layer_dots:
            ln = layer_score.norm(dim=1, keepdim=True).clamp(min=1e-8)
            layer_score = layer_score / ln
        dots.add_(layer_score)

    return dots


def _accumulate_batch_tracin(
    scores_acc: defaultdict,
    sample_order: list[int],
    ckpt_idx: int,
    batch_ids: object,
    dot_per_train_sample: Union[np.ndarray, torch.Tensor],
    lr: float,
) -> None:
    """Add lr * dot_per_train_sample[j] to scores_acc[batch_ids[j]]."""
    if isinstance(dot_per_train_sample, torch.Tensor):
        row = (
            dot_per_train_sample.detach().cpu().numpy().astype(np.float64).ravel()
        )
    else:
        row = np.asarray(dot_per_train_sample, dtype=np.float64).ravel()
    ids = [int(x) for x in batch_ids]
    if ckpt_idx == 0:
        sample_order.extend(ids)
    for j, sid in enumerate(ids):
        scores_acc[sid] += lr * float(row[j])


def _extract_ghost_vectors(
    model: nn.Module,
    ghost_layers: list[nn.Module],
    training_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: str,
    *,
    max_spatial_positions: Optional[int] = None,
    rnn_modules: Optional[list[nn.Module]] = None,
    inplace_modules: Optional[list[nn.Module]] = None,
) -> np.ndarray:
    """Forward + backward through model, return concatenated ghost vectors for given batch."""
    raw_A, raw_E = _run_forward_backward(
        model,
        ghost_layers,
        training_loss_fn,
        inputs,
        targets,
        device,
        raw=True,
        max_spatial_positions=max_spatial_positions,
        rnn_modules=rnn_modules,
        inplace_modules=inplace_modules,
    )
    if _ghost_layers_need_per_layer_path(ghost_layers, raw_A):
        return _ghost_matrix_from_raw_blocks(
            raw_A, raw_E, ghost_layers=ghost_layers,
        )
    A_list = [a.detach().cpu().numpy().astype(np.float32) for a in raw_A]
    E_list = [e.detach().cpu().numpy().astype(np.float32) for e in raw_E]
    return form_multi_layer_ghost_vectors(A_list, E_list)


def compute_ghost_tracin_scores(
    model: nn.Module,
    ghost_layers: list[nn.Module],
    training_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    query_inputs: torch.Tensor,
    query_targets: torch.Tensor,
    checkpoints: list[dict],
    adam_param_keys: Optional[list[Union[int, str]]] = None,
    projection_dim: Optional[int] = None,
    projection_type: str = "sjlt",
    projection_seed: int = 42,
    device: str = "cpu",
    load_weights_fn: Optional[Callable[[nn.Module, str, str], None]] = None,
    normalize_layer_dots: bool = False,
    max_spatial_positions: Optional[int] = None,
    auto_fallback: bool = True,
) -> dict[int, float]:
    """Model-agnostic Ghost TracIn with per-checkpoint fidelity.

    Implements the TracInCP formula using multi-layer ghost vectors extracted
    via backward hooks.  Both training and query ghosts are recomputed at each
    checkpoint, matching the paper:

        Score(z, z') = sum_t  eta_t  <g_train_t, g_query_t>

    averaged over the query set.

    When ``adam_param_keys`` is *None* (no Adam correction), exploits the
    ghost factorisation identity to avoid materializing d-dimensional ghost
    vectors entirely:

        <g_i, g_j> = sum_l  <A_i^l, A_j^l> * <E_i^l, E_j^l>

    This is **exact**, uses no projection, and reduces compute from
    O(N * d) to O(N * (H + C)) per layer — orders of magnitude faster
    for wide models.

    Args:
        model: PyTorch model (weights overwritten per checkpoint).
        ghost_layers: Layers to hook, e.g. ``[model.fc1, model.fc2]``.
        training_loss_fn: Same scalar loss used during training (e.g. ``nn.MSELoss()``).
        data_loader: Yields ``(inputs, targets, sample_ids)``.
        query_inputs: (Q, ...) query inputs.
        query_targets: (Q, ...) query targets.
        checkpoints: List of dicts with ``weights_path``, ``learning_rate``,
            and optionally ``optimizer_state_path``.
        adam_param_keys: Optimizer state keys for each layer's weight, same
            order as ``ghost_layers``.  If None, Adam correction is skipped
            and the fast factored path is used.  If set and all layers use
            the 2D factored ghost (no 3D conv / Embedding / LayerNorm /
            BatchNorm2d raw path), uses an **Adam-corrected factored** path
            (no full ghost materialization) unless ``projection_dim`` is
            smaller than the concatenated ghost dimension (then materializes).
        projection_dim: Target dimension for SJLT/dense projection.
            Ignored when the fast factored path is used.
        projection_type: ``"sjlt"`` or ``"dense"``.
        projection_seed: RNG seed for projection matrix.
        device: Torch device (``"auto"`` resolves to cuda/cpu).
        load_weights_fn: Custom ``(model, path, device) -> None`` loader.
        normalize_layer_dots: If True, scale each layer's contribution so its
            Frobenius norm over queries is 1 (2D factored, 3D SDP, and 3D
            materialised paths), reducing dominance by high-magnitude layers.
        max_spatial_positions: If set, Conv2d layers with more output spatial sites
            than this use mean-pooled raw blocks (approximate) to limit memory.
        auto_fallback: If True, parameters not covered by ``ghost_layers`` get
            per-sample autograd gradients and their dot products are added to scores
            (exact for those params, model-agnostic).

    Returns:
        Mapping ``sample_id -> float`` influence score.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _default_load(m: nn.Module, path: str, dev: str) -> None:
        state = torch.load(path, map_location=dev, weights_only=True)
        m.load_state_dict(state)

    load_fn = load_weights_fn or _default_load

    if query_inputs.dim() == 1:
        query_inputs = query_inputs.unsqueeze(0)
    if query_targets.dim() == 0:
        query_targets = query_targets.unsqueeze(0)
    n_q = int(query_inputs.shape[0])
    if n_q == 0:
        return {}

    model.to(device)
    n_layers = len(ghost_layers)

    scores_acc: defaultdict[float] = defaultdict(float)
    sample_order: list[int] = []
    _rnn_cached = [m for m in model.modules() if isinstance(m, nn.RNNBase)]
    _inplace_cached = [
        m for m in model.modules() if getattr(m, "inplace", False)
    ]
    _mod_kw = {
        "rnn_modules": _rnn_cached,
        "inplace_modules": _inplace_cached,
    }

    uncovered_params: list[tuple[str, nn.Parameter]] = []
    if auto_fallback:
        uncovered_params = _find_uncovered_params(model, ghost_layers)
        if uncovered_params:
            logger.info(
                "Ghost TracIn: auto_fallback for %d uncovered parameter tensors",
                len(uncovered_params),
            )

    use_factored_no_adam = adam_param_keys is None
    use_adam_factored = False
    if adam_param_keys is not None and len(adam_param_keys) == n_layers:
        load_fn(model, checkpoints[0]["weights_path"], device)
        model.eval()
        raw_probe_a, raw_probe_e = _run_forward_backward(
            model,
            ghost_layers,
            training_loss_fn,
            query_inputs,
            query_targets,
            device,
            raw=True,
            max_spatial_positions=max_spatial_positions,
            **_mod_kw,
        )
        use_adam_factored = not _ghost_layers_need_per_layer_path(ghost_layers, raw_probe_a)
        adam_fd_dim = sum(_ghost_dim_for_layer(l) for l in ghost_layers)
        if projection_dim is not None and projection_dim < adam_fd_dim:
            use_adam_factored = False
        if use_adam_factored:
            opt0 = checkpoints[0].get("optimizer_state_path")
            w_shapes_probe = _weight_shapes_for_layers(ghost_layers)
            if not opt0:
                use_adam_factored = False
            else:
                try:
                    for pk, ws in zip(adam_param_keys, w_shapes_probe):
                        load_adam_inverse_sqrt_scale_matrix_ghost_layout(
                            opt0, pk, weight_shape=ws,
                        )
                except Exception as e:
                    logger.warning(
                        "Ghost TracIn: Adam factored probe failed; using materialized path: %s",
                        e,
                    )
                    use_adam_factored = False
        if use_adam_factored:
            logger.info(
                "Ghost TracIn: Adam-corrected factored dot-product (2D layers, no projection)",
            )

    if use_factored_no_adam:
        logger.info("Ghost TracIn: using factored dot-product (no projection needed)")

    for ckpt_idx, ckpt in enumerate(checkpoints):
        lr = float(ckpt["learning_rate"])
        load_fn(model, ckpt["weights_path"], device)
        model.eval()

        fb_q: Optional[torch.Tensor] = None
        if uncovered_params:
            fb_q = _per_sample_fallback_grad_matrix(
                model,
                training_loss_fn,
                query_inputs,
                query_targets,
                uncovered_params,
                device,
                rnn_modules=_rnn_cached,
                inplace_modules=_inplace_cached,
            )

        if use_factored_no_adam:
            # ----- Fast path: factored <A,A>*<E,E> per layer -----
            # Probe raw tensor shapes to detect 3D layers that need
            # sum-of-outer-products instead of the factored identity.
            raw_q_A, raw_q_E = _run_forward_backward(
                model,
                ghost_layers,
                training_loss_fn,
                query_inputs,
                query_targets,
                device,
                raw=True,
                max_spatial_positions=max_spatial_positions,
                **_mod_kw,
            )
            factored_idx = _factored_2d_linear_indices(ghost_layers, raw_q_A)
            factored_set = set(factored_idx)
            per_layer_idx = [i for i in range(n_layers) if i not in factored_set]
            pure_per_layer = len(factored_idx) == 0
            pure_factored = len(per_layer_idx) == 0

            use_cuda_path = str(device).startswith("cuda") and torch.cuda.is_available()
            cuda_gc_every = 20
            batch_i = 0

            def _warn_large_spatial() -> None:
                if ckpt_idx != 0:
                    return
                for a in raw_q_A:
                    if a.dim() == 3 and int(a.shape[1]) >= 1024:
                        logger.warning(
                            "Ghost TracIn: hooked layer has L=%d spatial/seq positions; "
                            "memory scales with L. Pass max_spatial_positions to cap if needed.",
                            int(a.shape[1]),
                        )
                        break

            if pure_per_layer:
                logger.info(
                    "Ghost TracIn: per-layer ghost dots (3D / Embedding / LayerNorm / mixed)",
                )
                _warn_large_spatial()
                for inputs, targets, batch_ids in data_loader:
                    raw_A, raw_E = _run_forward_backward(
                        model,
                        ghost_layers,
                        training_loss_fn,
                        inputs,
                        targets,
                        device,
                        raw=True,
                        max_spatial_positions=max_spatial_positions,
                        **_mod_kw,
                    )
                    dots = _layer_ghost_dots_from_raw_blocks(
                        raw_q_A,
                        raw_q_E,
                        raw_A,
                        raw_E,
                        ghost_layers=ghost_layers,
                        normalize_layer_dots=normalize_layer_dots,
                    )
                    del raw_A, raw_E
                    row = dots.mean(dim=0).detach().cpu().numpy().astype(np.float64)
                    del dots
                    if fb_q is not None:
                        fb_t = _per_sample_fallback_grad_matrix(
                            model,
                            training_loss_fn,
                            inputs,
                            targets,
                            uncovered_params,
                            device,
                            rnn_modules=_rnn_cached,
                            inplace_modules=_inplace_cached,
                        )
                        fdots = torch.mm(fb_q, fb_t.T)
                        row = row + fdots.mean(dim=0).detach().cpu().numpy().astype(
                            np.float64,
                        )
                        del fb_t, fdots
                    _accumulate_batch_tracin(
                        scores_acc, sample_order, ckpt_idx, batch_ids, row, lr,
                    )
                    batch_i += 1
                    if (
                        use_cuda_path
                        and batch_i % cuda_gc_every == 0
                        and torch.cuda.is_available()
                    ):
                        torch.cuda.empty_cache()

            elif pure_factored and use_cuda_path:
                q_A_list = [a.float() for a in raw_q_A]
                q_E_list = [e.float() for e in raw_q_E]
                for inputs, targets, batch_ids in data_loader:
                    A_list, E_list = _run_forward_backward(
                        model,
                        ghost_layers,
                        training_loss_fn,
                        inputs,
                        targets,
                        device,
                        as_torch=True,
                        **_mod_kw,
                    )
                    bs = int(A_list[0].shape[0])
                    dots = torch.zeros(
                        n_q, bs, device=device, dtype=torch.float32,
                    )
                    for l_idx in range(n_layers):
                        qa = q_A_list[l_idx]
                        qe = q_E_list[l_idx]
                        a = A_list[l_idx]
                        e = E_list[l_idx]
                        layer_score = torch.mm(qa, a.T) * torch.mm(qe, e.T)
                        if normalize_layer_dots:
                            ln = layer_score.norm(dim=1, keepdim=True).clamp(min=1e-8)
                            layer_score = layer_score / ln
                        dots.add_(layer_score)
                    del A_list, E_list
                    row = dots.mean(dim=0).detach().cpu().numpy().astype(np.float64)
                    del dots
                    if fb_q is not None:
                        fb_t = _per_sample_fallback_grad_matrix(
                            model,
                            training_loss_fn,
                            inputs,
                            targets,
                            uncovered_params,
                            device,
                            rnn_modules=_rnn_cached,
                            inplace_modules=_inplace_cached,
                        )
                        fdots = torch.mm(fb_q, fb_t.T)
                        row = row + fdots.mean(dim=0).detach().cpu().numpy().astype(
                            np.float64,
                        )
                        del fb_t, fdots
                    _accumulate_batch_tracin(
                        scores_acc, sample_order, ckpt_idx, batch_ids, row, lr,
                    )
                    batch_i += 1
                    if batch_i % cuda_gc_every == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            elif pure_factored:
                q_A_stacked = [
                    a.detach().cpu().numpy().astype(np.float32) for a in raw_q_A
                ]
                q_E_stacked = [
                    e.detach().cpu().numpy().astype(np.float32) for e in raw_q_E
                ]

                for inputs, targets, batch_ids in data_loader:
                    A_list, E_list = _run_forward_backward(
                        model,
                        ghost_layers,
                        training_loss_fn,
                        inputs,
                        targets,
                        device,
                        **_mod_kw,
                    )
                    bs = A_list[0].shape[0]
                    dots = np.zeros((n_q, bs), dtype=np.float64)
                    for l_idx in range(n_layers):
                        dot_a = q_A_stacked[l_idx] @ A_list[l_idx].T
                        dot_e = q_E_stacked[l_idx] @ E_list[l_idx].T
                        layer_score = dot_a * dot_e
                        if normalize_layer_dots:
                            norms = np.linalg.norm(layer_score, axis=1, keepdims=True)
                            norms = np.maximum(norms, 1e-8)
                            layer_score = layer_score / norms
                        dots += layer_score
                    del A_list, E_list
                    row = dots.mean(axis=0)
                    if fb_q is not None:
                        fb_t = _per_sample_fallback_grad_matrix(
                            model,
                            training_loss_fn,
                            inputs,
                            targets,
                            uncovered_params,
                            device,
                            rnn_modules=_rnn_cached,
                            inplace_modules=_inplace_cached,
                        )
                        fdots = torch.mm(fb_q, fb_t.T)
                        row = row + fdots.mean(dim=0).detach().cpu().numpy().astype(
                            np.float64,
                        )
                        del fb_t, fdots
                    _accumulate_batch_tracin(
                        scores_acc, sample_order, ckpt_idx, batch_ids, row, lr,
                    )
                    batch_i += 1
                    if batch_i % cuda_gc_every == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            else:
                logger.info(
                    "Ghost TracIn: hybrid factored + per-layer path "
                    "(factored layers %s, per-layer %s)",
                    factored_idx,
                    per_layer_idx,
                )
                _warn_large_spatial()
                gl_per = [ghost_layers[i] for i in per_layer_idx]
                rq_A_p = [raw_q_A[i] for i in per_layer_idx]
                rq_E_p = [raw_q_E[i] for i in per_layer_idx]
                if use_cuda_path:
                    q_A_list = [a.float() for a in raw_q_A]
                    q_E_list = [e.float() for e in raw_q_E]
                    for inputs, targets, batch_ids in data_loader:
                        raw_A, raw_E = _run_forward_backward(
                            model,
                            ghost_layers,
                            training_loss_fn,
                            inputs,
                            targets,
                            device,
                            raw=True,
                            max_spatial_positions=max_spatial_positions,
                            **_mod_kw,
                        )
                        bs = int(raw_A[0].shape[0])
                        dots = torch.zeros(
                            n_q, bs, device=device, dtype=torch.float32,
                        )
                        for l_idx in factored_idx:
                            qa = q_A_list[l_idx]
                            qe = q_E_list[l_idx]
                            a = raw_A[l_idx].float()
                            e = raw_E[l_idx].float()
                            layer_score = torch.mm(qa, a.T) * torch.mm(qe, e.T)
                            if normalize_layer_dots:
                                ln = layer_score.norm(
                                    dim=1, keepdim=True,
                                ).clamp(min=1e-8)
                                layer_score = layer_score / ln
                            dots.add_(layer_score)
                        rA_p = [raw_A[i] for i in per_layer_idx]
                        rE_p = [raw_E[i] for i in per_layer_idx]
                        dots.add_(
                            _layer_ghost_dots_from_raw_blocks(
                                rq_A_p,
                                rq_E_p,
                                rA_p,
                                rE_p,
                                ghost_layers=gl_per,
                                normalize_layer_dots=normalize_layer_dots,
                            ),
                        )
                        del raw_A, raw_E, rA_p, rE_p
                        row = dots.mean(dim=0).detach().cpu().numpy().astype(
                            np.float64,
                        )
                        del dots
                        if fb_q is not None:
                            fb_t = _per_sample_fallback_grad_matrix(
                                model,
                                training_loss_fn,
                                inputs,
                                targets,
                                uncovered_params,
                                device,
                                rnn_modules=_rnn_cached,
                                inplace_modules=_inplace_cached,
                            )
                            fdots = torch.mm(fb_q, fb_t.T)
                            row = row + fdots.mean(dim=0).detach().cpu().numpy().astype(
                                np.float64,
                            )
                            del fb_t, fdots
                        _accumulate_batch_tracin(
                            scores_acc, sample_order, ckpt_idx, batch_ids, row, lr,
                        )
                        batch_i += 1
                        if batch_i % cuda_gc_every == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    q_A_stacked = [
                        raw_q_A[i].detach().cpu().numpy().astype(np.float32)
                        for i in factored_idx
                    ]
                    q_E_stacked = [
                        raw_q_E[i].detach().cpu().numpy().astype(np.float32)
                        for i in factored_idx
                    ]
                    fi = factored_idx
                    for inputs, targets, batch_ids in data_loader:
                        raw_A, raw_E = _run_forward_backward(
                            model,
                            ghost_layers,
                            training_loss_fn,
                            inputs,
                            targets,
                            device,
                            raw=True,
                            max_spatial_positions=max_spatial_positions,
                            **_mod_kw,
                        )
                        bs = int(raw_E[0].shape[0])
                        dots = np.zeros((n_q, bs), dtype=np.float64)
                        for j, l_idx in enumerate(fi):
                            dot_a = q_A_stacked[j] @ raw_A[l_idx].detach().cpu().numpy().astype(
                                np.float32,
                            ).T
                            dot_e = q_E_stacked[j] @ raw_E[l_idx].detach().cpu().numpy().astype(
                                np.float32,
                            ).T
                            layer_score = dot_a * dot_e
                            if normalize_layer_dots:
                                norms = np.linalg.norm(
                                    layer_score, axis=1, keepdims=True,
                                )
                                norms = np.maximum(norms, 1e-8)
                                layer_score = layer_score / norms
                            dots += layer_score
                        rA_p = [raw_A[i] for i in per_layer_idx]
                        rE_p = [raw_E[i] for i in per_layer_idx]
                        sub = _layer_ghost_dots_from_raw_blocks(
                            rq_A_p,
                            rq_E_p,
                            rA_p,
                            rE_p,
                            ghost_layers=gl_per,
                            normalize_layer_dots=normalize_layer_dots,
                        )
                        dots = dots + sub.detach().cpu().numpy().astype(np.float64)
                        del raw_A, raw_E, rA_p, rE_p, sub
                        row = dots.mean(axis=0)
                        if fb_q is not None:
                            fb_t = _per_sample_fallback_grad_matrix(
                                model,
                                training_loss_fn,
                                inputs,
                                targets,
                                uncovered_params,
                                device,
                                rnn_modules=_rnn_cached,
                                inplace_modules=_inplace_cached,
                            )
                            fdots = torch.mm(fb_q, fb_t.T)
                            row = row + fdots.mean(dim=0).detach().cpu().numpy().astype(
                                np.float64,
                            )
                            del fb_t, fdots
                        _accumulate_batch_tracin(
                            scores_acc, sample_order, ckpt_idx, batch_ids, row, lr,
                        )
                        batch_i += 1
                        if batch_i % cuda_gc_every == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()

        elif use_adam_factored:
            # ----- Adam + factored 2D: score = sum_{h,c} E_q E_t A_q A_t / (sqrt(v)+eps)^2 -----
            w_shapes = _weight_shapes_for_layers(ghost_layers)
            opt_path = ckpt.get("optimizer_state_path")
            adam_M_list: list[torch.Tensor] = []
            for l_idx, layer in enumerate(ghost_layers):
                try:
                    if opt_path:
                        Mi = load_adam_inverse_sqrt_scale_matrix_ghost_layout(
                            opt_path,
                            adam_param_keys[l_idx],
                            weight_shape=w_shapes[l_idx],
                        )
                        Mi_sq = Mi * Mi
                        adam_M_list.append(
                            torch.from_numpy(Mi_sq).to(device=device, dtype=torch.float32),
                        )
                    else:
                        adam_M_list.append(
                            _adam_scale_matrix_for_layer(
                                layer,
                                torch.device(device),
                                torch.float32,
                            ),
                        )
                except Exception as e:
                    logger.warning(
                        "Ghost TracIn: ckpt %d layer %d Adam matrix load failed, using ones: %s",
                        ckpt_idx,
                        l_idx,
                        e,
                    )
                    adam_M_list.append(
                        _adam_scale_matrix_for_layer(
                            layer,
                            torch.device(device),
                            torch.float32,
                        ),
                    )

            raw_q_adam_A, raw_q_adam_E = _run_forward_backward(
                model,
                ghost_layers,
                training_loss_fn,
                query_inputs,
                query_targets,
                device,
                raw=True,
                max_spatial_positions=max_spatial_positions,
                **_mod_kw,
            )
            q_A_list = [a.float() for a in raw_q_adam_A]
            q_E_list = [e.float() for e in raw_q_adam_E]
            adam_batch_i = 0
            for inputs, targets, batch_ids in data_loader:
                A_list, E_list = _run_forward_backward(
                    model,
                    ghost_layers,
                    training_loss_fn,
                    inputs,
                    targets,
                    device,
                    as_torch=True,
                    **_mod_kw,
                )
                bs = int(A_list[0].shape[0])
                dots = torch.zeros(
                    n_q, bs, device=device, dtype=torch.float32,
                )
                for l_idx in range(n_layers):
                    Ml = adam_M_list[l_idx]
                    qa = q_A_list[l_idx]
                    qe = q_E_list[l_idx]
                    a = A_list[l_idx].float()
                    e = E_list[l_idx].float()
                    e_outer = qe[:, None, :] * e[None, :, :]
                    a_outer = qa[:, None, :] * a[None, :, :]
                    t = torch.einsum("qbc,hc->qbh", e_outer, Ml)
                    layer_score = (t * a_outer).sum(dim=-1)
                    if normalize_layer_dots:
                        ln = layer_score.norm(dim=1, keepdim=True).clamp(min=1e-8)
                        layer_score = layer_score / ln
                    dots.add_(layer_score)
                del A_list, E_list
                row = dots.mean(dim=0).detach().cpu().numpy().astype(np.float64)
                del dots
                if fb_q is not None:
                    fb_t = _per_sample_fallback_grad_matrix(
                        model,
                        training_loss_fn,
                        inputs,
                        targets,
                        uncovered_params,
                        device,
                        rnn_modules=_rnn_cached,
                        inplace_modules=_inplace_cached,
                    )
                    fdots = torch.mm(fb_q, fb_t.T)
                    row = row + fdots.mean(dim=0).detach().cpu().numpy().astype(
                        np.float64,
                    )
                    del fb_t, fdots
                _accumulate_batch_tracin(
                    scores_acc, sample_order, ckpt_idx, batch_ids, row, lr,
                )
                adam_batch_i += 1
                if (
                    adam_batch_i % 20 == 0
                    and str(device).startswith("cuda")
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

        else:
            # ----- Slow path: materialize ghost vectors + Adam + project -----
            w_shapes = _weight_shapes_for_layers(ghost_layers)
            adam_v: Optional[np.ndarray] = None
            opt_path = ckpt.get("optimizer_state_path")
            if adam_param_keys is not None and opt_path:
                try:
                    adam_v = concatenate_adam_second_moments(
                        opt_path, list(adam_param_keys), w_shapes,
                    )
                except Exception as e:
                    logger.warning("Ghost TracIn: Adam state failed ckpt %d: %s", ckpt_idx, e)

            # Probe ghost dim for projection (first checkpoint only)
            if ckpt_idx == 0:
                probe_in, probe_tg, _ = next(iter(data_loader))
                g_probe = _extract_ghost_vectors(
                    model,
                    ghost_layers,
                    training_loss_fn,
                    probe_in,
                    probe_tg,
                    device,
                    max_spatial_positions=max_spatial_positions,
                    rnn_modules=_rnn_cached,
                    inplace_modules=_inplace_cached,
                )
                ghost_dim = int(g_probe.shape[1])
                proj_dim = projection_dim if projection_dim is not None and projection_dim < ghost_dim else ghost_dim
                P: object = None
                if proj_dim < ghost_dim:
                    if projection_type == "sjlt":
                        P = build_sjlt_matrix(ghost_dim, proj_dim, seed=projection_seed)
                    else:
                        P = build_dense_projection(ghost_dim, proj_dim, seed=projection_seed)

            g_query_full = _extract_ghost_vectors(
                model,
                ghost_layers,
                training_loss_fn,
                query_inputs,
                query_targets,
                device,
                max_spatial_positions=max_spatial_positions,
                rnn_modules=_rnn_cached,
                inplace_modules=_inplace_cached,
            )
            if adam_v is not None:
                g_query_full = apply_adam_correction(g_query_full, adam_v)
            g_query = g_query_full if P is None else project(g_query_full, P)

            slow_batch_i = 0
            for inputs, targets, batch_ids in data_loader:
                g_train = _extract_ghost_vectors(
                    model,
                    ghost_layers,
                    training_loss_fn,
                    inputs,
                    targets,
                    device,
                    max_spatial_positions=max_spatial_positions,
                    rnn_modules=_rnn_cached,
                    inplace_modules=_inplace_cached,
                )
                if adam_v is not None:
                    g_train = apply_adam_correction(g_train, adam_v)
                g_train_proj = g_train if P is None else project(g_train, P)
                del g_train
                dots = g_query @ g_train_proj.T
                dot_mean = dots.mean(axis=0)
                row = np.asarray(dot_mean, dtype=np.float64).ravel()
                del dots, g_train_proj
                if fb_q is not None:
                    fb_t = _per_sample_fallback_grad_matrix(
                        model,
                        training_loss_fn,
                        inputs,
                        targets,
                        uncovered_params,
                        device,
                        rnn_modules=_rnn_cached,
                        inplace_modules=_inplace_cached,
                    )
                    fdots = torch.mm(fb_q, fb_t.T)
                    row = row + fdots.mean(dim=0).detach().cpu().numpy().astype(
                        np.float64,
                    )
                    del fb_t, fdots
                _accumulate_batch_tracin(
                    scores_acc, sample_order, ckpt_idx, batch_ids, row, lr,
                )
                slow_batch_i += 1
                if (
                    slow_batch_i % 20 == 0
                    and str(device).startswith("cuda")
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

        logger.info("Ghost TracIn: checkpoint %d/%d done (lr=%.6f)",
                     ckpt_idx + 1, len(checkpoints), lr)

    if not sample_order:
        return {}
    out: dict[int, float] = {}
    for sid in sample_order:
        if sid not in out:
            out[sid] = float(scores_acc[sid])
    return out
