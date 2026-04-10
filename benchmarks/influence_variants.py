"""Compute all six benchmark variants (A\u2013F) for Ghost vs Original TracIn diagnostics."""

from __future__ import annotations

from typing import Callable, Final, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from benchmarks.exact_tracin import (
    compute_exact_tracin_scores,
    compute_true_tracin_ghost_scores,
)
from benchmarks.full_gradient_tracin import compute_full_gradient_tracin_scores
from benchmarks.ghost_faiss import _ghost_dim_for_layer, compute_ghost_faiss_scores

VARIANT_ORDER: Final[tuple[str, ...]] = (
    "A_ghost_faiss",
    "B_ghost_noproj",
    "C_ghost_noproj_noadam",
    "D_ghost_multi_ckpt",
    "E_ghost_multi_ckpt_noadam",
    "F_fullgrad_multi_ckpt",
)


def model_ghost_coverage(
    model: nn.Module,
    target_layer: nn.Module,
    ghost_layers: Optional[list[nn.Module]] = None,
) -> dict:
    """Fraction of total parameters covered by hooked ghost layer(s)."""
    layers = ghost_layers if ghost_layers is not None else [target_layer]
    total = sum(int(p.numel()) for p in model.parameters())

    def _layer_param_count_for_coverage(layer: nn.Module) -> int:
        if isinstance(layer, nn.RNNBase):
            return sum(
                int(p.numel())
                for name, p in layer.named_parameters()
                if "weight_ih" in name or "bias_ih" in name
            )
        return sum(int(p.numel()) for p in layer.parameters())

    ghost_params = sum(_layer_param_count_for_coverage(layer) for layer in layers)
    # ghost_dim: matches materialised ghost / _ghost_dim_for_layer (weights + bias aug)
    ghost_dim = sum(_ghost_dim_for_layer(layer) for layer in layers)
    pct = 100.0 * ghost_params / total if total else 0.0
    return {
        "total_params": total,
        "ghost_layer_params": ghost_params,
        "ghost_coverage_pct": round(pct, 4),
        "ghost_dim": ghost_dim,
        "layers": len(layers),
    }


def compute_all_six_variants(
    model: nn.Module,
    target_layer: nn.Module,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    checkpoints: list[dict],
    sample_metadata: dict[int, str],
    query_inputs: torch.Tensor,
    query_targets: torch.Tensor,
    checkpoint_path: str,
    optimizer_state_path: Optional[str],
    index_dir: str,
    adam_param_key: Union[int, str] = 2,
    device: str = "cpu",
    projection_dim: Optional[int] = 1280,
    projection_type: str = "sjlt",
    projection_seed: int = 42,
    faiss_index_filename: str = "bench_faiss_index",
    faiss_meta_filename: str = "bench_faiss_meta.json",
) -> dict[str, dict[int, float]]:
    """Return score dicts for variants A–F (keys: A_ghost_faiss … F_fullgrad_multi_ckpt)."""
    out: dict[str, dict[int, float]] = {}

    out["A_ghost_faiss"] = compute_ghost_faiss_scores(
        model,
        target_layer,
        error_fn,
        data_loader,
        checkpoints,
        sample_metadata,
        query_inputs,
        query_targets,
        checkpoint_path,
        optimizer_state_path,
        index_dir,
        index_filename=faiss_index_filename,
        metadata_filename=faiss_meta_filename,
        projection_dim=projection_dim,
        projection_type=projection_type,
        projection_seed=projection_seed,
        adam_param_key=adam_param_key,
        device=device,
    )

    out["B_ghost_noproj"] = compute_exact_tracin_scores(
        model,
        target_layer,
        error_fn,
        data_loader,
        query_inputs,
        query_targets,
        checkpoints,
        adam_param_key=adam_param_key,
        optimizer_state_path=optimizer_state_path,
        use_adam=True,
        device=device,
    )

    out["C_ghost_noproj_noadam"] = compute_exact_tracin_scores(
        model,
        target_layer,
        error_fn,
        data_loader,
        query_inputs,
        query_targets,
        checkpoints,
        adam_param_key=adam_param_key,
        optimizer_state_path=optimizer_state_path,
        use_adam=False,
        device=device,
    )

    out["D_ghost_multi_ckpt"] = compute_true_tracin_ghost_scores(
        model,
        target_layer,
        error_fn,
        data_loader,
        query_inputs,
        query_targets,
        checkpoints,
        adam_param_key=adam_param_key,
        use_adam=True,
        device=device,
    )

    out["E_ghost_multi_ckpt_noadam"] = compute_true_tracin_ghost_scores(
        model,
        target_layer,
        error_fn,
        data_loader,
        query_inputs,
        query_targets,
        checkpoints,
        adam_param_key=adam_param_key,
        use_adam=False,
        device=device,
    )

    out["F_fullgrad_multi_ckpt"] = compute_full_gradient_tracin_scores(
        model,
        loss_fn,
        data_loader,
        query_inputs,
        query_targets,
        checkpoints,
        device=device,
    )

    return out


def compute_diagnostic_middle_variants(
    model: nn.Module,
    target_layer: nn.Module,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    checkpoints: list[dict],
    query_inputs: torch.Tensor,
    query_targets: torch.Tensor,
    checkpoint_path: str,
    optimizer_state_path: Optional[str],
    adam_param_key: Union[int, str] = 2,
    device: str = "cpu",
) -> dict[str, dict[int, float]]:
    """Variants B–E only (for ``--diagnostic`` when A/F already computed with profiling)."""
    out: dict[str, dict[int, float]] = {}
    out["B_ghost_noproj"] = compute_exact_tracin_scores(
        model,
        target_layer,
        error_fn,
        data_loader,
        query_inputs,
        query_targets,
        checkpoints,
        adam_param_key=adam_param_key,
        optimizer_state_path=optimizer_state_path,
        use_adam=True,
        device=device,
    )
    out["C_ghost_noproj_noadam"] = compute_exact_tracin_scores(
        model,
        target_layer,
        error_fn,
        data_loader,
        query_inputs,
        query_targets,
        checkpoints,
        adam_param_key=adam_param_key,
        optimizer_state_path=optimizer_state_path,
        use_adam=False,
        device=device,
    )
    out["D_ghost_multi_ckpt"] = compute_true_tracin_ghost_scores(
        model,
        target_layer,
        error_fn,
        data_loader,
        query_inputs,
        query_targets,
        checkpoints,
        adam_param_key=adam_param_key,
        use_adam=True,
        device=device,
    )
    out["E_ghost_multi_ckpt_noadam"] = compute_true_tracin_ghost_scores(
        model,
        target_layer,
        error_fn,
        data_loader,
        query_inputs,
        query_targets,
        checkpoints,
        adam_param_key=adam_param_key,
        use_adam=False,
        device=device,
    )
    return out
