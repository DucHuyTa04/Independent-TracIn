"""Online attribution: query → ghost → FAISS search → rights-holder percentages."""

import logging
from collections import defaultdict
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from src.config_utils import smart_load_weights_into_model
from src.faiss_store import FAISSStore
from src.hooks_manager import HookManager
from src.math_utils import (
    apply_adam_correction,
    build_dense_projection,
    build_sjlt_matrix,
    form_ghost_vectors,
    load_adam_second_moment_with_bias,
    project,
)

logger = logging.getLogger(__name__)


def _weight_shape_for_adam(layer: nn.Module) -> Optional[tuple[int, int]]:
    """(out_features, in_features) for a 2D weight tensor, else None."""
    w = getattr(layer, "weight", None)
    if w is not None and w.dim() == 2:
        return (int(w.shape[0]), int(w.shape[1]))
    return None


def attribute(
    model: nn.Module,
    target_layer: nn.Module,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    query_inputs: torch.Tensor,
    query_targets: torch.Tensor,
    index_path: str,
    metadata_path: str,
    checkpoint_path: str,
    projection_dim: Optional[int] = 1280,
    projection_type: str = "sjlt",
    projection_seed: int = 42,
    optimizer_state_path: Optional[str] = None,
    adam_param_key: Union[int, str] = 1,
    adam_bias_param_key: Optional[Union[int, str]] = None,
    top_k: int = 20,
    index_type: str = "flat",
    device: str = "auto",
    load_weights_fn: Optional[Callable[[nn.Module, str, str], None]] = None,
) -> list[dict]:
    """Compute attribution for generated output(s) against indexed training data.

    This is the online phase of the TracIn attribution pipeline.
    Computes the ghost vector for the query, searches the FAISS index,
    and returns per-rights-holder attribution percentages.

    Args:
        model: PyTorch model (will NOT be modified permanently).
        target_layer: The nn.Module layer to hook (must match indexer).
        error_fn: Callable (logits, targets) -> error signal E, shape (batch, C).
                  Must be the same function used during indexing.
        query_inputs: (batch, ...) the generated outputs to attribute.
        query_targets: (batch, ...) reconstruction targets
                       (classification: labels, diffusion: noise, LLM: token_ids).
        index_path: Path to the FAISS index built by build_index().
        metadata_path: Path to the metadata JSON built by build_index().
        checkpoint_path: Path to model weights for the query.
        projection_dim: Must match the value used in build_index().
        projection_type: Must match the value used in build_index().
        projection_seed: Must match the value used in build_index().
        optimizer_state_path: Optional path to optimizer state for Adam correction.
        adam_param_key: Key for target layer params in optimizer state.
        top_k: Number of top training samples to retrieve.
        index_type: FAISS index type (must match build_index).
        device: "auto", "cuda", or "cpu".
        load_weights_fn: Optional custom weight loader. Default: torch.load + load_state_dict.

    Returns:
        List of dicts (one per query sample):
            {
                "rights_holder_attribution": {"holder_name": percentage, ...},
                "top_samples": [(sample_id, raw_score), ...]
            }
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _default_load_weights(m: nn.Module, path: str, dev: str) -> None:
        smart_load_weights_into_model(m, path, dev)

    load_fn = load_weights_fn or _default_load_weights

    # Ensure batch dimensions
    if query_inputs.dim() == 1:
        query_inputs = query_inputs.unsqueeze(0)
    if query_targets.dim() == 0:
        query_targets = query_targets.unsqueeze(0)

    query_inputs = query_inputs.to(device)
    query_targets = query_targets.to(device)

    # Load model
    model.to(device)
    load_fn(model, checkpoint_path, device)
    model.eval()

    # Extract A and E via hooks
    with HookManager(model, target_layer) as hm:
        with torch.no_grad():
            logits = model(query_inputs)
        A = hm.activation.cpu().numpy().astype(np.float32)
        E_tensor = error_fn(logits, query_targets)
        E = E_tensor.detach().cpu().numpy().astype(np.float32)

    # Ghost → Adam correct → project
    g = form_ghost_vectors(A, E)

    adam_v = None
    if optimizer_state_path:
        try:
            adam_v = load_adam_second_moment_with_bias(
                optimizer_state_path,
                adam_param_key,
                adam_bias_param_key,
                weight_shape=_weight_shape_for_adam(target_layer),
            )
        except Exception as e:
            logger.warning("Could not load Adam state: %s", e)

    if adam_v is not None:
        g = apply_adam_correction(g, adam_v)

    # Build same projection as indexer (deterministic via seed)
    ghost_dim = g.shape[1]
    if projection_dim is not None and projection_dim < ghost_dim:
        if projection_type == "sjlt":
            P = build_sjlt_matrix(ghost_dim, projection_dim, seed=projection_seed)
        else:
            P = build_dense_projection(ghost_dim, projection_dim, seed=projection_seed)
        g_proj = project(g, P)
    else:
        g_proj = g

    # Load FAISS index and query
    store = FAISSStore(index_type=index_type)
    store.load(index_path, metadata_path)
    scores, indices, sample_ids_per_query = store.query(g_proj, top_k)

    sample_id_to_holder = store.metadata.get("sample_id_to_rights_holder", {})

    # Clamp negatives, normalize, group by rights holder
    results = []
    for q in range(g_proj.shape[0]):
        s = scores[q]
        ids = sample_ids_per_query[q]

        # Clamp negatives to 0 (only positive influence counts)
        scores_pos = np.maximum(0, s.astype(np.float64))
        total = scores_pos.sum()

        if total <= 0:
            results.append({
                "rights_holder_attribution": {},
                "top_samples": list(zip(ids, s.tolist())),
            })
            continue

        pct = scores_pos / total

        # Group by rights holder
        holder_pct: dict[str, float] = {}
        for sid, p in zip(ids, pct):
            holder = sample_id_to_holder.get(str(sid), f"sample_{sid}")
            holder_pct[holder] = holder_pct.get(holder, 0.0) + float(p)

        results.append({
            "rights_holder_attribution": holder_pct,
            "top_samples": list(zip(ids, s.tolist())),
        })

    del A, E, E_tensor, g, g_proj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def attribute_multi_checkpoint(
    model: nn.Module,
    target_layer: nn.Module,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    query_inputs: torch.Tensor,
    query_targets: torch.Tensor,
    checkpoint_index_specs: list[dict],
    projection_dim: Optional[int] = 1280,
    projection_type: str = "sjlt",
    projection_seed: int = 42,
    adam_param_key: Union[int, str] = 1,
    adam_bias_param_key: Optional[Union[int, str]] = None,
    top_k: int = 20,
    index_type: str = "flat",
    device: str = "auto",
    load_weights_fn: Optional[Callable[[nn.Module, str, str], None]] = None,
) -> list[dict]:
    """Attribution with one FAISS index per checkpoint (``build_multi_checkpoint_index``).

    For each checkpoint: load weights, compute query ghost, query that index for **all**
    training vectors, sum inner products across checkpoints. Then clamp, normalize, and
    return top-``top_k`` (same result shape as ``attribute``).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not checkpoint_index_specs:
        raise ValueError("checkpoint_index_specs is empty")

    def _default_load_weights(m: nn.Module, path: str, dev: str) -> None:
        smart_load_weights_into_model(m, path, dev)

    load_fn = load_weights_fn or _default_load_weights

    if query_inputs.dim() == 1:
        query_inputs = query_inputs.unsqueeze(0)
    if query_targets.dim() == 0:
        query_targets = query_targets.unsqueeze(0)

    query_inputs = query_inputs.to(device)
    query_targets = query_targets.to(device)

    meta_store = FAISSStore(index_type=index_type)
    meta_store.load(
        checkpoint_index_specs[0]["index_path"],
        checkpoint_index_specs[0]["metadata_path"],
    )
    sample_id_to_holder = meta_store.metadata.get("sample_id_to_rights_holder", {})
    del meta_store

    model.to(device)
    results: list[dict] = []

    for q in range(query_inputs.shape[0]):
        q_in = query_inputs[q : q + 1]
        q_tg = query_targets[q : q + 1]
        total_scores: dict[int, float] = defaultdict(float)

        for spec in checkpoint_index_specs:
            wpath = spec["weights_path"]
            opath = spec.get("optimizer_state_path")
            idx_p = spec["index_path"]
            meta_p = spec["metadata_path"]

            load_fn(model, wpath, device)
            model.eval()

            with HookManager(model, target_layer) as hm:
                with torch.no_grad():
                    logits = model(q_in)
                A = hm.activation.cpu().numpy().astype(np.float32)
                E_tensor = error_fn(logits, q_tg)
                E = E_tensor.detach().cpu().numpy().astype(np.float32)

            g = form_ghost_vectors(A, E)

            adam_v = None
            if opath:
                try:
                    adam_v = load_adam_second_moment_with_bias(
                        opath,
                        adam_param_key,
                        adam_bias_param_key,
                        weight_shape=_weight_shape_for_adam(target_layer),
                    )
                except Exception as e:
                    logger.warning("Multi-ckpt query: could not load Adam state: %s", e)

            if adam_v is not None:
                g = apply_adam_correction(g, adam_v)

            ghost_dim = g.shape[1]
            if projection_dim is not None and projection_dim < ghost_dim:
                if projection_type == "sjlt":
                    P = build_sjlt_matrix(ghost_dim, projection_dim, seed=projection_seed)
                else:
                    P = build_dense_projection(ghost_dim, projection_dim, seed=projection_seed)
                g_proj = project(g, P)
            else:
                g_proj = g

            store = FAISSStore(index_type=index_type)
            store.load(idx_p, meta_p)
            n_all = len(store._sample_ids)
            scores, _, sid_lists = store.query(g_proj, top_k=n_all)
            for sid, sc in zip(sid_lists[0], scores[0]):
                total_scores[int(sid)] += float(sc)

            del A, E, E_tensor, g, g_proj, logits, store
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ranked = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ids = [x[0] for x in ranked]
        s = np.array([x[1] for x in ranked], dtype=np.float64)
        scores_pos = np.maximum(0, s)
        tot = scores_pos.sum()

        if tot <= 0:
            results.append({
                "rights_holder_attribution": {},
                "top_samples": list(zip(ids, s.tolist())),
            })
            continue

        pct = scores_pos / tot
        holder_pct: dict[str, float] = {}
        for sid, p in zip(ids, pct):
            holder = sample_id_to_holder.get(str(sid), f"sample_{sid}")
            holder_pct[holder] = holder_pct.get(holder, 0.0) + float(p)

        results.append({
            "rights_holder_attribution": holder_pct,
            "top_samples": list(zip(ids, s.tolist())),
        })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results
