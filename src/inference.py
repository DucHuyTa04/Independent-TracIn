"""Online attribution: generated output -> rights holder percentages.

Pipeline:
    1. Load model from checkpoint
    2. Hook → forward query → capture A → error_fn → E
    3. Ghost → Adam correct → project (same params as indexer)
    4. FAISS inner-product search
    5. Clamp negatives, normalize to attribution percentages
    6. Group by rights holder for revenue sharing
"""

import logging
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from src.faiss_store import FAISSStore
from src.hooks_manager import HookManager
from src.math_utils import (
    apply_adam_correction,
    build_dense_projection,
    build_sjlt_matrix,
    form_ghost_vectors,
    load_adam_second_moment,
    project,
)

logger = logging.getLogger(__name__)


def attribute(
    model: nn.Module,
    target_layer: nn.Module,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    query_inputs: torch.Tensor,
    query_targets: torch.Tensor,
    index_path: str,
    metadata_path: str,
    checkpoint_path: str,
    projection_dim: int = 1280,
    projection_type: str = "sjlt",
    projection_seed: int = 42,
    optimizer_state_path: Optional[str] = None,
    adam_param_key: Union[int, str] = 1,
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
        state = torch.load(path, map_location=dev, weights_only=True)
        m.load_state_dict(state)

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
            adam_v = load_adam_second_moment(optimizer_state_path, adam_param_key)
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

    # Rule 1: free memory
    del A, E, E_tensor, g, g_proj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results
