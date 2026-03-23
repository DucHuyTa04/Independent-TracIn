"""Offline indexer: process copyrighted training data into a FAISS index.

Pipeline per checkpoint (from TracIn):
    1. Load model weights, set eval mode
    2. For each batch: Hook → capture A → error_fn → E → ghost → adam correct → project
    3. Accumulate lr-weighted projected vectors across checkpoints
    4. Build FAISS IndexFlatIP, save index + metadata

Memory management (Rule 1):
    - torch.no_grad() globally during extraction
    - del intermediate tensors after projection
    - torch.cuda.empty_cache() every batch
"""

import logging
import os
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from torch.utils.data import DataLoader

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


def _resolve_device(device: str) -> str:
    """Resolve 'auto' device to cuda/cpu."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _build_projection(
    ghost_dim: int,
    projection_dim: Optional[int],
    projection_type: str,
    projection_seed: int,
) -> tuple[int, Optional[Union[sparse.spmatrix, np.ndarray]]]:
    """Build projection matrix P based on config.

    Returns:
        (output_dim, P) where P is None if no projection needed.
    """
    if projection_dim is None or projection_dim >= ghost_dim:
        return ghost_dim, None

    if projection_type == "sjlt":
        P = build_sjlt_matrix(ghost_dim, projection_dim, seed=projection_seed)
    else:
        P = build_dense_projection(ghost_dim, projection_dim, seed=projection_seed)

    return projection_dim, P


def _probe_ghost_dim(
    model: nn.Module,
    target_layer: nn.Module,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    device: str,
) -> int:
    """Run one batch to determine ghost vector dimensionality H*C.

    Args:
        model: The model in eval mode.
        target_layer: Layer to hook.
        error_fn: (logits, targets) -> E tensor of shape (batch, C).
        data_loader: Training data loader.
        device: Torch device string.

    Returns:
        Ghost dimension H * C.
    """
    inputs, targets, _ = next(iter(data_loader))
    inputs = inputs.to(device)
    targets = targets.to(device)

    with HookManager(model, target_layer) as hm:
        with torch.no_grad():
            logits = model(inputs)
        A = hm.activation
        E = error_fn(logits, targets)

    H = A.shape[1]
    C = E.shape[1] if E.dim() > 1 else 1
    return H * C


def build_index(
    model: nn.Module,
    target_layer: nn.Module,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    checkpoints: list[dict],
    sample_metadata: dict[int, str],
    projection_dim: int = 1280,
    projection_type: str = "sjlt",
    projection_seed: int = 42,
    adam_param_key: Union[int, str] = 1,
    output_dir: str = "outputs",
    index_filename: str = "faiss_index",
    metadata_filename: str = "faiss_metadata.json",
    index_type: str = "flat",
    device: str = "auto",
    load_weights_fn: Optional[Callable[[nn.Module, str, str], None]] = None,
) -> str:
    """Build a FAISS index from training data ghost vectors.

    This is the offline phase of the TracIn attribution pipeline.
    Processes the copyrighted dataset through the model at each checkpoint,
    extracts ghost vectors, applies corrections, and stores in FAISS.

    Args:
        model: PyTorch model (will NOT be modified).
        target_layer: The nn.Module layer to hook (e.g., model.fc2).
        error_fn: Callable (logits, targets) -> error signal E, shape (batch, C).
                  For classification: softmax(logits) - one_hot(targets).
                  For regression: logits - targets.
        data_loader: DataLoader yielding (inputs, targets, sample_ids) 3-tuples.
        checkpoints: List of dicts with keys:
                     - "weights_path" (str): path to model weights
                     - "learning_rate" (float): lr at this checkpoint
                     - "optimizer_state_path" (str, optional): path to optimizer state
        sample_metadata: Mapping sample_id -> rights_holder_id for attribution.
        projection_dim: Target dimension for SJLT/dense projection.
        projection_type: "sjlt" or "dense".
        projection_seed: RNG seed for projection matrix.
        adam_param_key: Key for the target layer's params in optimizer state.
        output_dir: Directory to save index and metadata.
        index_filename: Filename for the FAISS index.
        metadata_filename: Filename for the metadata JSON.
        index_type: FAISS index type ("flat" or "ivf").
        device: "auto", "cuda", or "cpu".
        load_weights_fn: Optional custom weight loader (model, path, device) -> None.
                         Default: torch.load + load_state_dict.

    Returns:
        Path to the saved FAISS index file.
    """
    device = _resolve_device(device)

    if not checkpoints:
        raise ValueError("checkpoints list is empty")

    def _default_load_weights(m: nn.Module, path: str, dev: str) -> None:
        state = torch.load(path, map_location=dev, weights_only=True)
        m.load_state_dict(state)

    load_fn = load_weights_fn or _default_load_weights

    # Collect all sample_ids in order
    all_sample_ids: list[int] = []
    for _, _, batch_ids in data_loader:
        all_sample_ids.extend(int(x) for x in batch_ids)
    n_train = len(all_sample_ids)
    logger.info("Training samples to index: %d", n_train)

    # Probe ghost dim from one batch
    model.to(device)
    load_fn(model, checkpoints[0]["weights_path"], device)
    model.eval()
    ghost_dim = _probe_ghost_dim(model, target_layer, error_fn, data_loader, device)

    # Build projection matrix
    proj_dim, P = _build_projection(
        ghost_dim, projection_dim, projection_type, projection_seed,
    )
    logger.info("Ghost dim=%d, projection dim=%d", ghost_dim, proj_dim)

    # Initialize accumulation buffer
    accumulated = np.zeros((n_train, proj_dim), dtype=np.float32)

    # Process each checkpoint
    for ckpt_idx, ckpt in enumerate(checkpoints):
        weights_path = ckpt["weights_path"]
        opt_path = ckpt.get("optimizer_state_path")
        lr = float(ckpt["learning_rate"])

        load_fn(model, weights_path, device)
        model.eval()

        # Load Adam state if available
        adam_v = None
        if opt_path:
            try:
                adam_v = load_adam_second_moment(opt_path, adam_param_key)
            except Exception as e:
                logger.warning("Could not load Adam state from %s: %s", opt_path, e)

        offset = 0
        for inputs, targets, batch_ids in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Extract A and E via hooks
            with HookManager(model, target_layer) as hm:
                with torch.no_grad():
                    logits = model(inputs)
                A = hm.activation.cpu().numpy().astype(np.float32)
                E_tensor = error_fn(logits, targets)
                E = E_tensor.detach().cpu().numpy().astype(np.float32)

            # Ghost → Adam correct → project
            g = form_ghost_vectors(A, E)

            if adam_v is not None:
                g = apply_adam_correction(g, adam_v)

            g_proj = g if P is None else project(g, P)

            batch_size = g_proj.shape[0]
            accumulated[offset: offset + batch_size] += lr * g_proj
            offset += batch_size

            # Rule 1: free memory
            del A, E, E_tensor, g, g_proj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("Checkpoint %d/%d done (lr=%.6f)", ckpt_idx + 1, len(checkpoints), lr)

    # Build and save FAISS index
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, index_filename)
    metadata_path = os.path.join(output_dir, metadata_filename)

    metadata_extra = {}
    if sample_metadata:
        metadata_extra["sample_id_to_rights_holder"] = {
            str(k): v for k, v in sample_metadata.items()
        }

    store = FAISSStore(index_type=index_type)
    store.build_and_save(
        accumulated,
        all_sample_ids,
        index_path,
        metadata_path,
        metadata_extra=metadata_extra if metadata_extra else None,
    )

    logger.info("Index saved to: %s", index_path)
    return index_path
