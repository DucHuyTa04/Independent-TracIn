"""Mathematical engine for TracIn ghost dot product attribution.

Ghost vector formation, Adam optimizer correction, and SJLT/dense random projection.
"""

from typing import Optional, Union

import numpy as np
import torch
from scipy import sparse


# ---------------------------------------------------------------------------
# Ghost vector formation
# ---------------------------------------------------------------------------


def form_ghost_vectors(A: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Form ghost vectors via batch outer product: g_i = vec(E_i ⊗ A_i).

    Args:
        A: (N, H) activations.
        E: (N, C) error signals.

    Returns:
        (N, H*C) ghost vectors, float32.
    """
    N, H = A.shape
    C = E.shape[1]
    # (N, H, 1) * (N, 1, C) -> (N, H, C) -> (N, H*C)
    g = (A[:, :, np.newaxis] * E[:, np.newaxis, :]).reshape(N, H * C)
    return g.astype(np.float32)


def form_multi_layer_ghost_vectors(
    activations: list[np.ndarray],
    errors: list[np.ndarray],
) -> np.ndarray:
    """Concatenate ghost vectors from multiple layers (each A, E pair as in ``form_ghost_vectors``)."""
    if len(activations) != len(errors):
        raise ValueError("activations and errors must have the same length")
    parts = [
        form_ghost_vectors(
            np.asarray(A, dtype=np.float32),
            np.asarray(E, dtype=np.float32),
        )
        for A, E in zip(activations, errors)
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Adam second-moment correction
# ---------------------------------------------------------------------------


def apply_adam_correction(
    ghost_vectors: np.ndarray,
    adam_v: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply Adam second-moment correction: g / (sqrt(v_t) + eps).

    Applied to the full ghost vector — does not factorize along E/A dimensions.

    Args:
        ghost_vectors: (N, H*C) raw ghost vectors.
        adam_v:        (H*C,) flattened second-moment estimate.
        eps:           Adam epsilon.

    Returns:
        (N, H*C) corrected ghost vectors, float32.
    """
    av = np.asarray(adam_v, dtype=np.float32)
    scale = np.float32(1.0) / (np.sqrt(av) + np.float32(eps))
    return (ghost_vectors.astype(np.float32) * scale[np.newaxis, :]).astype(np.float32)


def load_adam_second_moment_matrix_ghost_layout(
    optimizer_state_path: str,
    param_key: Union[int, str],
    weight_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Load Adam ``exp_avg_sq`` as a (H, C) matrix in ghost layout (float32).

    Same layout as ``form_ghost_vectors`` row-major flatten: index ``h * C + c``.
    """
    raw = torch.load(optimizer_state_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "optimizer_state_dict" in raw:
        state = raw["optimizer_state_dict"]
    else:
        state = raw
    v = state["state"][param_key]["exp_avg_sq"]
    v_np = v.detach().cpu().numpy()

    if v_np.ndim == 4:
        c_out = int(v_np.shape[0])
        h_in = int(np.prod(v_np.shape[1:]))
        v_np = v_np.reshape(c_out, h_in)

    if v_np.ndim == 2:
        c_out, h_in = int(v_np.shape[0]), int(v_np.shape[1])
        if weight_shape is not None:
            wc, wh = int(weight_shape[0]), int(weight_shape[1])
            if (wc, wh) != (c_out, h_in):
                raise ValueError(
                    f"weight_shape {weight_shape} does not match exp_avg_sq.shape "
                    f"{v_np.shape}"
                )
        return v_np.reshape(c_out, h_in).T.astype(np.float32)

    if weight_shape is None:
        raise ValueError(
            "1D exp_avg_sq requires weight_shape for matrix ghost layout; "
            "use load_adam_second_moment for flattened layout.",
        )

    c_out, h_in = int(weight_shape[0]), int(weight_shape[1])
    flat = int(v_np.size)
    if flat != c_out * h_in:
        raise ValueError(
            f"exp_avg_sq size {flat} != product(weight_shape) {c_out * h_in}"
        )
    return v_np.reshape(c_out, h_in).T.astype(np.float32)


def load_adam_inverse_sqrt_scale_matrix_ghost_layout(
    optimizer_state_path: str,
    param_key: Union[int, str],
    weight_shape: Optional[tuple[int, int]] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-element ``1 / (sqrt(exp_avg_sq) + eps)`` as (H, C), ghost layout."""
    m = load_adam_second_moment_matrix_ghost_layout(
            optimizer_state_path, param_key, weight_shape=weight_shape,
        )
    return (1.0 / (np.sqrt(m) + eps)).astype(np.float32)


def load_adam_second_moment(
    optimizer_state_path: str,
    param_key: Union[int, str],
    weight_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Load Adam exp_avg_sq aligned to ghost vector layout (H, C) row-major.

    PyTorch Linear weight is (C, H); ghost layout is (H, C) → transpose before flatten.

    Args:
        optimizer_state_path: Path to saved optimizer state dict.
        param_key: Parameter index or name in optimizer state.
        weight_shape: (out_features, in_features); required if exp_avg_sq is 1D.

    Returns:
        (H*C,) flattened second-moment in ghost layout, float32.
    """
    raw = torch.load(optimizer_state_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "optimizer_state_dict" in raw:
        state = raw["optimizer_state_dict"]
    else:
        state = raw
    v = state["state"][param_key]["exp_avg_sq"]
    v_np = v.detach().cpu().numpy()

    if v_np.ndim == 4:
        c_out = int(v_np.shape[0])
        h_in = int(np.prod(v_np.shape[1:]))
        v_np = v_np.reshape(c_out, h_in)

    if v_np.ndim == 2:
        c_out, h_in = int(v_np.shape[0]), int(v_np.shape[1])
        if weight_shape is not None:
            wc, wh = int(weight_shape[0]), int(weight_shape[1])
            if (wc, wh) != (c_out, h_in):
                raise ValueError(
                    f"weight_shape {weight_shape} does not match exp_avg_sq.shape "
                    f"{v_np.shape}"
                )
        return v_np.reshape(c_out, h_in).T.flatten().astype(np.float32)

    if weight_shape is None:
        return v_np.flatten().astype(np.float32)

    c_out, h_in = int(weight_shape[0]), int(weight_shape[1])
    flat = int(v_np.size)
    if flat != c_out * h_in:
        raise ValueError(
            f"exp_avg_sq size {flat} != product(weight_shape) {c_out * h_in}"
        )
    return v_np.reshape(c_out, h_in).T.flatten().astype(np.float32)


def load_adam_second_moment_with_bias(
    optimizer_state_path: str,
    weight_param_key: Union[int, str],
    bias_param_key: Optional[Union[int, str]],
    weight_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Load Adam exp_avg_sq for weight + bias in ghost layout.

    When the hook manager appends a column of ones for bias augmentation,
    ghost vectors have dimension ``(H+1)*C`` instead of ``H*C``.  The
    trailing ``C`` elements correspond to the bias gradient, so we need
    to append the bias second-moment to the weight second-moment.

    Args:
        optimizer_state_path: Path to optimizer state dict.
        weight_param_key: Optimizer state key for the layer's weight.
        bias_param_key: Optimizer state key for the layer's bias,
            or ``None`` when the layer has no bias.
        weight_shape: ``(out_features, in_features)`` for the weight.

    Returns:
        Flattened second-moment in ghost layout, float32.
        Length ``H*C`` if no bias, ``(H+1)*C`` with bias.
    """
    v_weight = load_adam_second_moment(
        optimizer_state_path, weight_param_key, weight_shape=weight_shape,
    )
    if bias_param_key is None:
        return v_weight
    raw = torch.load(optimizer_state_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "optimizer_state_dict" in raw:
        state = raw["optimizer_state_dict"]
    else:
        state = raw
    v_bias = state["state"][bias_param_key]["exp_avg_sq"]
    v_bias_np = v_bias.detach().cpu().numpy().flatten().astype(np.float32)
    return np.concatenate([v_weight, v_bias_np])


def concatenate_adam_second_moments(
    optimizer_state_path: str,
    param_keys: list[Union[int, str]],
    weight_shapes: list[Optional[tuple[int, int]]],
    bias_param_keys: Optional[list[Optional[Union[int, str]]]] = None,
) -> np.ndarray:
    """Load and concatenate ``exp_avg_sq`` vectors (ghost layout) for multiple weight tensors."""
    if len(param_keys) != len(weight_shapes):
        raise ValueError("param_keys and weight_shapes length must match")
    bkeys = bias_param_keys or [None] * len(param_keys)
    parts = [
        load_adam_second_moment_with_bias(optimizer_state_path, k, bk, weight_shape=ws)
        for k, bk, ws in zip(param_keys, bkeys, weight_shapes)
    ]
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# Random projection (SJLT sparse + dense Gaussian)
# ---------------------------------------------------------------------------


def build_sjlt_matrix(
    dim_in: int,
    dim_out: int,
    seed: int = 42,
) -> sparse.csr_matrix:
    """Sparse Johnson-Lindenstrauss projection matrix (Achlioptas 2003).

    P[i,j] ∈ {+1, 0, -1} * sqrt(3/K) with probabilities {1/6, 2/3, 1/6}.

    Args:
        dim_in:  Original dimensionality.
        dim_out: Target dimensionality.
        seed:    RNG seed.

    Returns:
        (dim_out, dim_in) sparse CSR matrix, float32.
    """
    rng = np.random.RandomState(seed)
    scale = np.float32(np.sqrt(3.0 / dim_out))
    vals = rng.choice([-1, 0, 1], size=(dim_out, dim_in), p=[1 / 6, 2 / 3, 1 / 6])
    rows, cols = np.nonzero(vals)
    data = (vals[rows, cols].astype(np.float32) * scale).ravel()
    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(dim_out, dim_in),
        dtype=np.float32,
    )


def build_dense_projection(
    dim_in: int,
    dim_out: int,
    seed: int = 42,
) -> np.ndarray:
    """Dense Gaussian random projection: R[i,j] ~ N(0, 1/sqrt(K)).

    Returns:
        (dim_out, dim_in) dense projection matrix, float32.
    """
    rng = np.random.RandomState(seed)
    R = rng.randn(dim_out, dim_in).astype(np.float32)
    R /= np.sqrt(dim_out)
    return R


def project(
    vectors: np.ndarray,
    P: Union[sparse.spmatrix, np.ndarray],
) -> np.ndarray:
    """Project ghost vectors: (N, dim_in) @ P^T → (N, dim_out), float32."""
    if sparse.issparse(P):
        result = (P @ vectors.T).T
        if sparse.issparse(result):
            result = result.toarray()
        return result.astype(np.float32)
    else:
        return (vectors @ P.T).astype(np.float32)
