"""Mathematical engine for TracIn ghost dot product attribution.

Combines ghost vector formation, Adam optimizer correction, and
SJLT/dense random projection into a single module.

Mathematical foundations:
    - Ghost dot product: <g1, g2> = <A1, A2> * <E1, E2>
      where g = vec(E * A^T) is the ghost gradient vector.
    - Adam correction: g_corrected = g / (sqrt(v_t) + eps)
      aligns TracIn with Adam-trained models.
    - SJLT (Achlioptas 2003): sparse random projection preserving
      inner products via Johnson-Lindenstrauss lemma.
"""

from typing import Union

import numpy as np
import torch
from scipy import sparse


# ---------------------------------------------------------------------------
# Ghost vector formation
# ---------------------------------------------------------------------------


def form_ghost_vectors(A: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Form ghost gradient vectors via batch outer product.

    Math:
        g_i = vec(E_i * A_i^T) = E_i ⊗ A_i  (Kronecker product)

    The ghost dot product identity:
        <g_1, g_2> = <A_1, A_2> · <E_1, E_2>

    Args:
        A: (N, H) hidden activations (input to last linear layer).
        E: (N, C) error signals (dLoss/dLogits).

    Returns:
        (N, H*C) ghost vectors, float32.
    """
    N, H = A.shape
    C = E.shape[1]
    # (N, H, 1) * (N, 1, C) -> (N, H, C) -> (N, H*C)
    g = (A[:, :, np.newaxis] * E[:, np.newaxis, :]).reshape(N, H * C)
    return g.astype(np.float32)


# ---------------------------------------------------------------------------
# Adam second-moment correction
# ---------------------------------------------------------------------------


def apply_adam_correction(
    ghost_vectors: np.ndarray,
    adam_v: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply Adam second-moment correction to ghost vectors.

    Math:
        g_corrected = g / (sqrt(v_t) + eps)

    Important: correction is applied to the FULL ghost vector
    g = vec(E * A^T), not just to E. Adam's v_t is per-weight-element
    and does not factorize along E/A dimensions.

    Args:
        ghost_vectors: (N, H*C) raw ghost vectors.
        adam_v:        (H*C,) flattened second-moment estimate (exp_avg_sq).
        eps:           Adam epsilon for numerical stability.

    Returns:
        (N, H*C) corrected ghost vectors, float32.
    """
    scale = 1.0 / (np.sqrt(adam_v) + eps)
    return (ghost_vectors * scale[np.newaxis, :]).astype(np.float32)


def load_adam_second_moment(
    optimizer_state_path: str,
    param_key: Union[int, str],
) -> np.ndarray:
    """Load Adam's exp_avg_sq for a specific parameter from saved optimizer state.

    Args:
        optimizer_state_path: Path to torch.save'd optimizer.state_dict().
        param_key: Parameter index (int) or name in the optimizer state.
                   Standard PyTorch optimizers index params as 0, 1, 2, ...

    Returns:
        (H*C,) flattened second-moment array, float32.
    """
    state = torch.load(optimizer_state_path, map_location="cpu", weights_only=True)
    v = state["state"][param_key]["exp_avg_sq"]
    return v.numpy().flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Random projection (SJLT sparse + dense Gaussian)
# ---------------------------------------------------------------------------


def build_sjlt_matrix(
    dim_in: int,
    dim_out: int,
    seed: int = 42,
) -> sparse.csr_matrix:
    """Build a Sparse Johnson-Lindenstrauss projection matrix.

    Achlioptas (2003):
        P[i,j] = sqrt(3/K) * {+1 w.p. 1/6, 0 w.p. 2/3, -1 w.p. 1/6}

    ~1/3 non-zero entries. Preserves inner products: <Px, Py> ≈ <x, y>.

    Args:
        dim_in:  Original dimensionality (H * C).
        dim_out: Target dimensionality (K).
        seed:    RNG seed for reproducibility.

    Returns:
        (dim_out, dim_in) sparse CSR matrix, float32.
    """
    rng = np.random.RandomState(seed)
    scale = np.sqrt(3.0 / dim_out)
    vals = rng.choice([-1, 0, 1], size=(dim_out, dim_in), p=[1 / 6, 2 / 3, 1 / 6])
    P = sparse.csr_matrix(vals * scale, dtype=np.float32)
    return P


def build_dense_projection(
    dim_in: int,
    dim_out: int,
    seed: int = 42,
) -> np.ndarray:
    """Dense Gaussian random projection (fallback for small dims).

    R[i,j] ~ N(0, 1/sqrt(K)). Only use when dim_in * dim_out fits in memory.

    Args:
        dim_in:  Original dimensionality.
        dim_out: Target dimensionality.
        seed:    RNG seed for reproducibility.

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
    """Project ghost vectors using P (sparse SJLT or dense Gaussian).

    Args:
        vectors: (N, dim_in) ghost vectors to project.
        P:       (dim_out, dim_in) projection matrix (sparse or dense).

    Returns:
        (N, dim_out) projected vectors, float32.
    """
    if sparse.issparse(P):
        result = (P @ vectors.T).T
        if sparse.issparse(result):
            result = result.toarray()
        return result.astype(np.float32)
    else:
        return (vectors @ P.T).astype(np.float32)
