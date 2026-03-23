"""Tests for src/math_utils.py.

Validates:
    - Ghost vector formation: known A, E -> expected g = vec(E * A^T)
    - Adam correction: g / (sqrt(v) + eps) identity
    - SJLT sparsity (~2/3 zeros) and shape
    - Dense projection shape and scale
    - project() with sparse and dense matrices
    - Inner product preservation under projection (approximate)
"""

import numpy as np
import pytest
from scipy import sparse

from src.math_utils import (
    apply_adam_correction,
    build_dense_projection,
    build_sjlt_matrix,
    form_ghost_vectors,
    project,
)


# ---- Ghost vector formation ----


class TestFormGhostVectors:
    def test_known_outer_product(self):
        """A=[1,0], E=[0,1] -> g = vec(E * A^T) = vec([[0],[1]]) = [0,1]."""
        A = np.array([[1.0, 0.0]])  # (1, 2)
        E = np.array([[0.0, 1.0]])  # (1, 2)
        g = form_ghost_vectors(A, E)
        # g_i = A_i[:, None] * E_i[None, :] flattened
        # A[:,newaxis] = [[1],[0]], E[newaxis,:] = [[0,1]]
        # outer = [[0,1],[0,0]] -> flatten = [0,1,0,0]
        assert g.shape == (1, 4)
        np.testing.assert_allclose(g[0], [0.0, 1.0, 0.0, 0.0])

    def test_identity_outer_product(self):
        """A=[1,0], E=[1,0] -> g = [1,0,0,0]."""
        A = np.array([[1.0, 0.0]])
        E = np.array([[1.0, 0.0]])
        g = form_ghost_vectors(A, E)
        np.testing.assert_allclose(g[0], [1.0, 0.0, 0.0, 0.0])

    def test_batch_shape(self):
        """Multiple samples produce correct batch dimension."""
        A = np.random.randn(5, 8).astype(np.float32)
        E = np.random.randn(5, 3).astype(np.float32)
        g = form_ghost_vectors(A, E)
        assert g.shape == (5, 24)
        assert g.dtype == np.float32

    def test_ghost_dot_product_identity(self):
        """<g1, g2> == <A1, A2> * <E1, E2> (ghost dot product theorem)."""
        rng = np.random.RandomState(0)
        A = rng.randn(2, 10).astype(np.float32)
        E = rng.randn(2, 4).astype(np.float32)
        g = form_ghost_vectors(A, E)

        ghost_ip = np.dot(g[0], g[1])
        factored_ip = np.dot(A[0], A[1]) * np.dot(E[0], E[1])
        np.testing.assert_allclose(ghost_ip, factored_ip, rtol=1e-5)


# ---- Adam correction ----


class TestAdamCorrection:
    def test_uniform_v(self):
        """Uniform v = constant -> scale is uniform."""
        g = np.array([[2.0, 4.0, 6.0]])
        v = np.array([1.0, 1.0, 1.0])
        eps = 1e-8
        corrected = apply_adam_correction(g, v, eps)
        expected = g / (np.sqrt(v) + eps)
        np.testing.assert_allclose(corrected, expected, rtol=1e-6)

    def test_varying_v(self):
        """Non-uniform v scales each element differently."""
        g = np.array([[1.0, 1.0]])
        v = np.array([4.0, 16.0])
        eps = 0.0  # zero eps for cleaner math
        corrected = apply_adam_correction(g, v, eps)
        # g / sqrt(v) = [1/2, 1/4]
        np.testing.assert_allclose(corrected[0], [0.5, 0.25])

    def test_batch_correction(self):
        """Batch of samples corrected independently."""
        g = np.ones((3, 4), dtype=np.float32)
        v = np.ones(4, dtype=np.float32) * 4.0
        corrected = apply_adam_correction(g, v)
        assert corrected.shape == (3, 4)
        assert corrected.dtype == np.float32


# ---- SJLT projection ----


class TestBuildSJLTMatrix:
    def test_shape(self):
        P = build_sjlt_matrix(100, 20, seed=42)
        assert P.shape == (20, 100)
        assert sparse.issparse(P)

    def test_sparsity(self):
        """~2/3 of entries should be zero (Achlioptas distribution)."""
        P = build_sjlt_matrix(1000, 500, seed=0)
        nnz_frac = P.nnz / (1000 * 500)
        # Should be around 1/3 non-zero, allow ±5% tolerance
        assert 0.25 < nnz_frac < 0.40, f"nnz fraction {nnz_frac} outside expected range"

    def test_deterministic(self):
        """Same seed produces same matrix."""
        P1 = build_sjlt_matrix(50, 10, seed=99)
        P2 = build_sjlt_matrix(50, 10, seed=99)
        np.testing.assert_array_equal(P1.toarray(), P2.toarray())

    def test_values_scale(self):
        """Non-zero entries should be ±sqrt(3/K)."""
        K = 20
        P = build_sjlt_matrix(50, K, seed=42)
        expected_scale = np.sqrt(3.0 / K)
        dense = P.toarray()
        nonzero = dense[dense != 0]
        unique_abs = np.unique(np.abs(nonzero))
        assert len(unique_abs) == 1
        np.testing.assert_allclose(unique_abs[0], expected_scale, rtol=1e-6)


# ---- Dense projection ----


class TestBuildDenseProjection:
    def test_shape(self):
        R = build_dense_projection(100, 20, seed=42)
        assert R.shape == (20, 100)
        assert R.dtype == np.float32

    def test_deterministic(self):
        R1 = build_dense_projection(50, 10, seed=99)
        R2 = build_dense_projection(50, 10, seed=99)
        np.testing.assert_array_equal(R1, R2)


# ---- project() ----


class TestProject:
    def test_sparse_project_shape(self):
        P = build_sjlt_matrix(20, 5, seed=0)
        x = np.random.randn(3, 20).astype(np.float32)
        out = project(x, P)
        assert out.shape == (3, 5)
        assert out.dtype == np.float32

    def test_dense_project_shape(self):
        R = build_dense_projection(20, 5, seed=0)
        x = np.random.randn(3, 20).astype(np.float32)
        out = project(x, R)
        assert out.shape == (3, 5)
        assert out.dtype == np.float32

    def test_inner_product_preservation(self):
        """JL guarantee: projected norms should approximate original norms.

        ||Px|| ≈ ||x|| for random projection P (Johnson-Lindenstrauss).
        We test on many vectors and check the average relative error is small.
        """
        rng = np.random.RandomState(123)
        dim_in = 500
        dim_out = 200
        n_vectors = 50

        P = build_sjlt_matrix(dim_in, dim_out, seed=42)
        errors = []
        for _ in range(n_vectors):
            x = rng.randn(1, dim_in).astype(np.float32)
            xp = project(x, P)
            orig_norm = float(np.linalg.norm(x))
            proj_norm = float(np.linalg.norm(xp))
            if orig_norm > 1e-6:
                errors.append(abs(proj_norm - orig_norm) / orig_norm)

        avg_error = np.mean(errors)
        assert avg_error < 0.15, f"Average relative norm error {avg_error:.3f} too high"
