"""End-to-end mock pipeline test.

Validates the full math path without FAISS:
    1. Create a tiny nn.Linear(4, 2) model
    2. Generate 10 dummy training samples
    3. Extract ghost vectors via HookManager
    4. Apply Adam correction
    5. Project via SJLT
    6. Simulate query + inner product search
    7. Verify: scores >= 0 after clamp, attribution sums ~1.0

This avoids importing faiss (which may not be installed in CI)
by testing the mathematical pipeline directly.
"""

import numpy as np
import torch
import torch.nn as nn

from src.hooks_manager import HookManager
from src.math_utils import (
    apply_adam_correction,
    build_sjlt_matrix,
    form_ghost_vectors,
    project,
)


class TestMockPipeline:
    """End-to-end math pipeline with a tiny model."""

    def _make_model(self):
        torch.manual_seed(42)
        return nn.Linear(4, 2)

    def _error_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Classification error: softmax(logits) - one_hot(targets)."""
        probs = torch.softmax(logits, dim=-1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
        return probs - one_hot

    def test_full_pipeline_math(self):
        """Build ghost vectors for training data, query, compute scores."""
        model = self._make_model()
        model.eval()
        n_train = 10
        proj_dim = 4  # tiny projection

        # --- Indexing phase: extract ghost vectors for training data ---
        rng = np.random.RandomState(0)
        train_inputs = torch.randn(n_train, 4)
        train_targets = torch.randint(0, 2, (n_train,))

        with HookManager(model, model) as hm:
            with torch.no_grad():
                logits = model(train_inputs)
            A_train = hm.activation.cpu().numpy()
            E_train = self._error_fn(logits, train_targets).detach().cpu().numpy()

        g_train = form_ghost_vectors(A_train, E_train)
        assert g_train.shape == (n_train, 10)  # (4+1) * 2 with bias augmentation

        # Fake Adam v (uniform for simplicity)
        v = np.ones(10, dtype=np.float32)
        g_train = apply_adam_correction(g_train, v)

        # Project
        P = build_sjlt_matrix(10, proj_dim, seed=42)
        g_train_proj = project(g_train, P)
        assert g_train_proj.shape == (n_train, proj_dim)

        # --- Query phase ---
        query_input = torch.randn(1, 4)
        query_target = torch.tensor([0])

        with HookManager(model, model) as hm:
            with torch.no_grad():
                q_logits = model(query_input)
            A_q = hm.activation.cpu().numpy()
            E_q = self._error_fn(q_logits, query_target).detach().cpu().numpy()

        g_query = form_ghost_vectors(A_q, E_q)
        g_query = apply_adam_correction(g_query, v)
        g_query_proj = project(g_query, P)
        assert g_query_proj.shape == (1, proj_dim)

        # --- Inner product search (what FAISS does) ---
        scores = g_query_proj @ g_train_proj.T  # (1, n_train)
        scores = scores[0]

        # Clamp negatives, normalize
        scores_pos = np.maximum(0, scores)
        total = scores_pos.sum()

        if total > 0:
            attribution = scores_pos / total
            np.testing.assert_allclose(attribution.sum(), 1.0, atol=1e-6)
            assert all(a >= 0 for a in attribution)
        # If total == 0, all scores were negative -> no attribution (valid edge case)

    def test_ghost_vectors_reproducible(self):
        """Same input -> same ghost vectors."""
        model = self._make_model()
        model.eval()
        x = torch.randn(3, 4)
        targets = torch.tensor([0, 1, 0])

        def extract():
            with HookManager(model, model) as hm:
                with torch.no_grad():
                    logits = model(x)
                A = hm.activation.cpu().numpy()
                E = self._error_fn(logits, targets).detach().cpu().numpy()
            return form_ghost_vectors(A, E)

        g1 = extract()
        g2 = extract()
        np.testing.assert_array_equal(g1, g2)

    def test_different_queries_different_scores(self):
        """Different query inputs should give different attribution scores."""
        model = self._make_model()
        model.eval()

        # Training data
        train_x = torch.randn(5, 4)
        train_y = torch.randint(0, 2, (5,))

        with HookManager(model, model) as hm:
            with torch.no_grad():
                logits = model(train_x)
            A = hm.activation.cpu().numpy()
            E = self._error_fn(logits, train_y).detach().cpu().numpy()

        g_train = form_ghost_vectors(A, E)

        # Two different queries
        query1 = torch.randn(1, 4)
        query2 = torch.randn(1, 4) + 5.0  # shift to ensure difference

        scores = []
        for q in [query1, query2]:
            with HookManager(model, model) as hm:
                with torch.no_grad():
                    logits = model(q)
                Aq = hm.activation.cpu().numpy()
                Eq = self._error_fn(logits, torch.tensor([0])).detach().cpu().numpy()
            gq = form_ghost_vectors(Aq, Eq)
            s = gq @ g_train.T
            scores.append(s[0])

        # Scores should differ
        assert not np.allclose(scores[0], scores[1])

    def test_projection_preserves_ranking(self):
        """Top-1 by inner product should be same before and after projection."""
        model = self._make_model()
        model.eval()

        torch.manual_seed(99)
        train_x = torch.randn(8, 4)
        train_y = torch.randint(0, 2, (8,))

        with HookManager(model, model) as hm:
            with torch.no_grad():
                logits = model(train_x)
            A = hm.activation.cpu().numpy()
            E = self._error_fn(logits, train_y).detach().cpu().numpy()

        g_train = form_ghost_vectors(A, E)

        query_x = torch.randn(1, 4)
        with HookManager(model, model) as hm:
            with torch.no_grad():
                logits = model(query_x)
            Aq = hm.activation.cpu().numpy()
            Eq = self._error_fn(logits, torch.tensor([1])).detach().cpu().numpy()

        g_q = form_ghost_vectors(Aq, Eq)

        # Full-dim scores
        scores_full = (g_q @ g_train.T)[0]
        top1_full = int(np.argmax(scores_full))

        # Projected scores (dim 6 from 10 -- mild compression)
        ghost_dim = int(g_train.shape[1])
        P = build_sjlt_matrix(ghost_dim, 6, seed=42)
        scores_proj = (project(g_q, P) @ project(g_train, P).T)[0]
        top1_proj = int(np.argmax(scores_proj))

        # With such a small dim reduction, ranking should be preserved
        assert top1_proj == top1_full
