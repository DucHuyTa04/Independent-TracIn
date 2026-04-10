"""Benchmark utilities: Ghost+FAISS, Original TracIn, metrics, dashboard plots."""

from .comparison import build_comparison
from .exact_tracin import compute_exact_tracin_scores, compute_true_tracin_ghost_scores
from .full_gradient_tracin import compute_full_gradient_tracin_scores
from .ghost_faiss import compute_ghost_faiss_scores
from .influence_variants import VARIANT_ORDER, compute_all_six_variants, model_ghost_coverage
from .metrics import summarize_all_variants, summarize_metrics

__all__ = [
    "build_comparison",
    "compute_exact_tracin_scores",
    "compute_true_tracin_ghost_scores",
    "compute_full_gradient_tracin_scores",
    "compute_ghost_faiss_scores",
    "compute_all_six_variants",
    "model_ghost_coverage",
    "VARIANT_ORDER",
    "summarize_metrics",
    "summarize_all_variants",
]
