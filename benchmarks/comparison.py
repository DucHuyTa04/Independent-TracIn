"""Build 2-way comparison block: Ghost+FAISS vs Original TracIn (full-gradient)."""

from __future__ import annotations

from typing import Sequence

from benchmarks.benchmark_profiling import ProfileResult, file_size_mb
from benchmarks.metrics import spearman_correlation, top_k_overlap


def build_comparison(
    ghost_faiss_scores: dict[int, float],
    original_tracin_scores: dict[int, float],
    sample_ids: Sequence[int],
    ghost_profile: ProfileResult,
    original_tracin_profile: ProfileResult,
    n_train: int,
    total_params: int,
    ghost_vector_dim: int | None,
    faiss_index_path: str | None = None,
    k_pct: Sequence[float] = (1, 5, 10, 15),
    baseline_subset_n: int | None = None,
) -> dict:
    """Assemble ``comparison`` block for ``metrics.json``.

    Compares Ghost+FAISS (our pipeline) against Original TracIn
    (full-parameter, multi-checkpoint, textbook formula).
    """
    ids = list(sample_ids)
    n = len(ids)

    spearman = spearman_correlation(ghost_faiss_scores, original_tracin_scores, ids)

    top_k: dict = {}
    for pct in k_pct:
        k = max(1, round(pct / 100.0 * n)) if n else 0
        if k <= 0 or k > n:
            continue
        top_k[f"pct{pct}"] = {
            "k": k,
            "overlap": top_k_overlap(ghost_faiss_scores, original_tracin_scores, ids, k),
            "random_baseline": round(float(k / n), 4) if n else 0.0,
        }

    index_mb = file_size_mb(faiss_index_path) if faiss_index_path else None
    # Storage for full-gradient baseline scales with the number of samples scored (subset or full).
    theoretical_storage_mb = (
        n * total_params * 8 / (1024 * 1024) if n and total_params else None
    )

    out: dict = {
        "spearman_ghost_vs_original": spearman,
        "top_k_pct_overlap": top_k,
        "original_tracin": {
            "wall_time_s": round(float(original_tracin_profile.wall_time_s), 4),
            "peak_memory_mb": (
                round(float(original_tracin_profile.peak_memory_mb), 4)
                if original_tracin_profile.peak_memory_mb is not None
                else None
            ),
            "vector_dim_per_sample": int(total_params),
            "storage_mb_theoretical": (
                round(float(theoretical_storage_mb), 4)
                if theoretical_storage_mb is not None
                else None
            ),
        },
        "ghost_faiss": {
            "wall_time_s": round(float(ghost_profile.wall_time_s), 4),
            "peak_memory_mb": (
                round(float(ghost_profile.peak_memory_mb), 4)
                if ghost_profile.peak_memory_mb is not None
                else None
            ),
            "vector_dim_per_sample": ghost_vector_dim,
            "index_size_mb": round(float(index_mb), 4) if index_mb is not None else None,
        },
    }
    if baseline_subset_n is not None:
        out["baseline_subset_n"] = int(baseline_subset_n)
        out["n_train_full"] = int(n_train)
    return out
