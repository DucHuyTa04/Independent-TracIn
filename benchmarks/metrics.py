"""Metrics comparing influence score vectors."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _rank_average(x: np.ndarray) -> np.ndarray:
    """Average ranks for ties (1-based positions before averaging)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    sorter = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[sorter[j + 1]] == x[sorter[i]]:
            j += 1
        # ranks i..j (0-based positions) -> average rank (1-based)
        avg = (i + j) / 2.0 + 1.0
        ranks[sorter[i : j + 1]] = avg
        i = j + 1
    return ranks


def _aligned_arrays(
    scores_a: dict[int, float],
    scores_b: dict[int, float],
    sample_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([scores_a[i] for i in sample_ids], dtype=np.float64)
    b = np.array([scores_b[i] for i in sample_ids], dtype=np.float64)
    return a, b


def spearman_correlation(
    scores_a: dict[int, float],
    scores_b: dict[int, float],
    sample_ids: Sequence[int],
) -> float:
    """Spearman rank correlation between two score dicts over ``sample_ids``."""
    a, b = _aligned_arrays(scores_a, scores_b, sample_ids)
    if len(a) < 2:
        return float("nan")
    ra = _rank_average(a)
    rb = _rank_average(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return rho if not np.isnan(rho) else float("nan")


def top_k_overlap(
    scores_a: dict[int, float],
    scores_b: dict[int, float],
    sample_ids: Sequence[int],
    k: int,
) -> float:
    """Fraction of overlap between top-``k`` sample ids by each score (descending)."""
    k = min(k, len(sample_ids))
    if k <= 0:
        return float("nan")

    def topk_set(scores: dict[int, float]) -> set[int]:
        ranked = sorted(sample_ids, key=lambda i: scores[i], reverse=True)
        return set(ranked[:k])

    sa = topk_set(scores_a)
    sb = topk_set(scores_b)
    return len(sa & sb) / k


def summarize_metrics(
    ghost: dict[int, float],
    reference: dict[int, float],
    sample_ids: Iterable[int],
    k_pct: Sequence[float] = (1, 5, 10, 15),
    k_values: Sequence[int] | None = None,
) -> dict:
    """Spearman and top-k overlap between Ghost+FAISS and a reference scorer.

    Args:
        ghost: Predicted influence scores per sample id.
        reference: Reference scores (e.g. Original TracIn full-gradient).
        sample_ids: Iterable of sample ids (aligned order for arrays).
        k_pct: Percentages of n_train for top-k (e.g. 5 -> top 5% of n).
        k_values: If provided, also report absolute-k overlaps under ``top_k_overlap_abs``.
    """
    ids = list(sample_ids)
    n = len(ids)
    out: dict = {
        "n_train": n,
        "spearman": {
            "ghost_vs_original_tracin": spearman_correlation(ghost, reference, ids),
        },
        "top_k_pct_overlap": {},
    }
    for pct in k_pct:
        k = max(1, round(pct / 100.0 * n)) if n else 0
        if k <= 0 or k > n:
            continue
        random_baseline = k / n if n else 0.0
        out["top_k_pct_overlap"][f"pct{pct}"] = {
            "k": k,
            "ghost_vs_original_tracin": top_k_overlap(ghost, reference, ids, k),
            "random_baseline": round(float(random_baseline), 4),
        }
    if k_values:
        out["top_k_overlap_abs"] = {}
        for k in k_values:
            if k > n or k <= 0:
                continue
            out["top_k_overlap_abs"][f"k{k}"] = {
                "ghost_vs_original_tracin": top_k_overlap(ghost, reference, ids, k),
            }
    return out


def summarize_all_variants(
    variants: dict[str, dict[int, float]],
    reference: dict[int, float],
    sample_ids: Iterable[int],
    k_pct: Sequence[float] = (1, 5, 10, 15),
) -> dict:
    """Spearman and top-k% overlap for every variant score dict vs reference."""
    ids = list(sample_ids)
    n = len(ids)
    results: dict = {}
    for name, scores in variants.items():
        entry: dict = {
            "spearman_vs_reference": spearman_correlation(scores, reference, ids),
            "top_k_pct": {},
        }
        for pct in k_pct:
            k = max(1, round(pct / 100.0 * n)) if n else 0
            if k <= 0 or k > n:
                continue
            entry["top_k_pct"][f"pct{pct}"] = {
                "k": k,
                "overlap": top_k_overlap(scores, reference, ids, k),
                "random_baseline": round(float(k / n), 4) if n else 0.0,
            }
        results[name] = entry
    return results
