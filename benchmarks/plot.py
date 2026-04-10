"""Matplotlib figures for benchmark reports — Ghost+FAISS vs Original TracIn dashboard."""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import spearman_correlation, top_k_overlap


def _ranks_descending(scores: dict[int, float], ids: Sequence[int]) -> np.ndarray:
    """1-based average ranks; rank 1 = highest score (most influential). Ties averaged."""
    x = np.array([scores[i] for i in ids], dtype=np.float64)
    n = x.size
    sorter = np.argsort(-x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[sorter[j + 1]] == x[sorter[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[sorter[i : j + 1]] = avg
        i = j + 1
    return ranks


def _rank_scatter_panel(
    ax: plt.Axes,
    scores_x: dict[int, float],
    scores_y: dict[int, float],
    ids: Sequence[int],
    title: str,
    xlabel: str,
    ylabel: str,
    top_k: int,
) -> None:
    rx = _ranks_descending(scores_x, ids)
    ry = _ranks_descending(scores_y, ids)
    n = len(ids)
    order_x = np.argsort(-np.array([scores_x[i] for i in ids], dtype=np.float64), kind="mergesort")
    top_set = set(int(order_x[j]) for j in range(min(top_k, n)))
    in_top = np.array([j in top_set for j in range(n)], dtype=bool)

    ax.scatter(
        rx[~in_top],
        ry[~in_top],
        alpha=0.55,
        s=28,
        c="0.45",
        label=f"Other (n={n - in_top.sum()})",
        edgecolors="white",
        linewidths=0.3,
    )
    ax.scatter(
        rx[in_top],
        ry[in_top],
        alpha=0.85,
        s=36,
        c="C0",
        label=f"Top-{top_k} by Ghost",
        edgecolors="navy",
        linewidths=0.4,
    )
    lim = (0.5, n + 0.5)
    ax.plot(lim, lim, "k--", linewidth=1.2, alpha=0.7, label="Perfect rank match")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    rho = spearman_correlation(scores_x, scores_y, ids)
    rho_s = f"{rho:.3f}" if np.isfinite(rho) else "nan"
    ax.text(
        0.03,
        0.97,
        f"Spearman ρ = {rho_s}\nn = {n}\nDiagonal = same rank",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="wheat", alpha=0.85),
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.35)


def plot_diagnostic_variants(
    model_name: str,
    spearman_by_variant: dict[str, float],
    output_dir: str,
    variant_order: Sequence[str] | None = None,
) -> str:
    """Grouped bar chart of Spearman ρ vs Original TracIn for variants A–E (diagnostic)."""
    os.makedirs(output_dir, exist_ok=True)
    order = list(variant_order) if variant_order is not None else sorted(spearman_by_variant.keys())
    labels = [v.replace("_", "\n") for v in order]
    vals = [float(spearman_by_variant.get(v, float("nan"))) for v in order]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(order))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(order)))
    bars = ax.bar(x, np.nan_to_num(vals, nan=0.0), color=colors, edgecolor="0.2", linewidth=0.6)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.85, label="ρ = 0")
    ax.axhline(0.3, color="green", linestyle="--", linewidth=1.0, alpha=0.7, label="ρ = 0.3 (target)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_ylabel("Spearman ρ vs Original TracIn")
    ax.set_title(f"{model_name}: diagnostic — all influence variants vs Original TracIn")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.35)
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.02 * np.sign(b.get_height() or 1),
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7,
            )
    fig.tight_layout()
    out_path = os.path.join(output_dir, "diagnostic_variants.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_benchmark(
    model_name: str,
    ghost_scores: dict[int, float],
    reference_scores: dict[int, float],
    output_dir: str,
    sample_ids: Sequence[int] | None = None,
    k_values: Sequence[int] | None = None,
    k_pct: Sequence[float] = (1, 5, 10, 15),
    rank_highlight_k: int = 10,
) -> list[str]:
    """Save a 2×2 dashboard: Ghost+FAISS vs Original TracIn. Returns list of PNG path(s)."""
    os.makedirs(output_dir, exist_ok=True)
    ids = list(sample_ids) if sample_ids is not None else sorted(ghost_scores.keys())
    n = len(ids)
    if n == 0:
        return []

    top_k_vis = min(rank_highlight_k, n)
    if k_values is not None:
        ks = [k for k in k_values if k <= n]
    else:
        ks = [max(1, round(pct / 100.0 * n)) for pct in k_pct]
        # de-duplicate while preserving order
        seen: set[int] = set()
        ks = [k for k in ks if k <= n and not (k in seen or seen.add(k))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"{model_name}: benchmark dashboard\n"
        "Ghost+FAISS (our pipeline) vs Original TracIn (full-gradient)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    # Panel A: Ghost vs Original TracIn ranks
    _rank_scatter_panel(
        axes[0, 0],
        ghost_scores,
        reference_scores,
        ids,
        "A. Rank agreement: Ghost+FAISS vs Original TracIn",
        "Rank by Ghost+FAISS (1 = highest influence)",
        "Rank by Original TracIn (1 = highest influence)",
        top_k_vis,
    )

    # Panel B: Top-k overlap Ghost vs Original TracIn
    ax_b = axes[0, 1]
    if ks:
        g_l = [top_k_overlap(ghost_scores, reference_scores, ids, k) for k in ks]
        ax_b.plot(ks, g_l, "s-", label="Ghost vs Original TracIn", linewidth=2, markersize=7, color="C0")
        random_baseline = [k / n for k in ks]
        ax_b.plot(
            ks,
            random_baseline,
            "k:",
            linewidth=1.5,
            alpha=0.75,
            label=f"Random overlap (k/n, n={n})",
        )
        k_last = ks[-1]
        ax_b.annotate(
            f"{g_l[-1]:.2f}",
            (k_last, g_l[-1]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    else:
        ax_b.text(0.5, 0.5, "No k values ≤ n.", ha="center", va="center", transform=ax_b.transAxes)
    ax_b.set_xlabel("k (top-k size; from % of n when k_pct mode)")
    ax_b.set_ylabel("Overlap fraction |top-k_Ghost ∩ top-k_Orig| / k")
    ax_b.set_title("B. Top-k overlap vs k")
    ax_b.set_ylim(-0.05, 1.08)
    ax_b.legend(loc="lower right", fontsize=9)
    ax_b.grid(True, alpha=0.35)
    ax_b.text(
        0.03,
        0.03,
        "Good: line well above random.\nOriginal TracIn uses full-parameter gradients.",
        transform=ax_b.transAxes,
        fontsize=8,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    # Panel C: Box plots Ghost + Original TracIn (z-scored)
    ax_c = axes[1, 0]
    g = np.array([ghost_scores[i] for i in ids], dtype=np.float64)
    l_arr = np.array([reference_scores[i] for i in ids], dtype=np.float64)

    def _z(a: np.ndarray) -> np.ndarray:
        m, s = float(np.mean(a)), float(np.std(a))
        return (a - m) / (s + 1e-12) if s > 1e-15 else np.zeros_like(a)

    data_z = [_z(g), _z(l_arr)]
    bp = ax_c.boxplot(
        data_z,
        labels=["Ghost+FAISS", "Original TracIn"],
        patch_artist=True,
        medianprops=dict(color="darkred", linewidth=2),
    )
    for patch, color in zip(bp["boxes"], ["C0", "C2"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax_c.set_ylabel("Per-method z-score (mean=0, std=1)")
    ax_c.set_title("C. Score spread (z-scored per method)")
    ax_c.grid(True, axis="y", alpha=0.35)
    ax_c.text(
        0.5,
        -0.2,
        "Units differ between methods; z-score compares spread only.",
        transform=ax_c.transAxes,
        ha="center",
        fontsize=8,
        style="italic",
    )

    # Panel D: Summary text
    ax_d = axes[1, 1]
    ax_d.axis("off")
    rho = spearman_correlation(ghost_scores, reference_scores, ids)
    rho_s = f"{rho:.4f}" if np.isfinite(rho) else "nan"
    lines = [
        "D. Summary",
        "",
        f"n training samples: {n}",
        f"Spearman ρ (Ghost vs Original TracIn): {rho_s}",
        "",
        "Top-k overlap (Ghost vs Original TracIn):",
    ]
    for k in ks:
        ov = top_k_overlap(ghost_scores, reference_scores, ids, k)
        lines.append(f"  k={k}: {ov:.4f}")
    lines.extend(
        [
            "",
            "Interpretation:",
            "  • Higher ρ / overlap → Ghost rankings",
            "    align more with full-gradient TracIn.",
            "  • Original TracIn = textbook formula.",
            "  • See docs/benchmark_guide.md for detail.",
        ]
    )
    ax_d.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax_d.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", alpha=0.95),
    )

    fig.tight_layout()
    out_path = os.path.join(output_dir, "benchmark_dashboard.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out_path]
