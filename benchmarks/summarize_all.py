"""Aggregate benchmark metrics across models and plot comparison.

Writes ``summary.json`` and a single unified ``comparison_cross_model.png``:
Spearman rho, top-k heatmap, throughput, speedup, peak memory, memory ratio,
and summary table. Models are ordered small → medium → large (fixed tuple order).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Tier definitions  (keep in sync with run_all.py imports)
# ---------------------------------------------------------------------------
SMALL_MODELS: tuple[str, ...] = (
    "synth_regression",
    "linear_logistic",
    "mnist",
    "mnist_autoencoder",
    "multi_task",
)

# Medium: conv / recurrent / attention-heavy; ghost coverage often partial.
MEDIUM_MODELS: tuple[str, ...] = (
    "cifar10_cnn",
    "resnet_cifar100",
    "transformer_lm",
    "vae_fashion",
    "vit_cifar10",
    "encoder_transformer",
    "mlp_mixer_cifar10",
    "gru_lm",
    "unet_tiny",
)

# Large: ResNet50-scale (~20-25M params); stress-test ghost at production size.
LARGE_MODELS: tuple[str, ...] = (
    "resnet50_cifar100",
    "transformer_lm_large",
    "vit_large_cifar10",
)

DEFAULT_MODELS: tuple[str, ...] = SMALL_MODELS + MEDIUM_MODELS + LARGE_MODELS


def _parse_k_keys(top_k_overlap: dict) -> list[int]:
    """Extract sorted k integers from keys like 'k5', 'k10'."""
    ks: list[int] = []
    for key in top_k_overlap:
        m = re.match(r"^k(\d+)$", str(key))
        if m:
            ks.append(int(m.group(1)))
    return sorted(set(ks))


def _parse_pct_keys(top_k_pct_overlap: dict) -> list[int]:
    """Extract sorted percentage integers from keys like 'pct1', 'pct5'."""
    pcts: list[int] = []
    for key in top_k_pct_overlap:
        m = re.match(r"^pct(\d+)$", str(key))
        if m:
            pcts.append(int(m.group(1)))
    return sorted(set(pcts))


def _tier_tag(model_name: str) -> str:
    if model_name in SMALL_MODELS:
        return "small"
    if model_name in MEDIUM_MODELS:
        return "medium"
    if model_name in LARGE_MODELS:
        return "large"
    return "other"


def _tier_bar_color(model_name: str) -> str:
    return {"small": "#4E79A7", "medium": "#F28E2B", "large": "#E15759"}.get(
        _tier_tag(model_name), "#BAB0AC",
    )


def _has_comparison_block(data: dict) -> bool:
    cmp = data.get("comparison")
    if not isinstance(cmp, dict):
        return False
    return cmp.get("spearman_ghost_vs_original") is not None


def _canonical_ordered_labels(
    model_list: list[str],
    summary: dict[str, dict | None],
) -> list[str]:
    """Models that have metrics and a comparison block, in DEFAULT_MODELS order."""
    out: list[str] = []
    for m in DEFAULT_MODELS:
        if m not in model_list:
            continue
        data = summary.get(m)
        if data is None or not _has_comparison_block(data):
            continue
        out.append(m)
    return out


def _tier_separator_ys(labels: list[str]) -> list[float]:
    """Y positions (between bar centers) for horizontal tier dividers."""
    seps: list[float] = []
    prev: str | None = None
    for i, name in enumerate(labels):
        t = _tier_tag(name)
        if prev is not None and t != prev:
            seps.append(i - 0.5)
        prev = t
    return seps


def _add_tier_lines(ax: plt.Axes, n: int, seps: list[float]) -> None:
    for sy in seps:
        ax.axhline(sy, color="#cccccc", linewidth=1.0, linestyle="-", zorder=0)


def summarize(
    output_dir: str | Path,
    models: list[str] | None = None,
) -> None:
    """Write summary.json and comparison_cross_model.png under ``output_dir``."""
    out_base = Path(output_dir)
    model_list = models if models is not None else list(DEFAULT_MODELS)

    all_data: dict[str, dict | None] = {}
    for name in model_list:
        p = out_base / name / "metrics.json"
        if not p.is_file():
            all_data[name] = None
            continue
        with open(p, "r", encoding="utf-8") as f:
            all_data[name] = json.load(f)

    _summarize_unified(all_data, model_list, out_base)


def _summarize_unified(
    summary: dict[str, dict | None],
    model_list: list[str],
    out_base: Path,
) -> None:
    out_base.mkdir(parents=True, exist_ok=True)
    summary_path = out_base / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}  [All models]")

    ordered = _canonical_ordered_labels(model_list, summary)
    if not ordered:
        print("No models with comparison metrics; skip plots.")
        return

    ghost_vs_orig: list[float] = []
    per_model_tko: list[dict[str, dict]] = []
    for name in ordered:
        data = summary[name]
        assert data is not None
        sp = data.get("spearman", {})
        ghost_vs_orig.append(float(sp.get("ghost_vs_original_tracin", float("nan"))))
        tko = data.get("top_k_pct_overlap") or data.get("top_k_overlap") or {}
        per_model_tko.append(tko)

    all_pct: set[int] = set()
    all_k_abs: set[int] = set()
    for tko in per_model_tko:
        all_pct.update(_parse_pct_keys(tko))
        all_k_abs.update(_parse_k_keys(tko))
    use_pct = bool(all_pct)
    ks_sorted = sorted(all_pct) if use_pct else sorted(all_k_abs)

    n = len(ordered)
    heat = np.full((n, len(ks_sorted)), np.nan, dtype=np.float64)
    for i, tko in enumerate(per_model_tko):
        for j, keyval in enumerate(ks_sorted):
            if use_pct:
                entry = tko.get(f"pct{keyval}")
                if isinstance(entry, dict):
                    v = entry.get("ghost_vs_original_tracin")
                    if v is not None:
                        heat[i, j] = float(v)
            else:
                entry = tko.get(f"k{keyval}")
                if isinstance(entry, dict):
                    v = entry.get("ghost_vs_original_tracin")
                    if v is not None:
                        heat[i, j] = float(v)

    # Timing / memory / table series
    sp_ghost: list[float] = []
    t_ghost: list[float] = []
    t_orig: list[float] = []
    m_ghost: list[float] = []
    m_orig: list[float] = []
    n_trains: list[int] = []
    baseline_ns: list[int] = []
    ghost_dims: list[int] = []
    coverages: list[float] = []
    total_params: list[int] = []

    for name in ordered:
        data = summary[name]
        assert data is not None
        cmp = data["comparison"]
        gh = cmp.get("ghost_faiss") or {}
        ot = cmp.get("original_tracin") or {}
        sg = cmp.get("spearman_ghost_vs_original")
        sp_ghost.append(float(sg) if sg is not None and np.isfinite(sg) else 0.0)
        t_ghost.append(float(gh.get("wall_time_s") or 1e-6))
        t_orig.append(float(ot.get("wall_time_s") or 1e-6))
        m_ghost.append(float(gh.get("peak_memory_mb") or 0.0))
        m_orig.append(float(ot.get("peak_memory_mb") or 0.0))
        nt = int(data.get("n_train", 0))
        n_trains.append(nt)
        bn = int(cmp.get("baseline_subset_n") or nt)
        baseline_ns.append(bn)
        mi = data.get("model_info") or {}
        gd = int(mi.get("ghost_dim", 0))
        tp = int(mi.get("total_params", 0))
        ghost_dims.append(gd)
        total_params.append(tp)
        cov_pct = mi.get("ghost_coverage_pct")
        if cov_pct is not None:
            coverages.append(float(cov_pct))
        else:
            coverages.append(round(100.0 * gd / tp, 4) if tp else 0.0)

    ms_ghost = [1000 * t / nt if nt else 0 for t, nt in zip(t_ghost, n_trains)]
    ms_orig = [1000 * t / bn if bn else 0 for t, bn in zip(t_orig, baseline_ns)]
    speedups = []
    for i in range(n):
        est_full = t_orig[i] * (n_trains[i] / baseline_ns[i]) if baseline_ns[i] else t_orig[i]
        speedups.append(est_full / t_ghost[i] if t_ghost[i] > 0 else 1.0)

    mem_ratio = [m_ghost[i] / m_orig[i] if m_orig[i] > 1e-6 else float("nan") for i in range(n)]
    mr_colors = []
    for r in mem_ratio:
        if not np.isfinite(r):
            mr_colors.append("#BAB0AC")
        elif r < 1.5:
            mr_colors.append("#59A14F")
        elif r < 3.0:
            mr_colors.append("#EDC948")
        else:
            mr_colors.append("#E15759")

    y = np.arange(n, dtype=np.float64)
    bar_h = 0.35
    tier_seps = _tier_separator_ys(ordered)

    fig_h = max(28.0, 2.8 * n + 14.0)
    fig = plt.figure(figsize=(20, fig_h))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.15, 1.15, 1.15, 0.95], hspace=0.35, wspace=0.28)

    supt = "Ghost+FAISS vs Full-gradient TracIn — cross-model benchmark"
    fig.suptitle(supt, fontsize=15, fontweight="bold", y=0.995)

    # Row 0: rho | heatmap
    ax_rho = fig.add_subplot(gs[0, 0])
    x_gl = np.nan_to_num(ghost_vs_orig, nan=0.0)
    bar_colors = [_tier_bar_color(name) for name in ordered]
    bars = ax_rho.barh(y, x_gl, height=0.55, color=bar_colors)
    ax_rho.axvline(0.0, color="gray", linewidth=0.8, linestyle="-", alpha=0.8)
    ax_rho.axvline(1.0, color="green", linewidth=0.9, linestyle="--", alpha=0.5)
    ax_rho.set_xlim(-1.05, 1.05)
    ax_rho.set_yticks(y)
    ax_rho.set_yticklabels(ordered, fontsize=9)
    ax_rho.set_xlabel("Spearman ρ (rank correlation)")
    ax_rho.set_title("Rank agreement (Ghost vs full-gradient)")
    ax_rho.legend(
        handles=[
            Patch(facecolor="#4E79A7", label="Small"),
            Patch(facecolor="#F28E2B", label="Medium"),
            Patch(facecolor="#E15759", label="Large"),
        ],
        loc="lower right",
        fontsize=8,
    )
    ax_rho.grid(True, axis="x", alpha=0.35)
    _add_tier_lines(ax_rho, n, tier_seps)
    for bar, val in zip(bars, ghost_vs_orig):
        if not np.isfinite(val):
            continue
        w = bar.get_width()
        x0 = bar.get_x() + w
        ax_rho.text(
            x0 + 0.02 if w >= 0 else x0 - 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left" if w >= 0 else "right",
            fontsize=7,
        )

    ax_heat = fig.add_subplot(gs[0, 1])
    if ks_sorted and np.any(np.isfinite(heat)):
        im = ax_heat.imshow(heat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
        ax_heat.set_xticks(np.arange(len(ks_sorted)))
        xtick = [f"{k}%" for k in ks_sorted] if use_pct else [str(k) for k in ks_sorted]
        ax_heat.set_xticklabels(xtick)
        ax_heat.set_yticks(np.arange(n))
        ax_heat.set_yticklabels(ordered, fontsize=9)
        ax_heat.set_xlabel("Top-k (% of n)" if use_pct else "k (top-k set size)")
        ax_heat.set_title("Top-k overlap: Ghost vs full-gradient")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="Overlap")
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                v = heat[i, j]
                if np.isfinite(v):
                    ax_heat.text(
                        j, i, f"{v:.2f}",
                        ha="center", va="center",
                        color="white" if v > 0.55 else "black",
                        fontsize=7,
                    )
        _add_tier_lines(ax_heat, n, tier_seps)
    else:
        ax_heat.text(0.5, 0.5, "No top-k overlap data.", ha="center", va="center", transform=ax_heat.transAxes)
        ax_heat.set_axis_off()

    # Row 1: throughput | speedup
    ax_thr = fig.add_subplot(gs[1, 0])
    ax_thr.barh(y - bar_h / 2, ms_orig, height=bar_h, label="Full-gradient TracIn", color="#E15759")
    ax_thr.barh(y + bar_h / 2, ms_ghost, height=bar_h, label="Ghost+FAISS", color="#4E79A7")
    ax_thr.set_yticks(y)
    ax_thr.set_yticklabels(ordered, fontsize=9)
    ax_thr.set_xlabel("Wall time per training sample (ms)")
    ax_thr.set_xscale("log")
    ax_thr.set_title("Throughput (lower is faster)")
    ax_thr.legend(loc="lower right", fontsize=8)
    ax_thr.grid(True, axis="x", alpha=0.35)
    _add_tier_lines(ax_thr, n, tier_seps)

    ax_spd = fig.add_subplot(gs[1, 1])
    sp_colors = ["#59A14F" if speedups[i] >= 1.0 else "#B07AA1" for i in range(n)]
    ax_spd.barh(y, speedups, height=0.55, color=sp_colors)
    ax_spd.axvline(1.0, color="gray", linestyle="--", linewidth=1.2, label="Break-even")
    ax_spd.set_yticks(y)
    ax_spd.set_yticklabels(ordered, fontsize=9)
    ax_spd.set_xlabel("Speedup (×) — full / Ghost, adjusted to n_train")
    ax_spd.set_xscale("log")
    ax_spd.set_title("Speedup (higher = Ghost faster)")
    ax_spd.legend(loc="lower right", fontsize=8)
    ax_spd.grid(True, axis="x", alpha=0.35)
    for i, v in enumerate(speedups):
        ax_spd.text(max(v, 1e-6) * 1.08, i, f"{v:.2f}×", va="center", fontsize=7)
    _add_tier_lines(ax_spd, n, tier_seps)

    # Row 2: memory | ratio
    ax_mem = fig.add_subplot(gs[2, 0])
    ax_mem.barh(y - bar_h / 2, m_orig, height=bar_h, label="Full-gradient peak", color="#E15759")
    ax_mem.barh(y + bar_h / 2, m_ghost, height=bar_h, label="Ghost+FAISS peak", color="#4E79A7")
    ax_mem.set_yticks(y)
    ax_mem.set_yticklabels(ordered, fontsize=9)
    ax_mem.set_xlabel("Peak memory (MiB)")
    ax_mem.set_xscale("log")
    ax_mem.set_title("Peak memory")
    ax_mem.legend(loc="lower right", fontsize=8)
    ax_mem.grid(True, axis="x", alpha=0.35)
    _add_tier_lines(ax_mem, n, tier_seps)

    ax_mrat = fig.add_subplot(gs[2, 1])
    ax_mrat.barh(y, mem_ratio, height=0.55, color=mr_colors)
    ax_mrat.axvline(1.0, color="gray", linestyle="--", linewidth=1.2)
    ax_mrat.set_yticks(y)
    ax_mrat.set_yticklabels(ordered, fontsize=9)
    ax_mrat.set_xlabel("Memory ratio (Ghost / Full)")
    ax_mrat.set_xscale("log")
    ax_mrat.set_title("Mem ratio — green <1.5×, yellow 1.5–3×, red >3×")
    ax_mrat.grid(True, axis="x", alpha=0.35)
    for i, r in enumerate(mem_ratio):
        if np.isfinite(r):
            ax_mrat.text(r * 1.06, i, f"{r:.2f}×", va="center", fontsize=7)
    _add_tier_lines(ax_mrat, n, tier_seps)

    # Row 3: table
    ax_tbl = fig.add_subplot(gs[3, :])
    ax_tbl.set_axis_off()

    def _fmt_mem(mb: float) -> str:
        if mb >= 1024:
            return f"{mb / 1024:.1f} GiB"
        return f"{mb:.0f} MiB"

    col_labels = [
        "Model", "Tier", "n_train", "Params", "Ghost dim", "Cov%", "ρ",
        "Ghost(s)", "Full(s)", "Speedup", "Ghost mem", "Full mem", "Mem×",
    ]
    cell_text = []
    for i in range(n):
        est_full = t_orig[i] * (n_trains[i] / baseline_ns[i]) if baseline_ns[i] else t_orig[i]
        su = est_full / t_ghost[i] if t_ghost[i] > 0 else 1.0
        mr = mem_ratio[i]
        mr_s = f"{mr:.2f}×" if np.isfinite(mr) else "—"
        cell_text.append([
            ordered[i],
            _tier_tag(ordered[i]),
            f"{n_trains[i]:,}",
            f"{total_params[i]:,}",
            f"{ghost_dims[i]:,}",
            f"{coverages[i]:.2f}",
            f"{sp_ghost[i]:.4f}",
            f"{t_ghost[i]:.1f}",
            f"{t_orig[i]:.1f}",
            f"{su:.2f}×",
            _fmt_mem(m_ghost[i]),
            _fmt_mem(m_orig[i]),
            mr_s,
        ])

    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.02, 1.7)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    rho_col = col_labels.index("ρ")
    for i in range(len(cell_text)):
        base = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(base)
        rv = sp_ghost[i]
        if rv >= 0.95:
            tbl[i + 1, rho_col].set_facecolor("#C6EFCE")
        elif rv >= 0.8:
            tbl[i + 1, rho_col].set_facecolor("#FFEB9C")
        elif rv < 0.5:
            tbl[i + 1, rho_col].set_facecolor("#FFC7CE")

    ax_tbl.set_title("Benchmark summary", fontsize=12, fontweight="bold", pad=12)

    plot_path = out_base / "comparison_cross_model.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize all benchmark metrics.json")
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "benchmarks"),
        help="Directory containing <model>/metrics.json",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=list(DEFAULT_MODELS),
        help="Model subfolder names under output-dir",
    )
    args = ap.parse_args()
    summarize(args.output_dir, list(args.models))


if __name__ == "__main__":
    main()
