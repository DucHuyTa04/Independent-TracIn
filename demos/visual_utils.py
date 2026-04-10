"""Visualization helpers for the interactive TracIn Ghost demo."""

from __future__ import annotations

import textwrap
from typing import Sequence

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_INFLUENCE_CMAP = LinearSegmentedColormap.from_list(
    "influence", ["#e0e0e0", "#4E79A7"], N=256,
)
_ACCENT = "#4E79A7"
_BG = "#fafafa"
_DARK = "#222222"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def denormalize_cifar(t: torch.Tensor) -> np.ndarray:
    """Undo mean/std=(0.5,0.5,0.5) normalisation → [0,1] HWC uint8-ready."""
    img = t.detach().cpu().float() * 0.5 + 0.5
    img = img.clamp(0, 1)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    return img.numpy()


def tensor_to_gray(t: torch.Tensor) -> np.ndarray:
    """Convert (1,H,W) or (H,W) tensor in [0,1] to HW numpy."""
    img = t.detach().cpu().float().clamp(0, 1)
    if img.ndim == 3:
        img = img.squeeze(0)
    return img.numpy()


# ---------------------------------------------------------------------------
# Interactive image selection grid
# ---------------------------------------------------------------------------

def show_image_selection_grid(
    images: list[np.ndarray],
    labels: list[str],
    title: str = "Click an image to select it",
    n_cols: int = 5,
) -> int | None:
    """Display a grid of images.  User clicks one.  Returns selected index or None."""
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.4 * n_cols, 2.8 * n_rows),
        facecolor=_BG,
    )
    axes = np.atleast_2d(axes)
    fig.suptitle(title, fontsize=14, fontweight="bold", color=_DARK, y=0.98)

    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        ax.set_facecolor(_BG)
        if i < n:
            cmap = "gray" if images[i].ndim == 2 else None
            ax.imshow(images[i], cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"[{i}] {labels[i]}", fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.manager.set_window_title("Select an image")

    selected: list[int | None] = [None]

    def _on_click(event):
        if event.inaxes is None:
            return
        for i in range(n):
            r, c = divmod(i, n_cols)
            if axes[r, c] == event.inaxes:
                selected[0] = i
                plt.close(fig)
                return

    fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.show(block=True)
    return selected[0]


# ---------------------------------------------------------------------------
# Text prompt selection (terminal-based, for text gen task)
# ---------------------------------------------------------------------------

def show_text_prompt_menu(
    prompts: list[str],
    allow_custom: bool = True,
) -> str:
    """Print numbered prompt list; return chosen text."""
    print("\n╔══════════════════════════════════════════╗")
    print("║        Choose a text prompt              ║")
    print("╚══════════════════════════════════════════╝")
    for i, p in enumerate(prompts):
        print(f"  [{i + 1}] {p}")
    if allow_custom:
        print(f"  [{len(prompts) + 1}] (type your own)")
    while True:
        raw = input("\nEnter number: ").strip()
        if not raw.isdigit():
            print("  Please enter a number.")
            continue
        choice = int(raw)
        if 1 <= choice <= len(prompts):
            return prompts[choice - 1]
        if allow_custom and choice == len(prompts) + 1:
            custom = input("  Type prompt: ").strip()
            if custom:
                return custom
            print("  Empty prompt, try again.")
        else:
            print(f"  Choose 1–{len(prompts) + (1 if allow_custom else 0)}.")


# ---------------------------------------------------------------------------
# Attribution result figure
# ---------------------------------------------------------------------------

def show_attribution_result(
    query_visual: np.ndarray | str,
    top_k_visuals: list[np.ndarray | str],
    top_k_scores: list[float],
    top_k_labels: list[str],
    task_title: str,
    save_path: str | None = None,
) -> None:
    """Show query on the left, top-K influential training samples on the right
    with influence-percentage bars.  Works for images (ndarray) or text (str).

    ``top_k_scores`` are raw scores — they will be normalised to percentages internally.
    """
    k = len(top_k_visuals)
    is_image = isinstance(query_visual, np.ndarray)

    # Normalise scores to percentages
    total = sum(abs(s) for s in top_k_scores)
    pcts = [abs(s) / total * 100 if total > 0 else 0.0 for s in top_k_scores]

    # ── layout ──────────────────────────────────────────────────────────
    # 1 column for query  |  k columns for top-K  |  below: stacked bar
    n_cols_right = min(k, 5)
    n_rows_right = (k + n_cols_right - 1) // n_cols_right

    fig_w = 3.2 + 2.4 * n_cols_right
    fig_h = max(3.5, 2.8 * n_rows_right + 2.0)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(task_title)

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[max(1, n_rows_right), 0.35],
        hspace=0.35,
    )

    # ── top row: query + top-K ──────────────────────────────────────────
    top_gs = gridspec.GridSpecFromSubplotSpec(
        n_rows_right, 1 + n_cols_right,
        subplot_spec=outer[0],
        width_ratios=[1.3] + [1.0] * n_cols_right,
        wspace=0.25, hspace=0.45,
    )

    # Query panel (spans all rows on the left)
    ax_q = fig.add_subplot(top_gs[:, 0])
    _draw_sample(ax_q, query_visual, "QUERY", accent=_ACCENT, fontsize=11)

    # Top-K panels
    for i in range(k):
        r, c = divmod(i, n_cols_right)
        ax = fig.add_subplot(top_gs[r, 1 + c])
        pct = pcts[i]
        colour = _INFLUENCE_CMAP(pct / 100.0)
        label = f"#{i + 1}  {top_k_labels[i]}\n{pct:.1f}%"
        _draw_sample(ax, top_k_visuals[i], label, accent=colour, fontsize=8)

    # ── bottom row: influence bar ───────────────────────────────────────
    ax_bar = fig.add_subplot(outer[1])
    _draw_influence_bar(ax_bar, pcts, top_k_labels)

    fig.suptitle(
        task_title,
        fontsize=13, fontweight="bold", color=_DARK,
        y=0.99,
    )
    fig.subplots_adjust(top=0.92)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved → {save_path}")
    if matplotlib.get_backend().lower() != "agg":
        plt.show(block=True)
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Private drawing helpers
# ---------------------------------------------------------------------------

def _draw_sample(
    ax: plt.Axes,
    visual: np.ndarray | str,
    label: str,
    accent: str | tuple = _ACCENT,
    fontsize: int = 9,
) -> None:
    """Draw an image or a styled text box in ``ax``."""
    ax.set_facecolor(_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(accent if isinstance(accent, str) else "gray")
        spine.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    if isinstance(visual, np.ndarray):
        cmap = "gray" if visual.ndim == 2 else None
        ax.imshow(visual, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        wrapped = textwrap.fill(str(visual), width=28)
        ax.text(
            0.5, 0.5, wrapped,
            ha="center", va="center",
            fontsize=max(7, fontsize - 1),
            fontfamily="monospace",
            wrap=True,
            transform=ax.transAxes,
        )
    ax.set_title(label, fontsize=fontsize, pad=4, color=_DARK)


def _draw_influence_bar(
    ax: plt.Axes,
    pcts: list[float],
    labels: list[str],
) -> None:
    """Horizontal stacked bar showing influence distribution."""
    ax.set_facecolor("white")
    left = 0.0
    for i, (p, lab) in enumerate(zip(pcts, labels)):
        colour = _INFLUENCE_CMAP(p / 100.0)
        ax.barh(0, p, left=left, height=0.6, color=colour, edgecolor="white", linewidth=0.5)
        if p > 4:
            ax.text(
                left + p / 2, 0, f"{p:.0f}%",
                ha="center", va="center", fontsize=7, fontweight="bold", color="white",
            )
        left += p
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Influence share (%)", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)


# ---------------------------------------------------------------------------
# Continue / quit prompt
# ---------------------------------------------------------------------------

def ask_continue(task_name: str) -> bool:
    """Ask user whether to run another query or move on."""
    while True:
        resp = input(f"\n  [{task_name}] Another query? (y/n): ").strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no", ""):
            return False
