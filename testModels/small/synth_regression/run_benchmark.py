"""Benchmark Ghost+FAISS vs Original TracIn (synthetic regression)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.benchmark_profiling import profile_block
from benchmarks.train_utils import train_with_convergence
from benchmarks.comparison import build_comparison
from benchmarks.full_gradient_tracin import compute_full_gradient_tracin_scores
from benchmarks.ghost_faiss import auto_ghost_layers, compute_ghost_tracin_scores
from benchmarks.subset_loader import make_baseline_loader
from benchmarks.influence_variants import (
    VARIANT_ORDER,
    compute_diagnostic_middle_variants,
    model_ghost_coverage,
)
from benchmarks.metrics import summarize_all_variants, summarize_metrics
from benchmarks.plot import plot_diagnostic_variants, plot_model_benchmark
from src.error_functions import regression_error
from testModels.small.synth_regression.data import SynthDataset
from testModels.small.synth_regression.model import SynthRegressionMLP


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/benchmarks/synth_regression")
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-queries", type=int, default=16)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument(
        "--max-baseline-n",
        type=int,
        default=5000,
        help="Max training samples for full-gradient original TracIn (subsample if larger).",
    )
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_train = args.n_train
    out = args.output_dir
    ckpt_dir = os.path.join(out, "checkpoints")
    faiss_dir = os.path.join(out, "faiss_tmp")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(faiss_dir, exist_ok=True)

    torch.manual_seed(args.seed)

    base_ds = SynthDataset(n_train, seed=args.seed)
    loader = DataLoader(base_ds, batch_size=32, shuffle=False, num_workers=0)
    baseline_loader, compare_ids, baseline_subset_n = make_baseline_loader(
        base_ds, loader, n_train, args.max_baseline_n, args.seed, 32,
    )
    sample_meta = {
        i: "positive" if float(base_ds.Y[i].item()) >= 0 else "negative"
        for i in range(n_train)
    }

    def model_factory() -> SynthRegressionMLP:
        torch.manual_seed(args.seed)
        return SynthRegressionMLP()


    m = model_factory().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    crit = nn.MSELoss()
    g_full = torch.Generator()
    g_full.manual_seed(int(args.seed))
    dl_full = DataLoader(
        base_ds, batch_size=32, shuffle=True, generator=g_full, num_workers=0,
    )
    ckpt_epochs = train_with_convergence(
        m, opt, lambda mdl, x, y: crit(mdl(x), y.unsqueeze(1)),
        dl_full, device, ckpt_dir, max_epochs=args.epochs,
    )
    del m
    checkpoints = [
        {
            "weights_path": os.path.join(ckpt_dir, f"ckpt_{e}.pt"),
            "learning_rate": args.lr,
            "optimizer_state_path": os.path.join(ckpt_dir, f"optim_{e}.pt"),
        }
        for e in ckpt_epochs
    ]
    last_w = checkpoints[-1]["weights_path"]
    last_o = checkpoints[-1]["optimizer_state_path"]

    test_ds = SynthDataset(50, seed=args.seed + 1)
    nq = min(args.num_queries, len(test_ds))
    q_in = torch.stack([test_ds[i][0] for i in range(nq)], dim=0)
    q_tg = torch.stack([test_ds[i][1] for i in range(nq)], dim=0)

    projection_dim = None
    print("Ghost TracIn (profiled) …")
    mod_g = model_factory()
    ghost_layers = auto_ghost_layers(mod_g, target_coverage=1.0)

    def mse_loss_synth(pred, target):
        return nn.MSELoss()(pred, target.view_as(pred))

    with profile_block(device) as ghost_prof:
        ghost_scores = compute_ghost_tracin_scores(
            mod_g,
            ghost_layers=ghost_layers,
            training_loss_fn=mse_loss_synth,
            data_loader=loader,
            query_inputs=q_in,
            query_targets=q_tg,
            checkpoints=checkpoints,
            adam_param_keys=None,
            projection_dim=projection_dim,
            device=device,
        )

    print("Original TracIn / full-gradient (profiled) …")
    mod_f = model_factory()
    with profile_block(device) as full_prof:
        original_scores = compute_full_gradient_tracin_scores(
            mod_f,
            crit,
            baseline_loader,
            q_in,
            q_tg,
            checkpoints,
            device=device,
        )

    ghost_cmp = {i: ghost_scores[i] for i in compare_ids}
    variant_scores: dict[str, dict[int, float]] = {
        "A_ghost_faiss": ghost_cmp,
        "F_fullgrad_multi_ckpt": original_scores,
    }
    if args.diagnostic:
        mod_d = model_factory()
        variant_scores.update(
            compute_diagnostic_middle_variants(
                mod_d,
                mod_d.fc2,
                regression_error,
                baseline_loader,
                checkpoints,
                q_in,
                q_tg,
                last_w,
                last_o,
                adam_param_key=2,
                device=device,
            )
        )

    metrics = summarize_metrics(ghost_cmp, original_scores, compare_ids)
    metrics["n_train"] = n_train
    mf = model_factory()
    metrics["model_info"] = model_ghost_coverage(
        mf, mf.fc2, ghost_layers=auto_ghost_layers(mf, target_coverage=1.0),
    )
    total_p = metrics["model_info"]["total_params"]
    ghost_vec_dim = metrics["model_info"].get("ghost_dim")
    metrics["comparison"] = build_comparison(
        ghost_cmp,
        original_scores,
        compare_ids,
        ghost_prof,
        full_prof,
        n_train,
        total_p,
        ghost_vec_dim,
        baseline_subset_n=baseline_subset_n,
    )
    metrics["benchmark_settings"] = {
        "seed": args.seed,
        "num_queries": nq,
        "diagnostic": args.diagnostic,
        "max_baseline_n": args.max_baseline_n,
    }
    metrics["variants"] = summarize_all_variants(
        variant_scores, original_scores, compare_ids,
    )
    metrics["primary_variant"] = "A_ghost_faiss"
    spearman_diag = {
        k: metrics["variants"][k]["spearman_vs_reference"]
        for k in VARIANT_ORDER
        if k in metrics["variants"]
    }
    plot_paths = plot_model_benchmark(
        "synth_regression", ghost_cmp, original_scores, out, sample_ids=compare_ids,
    )
    if args.diagnostic:
        plot_paths.append(
            plot_diagnostic_variants(
                "synth_regression", spearman_diag, out, variant_order=VARIANT_ORDER
            )
        )

    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics →", os.path.join(out, "metrics.json"))
    print("Figures:", plot_paths)


if __name__ == "__main__":
    main()
