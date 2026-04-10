"""Benchmark Ghost+FAISS vs Original TracIn (VAE, Fashion-MNIST)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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
from testModels.medium.vae_fashion.data import FashionMnistDataset
from testModels.medium.vae_fashion.model import FashionVAE


class SubsetDataset(Dataset):
    """Subset with re-indexed sample IDs."""

    def __init__(self, ds: FashionMnistDataset, indices: List[int]) -> None:
        self.ds = ds
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x, target, _ = self.ds[self.indices[i]]
        return x, target, i


def _make_elbo_loss(model: FashionVAE):
    """Closure that reads model._last_mu/logvar set during forward()."""

    def elbo(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(recon, target.view_as(recon), reduction="mean")
        mu = model._last_mu
        logvar = model._last_logvar
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kl

    return elbo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/benchmarks/vae_fashion")
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-queries", type=int, default=16)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--max-baseline-n", type=int, default=5000)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out = args.output_dir
    ckpt_dir = os.path.join(out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data
    full_train = FashionMnistDataset(train=True, root=args.data_root)
    n_avail = len(full_train)
    n_train = min(args.n_train, n_avail)
    g = torch.Generator()
    g.manual_seed(args.seed)
    indices = torch.randperm(n_avail, generator=g)[:n_train].tolist()
    base_ds = SubsetDataset(full_train, indices)
    loader = DataLoader(base_ds, batch_size=64, shuffle=False, num_workers=0)
    baseline_loader, compare_ids, baseline_subset_n = make_baseline_loader(
        base_ds, loader, n_train, args.max_baseline_n, args.seed, 64,
    )

    def model_factory() -> FashionVAE:
        torch.manual_seed(args.seed)
        return FashionVAE()


    # Train
    m = model_factory().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    elbo_train = _make_elbo_loss(m)
    g_full = torch.Generator()
    g_full.manual_seed(int(args.seed))
    dl_full = DataLoader(
        base_ds, batch_size=64, shuffle=True, generator=g_full, num_workers=0,
    )
    ckpt_epochs = train_with_convergence(
        m, opt, lambda mdl, x, y: elbo_train(mdl(x), y),
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

    mod_g = model_factory().to(device)
    elbo_ghost = _make_elbo_loss(mod_g)
    mod_g.load_state_dict(torch.load(last_w, map_location=device, weights_only=True))
    ghost_layers = auto_ghost_layers(mod_g, target_coverage=0.9)

    # Queries from test set
    test_ds = FashionMnistDataset(train=False, root=args.data_root)
    nq = min(args.num_queries, len(test_ds))
    q_ins: List[torch.Tensor] = []
    q_tgs: List[torch.Tensor] = []
    for i in range(nq):
        x, target, _ = test_ds[i]
        q_ins.append(x)
        q_tgs.append(target)
    q_in = torch.stack(q_ins, dim=0)
    q_tg = torch.stack(q_tgs, dim=0)

    # Ghost TracIn
    print("Ghost TracIn (profiled) …")
    with profile_block(device) as ghost_prof:
        ghost_scores = compute_ghost_tracin_scores(
            mod_g,
            ghost_layers=ghost_layers,
            training_loss_fn=elbo_ghost,
            data_loader=loader,
            query_inputs=q_in,
            query_targets=q_tg,
            checkpoints=checkpoints,
            adam_param_keys=None,
            projection_dim=None,
            device=device,
        )

    # Full-gradient TracIn
    print("Original TracIn / full-gradient (profiled) …")
    mod_f = model_factory()
    elbo_full = _make_elbo_loss(mod_f)
    with profile_block(device) as full_prof:
        original_scores = compute_full_gradient_tracin_scores(
            mod_f,
            elbo_full,
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

    metrics = summarize_metrics(ghost_cmp, original_scores, compare_ids)
    metrics["n_train"] = n_train
    mf_info = model_factory()
    metrics["model_info"] = model_ghost_coverage(
        mf_info, ghost_layers[0], ghost_layers=ghost_layers,
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
        "vae_fashion", ghost_cmp, original_scores, out, sample_ids=compare_ids,
    )
    if args.diagnostic:
        plot_paths.append(
            plot_diagnostic_variants(
                "vae_fashion", spearman_diag, out, variant_order=VARIANT_ORDER
            )
        )

    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics →", os.path.join(out, "metrics.json"))
    print("Figures:", plot_paths)


if __name__ == "__main__":
    main()
