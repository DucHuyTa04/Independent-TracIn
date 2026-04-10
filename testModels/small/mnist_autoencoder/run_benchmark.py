"""Benchmark Ghost+FAISS vs Original TracIn (MNIST autoencoder)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.benchmark_profiling import profile_block
from benchmarks.checkpoint_schedule import evenly_spaced_checkpoint_epochs
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
from testModels.small.mnist.data import MnistDataset
from testModels.small.mnist_autoencoder.model import MnistAutoencoder


class ReconstructionDataset(Dataset):
    """Wraps MnistDataset so that targets = flattened images (for MSE reconstruction)."""

    def __init__(self, mnist_ds: MnistDataset, indices: List[int] | None = None) -> None:
        self.mnist = mnist_ds
        self.indices = indices if indices is not None else list(range(len(mnist_ds)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        real = self.indices[i]
        x, _, _ = self.mnist[real]
        # x: (1, 28, 28) → target: (784,) flattened image
        target = x.view(-1)
        return x, target, i


def pick_subset_indices(mnist_ds: MnistDataset, n_train: int, seed: int) -> List[int]:
    """Pick n_train random indices from the MNIST training set."""
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(mnist_ds), generator=g)
    return perm[:n_train].tolist()


def main() -> None:
    p = argparse.ArgumentParser(
        description="MNIST autoencoder benchmark (Ghost+FAISS vs Original TracIn)"
    )
    p.add_argument("--output-dir", default="outputs/benchmarks/mnist_autoencoder")
    p.add_argument("--n-train", type=int, default=5000, help="number of training samples")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--data-root", default="data")
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

    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out = args.output_dir
    ckpt_dir = os.path.join(out, "checkpoints")
    faiss_dir = os.path.join(out, "faiss_tmp")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(faiss_dir, exist_ok=True)

    # Data: reuse MNIST, wrap so target = flattened image
    mnist_full = MnistDataset(train=True, root=args.data_root)
    positions = pick_subset_indices(mnist_full, args.n_train, args.seed)
    n_train = len(positions)
    base_ds = ReconstructionDataset(mnist_full, positions)
    loader = DataLoader(base_ds, batch_size=64, shuffle=False, num_workers=0)
    baseline_loader, compare_ids, baseline_subset_n = make_baseline_loader(
        base_ds, loader, n_train, args.max_baseline_n, args.seed, 64,
    )
    sample_meta = {
        i: f"digit_{int(mnist_full[positions[i]][1].item())}" for i in range(n_train)
    }

    def model_factory() -> MnistAutoencoder:
        torch.manual_seed(args.seed)
        return MnistAutoencoder()

    save_at = set(evenly_spaced_checkpoint_epochs(args.epochs))

    # Training
    m = model_factory().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    crit = nn.MSELoss()
    g_full = torch.Generator()
    g_full.manual_seed(int(args.seed))
    dl_full = DataLoader(
        base_ds, batch_size=64, shuffle=True, generator=g_full, num_workers=0,
    )
    m.train()
    for e in range(args.epochs):
        for x, target, _ in dl_full:
            x, target = x.to(device), target.to(device)
            opt.zero_grad()
            recon = m(x)
            crit(recon, target).backward()
            opt.step()
        if e in save_at:
            torch.save(m.state_dict(), os.path.join(ckpt_dir, f"ckpt_{e}.pt"))
            torch.save(opt.state_dict(), os.path.join(ckpt_dir, f"optim_{e}.pt"))
    del m

    ckpt_epochs = sorted(save_at)
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

    # Query: use test set images (reconstruction targets = themselves)
    mnist_test = MnistDataset(train=False, root=args.data_root)
    nq = min(args.num_queries, len(mnist_test))
    q_ins: List[torch.Tensor] = []
    q_tgs: List[torch.Tensor] = []
    for i in range(nq):
        x, _, _ = mnist_test[i]
        q_ins.append(x)
        q_tgs.append(x.view(-1))  # target = flattened image
    q_in = torch.stack(q_ins, dim=0)
    q_tg = torch.stack(q_tgs, dim=0)

    def mse_loss_autoenc(pred, target):
        return nn.MSELoss()(pred, target.view_as(pred))

    mod_g = model_factory().to(device)
    mod_g.load_state_dict(torch.load(last_w, map_location=device, weights_only=True))
    ghost_layers = auto_ghost_layers(mod_g, target_coverage=0.95)

    print("Ghost TracIn (profiled) …")

    with profile_block(device) as ghost_prof:
        ghost_scores = compute_ghost_tracin_scores(
            mod_g,
            ghost_layers=ghost_layers,
            training_loss_fn=mse_loss_autoenc,
            data_loader=loader,
            query_inputs=q_in,
            query_targets=q_tg,
            checkpoints=checkpoints,
            adam_param_keys=None,
            projection_dim=None,
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
                mod_d.fc_out,
                regression_error,
                baseline_loader,
                checkpoints,
                q_in,
                q_tg,
                last_w,
                last_o,
                adam_param_key=6,
                device=device,
            )
        )

    metrics = summarize_metrics(ghost_cmp, original_scores, compare_ids)
    metrics["n_train"] = n_train
    mf_info = model_factory()
    metrics["model_info"] = model_ghost_coverage(
        mf_info,
        ghost_layers[0],
        ghost_layers=ghost_layers,
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
        "mnist_autoencoder", ghost_cmp, original_scores, out, sample_ids=compare_ids,
    )
    if args.diagnostic:
        plot_paths.append(
            plot_diagnostic_variants(
                "mnist_autoencoder", spearman_diag, out, variant_order=VARIANT_ORDER
            )
        )

    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics →", os.path.join(out, "metrics.json"))
    print("Figures:", plot_paths)


if __name__ == "__main__":
    main()
