"""Benchmark convex MNIST logistic regression (100%% ghost coverage)."""

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
from benchmarks.ghost_faiss import compute_ghost_tracin_scores
from benchmarks.influence_variants import (
    VARIANT_ORDER,
    compute_diagnostic_middle_variants,
    model_ghost_coverage,
)
from benchmarks.metrics import summarize_all_variants, summarize_metrics
from benchmarks.plot import plot_diagnostic_variants, plot_model_benchmark
from benchmarks.subset_loader import make_baseline_loader
from src.error_functions import classification_error
from testModels.small.linear_logistic.model import MnistLinear
from testModels.small.mnist.data import MnistDataset


class IndexListDataset(Dataset):
    def __init__(self, mnist_train: MnistDataset, positions: List[int]) -> None:
        self.mnist = mnist_train
        self.positions = positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int):
        real = self.positions[i]
        x, y, _ = self.mnist[real]
        return x, y, i


def pick_subset_indices(mnist_train: MnistDataset, per_class: int, seed: int) -> List[int]:
    torch.manual_seed(seed)
    by_c: dict[int, List[int]] = {c: [] for c in range(10)}
    for idx in range(len(mnist_train)):
        _, y, _ = mnist_train[idx]
        c = int(y.item())
        if len(by_c[c]) < per_class:
            by_c[c].append(idx)
        if all(len(by_c[c]) >= per_class for c in range(10)):
            break
    out = []
    for c in range(10):
        out.extend(by_c[c])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Linear MNIST benchmark (convex, full ghost coverage)")
    p.add_argument("--output-dir", default="outputs/benchmarks/linear_logistic")
    p.add_argument("--per-class", type=int, default=150)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
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
    os.makedirs(ckpt_dir, exist_ok=True)

    mnist_full = MnistDataset(train=True, root=args.data_root)
    positions = pick_subset_indices(mnist_full, args.per_class, args.seed)
    n_train = len(positions)
    base_ds = IndexListDataset(mnist_full, positions)
    loader = DataLoader(base_ds, batch_size=64, shuffle=False, num_workers=0)
    baseline_loader, compare_ids, baseline_subset_n = make_baseline_loader(
        base_ds, loader, n_train, args.max_baseline_n, args.seed, 64,
    )

    def model_factory() -> MnistLinear:
        torch.manual_seed(args.seed)
        return MnistLinear()

    save_at = set(evenly_spaced_checkpoint_epochs(args.epochs))

    m = model_factory().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    g_full = torch.Generator()
    g_full.manual_seed(int(args.seed))
    dl_full = DataLoader(
        base_ds, batch_size=64, shuffle=True, generator=g_full, num_workers=0,
    )
    m.train()
    for e in range(args.epochs):
        for x, y, _ in dl_full:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(m(x), y).backward()
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

    mnist_test = MnistDataset(train=False, root=args.data_root)
    q_ins: List[torch.Tensor] = []
    q_tgs: List[torch.Tensor] = []
    for i in range(len(mnist_test)):
        if len(q_ins) >= args.num_queries:
            break
        x, y, _ = mnist_test[i]
        if int(y.item()) == 0:
            q_ins.append(x)
            q_tgs.append(torch.tensor(0, dtype=torch.long))
    assert q_ins
    q_in = torch.stack(q_ins, dim=0)
    q_tg = torch.stack(q_tgs, dim=0)

    adam_key = 0

    print("Ghost TracIn (profiled) …")
    mod_g = model_factory()
    with profile_block(device) as ghost_prof:
        ghost_scores = compute_ghost_tracin_scores(
            mod_g,
            ghost_layers=[mod_g.fc],
            training_loss_fn=nn.CrossEntropyLoss(),
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
                mod_d.fc,
                classification_error,
                baseline_loader,
                checkpoints,
                q_in,
                q_tg,
                last_w,
                last_o,
                adam_param_key=adam_key,
                device=device,
            )
        )

    metrics = summarize_metrics(ghost_cmp, original_scores, compare_ids)
    metrics["n_train"] = n_train
    mf_info = model_factory()
    metrics["model_info"] = model_ghost_coverage(mf_info, mf_info.fc, ghost_layers=[mf_info.fc])
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
        "num_queries": args.num_queries,
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
        "linear_logistic", ghost_cmp, original_scores, out, sample_ids=compare_ids,
    )
    if args.diagnostic:
        plot_paths.append(
            plot_diagnostic_variants(
                "linear_logistic", spearman_diag, out, variant_order=VARIANT_ORDER
            )
        )

    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics →", os.path.join(out, "metrics.json"))
    print("Figures:", plot_paths)


if __name__ == "__main__":
    main()
