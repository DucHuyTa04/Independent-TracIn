"""Benchmark Ghost+FAISS vs Original TracIn (ViT-Micro, CIFAR-10)."""

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
from src.error_functions import classification_error
from testModels.medium.vit_cifar10.data import CifarDataset
from testModels.medium.vit_cifar10.model import ViTMicro


def _label_to_int(y) -> int:
    return int(y) if not isinstance(y, torch.Tensor) else int(y.item())


class IndexListDataset(Dataset):
    def __init__(self, ds: CifarDataset, positions: List[int]) -> None:
        self.ds = ds
        self.positions = positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int):
        real = self.positions[i]
        x, y, _ = self.ds[real]
        return x, y, i


def pick_subset_indices(ds: CifarDataset, per_class: int, seed: int) -> List[int]:
    torch.manual_seed(seed)
    by_c: dict[int, List[int]] = {c: [] for c in range(10)}
    for idx in range(len(ds)):
        _, y, _ = ds[idx]
        c = _label_to_int(y)
        if len(by_c[c]) < per_class:
            by_c[c].append(idx)
        if all(len(v) >= per_class for v in by_c.values()):
            break
    out: List[int] = []
    for c in range(10):
        out.extend(by_c[c])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/benchmarks/vit_cifar10")
    p.add_argument("--per-class", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
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

    full_train = CifarDataset(train=True, root=args.data_root)
    positions = pick_subset_indices(full_train, args.per_class, args.seed)
    n_train = len(positions)
    base_ds = IndexListDataset(full_train, positions)
    loader = DataLoader(base_ds, batch_size=64, shuffle=False, num_workers=0)
    baseline_loader, compare_ids, baseline_subset_n = make_baseline_loader(
        base_ds, loader, n_train, args.max_baseline_n, args.seed, 64,
    )
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    sample_meta = {
        i: classes[_label_to_int(full_train[positions[i]][1])] for i in range(n_train)
    }

    def model_factory() -> ViTMicro:
        torch.manual_seed(args.seed)
        return ViTMicro()


    m = model_factory().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    g_full = torch.Generator()
    g_full.manual_seed(int(args.seed))
    dl_full = DataLoader(
        base_ds, batch_size=64, shuffle=True, generator=g_full, num_workers=0,
    )
    ckpt_epochs = train_with_convergence(
        m, opt, lambda mdl, x, y: crit(mdl(x), y),
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

    # Scored layer selection uses trained weights + one calibration batch
    mod_g = model_factory().to(device)
    mod_g.load_state_dict(torch.load(last_w, map_location=device, weights_only=True))
    ghost_layers = auto_ghost_layers(mod_g, target_coverage=1.0)

    test_ds = CifarDataset(train=False, root=args.data_root)
    q_ins: List[torch.Tensor] = []
    q_tgs: List[torch.Tensor] = []
    for i in range(len(test_ds)):
        if len(q_ins) >= args.num_queries:
            break
        x, y, _ = test_ds[i]
        if _label_to_int(y) == 0:
            q_ins.append(x)
            q_tgs.append(torch.tensor(0, dtype=torch.long))
    assert q_ins, "Need at least one class-0 test image"
    q_in = torch.stack(q_ins, dim=0)
    q_tg = torch.stack(q_tgs, dim=0)

    print("Ghost TracIn (profiled) …")
    with profile_block(device) as ghost_prof:
        ghost_scores = compute_ghost_tracin_scores(
            mod_g,
            ghost_layers=ghost_layers,
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
                mod_d.head,
                classification_error,
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
        "vit_cifar10", ghost_cmp, original_scores, out, sample_ids=compare_ids,
    )
    if args.diagnostic:
        plot_paths.append(
            plot_diagnostic_variants(
                "vit_cifar10", spearman_diag, out, variant_order=VARIANT_ORDER
            )
        )

    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics →", os.path.join(out, "metrics.json"))
    print("Figures:", plot_paths)


if __name__ == "__main__":
    main()
