"""Benchmark Ghost+FAISS vs Original TracIn (Large GPT char-level LM, ~26M params)."""

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
from benchmarks.influence_variants import model_ghost_coverage
from benchmarks.metrics import summarize_all_variants, summarize_metrics
from benchmarks.plot import plot_model_benchmark
from testModels.large.transformer_lm_large.data import CharLMDataset
from testModels.large.transformer_lm_large.model import build_large_gpt


class ReindexedSubset(Dataset):
    """Subset of a dataset with contiguous re-indexed sample IDs."""

    def __init__(self, ds: CharLMDataset, indices: List[int]) -> None:
        self.ds = ds
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        x, y, _ = self.ds[self.indices[i]]
        return x, y, i


def _lm_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CrossEntropy for (B, T, V) logits and (B, T) targets."""
    B, T, V = logits.shape
    return nn.functional.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/benchmarks/transformer_lm_large")
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-queries", type=int, default=16)
    p.add_argument("--max-baseline-n", type=int, default=2000)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out = args.output_dir
    ckpt_dir = os.path.join(out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data — longer context (128) to match larger model capacity
    full_train = CharLMDataset(root=args.data_root, ctx_len=128, train=True)
    vocab_size = full_train.vocab_size
    n_avail = len(full_train)
    n_train = min(args.n_train, n_avail)
    g = torch.Generator()
    g.manual_seed(args.seed)
    indices = torch.randperm(n_avail, generator=g)[:n_train].tolist()
    base_ds = ReindexedSubset(full_train, indices)
    loader = DataLoader(base_ds, batch_size=16, shuffle=False, num_workers=0)
    baseline_loader, compare_ids, baseline_subset_n = make_baseline_loader(
        base_ds, loader, n_train, args.max_baseline_n, args.seed, 16,
    )

    def model_factory() -> nn.Module:
        torch.manual_seed(args.seed)
        return build_large_gpt(vocab_size=vocab_size, ctx_len=128)

    save_at = set(evenly_spaced_checkpoint_epochs(args.epochs))

    # Train
    m = model_factory().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    g_full = torch.Generator()
    g_full.manual_seed(int(args.seed))
    dl_full = DataLoader(
        base_ds, batch_size=16, shuffle=True, generator=g_full, num_workers=0,
    )
    m.train()
    for e in range(args.epochs):
        for x, y, _ in dl_full:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            _lm_cross_entropy(m(x), y).backward()
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

    # Queries from test split
    test_ds = CharLMDataset(root=args.data_root, ctx_len=128, train=False)
    nq = min(args.num_queries, len(test_ds))
    q_ins: List[torch.Tensor] = []
    q_tgs: List[torch.Tensor] = []
    for i in range(nq):
        x, y, _ = test_ds[i]
        q_ins.append(x)
        q_tgs.append(y)
    q_in = torch.stack(q_ins, dim=0)
    q_tg = torch.stack(q_tgs, dim=0)

    # Ghost TracIn
    print("Ghost TracIn (profiled) …")
    mod_g = model_factory().to(device)
    mod_g.load_state_dict(torch.load(last_w, map_location=device, weights_only=True))
    ghost_layers = auto_ghost_layers(mod_g, target_coverage=0.95)
    with profile_block(device) as ghost_prof:
        ghost_scores = compute_ghost_tracin_scores(
            mod_g,
            ghost_layers=ghost_layers,
            training_loss_fn=_lm_cross_entropy,
            data_loader=loader,
            query_inputs=q_in,
            query_targets=q_tg,
            checkpoints=checkpoints,
            adam_param_keys=None,
            projection_dim=None,
            device=device,
        )

    # Full-gradient TracIn (baseline)
    print("Original TracIn / full-gradient (profiled) …")
    mod_f = model_factory()
    with profile_block(device) as full_prof:
        original_scores = compute_full_gradient_tracin_scores(
            mod_f,
            _lm_cross_entropy,
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
    mf_layers = auto_ghost_layers(mf_info, target_coverage=0.95)
    metrics["model_info"] = model_ghost_coverage(
        mf_info, mf_layers[0], ghost_layers=mf_layers,
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
        "diagnostic": False,
        "max_baseline_n": args.max_baseline_n,
    }
    metrics["variants"] = summarize_all_variants(
        variant_scores, original_scores, compare_ids,
    )
    metrics["primary_variant"] = "A_ghost_faiss"

    plot_model_benchmark(
        "transformer_lm_large", ghost_cmp, original_scores, out, sample_ids=compare_ids,
    )

    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics →", os.path.join(out, "metrics.json"))


if __name__ == "__main__":
    main()
