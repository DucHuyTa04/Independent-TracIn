#!/usr/bin/env python3
"""CIFAR-10 classification demo: attribute test images to top-k training influences."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from demos.demo_utils import (
    CIFAR_CLASSES,
    ensure_faiss_index,
    format_attribution_lines,
    last_ckpt_paths,
    resolve_device,
    run_attribute,
    train_with_tracin_checkpoints,
    write_demo_config,
)
from src.config_utils import (
    find_adam_bias_param_key,
    find_adam_param_key,
    resolve_target_layer,
    smart_load_weights_into_model,
)
from src.error_functions import get_error_fn
from testModels.medium.cifar10_cnn.data import CifarDataset, make_loaders
from testModels.medium.cifar10_cnn.model import CifarSmallCNN


def main() -> None:
    p = argparse.ArgumentParser(description="CIFAR-10 TracIn Ghost attribution demo")
    p.add_argument("--device", default="cuda", help="cuda | cpu | auto (default: cuda)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--num-queries", type=int, default=5)
    p.add_argument("--max-train", type=int, default=8000, help="Subset size for fast demo")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--projection-dim", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--force-reindex", action="store_true")
    args = p.parse_args()

    device = resolve_device(args.device)
    base = os.path.join(ROOT, "demos", "outputs", "cifar10_classification")
    ckpt_dir = os.path.join(base, "checkpoints")
    index_dir = base
    demo_cfg = os.path.join(base, "demo_config.yaml")

    os.makedirs(base, exist_ok=True)
    g = torch.Generator().manual_seed(args.seed)

    full_train = CifarDataset(train=True, root=args.data_root)
    n = min(args.max_train, len(full_train))
    idx = torch.randperm(len(full_train), generator=g)[:n].tolist()
    train_subset = Subset(full_train, idx)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=False, num_workers=0)
    _, test_loader, _ = make_loaders(batch_size=1, data_root=args.data_root)

    sample_meta: dict[int, str] = {}
    for j, i in enumerate(idx):
        _, y, _ = full_train[i]
        sample_meta[j] = CIFAR_CLASSES[int(y)]

    meta_path = os.path.join(ckpt_dir, "tracin_checkpoints_metadata.json")
    need_train = args.force_retrain or not os.path.isfile(meta_path)

    model = CifarSmallCNN(num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    def loss_step(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return crit(m(x), y)

    if need_train:
        print("Training CIFAR-10 CNN (subset) …")
        train_with_tracin_checkpoints(
            model, opt, train_loader, loss_step, ckpt_dir, args.epochs, device,
            save_every=max(1, args.epochs // 5),
        )
    else:
        print("Skipping training (checkpoints present). Use --force-retrain to retrain.")

    write_demo_config(
        os.path.abspath(ckpt_dir),
        os.path.abspath(index_dir),
        demo_cfg,
    )
    if not need_train:
        w0, _ = last_ckpt_paths(demo_cfg)
        smart_load_weights_into_model(model, w0, device)

    _, target_layer = resolve_target_layer(model, None)
    error_fn = get_error_fn("classification")
    adam_key = find_adam_param_key(model, target_layer)
    adam_bias_key = find_adam_bias_param_key(model, target_layer)

    ensure_faiss_index(
        model=model,
        target_layer=target_layer,
        error_fn=error_fn,
        train_loader=train_loader,
        sample_meta=sample_meta,
        demo_config_path=demo_cfg,
        index_dir=index_dir,
        index_name="faiss_index_cifar10_demo",
        meta_name="faiss_metadata_cifar10_demo.json",
        projection_dim=args.projection_dim,
        projection_type="sjlt",
        projection_seed=42,
        device=device,
        force=args.force_reindex or need_train,
    )

    wpath, opath = last_ckpt_paths(demo_cfg)
    model.eval()

    # Random test queries
    test_ds = test_loader.dataset
    n_test = len(test_ds)
    q_idx = torch.randperm(n_test, generator=g)[: min(args.num_queries, n_test)].tolist()

    print("\nAttribution (Ghost + FAISS) on random CIFAR-10 test images:\n")
    for k, ti in enumerate(q_idx):
        x, y, _ = test_ds[ti]
        q_in = x.unsqueeze(0)
        q_tg = torch.tensor([int(y)])
        pred = int(torch.argmax(model(q_in.to(device)), dim=-1).item())
        results = run_attribute(
            model=model,
            target_layer=target_layer,
            error_fn=error_fn,
            query_inputs=q_in,
            query_targets=q_tg,
            index_dir=index_dir,
            index_name="faiss_index_cifar10_demo",
            meta_name="faiss_metadata_cifar10_demo.json",
            ckpt_weights=wpath,
            ckpt_opt=opath,
            adam_key=adam_key,
            adam_bias_key=adam_bias_key,
            top_k=args.top_k,
            projection_dim=args.projection_dim,
            projection_type="sjlt",
            projection_seed=42,
            device=device,
        )
        print(f"=== Query {k + 1}/{len(q_idx)}  true={CIFAR_CLASSES[int(y)]}  pred={CIFAR_CLASSES[pred]} ===")
        for line in format_attribution_lines(results, sample_meta, args.top_k):
            print(line)


if __name__ == "__main__":
    main()
