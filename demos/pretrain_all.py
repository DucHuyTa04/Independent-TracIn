#!/usr/bin/env python3
"""One-shot pre-training script for the interactive demo.

Trains all three demo models (CIFAR-10 CNN, TinyGPT, Fashion-VAE),
saves TracIn checkpoints, and builds FAISS ghost indices.

Run this ONCE before ``interactive_demo.py``::

    python demos/pretrain_all.py --device cpu --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from demos.demo_utils import (
    CIFAR_CLASSES,
    FASHION_LABELS,
    ReindexedSubset as _ReindexedSubset,
    ensure_faiss_index,
    lm_pooled_classification_error,
    resolve_device,
    train_with_tracin_checkpoints,
    write_demo_config,
)
from src.config_utils import resolve_target_layer
from src.error_functions import get_error_fn


# ── helpers ────────────────────────────────────────────────────────────────


# ── task 1 ─────────────────────────────────────────────────────────────────

def pretrain_classification(args) -> None:
    print("\n" + "─" * 50)
    print("  [1/3] CIFAR-10 Classification")
    print("─" * 50)

    from testModels.medium.cifar10_cnn.data import CifarDataset
    from testModels.medium.cifar10_cnn.model import CifarSmallCNN

    base = os.path.join(ROOT, "demos", "outputs", "cifar10_classification")
    ckpt_dir = os.path.join(base, "checkpoints")
    demo_cfg = os.path.join(base, "demo_config.yaml")
    os.makedirs(base, exist_ok=True)

    full_train = CifarDataset(train=True, root=args.data_root)
    g = torch.Generator().manual_seed(42)
    n = min(args.max_train, len(full_train))
    idx = torch.randperm(len(full_train), generator=g)[:n].tolist()
    train_subset = Subset(full_train, idx)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=False, num_workers=0)

    sample_meta: dict[int, str] = {}
    for j, i in enumerate(idx):
        _, y, _ = full_train[i]
        sample_meta[j] = CIFAR_CLASSES[int(y)]

    model = CifarSmallCNN(num_classes=10).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    def loss_step(m, x, y):
        return crit(m(x), y)

    train_with_tracin_checkpoints(
        model, opt, train_loader, loss_step, ckpt_dir,
        args.cifar_epochs, args.device,
        save_every=max(1, args.cifar_epochs // 5),
    )

    write_demo_config(os.path.abspath(ckpt_dir), os.path.abspath(base), demo_cfg)
    _, target_layer = resolve_target_layer(model, None)
    error_fn = get_error_fn("classification")

    ensure_faiss_index(
        model=model, target_layer=target_layer, error_fn=error_fn,
        train_loader=train_loader, sample_meta=sample_meta,
        demo_config_path=demo_cfg, index_dir=base,
        index_name="faiss_index_cifar10_demo",
        meta_name="faiss_metadata_cifar10_demo.json",
        projection_dim=args.projection_dim, projection_type="sjlt",
        projection_seed=42, device=args.device, force=True,
    )
    print("  ✓ CIFAR-10 done\n")


# ── task 2 ─────────────────────────────────────────────────────────────────

def pretrain_text_generation(args) -> None:
    print("\n" + "─" * 50)
    print("  [2/3] TinyGPT Text Generation")
    print("─" * 50)

    from testModels.medium.transformer_lm.data import CharLMDataset
    from testModels.medium.transformer_lm.model import TinyGPT

    base = os.path.join(ROOT, "demos", "outputs", "tinygpt_demo")
    ckpt_dir = os.path.join(base, "checkpoints")
    demo_cfg = os.path.join(base, "demo_config.yaml")
    os.makedirs(base, exist_ok=True)

    full_train = CharLMDataset(root=args.data_root, train=True)
    vocab_size = full_train.vocab_size
    g = torch.Generator().manual_seed(42)
    n_train = min(args.n_train, len(full_train))
    indices = torch.randperm(len(full_train), generator=g)[:n_train].tolist()
    base_ds = _ReindexedSubset(full_train, indices)
    train_loader = DataLoader(base_ds, batch_size=32, shuffle=False, num_workers=0)
    sample_meta = {i: f"seq_{i}" for i in range(n_train)}

    model = TinyGPT(vocab_size=vocab_size).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    def loss_step(m, x, y):
        logits = m(x)
        b, t, v = logits.shape
        return nn.functional.cross_entropy(logits.reshape(-1, v), y.reshape(-1))

    train_with_tracin_checkpoints(
        model, opt, train_loader, loss_step, ckpt_dir,
        args.gpt_epochs, args.device,
        save_every=max(1, args.gpt_epochs // 5),
    )

    write_demo_config(os.path.abspath(ckpt_dir), os.path.abspath(base), demo_cfg)
    _, target_layer = resolve_target_layer(model, "output_proj")

    ensure_faiss_index(
        model=model, target_layer=target_layer,
        error_fn=lm_pooled_classification_error,
        train_loader=train_loader, sample_meta=sample_meta,
        demo_config_path=demo_cfg, index_dir=base,
        index_name="faiss_index_tinygpt_demo",
        meta_name="faiss_metadata_tinygpt_demo.json",
        projection_dim=args.projection_dim, projection_type="sjlt",
        projection_seed=43, device=args.device, force=True,
    )
    print("  ✓ TinyGPT done\n")


# ── task 3 ─────────────────────────────────────────────────────────────────

def pretrain_image_generation(args) -> None:
    print("\n" + "─" * 50)
    print("  [3/3] Fashion-MNIST VAE")
    print("─" * 50)

    from testModels.medium.vae_fashion.data import FashionMnistDataset
    from testModels.medium.vae_fashion.model import FashionVAE

    base = os.path.join(ROOT, "demos", "outputs", "vae_fashion_demo")
    ckpt_dir = os.path.join(base, "checkpoints")
    demo_cfg = os.path.join(base, "demo_config.yaml")
    os.makedirs(base, exist_ok=True)

    full_train = FashionMnistDataset(train=True, root=args.data_root)
    g = torch.Generator().manual_seed(42)
    n = min(args.max_train, len(full_train))
    idx = torch.randperm(len(full_train), generator=g)[:n].tolist()
    train_subset = Subset(full_train, idx)
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=False, num_workers=0)

    sample_meta: dict[int, str] = {}
    for j, i in enumerate(idx):
        _, lbl = full_train.ds[i]
        sample_meta[j] = FASHION_LABELS[int(lbl)]

    model = FashionVAE().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def loss_step(m, x, y):
        recon = m(x)
        bce = F.binary_cross_entropy(recon, y.view_as(recon), reduction="mean")
        mu = m._last_mu
        logvar = m._last_logvar
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kl

    train_with_tracin_checkpoints(
        model, opt, train_loader, loss_step, ckpt_dir,
        args.vae_epochs, args.device,
        save_every=max(1, args.vae_epochs // 5),
    )

    write_demo_config(os.path.abspath(ckpt_dir), os.path.abspath(base), demo_cfg)
    _, target_layer = resolve_target_layer(model, "dec_out")
    error_fn = get_error_fn("regression")

    ensure_faiss_index(
        model=model, target_layer=target_layer, error_fn=error_fn,
        train_loader=train_loader, sample_meta=sample_meta,
        demo_config_path=demo_cfg, index_dir=base,
        index_name="faiss_index_vae_demo",
        meta_name="faiss_metadata_vae_demo.json",
        projection_dim=args.projection_dim, projection_type="sjlt",
        projection_seed=44, device=args.device, force=True,
    )
    print("  ✓ Fashion VAE done\n")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-train all demo models and build FAISS indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run this ONCE before interactive_demo.py.",
    )
    p.add_argument("--device", default="cuda", help="cuda | cpu | auto (default: cuda)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--max-train", type=int, default=8000, help="Subset size (classification & VAE)")
    p.add_argument("--n-train", type=int, default=8000, help="Training sequences for text gen")
    p.add_argument("--cifar-epochs", type=int, default=500)
    p.add_argument("--gpt-epochs", type=int, default=1000)
    p.add_argument("--vae-epochs", type=int, default=500)
    p.add_argument("--projection-dim", type=int, default=512)
    p.add_argument("--skip", default="", help="Comma-separated tasks to skip: cifar,gpt,vae")
    args = p.parse_args()
    args.device = resolve_device(args.device)

    skip = set(s.strip().lower() for s in args.skip.split(",") if s.strip())

    print("╔══════════════════════════════════════════════════╗")
    print("║   TracIn Ghost Demo — Pre-training Pipeline     ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Device: {args.device}")

    if "cifar" not in skip:
        pretrain_classification(args)
    if "gpt" not in skip:
        pretrain_text_generation(args)
    if "vae" not in skip:
        pretrain_image_generation(args)

    print("=" * 50)
    print("  All pre-training complete.")
    print("  Outputs in: demos/outputs/")
    print("  Now run: python demos/interactive_demo.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
