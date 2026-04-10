#!/usr/bin/env python3
"""Fashion-MNIST VAE demo: sample latent → decode image → TracIn attribution."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from demos.demo_utils import (
    FASHION_LABELS,
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
from testModels.medium.vae_fashion.data import FashionMnistDataset
from testModels.medium.vae_fashion.model import FashionVAE


def make_elbo_loss(model: FashionVAE):
    def elbo(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(recon, target.view_as(recon), reduction="mean")
        mu = model._last_mu
        logvar = model._last_logvar
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kl

    return elbo


def main() -> None:
    p = argparse.ArgumentParser(description="Fashion VAE TracIn Ghost attribution demo")
    p.add_argument("--device", default="cuda", help="cuda | cpu | auto (default: cuda)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--num-images", type=int, default=3)
    p.add_argument("--max-train", type=int, default=12000)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--projection-dim", type=int, default=512)
    p.add_argument("--latent-seed", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--force-reindex", action="store_true")
    p.add_argument("--save-grid", action="store_true", help="Save PNG grid under demo output dir")
    args = p.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    base = os.path.join(ROOT, "demos", "outputs", "vae_fashion_demo")
    ckpt_dir = os.path.join(base, "checkpoints")
    index_dir = base
    demo_cfg = os.path.join(base, "demo_config.yaml")
    os.makedirs(base, exist_ok=True)

    full_train = FashionMnistDataset(train=True, root=args.data_root)
    n = min(args.max_train, len(full_train))
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(full_train), generator=g)[:n].tolist()
    train_subset = Subset(full_train, idx)
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=False, num_workers=0)

    sample_meta: dict[int, str] = {}
    for j, i in enumerate(idx):
        _, lbl = full_train.ds[i]
        sample_meta[j] = FASHION_LABELS[int(lbl)]

    meta_path = os.path.join(ckpt_dir, "tracin_checkpoints_metadata.json")
    need_train = args.force_retrain or not os.path.isfile(meta_path)

    model = FashionVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    elbo_fn = make_elbo_loss(model)

    def loss_step(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        recon = m(x)
        return elbo_fn(recon, y)

    if need_train:
        print("Training FashionVAE (subset) …")
        train_with_tracin_checkpoints(
            model, opt, train_loader, loss_step, ckpt_dir, args.epochs, device,
            save_every=max(1, args.epochs // 5),
        )
    else:
        print("Skipping training. Use --force-retrain to retrain.")

    write_demo_config(os.path.abspath(ckpt_dir), os.path.abspath(index_dir), demo_cfg)
    if not need_train:
        w0, _ = last_ckpt_paths(demo_cfg)
        smart_load_weights_into_model(model, w0, device)

    _, target_layer = resolve_target_layer(model, "dec_out")
    error_fn = get_error_fn("regression")
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
        index_name="faiss_index_vae_demo",
        meta_name="faiss_metadata_vae_demo.json",
        projection_dim=args.projection_dim,
        projection_type="sjlt",
        projection_seed=44,
        device=device,
        force=args.force_reindex or need_train,
    )

    wpath, opath = last_ckpt_paths(demo_cfg)
    model.eval()

    gen_rng = torch.Generator(device="cpu")
    if args.latent_seed is not None:
        gen_rng.manual_seed(int(args.latent_seed))

    to_save: list[tuple[str, torch.Tensor]] = []

    print("\n=== Latent samples → attribution ===\n")
    for im in range(args.num_images):
        z = torch.randn(1, model.latent_dim, generator=gen_rng).to(device)
        with torch.no_grad():
            recon = model.decode(z)
        x_vis = recon.view(1, 1, 28, 28).clamp(0, 1).cpu()
        to_save.append((f"gen_{im}", x_vis.squeeze().cpu()))

        q_in = x_vis.to(device)
        q_tg = recon.detach()

        results = run_attribute(
            model=model,
            target_layer=target_layer,
            error_fn=error_fn,
            query_inputs=q_in,
            query_targets=q_tg,
            index_dir=index_dir,
            index_name="faiss_index_vae_demo",
            meta_name="faiss_metadata_vae_demo.json",
            ckpt_weights=wpath,
            ckpt_opt=opath,
            adam_key=adam_key,
            adam_bias_key=adam_bias_key,
            top_k=args.top_k,
            projection_dim=args.projection_dim,
            projection_type="sjlt",
            projection_seed=44,
            device=device,
        )
        print(f"--- Generated image {im + 1}/{args.num_images} ---")
        for line in format_attribution_lines(results, sample_meta, args.top_k):
            print(line)

    if args.save_grid:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skip --save-grid")
            return

        ncols = min(4, args.num_images)
        nrows = (args.num_images + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = np.atleast_2d(axes)
        for i, (title, timg) in enumerate(to_save):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            ax.imshow(timg.numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")
        for j in range(len(to_save), nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r, c].axis("off")
        outp = os.path.join(base, "generated_grid.png")
        fig.tight_layout()
        fig.savefig(outp, dpi=120)
        plt.close(fig)
        print(f"\nSaved grid → {outp}")


if __name__ == "__main__":
    main()