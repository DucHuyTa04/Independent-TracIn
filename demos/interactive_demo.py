#!/usr/bin/env python3
"""Unified interactive demo for TracIn Ghost attribution.

Walks through three tasks sequentially:
  1. **Image Classification** (CIFAR-10 CNN)
  2. **Text Generation** (TinyGPT on Shakespeare)
  3. **Image Generation** (Fashion-MNIST VAE)

For each task the user selects a query interactively, the Ghost+FAISS pipeline
runs attribution, and a polished visual shows the query alongside the top-K most
influential training samples with influence percentages.

Pre-train models first with ``demos/pretrain_all.py``.
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
    autoregressive_generate_chars,
    ensure_faiss_index,
    last_ckpt_paths,
    lm_pooled_classification_error,
    resolve_device,
    run_attribute,
    write_demo_config,
)
from demos.visual_utils import (
    ask_continue,
    denormalize_cifar,
    show_attribution_result,
    show_image_selection_grid,
    show_text_prompt_menu,
    tensor_to_gray,
)

# Will be set True by --headless flag
_HEADLESS = False
from src.config_utils import (
    find_adam_bias_param_key,
    find_adam_param_key,
    resolve_target_layer,
    smart_load_weights_into_model,
)
from src.error_functions import get_error_fn

DEFAULT_PROMPTS = [
    "To be or not to be",
    "The king shall",
    "All that glitters",
    "Once upon a time",
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _check_checkpoints(base_dir: str, task_name: str) -> bool:
    meta = os.path.join(base_dir, "checkpoints", "tracin_checkpoints_metadata.json")
    if os.path.isfile(meta):
        return True
    print(f"\n⚠  No pre-trained checkpoints found for {task_name}.")
    print(f"   Expected: {meta}")
    print("   Run  python demos/pretrain_all.py  first.\n")
    return False


def _attribution_to_visuals(
    results: list[dict],
    fetch_visual,
    sample_meta: dict[int, str],
    top_k: int,
):
    """Unpack attribution results into lists of visuals, scores, labels."""
    tops = results[0].get("top_samples", [])[:top_k]
    visuals = [fetch_visual(int(sid)) for sid, _ in tops]
    scores = [float(sc) for _, sc in tops]
    labels = [sample_meta.get(int(sid), "?") for sid, _ in tops]
    return visuals, scores, labels


# ═══════════════════════════════════════════════════════════════════════════
# Task 1 — CIFAR-10 Classification
# ═══════════════════════════════════════════════════════════════════════════

def run_classification(args) -> None:
    print("\n" + "=" * 60)
    print("  Task 1 / 3 — Image Classification (CIFAR-10)")
    print("=" * 60)

    from testModels.medium.cifar10_cnn.data import CifarDataset, make_loaders
    from testModels.medium.cifar10_cnn.model import CifarSmallCNN

    base = os.path.join(ROOT, "demos", "outputs", "cifar10_classification")
    if not _check_checkpoints(base, "CIFAR-10 Classification"):
        return

    device = args.device
    ckpt_dir = os.path.join(base, "checkpoints")
    demo_cfg = os.path.join(base, "demo_config.yaml")

    # Data ------------------------------------------------------------------
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

    # Also build a mapping by original index (FAISS stores these)
    orig_to_label: dict[int, str] = {}
    for j, i in enumerate(idx):
        _, y, _ = full_train[i]
        orig_to_label[i] = CIFAR_CLASSES[int(y)]

    test_ds = CifarDataset(train=False, root=args.data_root)

    # Model -----------------------------------------------------------------
    model = CifarSmallCNN(num_classes=10).to(device)
    write_demo_config(os.path.abspath(ckpt_dir), os.path.abspath(base), demo_cfg)
    w0, _ = last_ckpt_paths(demo_cfg)
    smart_load_weights_into_model(model, w0, device)
    model.eval()

    _, target_layer = resolve_target_layer(model, None)
    error_fn = get_error_fn("classification")
    adam_key = find_adam_param_key(model, target_layer)
    adam_bias_key = find_adam_bias_param_key(model, target_layer)

    # Index -----------------------------------------------------------------
    ensure_faiss_index(
        model=model, target_layer=target_layer, error_fn=error_fn,
        train_loader=train_loader, sample_meta=sample_meta,
        demo_config_path=demo_cfg, index_dir=base,
        index_name="faiss_index_cifar10_demo",
        meta_name="faiss_metadata_cifar10_demo.json",
        projection_dim=args.projection_dim, projection_type="sjlt",
        projection_seed=42, device=device, force=args.force_rebuild,
    )
    wpath, opath = last_ckpt_paths(demo_cfg)

    # Interactive loop ------------------------------------------------------
    n_rounds = 1 if _HEADLESS else 999
    for _round in range(n_rounds):
        # Pick 20 random test images
        perm = torch.randperm(len(test_ds))[:20].tolist()
        imgs = [denormalize_cifar(test_ds[i][0]) for i in perm]
        lbls = [CIFAR_CLASSES[int(test_ds[i][1])] for i in perm]

        if _HEADLESS:
            sel = 0
        else:
            sel = show_image_selection_grid(imgs, lbls, title="CIFAR‑10 — Click a test image to attribute")
            if sel is None:
                print("  No selection — skipping classification task.")
                break

        real_idx = perm[sel]
        x, y, _ = test_ds[real_idx]
        pred = int(torch.argmax(model(x.unsqueeze(0).to(device)), -1).item())

        print(f"  Selected: true={CIFAR_CLASSES[int(y)]}  pred={CIFAR_CLASSES[pred]}")
        print("  Running attribution …")

        results = run_attribute(
            model=model, target_layer=target_layer, error_fn=error_fn,
            query_inputs=x.unsqueeze(0), query_targets=torch.tensor([int(y)]),
            index_dir=base, index_name="faiss_index_cifar10_demo",
            meta_name="faiss_metadata_cifar10_demo.json",
            ckpt_weights=wpath, ckpt_opt=opath,
            adam_key=adam_key, adam_bias_key=adam_bias_key,
            top_k=args.top_k, projection_dim=args.projection_dim,
            projection_type="sjlt", projection_seed=42, device=device,
        )

        def _fetch_train_img(sid: int):
            # sid is the original dataset index stored by FAISS
            img_t, _, _ = full_train[sid]
            return denormalize_cifar(img_t)

        top_vis, top_scores, top_labels = _attribution_to_visuals(
            results, _fetch_train_img, orig_to_label, args.top_k,
        )

        query_vis = denormalize_cifar(x)
        save = os.path.join(base, "attribution_classification.png") if args.save_figures else None
        show_attribution_result(
            query_vis, top_vis, top_scores, top_labels,
            task_title=f"Classification — true: {CIFAR_CLASSES[int(y)]}, pred: {CIFAR_CLASSES[pred]}",
            save_path=save,
        )

        if _HEADLESS or not ask_continue("Classification"):
            break


# ═══════════════════════════════════════════════════════════════════════════
# Task 2 — Text Generation
# ═══════════════════════════════════════════════════════════════════════════

def run_text_generation(args) -> None:
    print("\n" + "=" * 60)
    print("  Task 2 / 3 — Text Generation (TinyGPT + Shakespeare)")
    print("=" * 60)

    from testModels.medium.transformer_lm.data import CharLMDataset
    from testModels.medium.transformer_lm.model import TinyGPT

    base = os.path.join(ROOT, "demos", "outputs", "tinygpt_demo")
    if not _check_checkpoints(base, "TinyGPT Text Generation"):
        return

    device = args.device
    ckpt_dir = os.path.join(base, "checkpoints")
    demo_cfg = os.path.join(base, "demo_config.yaml")

    # Data ------------------------------------------------------------------
    full_train = CharLMDataset(root=args.data_root, train=True)
    vocab_size = full_train.vocab_size
    g = torch.Generator().manual_seed(42)
    n_train = min(args.n_train, len(full_train))
    indices = torch.randperm(len(full_train), generator=g)[:n_train].tolist()
    base_ds = _ReindexedSubset(full_train, indices)
    train_loader = DataLoader(base_ds, batch_size=32, shuffle=False, num_workers=0)
    sample_meta = {i: f"seq_{i}" for i in range(n_train)}

    # Model -----------------------------------------------------------------
    model = TinyGPT(vocab_size=vocab_size).to(device)
    write_demo_config(os.path.abspath(ckpt_dir), os.path.abspath(base), demo_cfg)
    w0, _ = last_ckpt_paths(demo_cfg)
    smart_load_weights_into_model(model, w0, device)
    model.eval()

    _, target_layer = resolve_target_layer(model, "output_proj")
    adam_key = find_adam_param_key(model, target_layer)
    adam_bias_key = find_adam_bias_param_key(model, target_layer)

    # Index -----------------------------------------------------------------
    ensure_faiss_index(
        model=model, target_layer=target_layer,
        error_fn=lm_pooled_classification_error,
        train_loader=train_loader, sample_meta=sample_meta,
        demo_config_path=demo_cfg, index_dir=base,
        index_name="faiss_index_tinygpt_demo",
        meta_name="faiss_metadata_tinygpt_demo.json",
        projection_dim=args.projection_dim, projection_type="sjlt",
        projection_seed=43, device=device, force=args.force_rebuild,
    )
    wpath, opath = last_ckpt_paths(demo_cfg)

    # Helper to get snippet text from training sample id
    def _snippet(sid: int, max_chars: int = 80) -> str:
        x, _, _ = base_ds[int(sid)]
        s = "".join(full_train.itos[int(t)] for t in x.tolist())
        return s[:max_chars].replace("\n", "↵ ")

    def _encode(text: str) -> torch.Tensor:
        ids = [full_train.stoi[c] for c in text if c in full_train.stoi]
        if not ids:
            ids = [0]
        return torch.tensor([ids], dtype=torch.long, device=device)

    # Interactive loop ------------------------------------------------------
    n_rounds = 1 if _HEADLESS else 999
    for _round in range(n_rounds):
        prompt_text = DEFAULT_PROMPTS[0] if _HEADLESS else show_text_prompt_menu(DEFAULT_PROMPTS, allow_custom=True)
        print(f"\n  Prompt: \"{prompt_text}\"")
        print("  Generating text …")

        ctx = _encode(prompt_text)
        gen_ids = autoregressive_generate_chars(
            model, ctx.cpu(), max_new_tokens=100, temperature=0.8,
            vocab_size=vocab_size, device=device,
        )
        gen_cpu = gen_ids[0]
        text_out = "".join(full_train.itos[int(i)] for i in gen_cpu.tolist())
        print(f"\n  Generated:\n  {text_out[:300]}\n")

        if gen_cpu.numel() < 2:
            print("  Generation too short for attribution.")
            if _HEADLESS or not ask_continue("Text Generation"):
                break
            continue

        print("  Running attribution …")
        # Truncate to model context length for attribution (pos_emb is fixed-size)
        max_t = getattr(model, "ctx_len", gen_cpu.numel() - 1)
        gen_trunc = gen_cpu[-(max_t + 1):]
        q_in = gen_trunc.unsqueeze(0)[:, :-1].long()
        q_tg = gen_trunc.unsqueeze(0)[:, 1:].long()

        results = run_attribute(
            model=model, target_layer=target_layer,
            error_fn=lm_pooled_classification_error,
            query_inputs=q_in, query_targets=q_tg,
            index_dir=base, index_name="faiss_index_tinygpt_demo",
            meta_name="faiss_metadata_tinygpt_demo.json",
            ckpt_weights=wpath, ckpt_opt=opath,
            adam_key=adam_key, adam_bias_key=adam_bias_key,
            top_k=args.top_k, projection_dim=args.projection_dim,
            projection_type="sjlt", projection_seed=43, device=device,
        )

        top_vis, top_scores, top_labels = _attribution_to_visuals(
            results, _snippet, sample_meta, args.top_k,
        )

        save = os.path.join(base, "attribution_text_gen.png") if args.save_figures else None
        show_attribution_result(
            text_out[:200], top_vis, top_scores, top_labels,
            task_title="Text Generation — Training Data Attribution",
            save_path=save,
        )

        if _HEADLESS or not ask_continue("Text Generation"):
            break


# ═══════════════════════════════════════════════════════════════════════════
# Task 3 — Image Generation (VAE)
# ═══════════════════════════════════════════════════════════════════════════

def run_image_generation(args) -> None:
    print("\n" + "=" * 60)
    print("  Task 3 / 3 — Image Generation (Fashion-MNIST VAE)")
    print("=" * 60)

    from testModels.medium.vae_fashion.data import FashionMnistDataset
    from testModels.medium.vae_fashion.model import FashionVAE

    base = os.path.join(ROOT, "demos", "outputs", "vae_fashion_demo")
    if not _check_checkpoints(base, "Fashion-MNIST VAE"):
        return

    device = args.device
    ckpt_dir = os.path.join(base, "checkpoints")
    demo_cfg = os.path.join(base, "demo_config.yaml")

    # Data ------------------------------------------------------------------
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

    # Also build mapping by original index (FAISS stores these)
    orig_to_label: dict[int, str] = {}
    for j, i in enumerate(idx):
        _, lbl = full_train.ds[i]
        orig_to_label[i] = FASHION_LABELS[int(lbl)]

    # Model -----------------------------------------------------------------
    model = FashionVAE().to(device)
    write_demo_config(os.path.abspath(ckpt_dir), os.path.abspath(base), demo_cfg)
    w0, _ = last_ckpt_paths(demo_cfg)
    smart_load_weights_into_model(model, w0, device)
    model.eval()

    _, target_layer = resolve_target_layer(model, "dec_out")
    error_fn = get_error_fn("regression")
    adam_key = find_adam_param_key(model, target_layer)
    adam_bias_key = find_adam_bias_param_key(model, target_layer)

    # Index -----------------------------------------------------------------
    ensure_faiss_index(
        model=model, target_layer=target_layer, error_fn=error_fn,
        train_loader=train_loader, sample_meta=sample_meta,
        demo_config_path=demo_cfg, index_dir=base,
        index_name="faiss_index_vae_demo",
        meta_name="faiss_metadata_vae_demo.json",
        projection_dim=args.projection_dim, projection_type="sjlt",
        projection_seed=44, device=device, force=args.force_rebuild,
    )
    wpath, opath = last_ckpt_paths(demo_cfg)

    def _fetch_train_img(sid: int) -> "np.ndarray":
        # sid is the original dataset index stored by FAISS
        img_t, _, _ = full_train[sid]
        return tensor_to_gray(img_t)

    # Interactive loop ------------------------------------------------------
    n_rounds = 1 if _HEADLESS else 999
    for _round in range(n_rounds):
        # Generate a grid of images from random latents
        n_gen = 12
        gen_rng = torch.Generator(device="cpu")
        latents = torch.randn(n_gen, model.latent_dim, generator=gen_rng).to(device)
        with torch.no_grad():
            recons = model.decode(latents)
        gen_imgs = [
            tensor_to_gray(recons[i].view(1, 28, 28).clamp(0, 1).cpu())
            for i in range(n_gen)
        ]
        gen_labels = [f"Generated #{i + 1}" for i in range(n_gen)]

        if _HEADLESS:
            sel = 0
        else:
            sel = show_image_selection_grid(
                gen_imgs, gen_labels,
                title="Fashion VAE — Click a generated image to attribute",
                n_cols=4,
            )
            if sel is None:
                print("  No selection — skipping image generation task.")
                break

        print(f"  Selected generated image #{sel + 1}")
        print("  Running attribution …")

        x_vis = recons[sel].view(1, 1, 28, 28).clamp(0, 1).cpu()
        q_in = x_vis.to(device)
        q_tg = recons[sel].detach().unsqueeze(0)

        results = run_attribute(
            model=model, target_layer=target_layer, error_fn=error_fn,
            query_inputs=q_in, query_targets=q_tg,
            index_dir=base, index_name="faiss_index_vae_demo",
            meta_name="faiss_metadata_vae_demo.json",
            ckpt_weights=wpath, ckpt_opt=opath,
            adam_key=adam_key, adam_bias_key=adam_bias_key,
            top_k=args.top_k, projection_dim=args.projection_dim,
            projection_type="sjlt", projection_seed=44, device=device,
        )

        top_vis, top_scores, top_labels = _attribution_to_visuals(
            results, _fetch_train_img, orig_to_label, args.top_k,
        )

        query_vis = tensor_to_gray(x_vis.squeeze(0))
        save = os.path.join(base, "attribution_image_gen.png") if args.save_figures else None
        show_attribution_result(
            query_vis, top_vis, top_scores, top_labels,
            task_title="Image Generation — Training Data Attribution",
            save_path=save,
        )

        if _HEADLESS or not ask_continue("Image Generation"):
            break


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Interactive TracIn Ghost attribution demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Presentation flow:\n"
            "  1. Run demos/pretrain_all.py once to prepare models + indices.\n"
            "  2. Run this script live during the demo.\n"
            "  3. Each task shows an interactive selection → attribution → visual result.\n"
        ),
    )
    p.add_argument("--device", default="cuda", help="cuda | cpu | auto (default: cuda)")
    p.add_argument("--data-root", default="data", help="Root for torchvision datasets")
    p.add_argument("--top-k", type=int, default=5, help="Number of top influential samples")
    p.add_argument("--max-train", type=int, default=8000, help="Max training subset (classification & VAE)")
    p.add_argument("--n-train", type=int, default=8000, help="Training sequences for text gen")
    p.add_argument("--projection-dim", type=int, default=512)
    p.add_argument("--skip-tasks", default="", help="Comma-separated tasks to skip: classification,text,image")
    p.add_argument("--save-figures", action="store_true", help="Save attribution figures as PNG")
    p.add_argument("--force-rebuild", action="store_true", help="Force rebuild FAISS indices")
    p.add_argument("--headless", action="store_true",
                   help="Non-interactive mode: auto-select queries, save figures, no display")
    args = p.parse_args()

    global _HEADLESS
    if args.headless:
        _HEADLESS = True
        args.save_figures = True
        import matplotlib
        matplotlib.use("Agg")

    args.device = resolve_device(args.device)
    skip = set(s.strip().lower() for s in args.skip_tasks.split(",") if s.strip())

    print("╔══════════════════════════════════════════════════════════╗")
    print("║      TracIn Ghost — Interactive Attribution Demo        ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  For each task:                                        ║")
    print("║    • Select a query (click image / choose prompt)      ║")
    print("║    • See which training samples influenced it most     ║")
    print("║    • Visualise the influence distribution               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Device: {args.device}  |  Top-k: {args.top_k}")

    if "classification" not in skip:
        run_classification(args)
    else:
        print("\n  [Skipped] Classification")

    if "text" not in skip:
        run_text_generation(args)
    else:
        print("\n  [Skipped] Text Generation")

    if "image" not in skip:
        run_image_generation(args)
    else:
        print("\n  [Skipped] Image Generation")

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
