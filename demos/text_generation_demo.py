#!/usr/bin/env python3
"""Tiny Shakespeare char-LM demo: generate text, then TracIn attribution."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from demos.demo_utils import (
    ReindexedSubset,
    autoregressive_generate_chars,
    ensure_faiss_index,
    format_attribution_lines,
    last_ckpt_paths,
    lm_pooled_classification_error,
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
from testModels.medium.transformer_lm.data import CharLMDataset
from testModels.medium.transformer_lm.model import TinyGPT


def _lm_ce_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    b, t, v = logits.shape
    return nn.functional.cross_entropy(logits.reshape(-1, v), y.reshape(-1))


def encode_prompt(ds: CharLMDataset, text: str, device: str) -> torch.Tensor:
    ids = [ds.stoi[c] for c in text if c in ds.stoi]
    if not ids:
        ids = [0]
    return torch.tensor([ids], dtype=torch.long, device=device)


def main() -> None:
    p = argparse.ArgumentParser(description="TinyGPT TracIn Ghost attribution demo")
    p.add_argument("--device", default="cuda", help="cuda | cpu | auto (default: cuda)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--prompt", default="To be or not to be")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--n-train", type=int, default=8000)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--projection-dim", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--force-reindex", action="store_true")
    args = p.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    base = os.path.join(ROOT, "demos", "outputs", "tinygpt_demo")
    ckpt_dir = os.path.join(base, "checkpoints")
    index_dir = base
    demo_cfg = os.path.join(base, "demo_config.yaml")
    os.makedirs(base, exist_ok=True)

    full_train = CharLMDataset(root=args.data_root, train=True)
    vocab_size = full_train.vocab_size
    n_avail = len(full_train)
    n_train = min(args.n_train, n_avail)
    g = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(n_avail, generator=g)[:n_train].tolist()
    base_ds = ReindexedSubset(full_train, indices)
    train_loader = DataLoader(base_ds, batch_size=32, shuffle=False, num_workers=0)

    sample_meta = {i: f"seq_{i}" for i in range(n_train)}

    meta_path = os.path.join(ckpt_dir, "tracin_checkpoints_metadata.json")
    need_train = args.force_retrain or not os.path.isfile(meta_path)

    model = TinyGPT(vocab_size=vocab_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if need_train:
        print("Training TinyGPT (subset) …")
        train_with_tracin_checkpoints(
            model, opt, train_loader, _lm_ce_loss, ckpt_dir, args.epochs, device,
            save_every=max(1, args.epochs // 5),
        )
    else:
        print("Skipping training. Use --force-retrain to retrain.")

    write_demo_config(os.path.abspath(ckpt_dir), os.path.abspath(index_dir), demo_cfg)
    if not need_train:
        w0, _ = last_ckpt_paths(demo_cfg)
        smart_load_weights_into_model(model, w0, device)

    _, target_layer = resolve_target_layer(model, "output_proj")
    adam_key = find_adam_param_key(model, target_layer)
    adam_bias_key = find_adam_bias_param_key(model, target_layer)

    ensure_faiss_index(
        model=model,
        target_layer=target_layer,
        error_fn=lm_pooled_classification_error,
        train_loader=train_loader,
        sample_meta=sample_meta,
        demo_config_path=demo_cfg,
        index_dir=index_dir,
        index_name="faiss_index_tinygpt_demo",
        meta_name="faiss_metadata_tinygpt_demo.json",
        projection_dim=args.projection_dim,
        projection_type="sjlt",
        projection_seed=43,
        device=device,
        force=args.force_reindex or need_train,
    )

    wpath, opath = last_ckpt_paths(demo_cfg)
    model.eval()

    ctx = encode_prompt(full_train, args.prompt, device)
    gen_ids = autoregressive_generate_chars(
        model, ctx.cpu(), args.max_new_tokens, args.temperature, vocab_size, device,
    )
    gen_cpu = gen_ids[0]
    chars = [full_train.itos[int(i)] for i in gen_cpu.tolist()]
    text_out = "".join(chars)
    print("\n=== Generated text ===\n", text_out[:500], ("…" if len(text_out) > 500 else ""))

    if gen_cpu.numel() < 2:
        print("Generation too short for attribution; need >= 2 tokens.")
        return
    # Truncate to model context length for attribution (pos_emb is fixed-size)
    max_t = getattr(model, "ctx_len", gen_cpu.numel() - 1)
    gen_trunc = gen_cpu[-(max_t + 1):]  # keep last ctx_len+1 tokens
    q_in = gen_trunc.unsqueeze(0)[:, :-1].long()
    q_tg = gen_trunc.unsqueeze(0)[:, 1:].long()

    results = run_attribute(
        model=model,
        target_layer=target_layer,
        error_fn=lm_pooled_classification_error,
        query_inputs=q_in,
        query_targets=q_tg,
        index_dir=index_dir,
        index_name="faiss_index_tinygpt_demo",
        meta_name="faiss_metadata_tinygpt_demo.json",
        ckpt_weights=wpath,
        ckpt_opt=opath,
        adam_key=adam_key,
        adam_bias_key=adam_bias_key,
        top_k=args.top_k,
        projection_dim=args.projection_dim,
        projection_type="sjlt",
        projection_seed=43,
        device=device,
    )

    def snippet(sid: int) -> str:
        x, _, _ = base_ds[int(sid)]
        s = "".join(full_train.itos[int(t)] for t in x.tolist())
        return s[:80].replace("\n", " ")

    print("\n=== Top influential training sequences (by index) ===")
    for line in format_attribution_lines(results, sample_meta, args.top_k):
        print(line)
    tops = results[0].get("top_samples", [])[: args.top_k]
    print("\nSnippets:")
    for rank, (sid, sc) in enumerate(tops, 1):
        print(f"  {rank}. id={sid} score={sc:.4f}  «{snippet(int(sid))}»")


if __name__ == "__main__":
    main()
