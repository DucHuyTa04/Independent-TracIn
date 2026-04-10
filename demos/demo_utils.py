"""Shared helpers for TracIn Ghost demos."""

from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config_utils import (
    TracInCheckpointCallback,
    find_adam_bias_param_key,
    find_adam_param_key,
    resolve_checkpoints,
    smart_load_weights_into_model,
)
from src.error_functions import classification_error
from src.indexer import build_index
from src.inference import attribute

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

CIFAR_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

FASHION_LABELS = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


class ReindexedSubset(Dataset):
    """Wrap a Subset (or dataset + indices) so ``__getitem__`` returns ``(x, y, new_index)``."""

    def __init__(self, dataset: Dataset, indices=None) -> None:
        if indices is not None:
            from torch.utils.data import Subset
            dataset = Subset(dataset, indices)
        self.subset = dataset

    def __len__(self) -> int:
        return len(self.subset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        item = self.subset[idx]
        # Underlying dataset may return (x, y) or (x, y, old_idx).
        # Either way, re-index with the sequential idx.
        return item[0], item[1], idx


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Demos require a GPU — run on a compute node "
            "(e.g. via Slurm) or pass --device cpu (not recommended for text generation)."
        )
    return device


def write_demo_config(ckpt_dir: str, outputs_dir: str, config_path: str) -> None:
    """Minimal YAML so ``resolve_checkpoints`` finds ``checkpoints_dir``."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(
            "paths:\n"
            f"  checkpoints_dir: {ckpt_dir}\n"
            f"  outputs_dir: {outputs_dir}\n"
        )


def checkpoints_from_demo_config(config_path: str) -> list[dict[str, Any]]:
    return resolve_checkpoints(os.path.abspath(config_path), {})


def train_with_tracin_checkpoints(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    loss_step: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    ckpt_dir: str,
    epochs: int,
    device: str,
    save_every: int = 1,
    patience: int = 15,
    min_rel_delta: float = 1e-4,
) -> None:
    """Train up to ``epochs`` with early stopping on convergence.

    Stops early when the relative loss improvement stays below
    ``min_rel_delta`` for ``patience`` consecutive epochs.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    cb = TracInCheckpointCallback(save_dir=ckpt_dir, save_every=max(1, save_every))
    model.train()
    avg = 0.0
    best_loss = float("inf")
    wait = 0
    final_epoch = 0
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for batch in train_loader:
            inputs, targets, _ = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_step(model, inputs, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            n += 1
        avg = total / max(n, 1)
        cb.on_epoch_end(epoch, model, optimizer, avg)
        final_epoch = epoch

        # convergence check
        rel_improvement = (best_loss - avg) / max(abs(best_loss), 1e-8)
        if avg < best_loss:
            best_loss = avg
        if rel_improvement < min_rel_delta:
            wait += 1
        else:
            wait = 0

        status = f"  epoch {epoch + 1}/{epochs}  loss={avg:.4f}"
        if wait > 0:
            status += f"  (plateau {wait}/{patience})"
        print(status)

        if wait >= patience:
            print(f"  ✓ Converged after {epoch + 1} epochs (loss plateau for {patience} epochs)")
            break
    else:
        print(f"  ✓ Completed all {epochs} epochs")
    cb.finalize(model, optimizer, final_epoch, avg)


def ensure_faiss_index(
    model: nn.Module,
    target_layer: nn.Module,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train_loader: DataLoader,
    sample_meta: dict[int, str],
    demo_config_path: str,
    index_dir: str,
    index_name: str,
    meta_name: str,
    projection_dim: int | None,
    projection_type: str,
    projection_seed: int,
    device: str,
    force: bool,
) -> None:
    index_path = os.path.join(index_dir, index_name)
    meta_path = os.path.join(index_dir, meta_name)
    if not force and os.path.isfile(index_path) and os.path.isfile(meta_path):
        print(f"Using existing index: {index_path}")
        return
    os.makedirs(index_dir, exist_ok=True)
    ckpts = checkpoints_from_demo_config(demo_config_path)
    adam_key = find_adam_param_key(model, target_layer)
    adam_bias_key = find_adam_bias_param_key(model, target_layer)
    build_index(
        model=model,
        target_layer=target_layer,
        error_fn=error_fn,
        data_loader=train_loader,
        checkpoints=ckpts,
        sample_metadata=sample_meta,
        projection_dim=projection_dim,
        projection_type=projection_type,
        projection_seed=projection_seed,
        adam_param_key=adam_key,
        adam_bias_param_key=adam_bias_key,
        output_dir=index_dir,
        index_filename=index_name,
        metadata_filename=meta_name,
        device=device,
    )
    print(f"Built index → {index_path}")


def run_attribute(
    model: nn.Module,
    target_layer: nn.Module,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    query_inputs: torch.Tensor,
    query_targets: torch.Tensor,
    index_dir: str,
    index_name: str,
    meta_name: str,
    ckpt_weights: str,
    ckpt_opt: str | None,
    adam_key: int,
    adam_bias_key: int | None,
    top_k: int,
    projection_dim: int | None,
    projection_type: str,
    projection_seed: int,
    device: str,
) -> list[dict[str, Any]]:
    return attribute(
        model=model,
        target_layer=target_layer,
        error_fn=error_fn,
        query_inputs=query_inputs,
        query_targets=query_targets,
        index_path=os.path.join(index_dir, index_name),
        metadata_path=os.path.join(index_dir, meta_name),
        checkpoint_path=ckpt_weights,
        projection_dim=projection_dim,
        projection_type=projection_type,
        projection_seed=projection_seed,
        optimizer_state_path=ckpt_opt,
        adam_param_key=adam_key,
        adam_bias_param_key=adam_bias_key,
        top_k=top_k,
        device=device,
    )


def last_ckpt_paths(demo_config_path: str) -> tuple[str, str | None]:
    ckpts = checkpoints_from_demo_config(demo_config_path)
    last = ckpts[-1]
    return last["weights_path"], last.get("optimizer_state_path")


def format_attribution_lines(
    results: list[dict],
    sample_meta: dict[int, str],
    top_k: int,
) -> list[str]:
    lines: list[str] = []
    for qi, r in enumerate(results):
        lines.append(f"\n--- Query {qi + 1} ---")
        tops = r.get("top_samples", [])[:top_k]
        rh = r.get("rights_holder_attribution", {})
        if rh:
            lines.append("Rights-holder / label mix:")
            for h, p in sorted(rh.items(), key=lambda x: -x[1]):
                lines.append(f"  {h}: {100.0 * p:.2f}%")
        lines.append(f"Top-{len(tops)} training samples (raw FAISS score → label):")
        for rank, (sid, score) in enumerate(tops, start=1):
            lab = sample_meta.get(int(sid), "?")
            lines.append(f"  {rank}. id={sid}  score={score:.4f}  {lab}")
    return lines


def lm_pooled_classification_error(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Match mean-pooled activations from HookManager on (B,T,H) inputs.

    ``logits`` (B, T, V), ``targets`` (B, T) long.
    Returns E of shape (B, V).
    """
    b, t, v = logits.shape
    e = classification_error(
        logits.reshape(-1, v),
        targets.reshape(-1),
    )
    return e.view(b, t, v).mean(dim=1)


def autoregressive_generate_chars(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    vocab_size: int,
    device: str,
) -> torch.Tensor:
    """Greedy / temperature sampling; ``prompt_ids`` shape (1, T). Returns (1, T+max_new_tokens)."""
    model.eval()
    ctx_len = getattr(model, "ctx_len", prompt_ids.shape[1])
    ctx = prompt_ids.to(device).long()
    for _ in range(max_new_tokens):
        # Truncate to model's max context length to avoid pos_emb OOB
        ctx_in = ctx[:, -ctx_len:]
        logits = model(ctx_in)  # (1, t, V)
        next_logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(next_logits, dim=-1)
        if temperature <= 0:
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_id = torch.multinomial(probs, num_samples=1)
        ctx = torch.cat([ctx, next_id], dim=1)
        if int(next_id.item()) >= vocab_size:
            break
    return ctx.cpu()
