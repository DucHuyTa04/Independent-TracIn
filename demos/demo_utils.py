"""Shared helpers for TracIn Ghost demos."""

from __future__ import annotations

import os
import shutil
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config_utils import (
    METADATA_FILENAME,
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
            "CUDA is not available. Use --device auto (CPU inference is fine after "
            "pretraining) or --device cpu. For long training runs, use a GPU node."
        )
    return device


def _clean_checkpoint_dir(ckpt_dir: str) -> None:
    """Remove stale ckpt_*.pt, metadata, and .pruned/ before a fresh training run."""
    if not os.path.isdir(ckpt_dir):
        return
    pruned = os.path.join(ckpt_dir, ".pruned")
    if os.path.isdir(pruned):
        shutil.rmtree(pruned, ignore_errors=True)
    meta = os.path.join(ckpt_dir, METADATA_FILENAME)
    if os.path.isfile(meta):
        try:
            os.remove(meta)
        except OSError:
            pass
    for name in os.listdir(ckpt_dir):
        if name.startswith("ckpt_") and name.endswith(".pt"):
            try:
                os.remove(os.path.join(ckpt_dir, name))
            except OSError:
                pass


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
    min_rel_delta: float = 5e-4,
) -> None:
    """Train up to ``epochs`` with cosine LR annealing and early stopping.

    Uses ``CosineAnnealingLR`` to smoothly decay the learning rate to near-zero,
    and stops early when the relative loss improvement over the last
    ``patience`` epochs drops below ``min_rel_delta``.
    """
    _clean_checkpoint_dir(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    cb = TracInCheckpointCallback(save_dir=ckpt_dir, save_every=max(1, save_every))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5,
    )
    model.train()
    avg = 0.0
    loss_history: list[float] = []
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
        scheduler.step()
        cb.on_epoch_end(epoch, model, optimizer, avg)
        final_epoch = epoch
        loss_history.append(avg)

        # convergence: compare current loss to loss `patience` epochs ago
        converged = False
        if len(loss_history) > patience:
            past = loss_history[-(patience + 1)]
            window_rel = (past - avg) / max(abs(past), 1e-8)
            converged = window_rel < min_rel_delta

        status = f"  epoch {epoch + 1}/{epochs}  loss={avg:.4f}"
        if len(loss_history) > patience and not converged:
            past = loss_history[-(patience + 1)]
            window_pct = 100.0 * (past - avg) / max(abs(past), 1e-8)
            status += f"  (window Δ {window_pct:.2f}%)"
        print(status)

        if converged:
            print(f"  ✓ Converged after {epoch + 1} epochs "
                  f"(< {min_rel_delta:.1%} improvement over {patience} epochs)")
            break
    else:
        print(f"  ✓ Completed all {epochs} epochs")
    cb.finalize(model, optimizer, final_epoch, avg)
    cb.select_best(keep=7)


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
    n_train: int | None = None,
) -> list[dict[str, Any]]:
    """Run attribution. If ``n_train`` is set, FAISS is queried for all ``n_train``
    vectors so ``total_positive_score`` (sum of positive inner products) reflects
    the full indexed training set; ``top_samples`` is still trimmed to ``top_k``.
    """
    effective_k = (
        max(int(n_train), int(top_k)) if n_train is not None else int(top_k)
    )
    raw = attribute(
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
        top_k=effective_k,
        device=device,
    )
    if n_train is not None:
        for r in raw:
            tops = list(r.get("top_samples", []))
            total_positive_score = sum(max(0.0, float(sc)) for _, sc in tops)
            r["total_positive_score"] = total_positive_score
            r["top_samples"] = tops[: int(top_k)]
    return raw


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
        ctx_in = ctx[:, -ctx_len:].clamp(0, vocab_size - 1)
        logits = model(ctx_in)  # (1, t, V)
        next_logits = logits[:, -1, :vocab_size] / max(temperature, 1e-6)
        probs = torch.softmax(next_logits, dim=-1)
        # Guard against NaN/inf from undertrained models
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.ones_like(probs) / probs.shape[-1]
        if temperature <= 0:
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_id = torch.multinomial(probs, num_samples=1)
        next_id = next_id.clamp(0, vocab_size - 1)
        ctx = torch.cat([ctx, next_id], dim=1)
    return ctx.cpu()
