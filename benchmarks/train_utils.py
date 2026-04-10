"""Shared training loop with convergence detection for benchmarks."""

from __future__ import annotations

import os
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_with_convergence(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    data_loader: DataLoader,
    device: str,
    ckpt_dir: str,
    max_epochs: int = 10000,
    num_checkpoints: int = 5,
    patience: int = 15,
    min_rel_delta: float = 1e-4,
) -> list[int]:
    """Train until convergence, saving evenly-spaced checkpoints.

    Args:
        loss_fn: ``loss_fn(model, x, y) -> scalar loss`` — handles forward
            pass and loss computation.
        max_epochs: Safety cap (early stopping should trigger first).
        num_checkpoints: How many checkpoints to retain after training.
        patience: Stop after this many epochs without relative improvement.
        min_rel_delta: Minimum relative loss drop to count as improvement.

    Returns:
        Sorted list of retained checkpoint epoch numbers.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    save_every = max(1, patience // 3)

    best_loss = float("inf")
    wait = 0
    saved_epochs: list[int] = []
    final_epoch = 0

    model.train()
    for e in range(max_epochs):
        total = 0.0
        n = 0
        for x, y, _ in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model, x, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        avg = total / max(n, 1)
        final_epoch = e

        # save checkpoint periodically
        if e % save_every == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"ckpt_{e}.pt"))
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, f"optim_{e}.pt"))
            saved_epochs.append(e)

        # convergence check
        rel_improvement = (best_loss - avg) / max(abs(best_loss), 1e-8)
        if avg < best_loss:
            best_loss = avg
        if rel_improvement < min_rel_delta:
            wait += 1
        else:
            wait = 0

        status = f"  epoch {e + 1}  loss={avg:.4f}"
        if wait > 0:
            status += f"  (plateau {wait}/{patience})"
        print(status)

        if wait >= patience:
            print(
                f"  \u2713 Converged after {e + 1} epochs "
                f"(loss plateau for {patience} epochs)"
            )
            break
    else:
        print(f"  \u2713 Completed all {max_epochs} epochs")

    # always save the final epoch
    if final_epoch not in saved_epochs:
        torch.save(
            model.state_dict(), os.path.join(ckpt_dir, f"ckpt_{final_epoch}.pt")
        )
        torch.save(
            optimizer.state_dict(), os.path.join(ckpt_dir, f"optim_{final_epoch}.pt")
        )
        saved_epochs.append(final_epoch)

    # select evenly-spaced checkpoints
    selected = _select_checkpoints(sorted(saved_epochs), num_checkpoints)

    # clean up unselected
    for se in saved_epochs:
        if se not in selected:
            for fname in (f"ckpt_{se}.pt", f"optim_{se}.pt"):
                p = os.path.join(ckpt_dir, fname)
                if os.path.exists(p):
                    os.remove(p)

    return sorted(selected)


def _select_checkpoints(saved: list[int], num_ckpts: int) -> set[int]:
    """Pick *num_ckpts* evenly-spaced epochs from *saved*, always keeping last."""
    if len(saved) <= num_ckpts:
        return set(saved)
    result = {saved[-1]}
    remaining = num_ckpts - 1
    if remaining > 0:
        step = (len(saved) - 1) / remaining
        for i in range(remaining):
            idx = round(i * step)
            result.add(saved[idx])
    return result
