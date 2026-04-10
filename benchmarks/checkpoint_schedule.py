"""Evenly spaced checkpoint epoch indices for TracIn-style benchmarks."""

from __future__ import annotations


def evenly_spaced_checkpoint_epochs(num_epochs: int, num_checkpoints: int = 5) -> list[int]:
    """Return 0-based epoch indices (after each full pass) at which to save weights.

    Uses ``spacing = max(1, num_epochs // num_checkpoints)`` and saves at
    ``min(num_epochs - 1, (i + 1) * spacing - 1)`` for ``i`` in ``0 .. num_checkpoints-1``,
    deduplicating adjacent ties for small ``num_epochs``.
    """
    if num_epochs <= 0:
        return []
    spacing = max(1, num_epochs // num_checkpoints)
    out: list[int] = []
    for i in range(num_checkpoints):
        e = min(num_epochs - 1, (i + 1) * spacing - 1)
        if not out or e != out[-1]:
            out.append(e)
    return out
