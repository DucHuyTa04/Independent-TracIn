"""Wall-clock timing and peak memory helpers for benchmark comparisons."""

from __future__ import annotations

import os
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch


@dataclass
class ProfileResult:
    """Result of a timed / memory-profiled benchmark block."""

    wall_time_s: float = 0.0
    peak_memory_mb: float | None = None
    extra: dict = field(default_factory=dict)


@contextmanager
def profile_block(device: str) -> Iterator[ProfileResult]:
    """Measure wall time and peak memory (CUDA if available, else tracemalloc).

    For CUDA, resets peak stats at entry and reads ``max_memory_allocated`` at exit.
    For CPU, uses ``tracemalloc`` peak since start.
    """
    pr = ProfileResult()
    dev = device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"

    use_cuda = dev.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    else:
        tracemalloc.start()

    t0 = time.perf_counter()
    try:
        yield pr
    finally:
        pr.wall_time_s = time.perf_counter() - t0
        if use_cuda:
            torch.cuda.synchronize()
            pr.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            pr.peak_memory_mb = peak / (1024 * 1024)


def file_size_mb(path: str) -> float | None:
    """Return file size in MiB, or None if missing."""
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return None
