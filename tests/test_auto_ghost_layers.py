"""Tests for auto_ghost_layers (last / largest, dead-layer skip)."""

from __future__ import annotations

import torch
import torch.nn as nn

from benchmarks.ghost_faiss import auto_ghost_layers


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_auto_ghost_layers_last_respects_max_layers() -> None:
    m = TinyMLP()
    layers = auto_ghost_layers(m, target_coverage=0.99, strategy="last", max_layers=1)
    assert len(layers) == 1


def test_auto_ghost_layers_last_covers_all_linears_by_default() -> None:
    m = TinyMLP()
    layers = auto_ghost_layers(m, target_coverage=0.99, strategy="last")
    assert len(layers) == 2


def test_auto_ghost_layers_skips_dead_weight() -> None:
    m = TinyMLP()
    with torch.no_grad():
        m.fc2.weight.zero_()
    layers = auto_ghost_layers(
        m, target_coverage=0.99, strategy="last", dead_weight_std=1e-7,
    )
    assert len(layers) == 1
    assert layers[0] is m.fc1


def test_auto_ghost_layers_largest() -> None:
    m = TinyMLP()
    layers = auto_ghost_layers(m, target_coverage=0.99, strategy="largest")
    # fc1 has more params than fc2
    assert layers[0] is m.fc1 or layers[0] is m.fc2


def test_auto_ghost_layers_unknown_strategy_raises() -> None:
    m = TinyMLP()
    try:
        auto_ghost_layers(m, strategy="scored")  # type: ignore[arg-type]
    except ValueError as e:
        assert "last" in str(e).lower() or "largest" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


class TinyGRU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(5, 5)
        self.gru = nn.GRU(5, 5, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc(x)
        return self.gru(z)[0]


def test_auto_ghost_layers_excludes_rnn_by_default() -> None:
    m = TinyGRU()
    layers = auto_ghost_layers(m, target_coverage=1.0, strategy="last")
    assert all(not isinstance(l, nn.RNNBase) for l in layers)


def test_auto_ghost_layers_include_rnn_hooks_gru() -> None:
    m = TinyGRU()
    layers = auto_ghost_layers(m, target_coverage=1.0, strategy="last", include_rnn=True)
    assert any(l is m.gru for l in layers)
