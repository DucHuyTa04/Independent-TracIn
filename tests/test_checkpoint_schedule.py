"""Tests for benchmark checkpoint spacing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.checkpoint_schedule import evenly_spaced_checkpoint_epochs


def test_forty_epochs_five_checkpoints():
    assert evenly_spaced_checkpoint_epochs(40) == [7, 15, 23, 31, 39]


def test_thirty_epochs_five_checkpoints():
    assert evenly_spaced_checkpoint_epochs(30) == [5, 11, 17, 23, 29]


def test_small_epochs_dedupes():
    assert evenly_spaced_checkpoint_epochs(3) == [0, 1, 2]


def test_zero_epochs():
    assert evenly_spaced_checkpoint_epochs(0) == []
