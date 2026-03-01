import json
import os
from typing import Dict

import torch


def save_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Save model state dict to a checkpoint file."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def save_meta(meta_path: str, payload: Dict) -> None:
    """Save JSON metadata for a checkpoint."""
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    """Load model state dict from a checkpoint file."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def read_learning_rate(meta_path: str) -> float:
    """Read learning rate from checkpoint metadata JSON."""
    with open(meta_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    return float(payload["learning_rate"])
