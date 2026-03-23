"""Build FAISS index from MNIST training data using src/ library.

Usage:
    python testModels/mnist/run_index.py [--config testModels/mnist/config.yaml]
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from testModels.mnist.model import MnistMLP
from testModels.mnist.data import make_loaders
from src.indexer import build_index


def classification_error(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """E = softmax(logits) - one_hot(targets)."""
    probs = torch.softmax(logits, dim=-1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, targets.unsqueeze(1).long(), 1.0)
    return probs - one_hot


def main() -> None:
    parser = argparse.ArgumentParser(description="Index MNIST training data")
    parser.add_argument("--config", default="testModels/mnist/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model = MnistMLP(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_classes=cfg["model"]["num_classes"],
    )

    train_loader, _, sample_meta = make_loaders(
        batch_size=cfg["dataset"]["batch_size"],
        data_root=cfg["paths"]["data_root"],
    )

    checkpoints = []
    for ckpt in cfg["checkpoints"]:
        entry = {
            "weights_path": ckpt["weights_path"],
            "learning_rate": ckpt["learning_rate"],
        }
        if ckpt.get("optimizer_state_path"):
            entry["optimizer_state_path"] = ckpt["optimizer_state_path"]
        checkpoints.append(entry)

    index_path = build_index(
        model=model,
        target_layer=model.fc2,
        error_fn=classification_error,
        data_loader=train_loader,
        checkpoints=checkpoints,
        sample_metadata=sample_meta,
        projection_dim=cfg["ghost"]["projection_dim"],
        projection_type=cfg["ghost"]["projection_type"],
        projection_seed=cfg["ghost"]["projection_seed"],
        output_dir=cfg["paths"]["outputs_dir"],
        index_filename=cfg["index"]["output_path"],
        metadata_filename=cfg["index"]["metadata_path"],
        device=cfg.get("device", "auto"),
    )
    print(f"Index built → {index_path}")


if __name__ == "__main__":
    main()
