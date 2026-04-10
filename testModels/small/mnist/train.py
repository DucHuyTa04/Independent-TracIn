"""Train MNIST MLP and save TracIn-ready unified checkpoints.

Usage:
    python testModels/small/mnist/train.py [--epochs 2] [--lr 0.01]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config_utils import TracInCheckpointCallback
from testModels.small.mnist.model import MnistMLP
from testModels.small.mnist.data import MnistDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST MLP")
    parser.add_argument("--output", default="testModels/small/mnist/checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--save-every", type=int, default=1, help="checkpoint every N epochs")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="after training, keep only top checkpoints (see --keep-checkpoints)",
    )
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MnistMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = MnistDataset(train=True, root=args.data_root)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

    cb = TracInCheckpointCallback(save_dir=args.output, save_every=args.save_every)

    model.train()
    last_avg = 0.0
    for epoch in range(args.epochs):
        total_loss = 0.0
        for inputs, targets, _ in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        last_avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs}  loss={last_avg:.4f}")
        path = cb.on_epoch_end(epoch, model, optimizer, last_avg)
        if path:
            print(f"Saved TracIn checkpoint → {path}")

    cb.finalize(model, optimizer, args.epochs - 1, last_avg)
    print(f"Final checkpoint → {os.path.join(args.output, f'ckpt_{args.epochs - 1}.pt')}")

    if args.prune:
        cb.select_best(keep=args.keep_checkpoints)
        print(f"Pruned checkpoints; kept up to {args.keep_checkpoints} (see checkpoints/.pruned/)")


if __name__ == "__main__":
    main()
