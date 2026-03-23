"""Train MNIST MLP and save checkpoint + optimizer state.

Usage:
    python testModels/mnist/train.py [--epochs 2] [--lr 0.01]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from testModels.mnist.model import MnistMLP
from testModels.mnist.data import MnistDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST MLP")
    parser.add_argument("--output", default="testModels/mnist/checkpoints")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MnistMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = MnistDataset(train=True, root=args.data_root)
    loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

    model.train()
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
        print(f"Epoch {epoch+1}/{args.epochs}  loss={total_loss/len(loader):.4f}")

    ckpt_path = os.path.join(args.output, "ckpt_0.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved weights  → {ckpt_path}")

    optim_path = os.path.join(args.output, "optim_0.pt")
    torch.save(optimizer.state_dict(), optim_path)
    print(f"Saved optimizer → {optim_path}")


if __name__ == "__main__":
    main()
