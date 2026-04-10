"""Train CIFAR-10 small CNN with TracIn-ready checkpoints."""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config_utils import TracInCheckpointCallback
from testModels.medium.cifar10_cnn.model import CifarSmallCNN
from testModels.medium.cifar10_cnn.data import CifarDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="testModels/medium/cifar10_cnn/checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--prune", action="store_true")
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CifarSmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    ds = CifarDataset(train=True, root=args.data_root)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)

    cb = TracInCheckpointCallback(save_dir=args.output, save_every=args.save_every)
    model.train()
    last_avg = 0.0
    for epoch in range(args.epochs):
        total = 0.0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total += loss.item()
        last_avg = total / len(loader)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={last_avg:.4f}")
        p = cb.on_epoch_end(epoch, model, opt, last_avg)
        if p:
            print(f"Saved → {p}")

    cb.finalize(model, opt, args.epochs - 1, last_avg)
    if args.prune:
        cb.select_best(keep=args.keep_checkpoints)


if __name__ == "__main__":
    main()
