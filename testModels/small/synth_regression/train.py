"""Train synthetic regression MLP with TracIn-ready checkpoints."""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config_utils import TracInCheckpointCallback
from testModels.small.synth_regression.model import SynthRegressionMLP
from testModels.small.synth_regression.data import SynthDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="testModels/synth_regression/checkpoints")
    p.add_argument(
        "--data-root",
        default="data",
        help="Unused (synthetic data). Accepted so Slurm/container scripts match MNIST/CIFAR.",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--prune", action="store_true")
    p.add_argument("--keep-checkpoints", type=int, default=5)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SynthRegressionMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()
    ds = SynthDataset(args.n_train, seed=0)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    cb = TracInCheckpointCallback(save_dir=args.output, save_every=args.save_every)
    model.train()
    last_avg = 0.0
    for epoch in range(args.epochs):
        total = 0.0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            opt.zero_grad()
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()
        last_avg = total / len(loader)
        print(f"Epoch {epoch + 1}/{args.epochs}  mse={last_avg:.6f}")
        path = cb.on_epoch_end(epoch, model, opt, last_avg)
        if path:
            print(f"Saved → {path}")

    cb.finalize(model, opt, args.epochs - 1, last_avg)
    if args.prune:
        cb.select_best(keep=args.keep_checkpoints)


if __name__ == "__main__":
    main()
