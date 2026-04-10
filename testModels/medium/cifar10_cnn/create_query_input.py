"""Save a CIFAR-10 test image as query .pt file."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from testModels.medium.cifar10_cnn.data import CifarDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/cifar10_query_input.pt")
    p.add_argument("--data-root", default="data")
    p.add_argument("--index", type=int, default=0)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ds = CifarDataset(train=False, root=args.data_root)
    x, y, _ = ds[args.index]
    torch.save({"inputs": x.unsqueeze(0), "targets": torch.tensor([y])}, args.output)
    print(f"Saved → {args.output} (label {y})")


if __name__ == "__main__":
    main()
