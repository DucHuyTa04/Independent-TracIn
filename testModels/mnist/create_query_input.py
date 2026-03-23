"""Create a query input file for testing attribution.

Picks a test sample from MNIST and saves it as a .pt file.

Usage:
    python testModels/mnist/create_query_input.py [--index 0] [--output outputs/query_input.pt]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from testModels.mnist.data import MnistDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create query input from MNIST test set")
    parser.add_argument("--output", default="outputs/query_input.pt")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--index", type=int, default=0, help="Test sample index")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    test_ds = MnistDataset(train=False, root=args.data_root)
    x, y, idx = test_ds[args.index]

    payload = {
        "inputs": x.unsqueeze(0),
        "targets": torch.tensor([y]),
    }
    torch.save(payload, args.output)
    print(f"Saved query input (sample {idx}, label {y.item()}) → {args.output}")


if __name__ == "__main__":
    main()
