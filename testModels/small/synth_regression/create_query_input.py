"""Save a synthetic test point as query .pt."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from testModels.small.synth_regression.data import SynthDataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/synth_query_input.pt")
    p.add_argument("--index", type=int, default=0)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ds = SynthDataset(50, seed=43)
    x, y, _ = ds[args.index]
    torch.save({"inputs": x.unsqueeze(0), "targets": y.unsqueeze(0)}, args.output)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
