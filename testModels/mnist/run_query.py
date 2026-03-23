"""Query attribution for a generated MNIST sample using src/ library.

Usage:
    python testModels/mnist/run_query.py --input outputs/query_input.pt [--config testModels/mnist/config.yaml]
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from testModels.mnist.model import MnistMLP
from src.inference import attribute


def classification_error(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """E = softmax(logits) - one_hot(targets)."""
    probs = torch.softmax(logits, dim=-1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, targets.unsqueeze(1).long(), 1.0)
    return probs - one_hot


def main() -> None:
    parser = argparse.ArgumentParser(description="Query MNIST attribution")
    parser.add_argument("--config", default="testModels/mnist/config.yaml")
    parser.add_argument("--input", required=True, help="Path to query_input.pt")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model = MnistMLP(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_classes=cfg["model"]["num_classes"],
    )

    query = torch.load(args.input, map_location="cpu", weights_only=False)
    query_inputs = query["inputs"]
    query_targets = query["targets"]

    outputs_dir = cfg["paths"]["outputs_dir"]
    query_cfg = cfg["query"]

    results = attribute(
        model=model,
        target_layer=model.fc2,
        error_fn=classification_error,
        query_inputs=query_inputs,
        query_targets=query_targets,
        index_path=os.path.join(outputs_dir, query_cfg["index_path"]),
        metadata_path=os.path.join(outputs_dir, query_cfg["metadata_path"]),
        checkpoint_path=cfg["checkpoints"][0]["weights_path"],
        projection_dim=cfg["ghost"]["projection_dim"],
        projection_type=cfg["ghost"]["projection_type"],
        projection_seed=cfg["ghost"]["projection_seed"],
        optimizer_state_path=cfg["checkpoints"][0].get("optimizer_state_path"),
        top_k=query_cfg.get("top_k", 20),
        device=cfg.get("device", "auto"),
    )

    results_path = os.path.join(outputs_dir, "attribution_results.json")
    os.makedirs(outputs_dir, exist_ok=True)

    # Convert for JSON serialization
    serializable = []
    for r in results:
        serializable.append({
            "rights_holder_attribution": r["rights_holder_attribution"],
            "top_samples": [[int(sid), float(score)] for sid, score in r["top_samples"]],
        })

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"Attribution results → {results_path}")
    for r in serializable:
        print("\nRights holder attribution:")
        for holder, pct in sorted(r["rights_holder_attribution"].items(), key=lambda x: -x[1]):
            print(f"  {holder}: {pct*100:.1f}%")


if __name__ == "__main__":
    main()
