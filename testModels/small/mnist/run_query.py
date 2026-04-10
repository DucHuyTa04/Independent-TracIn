"""Query attribution for a generated MNIST sample using src/ library.

Usage:
    python testModels/small/mnist/run_query.py --input outputs/query_input.pt [--config testModels/small/mnist/config.yaml]
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from testModels.small.mnist.model import MnistMLP
from testModels.pipeline_helpers import (
    abs_config_path,
    build_checkpoints_list,
    ghost_settings,
    outputs_dir_from_cfg,
)
from src.config_utils import find_adam_param_key, last_checkpoint_paths, resolve_target_layer
from src.error_functions import get_error_fn
from src.inference import attribute


def main() -> None:
    parser = argparse.ArgumentParser(description="Query MNIST attribution")
    parser.add_argument("--config", default="testModels/small/mnist/config.yaml")
    parser.add_argument("--input", required=True, help="Path to query_input.pt")
    args = parser.parse_args()

    cfg_path = abs_config_path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = MnistMLP(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_classes=cfg["model"]["num_classes"],
    )

    query = torch.load(args.input, map_location="cpu", weights_only=False)
    query_inputs = query["inputs"]
    query_targets = query["targets"]

    outputs_dir = outputs_dir_from_cfg(cfg)
    query_cfg = cfg["query"]
    checkpoints = build_checkpoints_list(cfg_path, cfg)
    wpath, opath = last_checkpoint_paths(checkpoints)

    _, target_layer = resolve_target_layer(model, cfg.get("target_layer"))
    error_fn = get_error_fn(cfg.get("loss_type", "classification"))
    adam_key = find_adam_param_key(model, target_layer)
    proj_dim, proj_type, proj_seed = ghost_settings(cfg)

    results = attribute(
        model=model,
        target_layer=target_layer,
        error_fn=error_fn,
        query_inputs=query_inputs,
        query_targets=query_targets,
        index_path=os.path.join(outputs_dir, query_cfg["index_path"]),
        metadata_path=os.path.join(outputs_dir, query_cfg["metadata_path"]),
        checkpoint_path=wpath,
        projection_dim=proj_dim,
        projection_type=proj_type,
        projection_seed=proj_seed,
        optimizer_state_path=opath,
        adam_param_key=adam_key,
        top_k=query_cfg.get("top_k", 20),
        device=cfg.get("device", "auto"),
    )

    results_path = os.path.join(outputs_dir, "attribution_results.json")
    os.makedirs(outputs_dir, exist_ok=True)

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
