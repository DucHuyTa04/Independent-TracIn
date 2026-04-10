"""Query attribution for CIFAR-10 CNN."""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from testModels.medium.cifar10_cnn.model import CifarSmallCNN
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="testModels/medium/cifar10_cnn/config.yaml")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    cfg_path = abs_config_path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = CifarSmallCNN(num_classes=cfg["model"]["num_classes"])
    query = torch.load(args.input, map_location="cpu", weights_only=False)
    checkpoints = build_checkpoints_list(cfg_path, cfg)
    wpath, opath = last_checkpoint_paths(checkpoints)
    proj_dim, proj_type, proj_seed = ghost_settings(cfg)

    _, target_layer = resolve_target_layer(model, cfg.get("target_layer"))
    error_fn = get_error_fn(cfg.get("loss_type", "classification"))
    adam_key = find_adam_param_key(model, target_layer)

    results = attribute(
        model=model,
        target_layer=target_layer,
        error_fn=error_fn,
        query_inputs=query["inputs"],
        query_targets=query["targets"],
        index_path=os.path.join(outputs_dir_from_cfg(cfg), cfg["query"]["index_path"]),
        metadata_path=os.path.join(outputs_dir_from_cfg(cfg), cfg["query"]["metadata_path"]),
        checkpoint_path=wpath,
        projection_dim=proj_dim,
        projection_type=proj_type,
        projection_seed=proj_seed,
        optimizer_state_path=opath,
        adam_param_key=adam_key,
        top_k=cfg["query"].get("top_k", 20),
        device=cfg.get("device", "auto"),
    )

    out_dir = outputs_dir_from_cfg(cfg)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "attribution_results.json")
    serializable = [
        {
            "rights_holder_attribution": r["rights_holder_attribution"],
            "top_samples": [[int(s), float(v)] for s, v in r["top_samples"]],
        }
        for r in results
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"Attribution → {path}")


if __name__ == "__main__":
    main()
