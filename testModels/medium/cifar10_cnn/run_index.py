"""Build FAISS index for CIFAR-10 CNN."""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from testModels.medium.cifar10_cnn.model import CifarSmallCNN
from testModels.medium.cifar10_cnn.data import make_loaders
from testModels.pipeline_helpers import (
    abs_config_path,
    build_checkpoints_list,
    ghost_settings,
    outputs_dir_from_cfg,
)
from src.config_utils import find_adam_param_key, resolve_target_layer
from src.error_functions import get_error_fn
from src.indexer import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="testModels/medium/cifar10_cnn/config.yaml")
    args = parser.parse_args()

    cfg_path = abs_config_path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = CifarSmallCNN(num_classes=cfg["model"]["num_classes"])
    train_loader, _, sample_meta = make_loaders(
        batch_size=cfg["dataset"]["batch_size"],
        data_root=cfg["paths"]["data_root"],
    )
    checkpoints = build_checkpoints_list(cfg_path, cfg)
    _, target_layer = resolve_target_layer(model, cfg.get("target_layer"))
    error_fn = get_error_fn(cfg.get("loss_type", "classification"))
    adam_key = find_adam_param_key(model, target_layer)
    proj_dim, proj_type, proj_seed = ghost_settings(cfg)

    path = build_index(
        model=model,
        target_layer=target_layer,
        error_fn=error_fn,
        data_loader=train_loader,
        checkpoints=checkpoints,
        sample_metadata=sample_meta,
        projection_dim=proj_dim,
        projection_type=proj_type,
        projection_seed=proj_seed,
        adam_param_key=adam_key,
        output_dir=outputs_dir_from_cfg(cfg),
        index_filename=cfg["index"]["output_path"],
        metadata_filename=cfg["index"]["metadata_path"],
        device=cfg.get("device", "auto"),
    )
    print(f"Index built → {path}")


if __name__ == "__main__":
    main()
