import argparse
import logging
import os
from datetime import datetime

import yaml

from core.aggregator import ScoreAggregator
from core.engine import InfluenceEngine
from core.index_builder import IndexBuilder
from utils.class_loader import load_class


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logger(logs_dir: str) -> logging.Logger:
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"run_{timestamp}.log")

    logger = logging.getLogger("tracin_ghost_tool")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logger initialized. Log file: %s", log_path)
    return logger


def ensure_output_directories(outputs_dir: str) -> None:
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "embeddings"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "results"), exist_ok=True)


def resolve_extractor_class(mode: str):
    if mode == "ghost":
        from adapters.extractor_ghost import GhostExtractor

        return GhostExtractor
    raise ValueError("Unsupported extraction mode. Strict mode allows only 'ghost'.")


def run_pipeline(config: dict) -> None:
    paths = config["paths"]
    outputs_dir = paths["outputs_dir"]
    logs_dir = paths["logs_dir"]
    ensure_output_directories(outputs_dir)

    logger = setup_logger(logs_dir)
    logger.info("Starting TracIn pipeline.")

    adapter_class_path = config["model"]["adapter_class"]
    adapter_class = load_class(adapter_class_path)
    model_adapter = adapter_class()
    logger.info("Loaded model adapter: %s", adapter_class_path)

    dataset_adapter_class_path = config["dataset"]["adapter_class"]
    dataset_adapter_class = load_class(dataset_adapter_class_path)
    dataset_adapter = dataset_adapter_class()
    logger.info("Loaded dataset adapter: %s", dataset_adapter_class_path)

    task_adapter_class_path = config["task"]["adapter_class"]
    task_adapter_class = load_class(task_adapter_class_path)
    task_adapter = task_adapter_class()
    logger.info("Loaded task adapter: %s", task_adapter_class_path)

    trainer_class_path = config["train"]["trainer_class"]
    trainer_class = load_class(trainer_class_path)
    trainer = trainer_class(
        config=config,
        logger=logger,
        model_adapter=model_adapter,
        dataset_adapter=dataset_adapter,
        task_adapter=task_adapter,
    )
    logger.info("Loaded trainer class: %s", trainer_class_path)
    checkpoint_records = trainer.train()
    logger.info("Training complete. Saved %d checkpoints.", len(checkpoint_records))

    extraction_mode = config["influence"]["extraction_mode"]
    extractor_class = resolve_extractor_class(extraction_mode)
    extractor = extractor_class(
        config=config,
        logger=logger,
        model_adapter=model_adapter,
        dataset_adapter=dataset_adapter,
        task_adapter=task_adapter,
    )
    extractor_output = extractor.extract(checkpoint_records=checkpoint_records)
    logger.info("Extraction complete with mode: %s", extraction_mode)

    index_builder = IndexBuilder(config=config, logger=logger)
    built_indices = index_builder.build(extractor_output=extractor_output)
    logger.info("Index build complete.")

    engine = InfluenceEngine(config=config, logger=logger)
    checkpoint_scores = engine.compute(
        extractor_output=extractor_output,
        built_indices=built_indices,
    )
    logger.info("Per-checkpoint influence scoring complete.")

    aggregator = ScoreAggregator(config=config, logger=logger)
    results_path = aggregator.aggregate(
        checkpoint_records=checkpoint_records,
        checkpoint_scores=checkpoint_scores,
        ids_path=extractor_output["ids_path"],
    )

    logger.info("Pipeline finished. Top-K result saved to %s", results_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TracIn Ghost Tool pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    configuration = load_config(arguments.config)
    run_pipeline(configuration)
