"""TracIn Ghost — Copyright Attribution via Training Influence.

CLI entry point that dispatches to model-specific scripts in testModels/<name>/.

Usage:
    python main.py --config testModels/mnist/config.yaml --mode index
    python main.py --config testModels/mnist/config.yaml --mode query --input outputs/query_input.pt
    python main.py --config testModels/mnist/config.yaml --mode full --input outputs/query_input.pt
"""

import argparse
import importlib
import logging
import os
import sys
from datetime import datetime

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(logs_dir: str = "logs") -> None:
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Log file: %s", log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="TracIn Ghost copyright attribution")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--mode", choices=["index", "query", "full"], required=True)
    parser.add_argument("--input", help="Query tensor .pt path (required for query/full)")
    parser.add_argument(
        "--model", default="mnist",
        help="Model name matching testModels/<name>/ folder (default: mnist)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("paths", {}).get("logs_dir", "logs"))
    logging.info("Config: %s | Mode: %s | Model: %s", args.config, args.mode, args.model)

    pkg = f"testModels.{args.model}"

    if args.mode in ("index", "full"):
        logging.info("Running index …")
        mod = importlib.import_module(f"{pkg}.run_index")
        saved_argv = sys.argv
        sys.argv = ["run_index", "--config", args.config]
        try:
            mod.main()
        finally:
            sys.argv = saved_argv

    if args.mode in ("query", "full"):
        if not args.input:
            parser.error("--input is required for query / full mode")
        logging.info("Running query …")
        mod = importlib.import_module(f"{pkg}.run_query")
        saved_argv = sys.argv
        sys.argv = ["run_query", "--config", args.config, "--input", args.input]
        try:
            mod.main()
        finally:
            sys.argv = saved_argv

    logging.info("Done.")


if __name__ == "__main__":
    main()
