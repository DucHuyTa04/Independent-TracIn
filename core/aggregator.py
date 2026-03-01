import csv
import json
import os
from typing import Dict

import numpy as np

from utils.checkpoint_manager import read_learning_rate


class ScoreAggregator:
    """Aggregate checkpoint score vectors with learning-rate weighting and save top-k."""

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger

    def aggregate(self, checkpoint_records, checkpoint_scores: Dict[int, np.ndarray], ids_path: str) -> str:
        with open(ids_path, "r", encoding="utf-8") as file:
            ids_payload = json.load(file)
        train_ids = ids_payload["train"]

        final_scores = np.zeros(len(train_ids), dtype=np.float64)

        for record in checkpoint_records:
            checkpoint_idx = int(record["index"])
            lr = read_learning_rate(record["meta_path"])
            vector = checkpoint_scores[checkpoint_idx].astype(np.float64)
            final_scores += lr * vector

        top_k = int(self.config["influence"]["top_k"])
        top_k = min(top_k, len(train_ids))

        ranking = np.argsort(-final_scores)[:top_k]

        results_dir = os.path.join(self.config["paths"]["outputs_dir"], "results")
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, "topk.csv")
        json_path = os.path.join(results_dir, "topk.json")

        rows = []
        for rank, row_idx in enumerate(ranking, start=1):
            rows.append(
                {
                    "rank": rank,
                    "train_row_index": int(row_idx),
                    "original_sample_id": int(train_ids[row_idx]),
                    "score": float(final_scores[row_idx]),
                }
            )

        with open(csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["rank", "train_row_index", "original_sample_id", "score"],
            )
            writer.writeheader()
            writer.writerows(rows)

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(rows, file, indent=2)

        self.logger.info("Saved Top-K results: %s", csv_path)
        return csv_path
