import json
import os
from typing import Dict, List

import numpy as np
import torch

from utils.checkpoint_manager import load_checkpoint


class BaseExtractor:
    """Base class for pluggable checkpoint extractors."""

    def __init__(self, config: dict, logger, model_adapter, dataset_adapter, task_adapter):
        self.config = config
        self.logger = logger
        self.model_adapter = model_adapter
        self.dataset_adapter = dataset_adapter
        self.task_adapter = task_adapter
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def extract(self, checkpoint_records: List[Dict]) -> Dict:
        raise NotImplementedError("Subclasses must implement extract().")

    def _build_model(self) -> torch.nn.Module:
        return self.model_adapter.build_model(config=self.config, device=self.device)

    def _get_data(self):
        return self.dataset_adapter.build_loaders(config=self.config)

    @staticmethod
    def _collect_ids(loader) -> List[int]:
        sample_ids: List[int] = []
        for _, _, batch_ids in loader:
            sample_ids.extend([int(value) for value in batch_ids])
        return sample_ids

    def _save_ids(self, train_ids: List[int], test_ids: List[int]) -> str:
        embeddings_dir = os.path.join(self.config["paths"]["outputs_dir"], "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        ids_path = os.path.join(embeddings_dir, "ids.json")

        with open(ids_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "train": train_ids,
                    "test": test_ids,
                },
                file,
                indent=2,
            )

        return ids_path

    def _load_ckpt(self, model: torch.nn.Module, checkpoint_path: str) -> None:
        load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=self.device)

    @staticmethod
    def _stack_rows(rows: List[np.ndarray]) -> np.ndarray:
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)

    def _subset_test_queries(self, test_matrix: np.ndarray) -> np.ndarray:
        query_count = int(self.config["influence"]["test_query_count"])
        if query_count <= 0 or query_count >= test_matrix.shape[0]:
            return test_matrix
        return test_matrix[:query_count]
