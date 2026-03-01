from typing import List, Dict


class BaseTrainer:
    """Base trainer interface for checkpoint-producing training jobs."""

    def __init__(self, config: dict, logger, model_adapter, dataset_adapter, task_adapter):
        self.config = config
        self.logger = logger
        self.model_adapter = model_adapter
        self.dataset_adapter = dataset_adapter
        self.task_adapter = task_adapter

    def train(self) -> List[Dict]:
        raise NotImplementedError("Subclasses must implement train().")
