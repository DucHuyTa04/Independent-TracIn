import os
from typing import Dict, List

import numpy as np
import torch

from adapters.base_adapter import BaseExtractor


class GhostExtractor(BaseExtractor):
    """Extract hidden activations A and logit-error vectors E for each checkpoint."""

    def extract(self, checkpoint_records: List[Dict]) -> Dict:
        data_bundle = self._get_data()
        train_ids = self._collect_ids(data_bundle.train_loader)
        test_ids = self._collect_ids(data_bundle.test_loader)
        ids_path = self._save_ids(train_ids=train_ids, test_ids=test_ids)

        all_a_train: List[np.ndarray] = []
        all_a_test: List[np.ndarray] = []
        all_e_train: List[np.ndarray] = []
        all_e_test: List[np.ndarray] = []

        for record in checkpoint_records:
            model = self._build_model()
            self._load_ckpt(model=model, checkpoint_path=record["checkpoint_path"])
            model.eval()

            a_train_rows: List[np.ndarray] = []
            e_train_rows: List[np.ndarray] = []
            a_test_rows: List[np.ndarray] = []
            e_test_rows: List[np.ndarray] = []

            with torch.no_grad():
                for images, labels, _ in data_bundle.train_loader:
                    images = images.to(self.device)
                    labels = self.task_adapter.prepare_targets(labels, self.device)

                    hidden = self.model_adapter.hidden_activation(model=model, images=images)
                    logits = self.model_adapter.forward_logits(model=model, images=images)
                    error_matrix = self.task_adapter.error_signal(logits=logits, targets=labels)

                    for i in range(images.size(0)):
                        a_train_rows.append(hidden[i].detach().cpu().numpy().astype(np.float32))
                        e_train_rows.append(error_matrix[i].detach().cpu().numpy().astype(np.float32))

                for images, labels, _ in data_bundle.test_loader:
                    images = images.to(self.device)
                    labels = self.task_adapter.prepare_targets(labels, self.device)

                    hidden = self.model_adapter.hidden_activation(model=model, images=images)
                    logits = self.model_adapter.forward_logits(model=model, images=images)
                    error_matrix = self.task_adapter.error_signal(logits=logits, targets=labels)

                    for i in range(images.size(0)):
                        a_test_rows.append(hidden[i].detach().cpu().numpy().astype(np.float32))
                        e_test_rows.append(error_matrix[i].detach().cpu().numpy().astype(np.float32))

            all_a_train.append(self._stack_rows(a_train_rows))
            all_a_test.append(self._stack_rows(a_test_rows))
            all_e_train.append(self._stack_rows(e_train_rows))
            all_e_test.append(self._stack_rows(e_test_rows))

        embeddings_dir = os.path.join(self.config["paths"]["outputs_dir"], "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)

        a_path = os.path.join(embeddings_dir, "A.npz")
        e_path = os.path.join(embeddings_dir, "E.npz")

        a_payload = {}
        e_payload = {}
        for idx in range(len(checkpoint_records)):
            a_payload[f"train_{idx}"] = all_a_train[idx]
            a_payload[f"test_{idx}"] = all_a_test[idx]
            e_payload[f"train_{idx}"] = all_e_train[idx]
            e_payload[f"test_{idx}"] = all_e_test[idx]

        np.savez(a_path, **a_payload)
        np.savez(e_path, **e_payload)

        self.logger.info("Saved ghost embeddings: %s and %s", a_path, e_path)
        return {
            "mode": "ghost",
            "ids_path": ids_path,
            "checkpoint_count": len(checkpoint_records),
            "A_path": a_path,
            "E_path": e_path,
            "A_train": all_a_train,
            "A_test": [self._subset_test_queries(x) for x in all_a_test],
            "E_train": all_e_train,
            "E_test": [self._subset_test_queries(x) for x in all_e_test],
        }
