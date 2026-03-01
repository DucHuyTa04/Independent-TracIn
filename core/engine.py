from typing import Dict, List

import numpy as np


class InfluenceEngine:
    """Compute per-checkpoint influence score vectors over training samples."""

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger

    @staticmethod
    def _to_full_score_vector(train_count: int, scores: np.ndarray, indices: np.ndarray) -> np.ndarray:
        vector = np.zeros(train_count, dtype=np.float32)
        vector[indices] = scores
        return vector

    def _compute_ghost_scores(self, entry: Dict, a_test: np.ndarray, e_test: np.ndarray) -> np.ndarray:
        train_count = int(entry["train_count"])
        accumulated = np.zeros(train_count, dtype=np.float32)

        for query_idx in range(a_test.shape[0]):
            query_a = a_test[query_idx : query_idx + 1].astype(np.float32)
            query_e = e_test[query_idx : query_idx + 1].astype(np.float32)

            a_scores, a_indices = entry["A_index"].search(query_a, train_count)
            e_scores, e_indices = entry["E_index"].search(query_e, train_count)

            a_vector = self._to_full_score_vector(train_count, a_scores[0], a_indices[0])
            e_vector = self._to_full_score_vector(train_count, e_scores[0], e_indices[0])

            accumulated += a_vector * e_vector

        return accumulated

    def compute(self, extractor_output: Dict, built_indices: List[Dict]) -> Dict[int, np.ndarray]:
        mode = extractor_output["mode"]
        if mode != "ghost":
            raise ValueError("Strict mode supports only ghost extraction.")

        checkpoint_scores: Dict[int, np.ndarray] = {}

        for entry in built_indices:
            checkpoint_idx = int(entry["checkpoint_idx"])
            a_test = extractor_output["A_test"][checkpoint_idx]
            e_test = extractor_output["E_test"][checkpoint_idx]
            scores = self._compute_ghost_scores(entry=entry, a_test=a_test, e_test=e_test)

            checkpoint_scores[checkpoint_idx] = scores

        self.logger.info("Computed per-checkpoint influence score vectors.")
        return checkpoint_scores
