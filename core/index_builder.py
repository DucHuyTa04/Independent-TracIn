from typing import Dict, List

import faiss
import numpy as np


class IndexBuilder:
    """Build exact inner-product FAISS indices for checkpoint embeddings."""

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger

    @staticmethod
    def _build_ip_index(vectors: np.ndarray):
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def build(self, extractor_output: Dict) -> List[Dict]:
        mode = extractor_output["mode"]
        if mode != "ghost":
            raise ValueError("Strict mode supports only ghost extraction.")

        checkpoint_count = int(extractor_output["checkpoint_count"])
        built: List[Dict] = []

        for checkpoint_idx in range(checkpoint_count):
            a_train = extractor_output["A_train"][checkpoint_idx]
            e_train = extractor_output["E_train"][checkpoint_idx]
            built.append(
                {
                    "checkpoint_idx": checkpoint_idx,
                    "mode": mode,
                    "A_index": self._build_ip_index(a_train),
                    "E_index": self._build_ip_index(e_train),
                    "train_count": a_train.shape[0],
                }
            )

        self.logger.info(
            "Built FAISS IndexFlatIP indices for %d checkpoints (mode=%s).",
            checkpoint_count,
            mode,
        )
        return built
