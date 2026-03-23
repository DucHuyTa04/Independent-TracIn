"""FAISS inner-product index with persistent storage and metadata.

Builds, saves, loads, and queries a FAISS IndexFlatIP (inner product)
index. Each vector is tagged with a sample_id for tracing back to
copyrighted training data and rights holders.

Important: uses inner product (IndexFlatIP), NOT cosine similarity.
TracIn scores are inner products — cosine discards magnitude information.
"""

import json
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FAISSStore:
    """FAISS index with persistent storage and metadata sidecar."""

    def __init__(
        self,
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
        top_k: int = 50,
    ) -> None:
        """Initialize the FAISS store.

        Args:
            index_type: "flat" for IndexFlatIP, "ivf" for IndexIVFFlat.
            nlist: Number of IVF clusters (only used if index_type="ivf").
            nprobe: Number of clusters to search (only used if index_type="ivf").
            top_k: Default number of neighbors to return in queries.
        """
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.top_k = top_k
        self._index = None
        self._sample_ids: list[int] = []
        self._metadata: dict = {}

    def build_and_save(
        self,
        vectors: np.ndarray,
        sample_ids: list[int],
        index_path: str,
        metadata_path: str,
        metadata_extra: Optional[dict] = None,
    ) -> None:
        """Build FAISS inner-product index and save to disk.

        Args:
            vectors: (N, K) projected ghost vectors.
            sample_ids: Per-vector sample IDs for tracing.
            index_path: Where to save the FAISS index binary.
            metadata_path: Where to save the sidecar JSON.
            metadata_extra: Extra metadata (e.g. sample_id_to_rights_holder).
        """
        import faiss

        dim = vectors.shape[1]
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(self.nlist, vectors.shape[0])
            index = faiss.IndexIVFFlat(
                quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT,
            )
            index.train(vectors)
            index.nprobe = self.nprobe
        else:
            index = faiss.IndexFlatIP(dim)

        index.add(vectors)

        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        faiss.write_index(index, index_path)

        meta: dict = {"sample_ids": sample_ids}
        if metadata_extra:
            meta.update(metadata_extra)
        os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self._index = index
        self._sample_ids = sample_ids
        self._metadata = meta

        logger.info(
            "FAISS index saved: %s (%d vectors, dim=%d)",
            index_path, len(sample_ids), dim,
        )

    def load(self, index_path: str, metadata_path: str) -> None:
        """Load previously saved index + metadata.

        Args:
            index_path: Path to the FAISS index binary.
            metadata_path: Path to the sidecar JSON.
        """
        import faiss

        self._index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self._sample_ids = meta["sample_ids"]
        self._metadata = meta
        logger.info(
            "FAISS index loaded: %d vectors from %s",
            len(self._sample_ids), index_path,
        )

    def query(
        self,
        query_vectors: np.ndarray,
        top_k: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
        """Query the index by inner product.

        Args:
            query_vectors: (Q, K) query ghost vectors.
            top_k: Number of neighbors to return (default: self.top_k).

        Returns:
            scores: (Q, top_k) inner product scores.
            indices: (Q, top_k) FAISS internal indices.
            sample_ids_per_query: List of length Q, each a list of sample_ids.
        """
        if self._index is None:
            raise RuntimeError("FAISSStore not loaded. Call load() first.")

        k = top_k or self.top_k
        k = min(k, len(self._sample_ids))

        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)
        scores, indices = self._index.search(query_vectors, k)

        sample_ids_per_query = []
        for q in range(indices.shape[0]):
            ids = [self._sample_ids[i] if i >= 0 else -1 for i in indices[q]]
            sample_ids_per_query.append(ids)

        return scores, indices, sample_ids_per_query

    @property
    def metadata(self) -> dict:
        """Access loaded metadata (includes sample_id_to_rights_holder, etc.)."""
        return self._metadata
