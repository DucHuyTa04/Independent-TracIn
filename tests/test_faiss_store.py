"""Integration tests for src/faiss_store.py (requires faiss)."""

import json
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("faiss")

from src.faiss_store import FAISSStore


class TestFAISSStoreRoundtrip:
    def test_build_and_load_roundtrip(self):
        dim = 8
        n = 12
        rng = np.random.RandomState(0)
        vectors = rng.randn(n, dim).astype(np.float32)
        sample_ids = list(range(100, 100 + n))

        with tempfile.TemporaryDirectory() as tmp:
            idx_path = os.path.join(tmp, "index.faiss")
            meta_path = os.path.join(tmp, "meta.json")
            store = FAISSStore(index_type="flat")
            store.build_and_save(vectors, sample_ids, idx_path, meta_path)

            store2 = FAISSStore(index_type="flat")
            store2.load(idx_path, meta_path)
            scores, _, ids_per_q = store2.query(vectors, top_k=n)
            for q in range(n):
                assert sample_ids[q] in ids_per_q[q]

    def test_query_returns_correct_ids(self):
        """Orthogonal one-hot vectors: query aligns with exactly one stored row."""
        dim = 6
        vectors = np.eye(dim, dtype=np.float32)
        sample_ids = list(range(dim))

        with tempfile.TemporaryDirectory() as tmp:
            idx_path = os.path.join(tmp, "index.faiss")
            meta_path = os.path.join(tmp, "meta.json")
            store = FAISSStore(index_type="flat")
            store.build_and_save(vectors, sample_ids, idx_path, meta_path)
            store.load(idx_path, meta_path)

            query = np.eye(dim, dtype=np.float32)[2:3]  # row e_2
            scores, _, ids_per_q = store.query(query, top_k=1)
            assert ids_per_q[0][0] == 2
            assert scores[0, 0] > 0.99

    def test_metadata_preserved(self):
        dim = 4
        vectors = np.random.randn(3, dim).astype(np.float32)
        sample_ids = [10, 20, 30]
        extra = {"sample_id_to_rights_holder": {"10": "alice", "20": "bob", "30": "alice"}}

        with tempfile.TemporaryDirectory() as tmp:
            idx_path = os.path.join(tmp, "index.faiss")
            meta_path = os.path.join(tmp, "meta.json")
            store = FAISSStore(index_type="flat")
            store.build_and_save(
                vectors, sample_ids, idx_path, meta_path, metadata_extra=extra,
            )
            store.load(idx_path, meta_path)
            assert store.metadata["sample_ids"] == sample_ids
            assert store.metadata["sample_id_to_rights_holder"]["20"] == "bob"

            with open(meta_path, "r", encoding="utf-8") as f:
                disk = json.load(f)
            assert disk["sample_id_to_rights_holder"]["10"] == "alice"


class TestFAISSStoreIVF:
    def test_ivf_index_build_and_query(self):
        dim = 8
        n = 50
        rng = np.random.RandomState(42)
        vectors = rng.randn(n, dim).astype(np.float32)
        sample_ids = list(range(n))

        with tempfile.TemporaryDirectory() as tmp:
            idx_path = os.path.join(tmp, "ivf.faiss")
            meta_path = os.path.join(tmp, "ivf_meta.json")
            store = FAISSStore(index_type="ivf", nlist=10, nprobe=4)
            store.build_and_save(vectors, sample_ids, idx_path, meta_path)
            store.load(idx_path, meta_path)
            q = vectors[7:8]
            scores, _, ids_per_q = store.query(q, top_k=5)
            assert 7 in ids_per_q[0]
            assert scores[0, 0] > 0.0
