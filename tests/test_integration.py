"""End-to-end pipeline: build_index + attribute with FAISS (requires faiss)."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

pytest.importorskip("faiss")

from src.indexer import build_index
from src.inference import attribute


class _SepDataset(Dataset):
    """Four points: two on x-axis (label 0), two on y-axis (label 1)."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int):
        xs = torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        ys = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        return xs[idx], ys[idx], idx


def _classification_error(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, targets.unsqueeze(1).long(), 1.0)
    return probs - one_hot


class TinyCls(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TestFullPipelineFAISS:
    def test_class0_query_attributes_mostly_class0(self):
        torch.manual_seed(0)
        model = TinyCls()
        opt = torch.optim.SGD(model.parameters(), lr=0.2)
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(_SepDataset(), batch_size=4, shuffle=False)

        model.train()
        for _ in range(80):
            for inputs, targets, _ in loader:
                opt.zero_grad()
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                loss.backward()
                opt.step()

        with tempfile.TemporaryDirectory() as tmp:
            wpath = os.path.join(tmp, "w.pt")
            torch.save(model.state_dict(), wpath)

            checkpoints = [{"weights_path": wpath, "learning_rate": 0.01}]
            sample_meta = {0: "class_0", 1: "class_0", 2: "class_1", 3: "class_1"}
            index_path = os.path.join(tmp, "faiss_index")
            meta_path = os.path.join(tmp, "faiss_metadata.json")

            build_index(
                model=model,
                target_layer=model.fc,
                error_fn=_classification_error,
                data_loader=loader,
                checkpoints=checkpoints,
                sample_metadata=sample_meta,
                projection_dim=64,
                projection_type="sjlt",
                projection_seed=0,
                output_dir=tmp,
                index_filename="faiss_index",
                metadata_filename="faiss_metadata.json",
                device="cpu",
            )

            query_inputs = torch.tensor([[2.2, 0.0, 0.0, 0.0]], dtype=torch.float32)
            query_targets = torch.tensor([0], dtype=torch.long)

            results = attribute(
                model=model,
                target_layer=model.fc,
                error_fn=_classification_error,
                query_inputs=query_inputs,
                query_targets=query_targets,
                index_path=index_path,
                metadata_path=meta_path,
                checkpoint_path=wpath,
                projection_dim=64,
                projection_type="sjlt",
                projection_seed=0,
                top_k=4,
                device="cpu",
            )

            r0 = results[0]
            top_ids = [sid for sid, _ in r0["top_samples"]]
            class0_in_top2 = sum(1 for sid in top_ids[:2] if sid in (0, 1))
            assert class0_in_top2 >= 1, f"expected class-0 samples in top-2, got {top_ids}"

            pct0 = r0["rights_holder_attribution"].get("class_0", 0.0)
            pct1 = r0["rights_holder_attribution"].get("class_1", 0.0)
            assert pct0 > pct1, (
                f"class_0 share should exceed class_1, got {pct0} vs {pct1}"
            )
