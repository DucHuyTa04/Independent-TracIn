import os
from typing import Dict, List

import torch
import torch.optim as optim

from trainer.base_trainer import BaseTrainer
from utils.checkpoint_manager import save_checkpoint, save_meta


class GenericTrainer(BaseTrainer):
    """Generic trainer using injected model adapter and SGD/CrossEntropy."""

    def __init__(self, config: dict, logger, model_adapter, dataset_adapter, task_adapter):
        super().__init__(
            config=config,
            logger=logger,
            model_adapter=model_adapter,
            dataset_adapter=dataset_adapter,
            task_adapter=task_adapter,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _save_checkpoint_pair(
        self,
        model: torch.nn.Module,
        checkpoints_dir: str,
        checkpoint_index: int,
        global_step: int,
        epoch: int,
        learning_rate: float,
    ) -> Dict:
        ckpt_name = f"ckpt_{checkpoint_index}.pt"
        meta_name = f"meta_{checkpoint_index}.json"
        ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
        meta_path = os.path.join(checkpoints_dir, meta_name)

        save_checkpoint(model, ckpt_path)
        save_meta(
            meta_path,
            {
                "checkpoint_index": checkpoint_index,
                "global_step": global_step,
                "epoch": epoch,
                "learning_rate": float(learning_rate),
            },
        )

        self.logger.info(
            "Saved checkpoint %s with meta %s (lr=%f)",
            ckpt_path,
            meta_path,
            learning_rate,
        )

        return {
            "index": checkpoint_index,
            "checkpoint_path": ckpt_path,
            "meta_path": meta_path,
        }

    def train(self) -> List[Dict]:
        seed = int(self.config["seed"])
        torch.manual_seed(seed)

        paths_cfg = self.config["paths"]
        dataset_cfg = self.config["dataset"]
        train_cfg = self.config["train"]

        outputs_dir = paths_cfg["outputs_dir"]
        checkpoints_dir = os.path.join(outputs_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)

        data_bundle = self.dataset_adapter.build_loaders(config=self.config)

        model = self.model_adapter.build_model(config=self.config, device=self.device)

        optimizer = optim.SGD(
            model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            momentum=float(train_cfg["momentum"]),
        )

        epochs = int(train_cfg["epochs"])
        save_every = int(train_cfg["checkpoint_every_steps"])

        checkpoint_records: List[Dict] = []
        checkpoint_index = 0
        global_step = 0

        model.train()
        for epoch in range(epochs):
            for images, labels, _ in data_bundle.train_loader:
                images = images.to(self.device)
                labels = self.task_adapter.prepare_targets(labels, self.device)

                optimizer.zero_grad()
                logits = self.model_adapter.forward_logits(model=model, images=images)
                loss = self.task_adapter.compute_loss(logits, labels)
                loss.backward()
                optimizer.step()

                global_step += 1

                if global_step % save_every == 0:
                    learning_rate = float(optimizer.param_groups[0]["lr"])
                    checkpoint_records.append(
                        self._save_checkpoint_pair(
                            model=model,
                            checkpoints_dir=checkpoints_dir,
                            checkpoint_index=checkpoint_index,
                            global_step=global_step,
                            epoch=epoch,
                            learning_rate=learning_rate,
                        )
                    )
                    checkpoint_index += 1

        if not checkpoint_records or checkpoint_records[-1]["index"] != checkpoint_index - 1:
            learning_rate = float(optimizer.param_groups[0]["lr"])
            checkpoint_records.append(
                self._save_checkpoint_pair(
                    model=model,
                    checkpoints_dir=checkpoints_dir,
                    checkpoint_index=checkpoint_index,
                    global_step=global_step,
                    epoch=epochs - 1,
                    learning_rate=learning_rate,
                )
            )

        return checkpoint_records
