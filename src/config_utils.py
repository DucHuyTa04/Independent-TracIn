"""Training checkpoint helpers and config resolution for TracIn Ghost.

Convention: place ``config.yaml`` next to a ``checkpoints/`` folder, or set
``checkpoints_dir`` in YAML to override.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

METADATA_FILENAME = "tracin_checkpoints_metadata.json"


def smart_load_weights_into_model(model: nn.Module, path: str, device: str) -> None:
    """Load weights from a bare ``state_dict`` or unified TracIn checkpoint dict."""
    data = torch.load(path, map_location=device, weights_only=False)
    if isinstance(data, dict) and "model_state_dict" in data:
        model.load_state_dict(data["model_state_dict"])
    else:
        model.load_state_dict(data)


def find_target_layer(model: nn.Module) -> tuple[str, nn.Module]:
    """Return ``(name, module)`` of the last ``nn.Linear`` (standard TracIn choice)."""
    last: Optional[tuple[str, nn.Module]] = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last = (name, mod)
    if last is None:
        raise ValueError(
            "No nn.Linear layer found in model; set target_layer in config to hook a layer."
        )
    return last


# Backward-compatible name
find_last_linear_layer = find_target_layer


def resolve_target_layer(model: nn.Module, layer_name: Optional[str]) -> tuple[str, nn.Module]:
    """Resolve ``layer_name`` to a module, or use last ``nn.Linear`` if ``None``."""
    if not layer_name or not str(layer_name).strip():
        return find_target_layer(model)
    name = str(layer_name).strip()
    mod = model
    for part in name.split("."):
        if not part:
            continue
        mod = getattr(mod, part)
    if not isinstance(mod, nn.Module):
        raise TypeError(f"target_layer {name!r} is not an nn.Module")
    return name, mod


def find_adam_param_key(model: nn.Module, target_layer: nn.Module) -> int:
    """Optimizer state index for ``target_layer.weight`` (same order as ``named_parameters``)."""
    w = getattr(target_layer, "weight", None)
    if w is None:
        raise ValueError("target_layer has no .weight; cannot infer adam_param_key")
    for idx, (_, p) in enumerate(model.named_parameters()):
        if p is w:
            return idx
    raise ValueError("target_layer.weight not found in model.named_parameters()")


def find_adam_bias_param_key(
    model: nn.Module, target_layer: nn.Module,
) -> Optional[int]:
    """Optimizer state index for ``target_layer.bias``, or ``None`` if no bias."""
    b = getattr(target_layer, "bias", None)
    if b is None:
        return None
    for idx, (_, p) in enumerate(model.named_parameters()):
        if p is b:
            return idx
    return None


class TracInCheckpointCallback:
    """Save unified checkpoints + JSON metadata during training.

    Call ``on_epoch_end`` after each epoch, then ``finalize`` after the loop.
    Optionally ``select_best`` to move low-value checkpoints to ``.pruned/``.
    """

    def __init__(self, save_dir: str, save_every: int = 5) -> None:
        self.save_dir = save_dir
        self.save_every = max(1, int(save_every))
        self._records: list[dict[str, Any]] = []
        os.makedirs(self.save_dir, exist_ok=True)

    def _metadata_path(self) -> str:
        return os.path.join(self.save_dir, METADATA_FILENAME)

    def _write_metadata(self) -> None:
        path = self._metadata_path()
        sorted_records = sorted(self._records, key=lambda r: int(r["epoch"]))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted_records, f, indent=2)

    def _save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch_loss: float,
    ) -> str:
        lr = float(optimizer.param_groups[0]["lr"])
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "learning_rate": lr,
            "epoch_loss": float(epoch_loss),
        }
        fname = f"ckpt_{epoch}.pt"
        fpath = os.path.join(self.save_dir, fname)
        torch.save(payload, fpath)
        rec = {
            "epoch": int(epoch),
            "filename": fname,
            "learning_rate": lr,
            "epoch_loss": float(epoch_loss),
        }
        # Replace existing record for same epoch
        self._records = [r for r in self._records if int(r["epoch"]) != int(epoch)]
        self._records.append(rec)
        self._write_metadata()
        logger.info("TracIn checkpoint saved: %s (lr=%s, loss=%.6f)", fpath, lr, epoch_loss)
        return fpath

    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch_loss: float,
    ) -> Optional[str]:
        """Save every ``save_every`` epochs (including epoch 0). Returns path or None."""
        if int(epoch) % self.save_every != 0:
            return None
        return self._save_checkpoint(epoch, model, optimizer, epoch_loss)

    def finalize(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        epoch_loss: float,
    ) -> str:
        """Always save the last epoch (required for TracIn-last query)."""
        return self._save_checkpoint(int(epoch), model, optimizer, epoch_loss)

    def select_best(self, keep: int = 5) -> None:
        """Keep ``keep`` checkpoints: final + top (keep-1) others by loss drop vs previous.

        Moves other ``ckpt_*.pt`` files to ``checkpoints/.pruned/``.
        """
        keep = max(1, int(keep))
        if not self._records:
            logger.warning("select_best: no checkpoint records")
            return
        sorted_recs = sorted(self._records, key=lambda r: int(r["epoch"]))
        if len(sorted_recs) <= keep:
            return

        final = sorted_recs[-1]
        # Candidates: all except final. Score by loss drop from previous checkpoint in time order.
        scored: list[tuple[float, dict[str, Any]]] = []
        for i, r in enumerate(sorted_recs[:-1]):
            if i == 0:
                scored.append((float("inf"), r))
            else:
                prev_loss = float(sorted_recs[i - 1]["epoch_loss"])
                cur_loss = float(r["epoch_loss"])
                scored.append((prev_loss - cur_loss, r))

        scored.sort(key=lambda x: -x[0])
        take = min(keep - 1, len(scored))
        selected = [r for _, r in scored[:take]]
        selected.append(final)
        selected = sorted({int(r["epoch"]): r for r in selected}.values(), key=lambda r: r["epoch"])

        keep_epochs = {int(r["epoch"]) for r in selected}
        pruned_dir = os.path.join(self.save_dir, ".pruned")
        os.makedirs(pruned_dir, exist_ok=True)

        for r in list(sorted_recs):
            ep = int(r["epoch"])
            if ep in keep_epochs:
                continue
            fname = r.get("filename") or f"ckpt_{ep}.pt"
            src = os.path.join(self.save_dir, fname)
            if os.path.isfile(src):
                dst = os.path.join(pruned_dir, fname)
                shutil.move(src, dst)
                logger.info("Pruned checkpoint moved to %s", dst)

        self._records = selected
        self._write_metadata()


def resolve_checkpoints(
    config_path: str,
    cfg: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Build ``checkpoints`` list for ``build_index`` / ``attribute``.

    Reads ``tracin_checkpoints_metadata.json`` if present; otherwise scans
    ``ckpt_*.pt`` and legacy ``optim_*.pt`` peers.

    Args:
        config_path: Path to ``config.yaml`` (directory sets defaults).
        cfg: Optional loaded YAML dict; may contain ``checkpoints_dir``.

    Returns:
        List of dicts with keys ``weights_path``, ``learning_rate``, and optionally
        ``optimizer_state_path`` (same as weights for unified format).
    """
    cfg = cfg or {}
    config_dir = os.path.dirname(os.path.abspath(config_path))
    rel_ckpt = cfg.get("checkpoints_dir") or cfg.get("paths", {}).get("checkpoints_dir")
    if rel_ckpt:
        ckpt_dir = rel_ckpt if os.path.isabs(rel_ckpt) else os.path.normpath(
            os.path.join(config_dir, rel_ckpt)
        )
    else:
        ckpt_dir = os.path.join(config_dir, "checkpoints")

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}. "
            "Use TracInCheckpointCallback or place ckpt_*.pt under checkpoints/ next to config."
        )

    meta_path = os.path.join(ckpt_dir, METADATA_FILENAME)
    if os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        out: list[dict[str, Any]] = []
        for r in sorted(records, key=lambda x: int(x["epoch"])):
            fname = r["filename"]
            wpath = os.path.join(ckpt_dir, fname)
            lr = float(r.get("learning_rate", 1.0))
            out.append({
                "weights_path": wpath,
                "optimizer_state_path": wpath,
                "learning_rate": lr,
            })
        if not out:
            raise ValueError(f"Empty checkpoint metadata: {meta_path}")
        return out

    # Legacy: scan ckpt_*.pt
    ckpt_pattern = re.compile(r"^ckpt_(\d+)\.pt$")
    files = [f for f in os.listdir(ckpt_dir) if ckpt_pattern.match(f)]
    if not files:
        raise FileNotFoundError(f"No ckpt_*.pt files in {ckpt_dir}")

    def epoch_key(name: str) -> int:
        m = ckpt_pattern.match(name)
        return int(m.group(1)) if m else -1

    files.sort(key=epoch_key)
    out = []
    warned_lr = False
    for fname in files:
        wpath = os.path.join(ckpt_dir, fname)
        ep = epoch_key(fname)
        opt_name = f"optim_{ep}.pt"
        opt_path = os.path.join(ckpt_dir, opt_name)
        if os.path.isfile(opt_path):
            opt_use = opt_path
        else:
            opt_use = wpath
        # Detect unified vs legacy by loading keys only lightly
        try:
            data = torch.load(wpath, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load {wpath}: {e}") from e

        if isinstance(data, dict) and "model_state_dict" in data:
            lr = float(data.get("learning_rate", 1.0))
            out.append({
                "weights_path": wpath,
                "optimizer_state_path": wpath,
                "learning_rate": lr,
            })
        else:
            if not warned_lr:
                logger.warning(
                    "Legacy checkpoint format without metadata; using learning_rate=1.0 "
                    "for all checkpoints under %s",
                    ckpt_dir,
                )
                warned_lr = True
            out.append({
                "weights_path": wpath,
                "optimizer_state_path": opt_use if os.path.isfile(opt_use) else None,
                "learning_rate": 1.0,
            })
    return out


def last_checkpoint_paths(checkpoints: list[dict[str, Any]]) -> tuple[str, Optional[str]]:
    """Return (weights_path, optimizer_state_path) for the last checkpoint in time order."""
    if not checkpoints:
        raise ValueError("checkpoints list is empty")
    last = checkpoints[-1]
    return last["weights_path"], last.get("optimizer_state_path")
