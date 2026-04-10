"""Shared wiring for testModels run_index / run_query (not part of core src/ API)."""

from __future__ import annotations

import os
from typing import Any

import yaml

from src.config_utils import resolve_checkpoints


def load_yaml_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def abs_config_path(config_path: str) -> str:
    return os.path.abspath(config_path)


def outputs_dir_from_cfg(cfg: dict[str, Any]) -> str:
    """Resolve ``paths.outputs_dir`` relative to process cwd if not absolute."""
    p = cfg.get("paths", {}).get("outputs_dir", "outputs")
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(os.getcwd(), p))


def ghost_settings(cfg: dict[str, Any]) -> tuple[Any, str, int]:
    """Return (projection_dim, projection_type, projection_seed).

    If ``ghost.projection_dim`` is omitted, defaults to ``1280``.
    If set to YAML ``null``, returns ``None`` (no projection when dim is large enough).
    """
    g = cfg.get("ghost") or {}
    if "projection_dim" in g:
        pdim: Any = g["projection_dim"]
    else:
        pdim = 1280
    return (
        pdim,
        g.get("projection_type", "sjlt"),
        int(g.get("projection_seed", 42)),
    )


def build_checkpoints_list(config_path: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Explicit YAML list or auto-resolve ``checkpoints/`` next to config."""
    if cfg.get("checkpoints") and isinstance(cfg["checkpoints"], list):
        out = []
        for ck in cfg["checkpoints"]:
            entry: dict[str, Any] = {
                "weights_path": ck["weights_path"],
                "learning_rate": float(ck.get("learning_rate", 1.0)),
            }
            if ck.get("optimizer_state_path"):
                entry["optimizer_state_path"] = ck["optimizer_state_path"]
            out.append(entry)
        return out
    return resolve_checkpoints(abs_config_path(config_path), cfg)
