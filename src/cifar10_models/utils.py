"""Shared utility helpers (logging, reproducibility, config loading)."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from cifar10_models.config import TrainConfig


def setup_logging(level: int = logging.INFO, fmt: str | None = None) -> logging.Logger:
    """Configure a consistent logger for the modeling package."""
    if fmt is None:
        fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    logger = logging.getLogger("cifar10_models")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: Path,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """Load a YAML config and apply optional overrides."""
    raw = load_yaml(config_path)
    if overrides:
        raw = deep_merge(raw, overrides)
    return TrainConfig.from_dict(raw)


def save_config(config: TrainConfig, output_path: Path) -> None:
    """Save a config to YAML for reproducibility."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False)
