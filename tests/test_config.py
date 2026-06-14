"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from cifar10_models.config import (
    AugmentationConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    TrainConfig,
)
from cifar10_models.utils import load_config, save_config


def test_model_config_validation() -> None:
    with pytest.raises(ValueError):
        ModelConfig(name="unknown_model")


def test_optimizer_config_validation() -> None:
    with pytest.raises(ValueError):
        OptimizerConfig(optimizer="lamb")
    with pytest.raises(ValueError):
        OptimizerConfig(scheduler="plateau")


def test_train_config_roundtrip(tmp_path: Path) -> None:
    config = TrainConfig(
        model=ModelConfig(name="vit"),
        data=DataConfig(batch_size=64),
        augmentation=AugmentationConfig(cutmix=False),
    )
    path = tmp_path / "config.yaml"
    save_config(config, path)
    loaded = load_config(path)
    assert loaded.model.name == "vit"
    assert loaded.data.batch_size == 64
    assert loaded.augmentation.cutmix is False


def test_load_default_config() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.model.name == "convmixer"
    assert config.data.batch_size == 128
