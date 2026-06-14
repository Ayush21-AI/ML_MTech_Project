"""End-to-end fast training sanity test."""

from __future__ import annotations

import pytest

from cifar10_models import fit
from cifar10_models.config import DataConfig, TrainConfig
from cifar10_models.data import get_dataloaders
from cifar10_models.evaluate import evaluate
from cifar10_models.models.model_factory import build_model
from cifar10_models.optim import AMPManager


def test_fast_training_run() -> None:
    """A fast dev run should complete one epoch and produce eval metrics."""
    config = TrainConfig(
        data=DataConfig(
            batch_size=32,
            fast_dev_run=True,
            fast_dev_size=100,
            num_workers=0,
            pin_memory=False,
        ),
        epochs=1,
        use_amp=False,
        compile_model=False,
    )
    config.optimizer.use_ema = False

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        augmentation_cfg=config.augmentation,
        data_cfg=config.data,
        num_classes=config.model.num_classes,
    )
    model = build_model(config.model)

    history, eval_model = fit(model, train_loader, val_loader, config)

    assert len(history["train_loss"]) == 1
    assert len(history["val_acc"]) == 1

    amp_manager = AMPManager(config.device.type, enabled=False)
    test_loss, test_acc = evaluate(eval_model, test_loader, config.device, amp_manager)
    assert 0 <= test_acc <= 1
