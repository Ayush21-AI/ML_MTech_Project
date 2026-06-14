"""Evaluation and test-time augmentation helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from cifar10_models.metrics import evaluate_loader
from cifar10_models.train import load_checkpoint

logger = logging.getLogger("cifar10_models")


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    amp_manager=None,
) -> tuple[float, float]:
    """Evaluate a model on the test set.

    Parameters
    ----------
    model: nn.Module
        Trained model.
    test_loader: DataLoader
        Test data loader.
    device: torch.device
        Device to evaluate on.
    amp_manager: AMPManager | None
        Optional AMP manager.

    Returns
    -------
    tuple[float, float]
        ``(test_loss, test_accuracy)``.
    """
    criterion = nn.CrossEntropyLoss()
    loss, accuracy = evaluate_loader(model, test_loader, criterion, device, amp_manager)
    logger.info("Test loss=%.4f accuracy=%.4f", loss, accuracy)
    return loss, accuracy


def _tta_basic(inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
    return model(inputs)


def _tta_mirror(inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
    return 0.5 * model(inputs) + 0.5 * model(inputs.flip(-1))


def _tta_mirror_translate(inputs: torch.Tensor, model: nn.Module) -> torch.Tensor:
    logits = _tta_mirror(inputs, model)
    padded = F.pad(inputs, (1,) * 4, "reflect")
    views = [padded[:, :, 0:32, 0:32], padded[:, :, 2:34, 2:34]]
    translated = torch.stack([_tta_mirror(v, model) for v in views]).mean(0)
    return 0.5 * logits + 0.5 * translated


def evaluate_with_tta(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    tta_level: int = 1,
    amp_manager=None,
) -> float:
    """Evaluate with test-time augmentation.

    Parameters
    ----------
    model: nn.Module
        Trained model.
    test_loader: DataLoader
        Test data loader.
    device: torch.device
        Device.
    tta_level: int
        ``0``: no TTA, ``1``: horizontal flip, ``2``: flip + 1px translations.
    amp_manager: AMPManager | None
        Optional AMP manager.

    Returns
    -------
    float
        Test accuracy.
    """
    model.eval()
    infer_fn = [_tta_basic, _tta_mirror, _tta_mirror_translate][min(tta_level, 2)]

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if amp_manager is not None:
                with amp_manager.autocast():
                    outputs = infer_fn(inputs, model)
            else:
                outputs = infer_fn(inputs, model)

            predicted = outputs.argmax(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    accuracy = correct / total
    logger.info("TTA level %d test accuracy=%.4f", tta_level, accuracy)
    return accuracy
