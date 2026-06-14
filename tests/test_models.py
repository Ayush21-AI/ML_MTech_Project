"""Sanity tests for the PyTorch CIFAR-10 models."""

from __future__ import annotations

import pytest
import torch

from cifar10_models import NUM_CLASSES
from cifar10_models.config import ModelConfig, get_device
from cifar10_models.models.model_factory import build_model, list_models


@pytest.fixture
def batch() -> torch.Tensor:
    """Small deterministic input batch for model tests."""
    torch.manual_seed(0)
    return torch.randn(4, 3, 32, 32)


@pytest.mark.parametrize("name", list_models())
def test_model_forward_shape(name: str, batch: torch.Tensor) -> None:
    """Every model must produce logits of shape (B, NUM_CLASSES)."""
    config = ModelConfig(name=name)
    model = build_model(config)
    logits = model(batch)
    assert logits.shape == (batch.shape[0], NUM_CLASSES)


@pytest.mark.parametrize("name", list_models())
def test_model_training_step(name: str, batch: torch.Tensor) -> None:
    """Every model must support one optimizer step."""
    config = ModelConfig(name=name)
    model = build_model(config)
    model.train()
    labels = torch.randint(0, NUM_CLASSES, (batch.shape[0],))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    logits = model(batch)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("name", list_models())
def test_model_on_device(name: str, batch: torch.Tensor) -> None:
    """Every model must run on the configured device."""
    device = get_device()
    config = ModelConfig(name=name)
    model = build_model(config).to(device)
    batch_device = batch.to(device)
    logits = model(batch_device)
    assert logits.shape == (batch.shape[0], NUM_CLASSES)
    assert logits.device == batch_device.device


def test_unknown_model() -> None:
    """Building an unknown model must raise ValueError."""
    with pytest.raises(ValueError):
        build_model(ModelConfig(name="not_a_model"))
