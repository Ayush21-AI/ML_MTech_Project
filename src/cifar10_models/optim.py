"""Optimizer, scheduler, EMA, and AMP helpers."""

from __future__ import annotations

import math

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)


def get_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> optim.Optimizer:
    """Create an optimizer for the model."""
    if optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int,
    lr_min: float,
) -> optim.lr_scheduler.LRScheduler | None:
    """Create a learning-rate scheduler with optional linear warmup.

    For the ``cosine`` scheduler, linear warmup is followed by cosine decay,
    all stepped per batch.

    Returns
    -------
    optim.lr_scheduler.LRScheduler | None
        The scheduler, or ``None`` for ``scheduler_name == "none"``.
    """
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if scheduler_name == "cosine":
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=max(warmup_steps, 1),
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=lr_min,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    if scheduler_name == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=[pg["lr"] for pg in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=warmup_epochs / max(epochs, 1),
            div_factor=25.0,
            final_div_factor=1e4,
        )

    return None


class EMAModel:
    """Exponential moving average shadow model wrapper.

    Uses PyTorch's built-in AveragedModel with an EMA average function.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay),
        )

    def update(self, model: nn.Module) -> None:
        """Update the EMA parameters from the live model."""
        self.ema_model.update_parameters(model)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.ema_model.load_state_dict(state_dict)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


class AMPManager:
    """Wrapper for automatic mixed precision training."""

    def __init__(self, device_type: str, enabled: bool = True) -> None:
        self.device_type = device_type
        self.enabled = enabled
        self.scaler = torch.amp.GradScaler(device_type, enabled=enabled)

    def autocast(self):
        """Return an autocast context manager for the configured device."""
        return torch.amp.autocast(self.device_type, enabled=self.enabled)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return self.scaler.scale(loss)

    def step(self, optimizer: optim.Optimizer) -> None:
        self.scaler.step(optimizer)
        self.scaler.update()

    def unscale(self, optimizer: optim.Optimizer) -> None:
        self.scaler.unscale_(optimizer)

    def state_dict(self) -> dict:
        return {"scaler": self.scaler.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.scaler.load_state_dict(state_dict["scaler"])
