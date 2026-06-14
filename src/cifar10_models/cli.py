"""Command-line interface for training CIFAR-10 models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from cifar10_models import (
    build_model,
    evaluate,
    evaluate_with_tta,
    export_to_onnx,
    fit,
    get_dataloaders,
    set_seed,
    setup_logging,
)
from cifar10_models.config import TrainConfig
from cifar10_models.distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_distributed,
    setup_distributed,
    setup_device_for_distributed,
)
from cifar10_models.export import NormalizeWrapper
from cifar10_models.optim import AMPManager
from cifar10_models.utils import load_config, save_config

logger = logging.getLogger("cifar10_models")


def _parse_overrides(overrides: list[str]) -> dict[str, Any]:
    """Parse dotted CLI overrides into a nested dict.

    Example: ``--override model.name=vit`` becomes ``{"model": {"name": "vit"}}``.
    """
    result: dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}', expected key=value")
        key, value = override.split("=", 1)
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})

        # Try common type casts.
        if value.lower() in {"true", "false"}:
            typed_value = value.lower() == "true"
        else:
            try:
                typed_value = int(value)
            except ValueError:
                try:
                    typed_value = float(value)
                except ValueError:
                    typed_value = value
        current[parts[-1]] = typed_value
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CIFAR-10 patch-based models",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config file",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted config override, e.g. model.name=vit epochs=10",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the trained model to ONNX after training",
    )
    parser.add_argument(
        "--test-tta",
        action="store_true",
        help="Evaluate with test-time augmentation after training",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run in distributed mode (launched with torchrun)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.distributed:
        setup_distributed()
        device = setup_device_for_distributed()
    else:
        device = None

    overrides = _parse_overrides(args.override)
    config = load_config(args.config, overrides)
    if device is not None:
        config.device = device

    setup_logging()
    set_seed(config.seed)

    rank = get_rank()
    if rank == 0:
        save_config(config, config.logging.checkpoint_dir / "config.yaml")

    logger.info("Using device: %s", config.device)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        augmentation_cfg=config.augmentation,
        data_cfg=config.data,
        num_classes=config.model.num_classes,
        distributed=args.distributed,
        rank=rank,
        world_size=get_world_size(),
    )

    model = build_model(config.model)
    logger.info("Model: %s | Params: %.2fM", config.model.name, model.count_parameters() / 1e6)

    history = fit(model, train_loader, val_loader, config)

    if rank == 0:
        amp_manager = AMPManager(config.device.type, enabled=config.use_amp)
        test_loss, test_acc = evaluate(model, test_loader, config.device, amp_manager)
        logger.info("Final test accuracy: %.4f", test_acc)

        if args.test_tta:
            tta_acc = evaluate_with_tta(model, test_loader, config.device, tta_level=2, amp_manager=amp_manager)
            logger.info("TTA test accuracy: %.4f", tta_acc)

        if args.export_onnx:
            export_path = config.logging.checkpoint_dir / f"{config.model.name}.onnx"
            export_to_onnx(model, export_path, config.export.opset_version, config.export.dynamic_batch)

    if args.distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
