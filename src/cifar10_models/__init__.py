"""Production PyTorch modeling package for CIFAR-10."""

from cifar10_models.config import (
    AugmentationConfig,
    DataConfig,
    ExportConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    TrainConfig,
    get_device,
)
from cifar10_eda import NUM_CLASSES
from cifar10_models.data import get_dataloaders
from cifar10_models.evaluate import evaluate, evaluate_with_tta
from cifar10_models.export import export_to_onnx
from cifar10_models.models.model_factory import build_model, list_models
from cifar10_models.train import fit
from cifar10_models.utils import set_seed, setup_logging

__all__ = [
    "AugmentationConfig",
    "DataConfig",
    "ExportConfig",
    "LoggingConfig",
    "ModelConfig",
    "NUM_CLASSES",
    "OptimizerConfig",
    "TrainConfig",
    "get_device",
    "get_dataloaders",
    "evaluate",
    "evaluate_with_tta",
    "export_to_onnx",
    "build_model",
    "list_models",
    "fit",
    "set_seed",
    "setup_logging",
]
