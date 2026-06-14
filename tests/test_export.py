"""ONNX export sanity tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cifar10_models.config import ModelConfig
from cifar10_models.export import export_to_onnx
from cifar10_models.models.model_factory import build_model, list_models


@pytest.mark.parametrize("name", list_models())
def test_export_onnx(tmp_path: Path, name: str) -> None:
    """Each model should export to ONNX and pass the runtime parity check."""
    torch.manual_seed(0)
    model = build_model(ModelConfig(name=name))
    export_path = tmp_path / f"{name}.onnx"
    exported = export_to_onnx(model, export_path, opset_version=17, dynamic_batch=True, verify=True)
    assert exported.is_file()
