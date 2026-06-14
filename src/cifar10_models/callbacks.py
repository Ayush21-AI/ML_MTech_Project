"""Training callbacks for checkpointing, early stopping, and experiment tracking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger("cifar10_models")


class Callback:
    """Base training callback."""

    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        metrics: dict[str, float],
    ) -> None:
        pass

    def on_train_end(self, model: nn.Module) -> None:
        pass


class CheckpointCallback(Callback):
    """Save the best model based on a validation metric."""

    def __init__(
        self,
        checkpoint_dir: Path,
        model_name: str,
        metric: str = "val_acc",
        mode: str = "max",
        save_last: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.metric = metric
        self.mode = mode
        self.save_last = save_last
        self.best_value = -float("inf") if mode == "max" else float("inf")
        self.best_path: Path | None = None
        self.last_path = self.checkpoint_dir / f"{model_name}_last.pt"

    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        metrics: dict[str, float],
    ) -> None:
        value = metrics.get(self.metric)
        if value is None:
            return

        state = {
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
        }

        if self.save_last:
            torch.save(state, self.last_path)

        if self._is_better(value):
            self.best_value = value
            self.best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(state, self.best_path)
            logger.info(
                "Epoch %d: new best %s=%.4f → saved %s",
                epoch,
                self.metric,
                value,
                self.best_path,
            )


class EarlyStoppingCallback(Callback):
    """Stop training when a metric stops improving."""

    def __init__(
        self,
        metric: str = "val_acc",
        mode: str = "max",
        patience: int = 10,
    ) -> None:
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.counter = 0
        self.best_value = -float("inf") if mode == "max" else float("inf")
        self.should_stop = False

    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        metrics: dict[str, float],
    ) -> None:
        value = metrics.get(self.metric)
        if value is None:
            return

        if self._is_better(value):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement",
                    self.patience,
                )
                self.should_stop = True


class MetricsLogger(Callback):
    """Save metrics to a JSON lines file."""

    def __init__(self, log_dir: Path, model_name: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{model_name}_metrics.jsonl"

    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        metrics: dict[str, float],
    ) -> None:
        record = {"epoch": epoch, **metrics}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")


class MLFlowLogger(Callback):
    """Log metrics and parameters to MLflow."""

    def __init__(self, experiment_name: str, run_name: str | None, config: dict[str, Any]) -> None:
        self.config = config
        self._active = False
        try:
            import mlflow

            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name)
            mlflow.log_params(self._flatten(config))
            self._active = True
        except Exception as exc:
            logger.warning("MLflow initialization failed: %s", exc)

    def _flatten(self, d: dict[str, Any], parent: str = "") -> dict[str, Any]:
        items: dict[str, Any] = {}
        for k, v in d.items():
            key = f"{parent}.{k}" if parent else k
            if isinstance(v, dict):
                items.update(self._flatten(v, key))
            else:
                items[key] = v
        return items

    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        metrics: dict[str, float],
    ) -> None:
        if not self._active:
            return
        import mlflow

        mlflow.log_metrics(metrics, step=epoch)

    def on_train_end(self, model: nn.Module) -> None:
        if not self._active:
            return
        import mlflow

        mlflow.end_run()


class WandbLogger(Callback):
    """Log metrics and parameters to Weights & Biases."""

    def __init__(self, project: str, run_name: str | None, config: dict[str, Any]) -> None:
        self.config = config
        self._active = False
        try:
            import wandb

            wandb.init(project=project, name=run_name, config=config)
            self._active = True
        except Exception as exc:
            logger.warning("W&B initialization failed: %s", exc)

    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        metrics: dict[str, float],
    ) -> None:
        if not self._active:
            return
        import wandb

        wandb.log(metrics, step=epoch)

    def on_train_end(self, model: nn.Module) -> None:
        if not self._active:
            return
        import wandb

        wandb.finish()
