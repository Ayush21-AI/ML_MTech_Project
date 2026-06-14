# CIFAR-10 EDA & Production Modeling

A production-grade CIFAR-10 research project with a unique focus: **comparing patch-based architectures** (PatchCNN, ConvMixer, ViT) against a strong CNN baseline (ResNet-18), all trained with modern production recipes on the local CIFAR-10 archive.

This is **not** a generic ResNet training repo. It is designed as a lightweight, transparent, plain-PyTorch testbed for studying how different patch-based inductive biases perform on small-scale image classification.

## What makes this project unique

- **Patch-architecture comparison**: Patch-based CNN, ConvMixer, Vision Transformer (with optional CNN stem + stochastic depth), and ResNet-18 in one factory.
- **Config-driven training**: YAML configs + CLI overrides, no heavy framework required.
- **Modern training recipe**: AdamW, warmup + cosine LR, gradient clipping, AMP, `torch.compile` support, EMA, early stopping, best checkpointing.
- **Modern augmentation**: RandAugment, CutMix, Mixup, RandomErasing, label smoothing.
- **Production tooling**: torchmetrics, MLflow / W&B logging, ONNX export + runtime parity check, DDP-ready launcher, test-time augmentation.
- **Pure plain PyTorch**: no Lightning, no Hydra, no timm — full control and minimal dependencies.

## Structure

```
Mtech_Project/
├── configs/
│   ├── default.yaml              # ConvMixer default recipe
│   ├── patch_cnn.yaml
│   ├── vit.yaml
│   ├── resnet18.yaml
│   └── ...
├── notebooks/
│   ├── CIFAR_EDA.ipynb           # Thin EDA orchestrator
│   └── CIFAR_Models.ipynb        # Thin training / export orchestrator
├── src/
│   ├── cifar10_eda/              # EDA package (unchanged)
│   └── cifar10_models/           # PyTorch modeling package
│       ├── __init__.py
│       ├── __main__.py           # python -m cifar10_models
│       ├── cli.py                # argparse CLI
│       ├── config.py             # dataclass-driven config
│       ├── data.py               # DataLoader + train/val split + DDP sampler
│       ├── augmentation.py       # RandAugment, CutMix, Mixup
│       ├── train.py              # Production training loop
│       ├── evaluate.py           # Evaluation + TTA
│       ├── metrics.py            # torchmetrics + simple tracker
│       ├── optim.py              # optimizer/scheduler/EMA/AMP
│       ├── export.py             # ONNX export + parity check
│       ├── distributed.py        # DDP helpers
│       ├── callbacks.py          # checkpointing, early stopping, loggers
│       ├── utils.py              # seed, YAML loading
│       └── models/
│           ├── __init__.py
│           ├── model_factory.py  # build_model(name, config)
│           ├── patch_cnn.py      # Patch-based CNN
│           ├── convmixer.py      # ConvMixer (strong patch-CNN baseline)
│           ├── vit.py            # Vision Transformer
│           └── resnet.py         # CIFAR-10 ResNet-18
├── tests/
│   ├── test_data_loader.py       # EDA data sanity
│   ├── test_models.py            # All model forward + train step
│   ├── test_config.py            # Config loading/validation
│   ├── test_export.py            # ONNX parity for every model
│   └── test_training.py          # One-epoch fast-dev run
├── pyproject.toml                # Dependencies + CLI entry point
└── README.md                     # This file
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies in editable mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Place the CIFAR-10 archive at the project root:
   ```
   Mtech_Project/cifar-10-python.tar.gz
   ```
   If it is missing, the notebooks / CLI will raise a clear `FileNotFoundError`.

## Running the EDA

```bash
jupyter notebook notebooks/CIFAR_EDA.ipynb
```

## Training from the command line

Train the default ConvMixer recipe:

```bash
python -m cifar10_models --config configs/default.yaml
```

Override any config field from the CLI:

```bash
python -m cifar10_models \
  --config configs/vit.yaml \
  --override model.use_cnn_stem=true \
  --override epochs=10 \
  --override data.batch_size=64
```

Fast dev run (small subset, useful for debugging):

```bash
python -m cifar10_models \
  --config configs/default.yaml \
  --override data.fast_dev_run=true \
  --override data.fast_dev_size=500 \
  --override epochs=2
```

Export to ONNX and evaluate with test-time augmentation:

```bash
python -m cifar10_models \
  --config configs/default.yaml \
  --export-onnx \
  --test-tta
```

## Training from the notebook

```bash
jupyter notebook notebooks/CIFAR_Models.ipynb
```

The notebook loads a config, builds the model, trains, evaluates, optionally runs TTA, and exports to ONNX.

## Distributed training

Launch with `torchrun` on a multi-GPU machine:

```bash
torchrun --nproc_per_node=2 -m cifar10_models \
  --config configs/default.yaml \
  --distributed \
  --override data.batch_size=256
```

## Experiment tracking

Enable MLflow or W&B via config overrides:

```bash
python -m cifar10_models \
  --config configs/default.yaml \
  --override logging.use_mlflow=true \
  --override logging.experiment_name=cifar10-convmixer
```

```bash
python -m cifar10_models \
  --config configs/default.yaml \
  --override logging.use_wandb=true \
  --override logging.experiment_name=cifar10-convmixer
```

## Running tests

```bash
pytest
```

## Design notes

- All paths are resolved relative to the project root via `pathlib`.
- Archive extraction is idempotent: it only runs if the extracted files are missing.
- Random seeds are set deterministically for reproducibility.
- PyTorch device selection is automatic: MPS → CUDA → CPU.
- The training loop uses AMP, EMA, gradient clipping, and cosine LR with linear warmup.
- Checkpoints save the best validation model and the last epoch.
- ONNX export bakes normalization into the graph and runs an ONNX Runtime parity check.
- No Google Drive dependency remains.
