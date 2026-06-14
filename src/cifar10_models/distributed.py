"""Distributed Data Parallel helpers."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, Subset


def is_distributed() -> bool:
    """Return True if the current process is part of a DDP group."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the current DDP rank, or 0 if not distributed."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Return the world size, or 1 if not distributed."""
    return dist.get_world_size() if is_distributed() else 1


def setup_distributed(backend: str = "nccl") -> None:
    """Initialize a DDP process group from environment variables.

    Expected env vars set by ``torchrun``:
    ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR``, ``MASTER_PORT``.
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available on this system")

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "RANK and WORLD_SIZE environment variables must be set. "
            "Launch with torchrun."
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup_distributed() -> None:
    """Destroy the DDP process group if initialized."""
    if is_distributed():
        dist.destroy_process_group()


def get_distributed_sampler(
    dataset: Subset,
    shuffle: bool,
    distributed: bool,
    rank: int,
    world_size: int,
    seed: int,
) -> DistributedSampler | None:
    """Build a DistributedSampler for DDP or return None for single-process."""
    if not distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
    )


def setup_device_for_distributed() -> torch.device:
    """Set the device for the current DDP rank."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")
