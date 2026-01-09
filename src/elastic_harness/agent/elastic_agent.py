"""Elastic agent configuration and distributed environment setup.

This module provides utilities for configuring and initializing PyTorch's
elastic distributed training with etcd rendezvous backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class ElasticTrainingConfig:
    """Configuration for elastic distributed training.

    Attributes:
        min_nodes: Minimum number of nodes required to start training.
        max_nodes: Maximum number of nodes allowed in the training job.
        rdzv_backend: Rendezvous backend type (default: etcd-v2).
        rdzv_endpoint: Rendezvous endpoint address (default: localhost:2379).
        rdzv_id: Unique identifier for this training job.
        heartbeat_interval: Interval between heartbeats in seconds (default: 30).
        heartbeat_timeout: Timeout for missed heartbeats in seconds (default: 90).
        nproc_per_node: Number of processes per node (typically GPUs per node).
        max_restarts: Maximum number of worker restarts on failure.
    """

    min_nodes: int = 1
    max_nodes: int = 4
    rdzv_backend: str = "etcd-v2"
    rdzv_endpoint: str = "localhost:2379"
    rdzv_id: str = "elastic-training"
    heartbeat_interval: int = 30
    heartbeat_timeout: int = 90
    nproc_per_node: int = field(default_factory=lambda: torch.cuda.device_count() or 1)
    max_restarts: int = 100

    def to_torchrun_args(self) -> list[str]:
        """Convert config to torchrun command line arguments.

        Returns:
            List of command line arguments for torchrun.
        """
        return [
            f"--nnodes={self.min_nodes}:{self.max_nodes}",
            f"--nproc-per-node={self.nproc_per_node}",
            f"--rdzv-backend={self.rdzv_backend}",
            f"--rdzv-endpoint={self.rdzv_endpoint}",
            f"--rdzv-id={self.rdzv_id}",
            f"--rdzv-conf=timeout={self.heartbeat_timeout},last_call_timeout=30",
            f"--max-restarts={self.max_restarts}",
        ]


@dataclass
class WorldInfo:
    """Information about the distributed world.

    Attributes:
        rank: Global rank of this process.
        local_rank: Local rank on this node.
        world_size: Total number of processes.
        local_world_size: Number of processes on this node.
        master_addr: Address of the master process.
        master_port: Port of the master process.
    """

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    master_addr: str
    master_port: int


def get_world_info() -> WorldInfo:
    """Get distributed world information from environment variables.

    This function reads the environment variables set by torchrun to determine
    the distributed configuration.

    Returns:
        WorldInfo containing rank, world size, and other distributed info.

    Raises:
        RuntimeError: If required environment variables are not set.
    """
    required_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    missing = [var for var in required_vars if var not in os.environ]

    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {missing}. "
            "Ensure you are running with torchrun or torch.distributed.launch."
        )

    return WorldInfo(
        rank=int(os.environ["RANK"]),
        local_rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
        master_addr=os.environ["MASTER_ADDR"],
        master_port=int(os.environ["MASTER_PORT"]),
    )


def setup_distributed_environment(
    backend: str = "nccl",
    init_method: str | None = None,
) -> WorldInfo:
    """Initialize the distributed environment for training.

    This function sets up PyTorch's distributed process group using the
    environment variables set by torchrun. It handles both GPU (NCCL) and
    CPU (Gloo) backends.

    Args:
        backend: The distributed backend to use. Options are 'nccl' (GPU),
            'gloo' (CPU/GPU), or 'auto' to auto-detect.
        init_method: Optional initialization method. If None, uses env:// method.

    Returns:
        WorldInfo containing the distributed configuration.

    Raises:
        RuntimeError: If distributed setup fails.
    """
    world_info = get_world_info()

    # Auto-detect backend if requested
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method or "env://",
            world_size=world_info.world_size,
            rank=world_info.rank,
        )

    # Set the device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(world_info.local_rank)

    return world_info


def cleanup_distributed() -> None:
    """Clean up the distributed environment.

    This function destroys the process group and should be called at the end
    of training to ensure clean shutdown.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0).

    Returns:
        True if this is rank 0 or if distributed is not initialized.
    """
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get the global rank of this process.

    Returns:
        Global rank, or 0 if distributed is not initialized.
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the total number of processes in the distributed world.

    Returns:
        World size, or 1 if distributed is not initialized.
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """Synchronize all processes.

    This function blocks until all processes have reached this point.
    """
    if dist.is_initialized():
        dist.barrier()


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a Python object from source rank to all other ranks.

    Args:
        obj: The object to broadcast (only used on source rank).
        src: The source rank to broadcast from.

    Returns:
        The broadcasted object on all ranks.
    """
    if not dist.is_initialized():
        return obj

    object_list = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]
