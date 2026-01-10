"""Checkpoint manager for multi-tier checkpoint storage.

This module provides the main checkpoint management interface, coordinating
between in-memory snapshots, local NVMe, and S3 storage for optimal
recovery time and durability.
"""

from __future__ import annotations

import copy
import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from elastic_harness.checkpoint.memory_snapshot import MemorySnapshotBackend
from elastic_harness.checkpoint.storage_backends import NVMeBackend, S3Backend, StorageBackend
from elastic_harness.errors import CheckpointSaveError

logger = logging.getLogger(__name__)


class CheckpointTier(Enum):
    """Storage tier for checkpoints."""

    MEMORY = "memory"  # In-memory snapshot (fastest)
    NVME = "nvme"  # Local NVMe/SSD (fast)
    S3 = "s3"  # S3/cloud storage (durable)


class AsyncSavePolicy:
    """Policies for handling async save failures."""

    WARN = "warn"  # Log warning and continue (default, backwards compatible)
    FAIL = "fail"  # Raise exception on failure
    RETRY = "retry"  # Retry failed saves with exponential backoff


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint behavior.

    Attributes:
        checkpoint_interval: Steps between NVMe checkpoints.
        memory_snapshot_interval: Steps between in-memory snapshots.
        nvme_path: Local path for NVMe checkpoints.
        s3_bucket: S3 bucket for durable storage (optional).
        s3_prefix: S3 key prefix.
        async_save: Whether to use async saves for S3.
        keep_last_n: Number of recent checkpoints to keep.
        async_save_policy: How to handle async save failures ('warn', 'fail', 'retry').
        async_save_retries: Number of retry attempts for 'retry' policy.
        async_save_retry_delay: Base delay between retries in seconds.
    """

    checkpoint_interval: int = 500
    memory_snapshot_interval: int = 50
    nvme_path: str | Path = "/tmp/checkpoints"
    s3_bucket: str | None = None
    s3_prefix: str = ""
    async_save: bool = True
    keep_last_n: int = 3
    async_save_policy: str = AsyncSavePolicy.WARN
    async_save_retries: int = 3
    async_save_retry_delay: float = 5.0


@dataclass
class CheckpointState:
    """Complete training state for checkpoint.

    This dataclass captures all state needed to resume training,
    including model weights, optimizer state, data loader position,
    and RNG states.

    Attributes:
        step: Current training step.
        epoch: Current epoch.
        model_state_dict: Model parameters.
        optimizer_state_dict: Optimizer state.
        lr_scheduler_state_dict: Learning rate scheduler state.
        dataset_state: Data loader/dataset state.
        rng_states: Random number generator states.
        metrics: Training metrics at checkpoint time.
        world_size: World size when checkpoint was created.
        timestamp: Unix timestamp of checkpoint creation.
    """

    step: int = 0
    epoch: int = 0
    model_state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    optimizer_state_dict: dict[str, Any] = field(default_factory=dict)
    lr_scheduler_state_dict: dict[str, Any] = field(default_factory=dict)
    dataset_state: dict[str, Any] = field(default_factory=dict)
    rng_states: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    world_size: int = 1
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary for saving."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "lr_scheduler_state_dict": self.lr_scheduler_state_dict,
            "dataset_state": self.dataset_state,
            "rng_states": self.rng_states,
            "metrics": self.metrics,
            "world_size": self.world_size,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        """Deserialize state from dictionary."""
        return cls(
            step=data.get("step", 0),
            epoch=data.get("epoch", 0),
            model_state_dict=data.get("model_state_dict", {}),
            optimizer_state_dict=data.get("optimizer_state_dict", {}),
            lr_scheduler_state_dict=data.get("lr_scheduler_state_dict", {}),
            dataset_state=data.get("dataset_state", {}),
            rng_states=data.get("rng_states", {}),
            metrics=data.get("metrics", {}),
            world_size=data.get("world_size", 1),
            timestamp=data.get("timestamp", time.time()),
        )

    def cpu_copy(self) -> CheckpointState:
        """Create a copy with all tensors on CPU.

        Returns:
            New CheckpointState with CPU tensors.
        """

        def to_cpu(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().clone()
            elif isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(to_cpu(v) for v in obj)
            return obj

        return CheckpointState(
            step=self.step,
            epoch=self.epoch,
            model_state_dict=to_cpu(self.model_state_dict),
            optimizer_state_dict=to_cpu(self.optimizer_state_dict),
            lr_scheduler_state_dict=to_cpu(self.lr_scheduler_state_dict),
            dataset_state=copy.deepcopy(self.dataset_state),
            rng_states=copy.deepcopy(self.rng_states),
            metrics=copy.deepcopy(self.metrics),
            world_size=self.world_size,
            timestamp=self.timestamp,
        )


class CheckpointManager:
    """Manages checkpoint lifecycle across multiple storage tiers.

    This manager coordinates between in-memory snapshots, local NVMe storage,
    and cloud storage (S3) to provide optimal recovery time while ensuring
    durability. The recovery hierarchy is:

    1. Memory snapshot (instant recovery for non-fatal errors)
    2. Local NVMe (2-5 second recovery)
    3. S3 (minutes, but durable across node failures)

    Example:
        >>> config = CheckpointConfig(
        ...     nvme_path="/nvme/checkpoints",
        ...     s3_bucket="training-checkpoints",
        ... )
        >>> manager = CheckpointManager(config)
        >>> manager.save_checkpoint(state, tier=CheckpointTier.NVME)
        >>> # On recovery...
        >>> restored = manager.load_checkpoint()
    """

    def __init__(self, config: CheckpointConfig):
        """Initialize checkpoint manager.

        Args:
            config: Checkpoint configuration.
        """
        self.config = config

        # Initialize backends
        self.memory_backend = MemorySnapshotBackend(max_snapshots=2)
        self.nvme_backend = NVMeBackend(config.nvme_path)

        self.s3_backend: S3Backend | None = None
        if config.s3_bucket:
            self.s3_backend = S3Backend(config.s3_bucket, config.s3_prefix)

        # Track pending async saves with metadata for retry
        self._pending_saves: list[dict] = []  # [{future, path, state, retry_count}]

    def save_checkpoint(
        self,
        state: CheckpointState,
        tier: CheckpointTier = CheckpointTier.NVME,
    ) -> str | None:
        """Save checkpoint to specified storage tier.

        Args:
            state: The checkpoint state to save.
            tier: Storage tier to use.

        Returns:
            Checkpoint path/key for NVME/S3 tiers, None for memory.
        """
        if tier == CheckpointTier.MEMORY:
            self.memory_backend.save(state)
            logger.debug(f"Memory snapshot saved at step {state.step}")
            return None

        elif tier == CheckpointTier.NVME:
            path = f"checkpoint_step_{state.step:08d}.pt"
            self.nvme_backend.save(state, path)
            logger.info(f"NVMe checkpoint saved: {path}")

            # Cleanup old checkpoints
            deleted = self.nvme_backend.cleanup_old_checkpoints(self.config.keep_last_n)
            if deleted:
                logger.debug(f"Cleaned up old checkpoints: {deleted}")

            return path

        elif tier == CheckpointTier.S3:
            if self.s3_backend is None:
                raise ValueError("S3 backend not configured")

            path = f"checkpoint_step_{state.step:08d}.pt"

            if self.config.async_save:
                cpu_state = state.cpu_copy()  # Keep a copy for potential retry
                future = self.s3_backend.save_async(state, path)
                self._pending_saves.append({
                    "future": future,
                    "path": path,
                    "state": cpu_state,
                    "retry_count": 0,
                })
                logger.info(f"S3 checkpoint upload started (async): {path}")
            else:
                self.s3_backend.save(state, path)
                logger.info(f"S3 checkpoint saved: {path}")

            return path

        else:
            raise ValueError(f"Unknown checkpoint tier: {tier}")

    def load_checkpoint(
        self,
        path: str | None = None,
        tier: CheckpointTier | None = None,
    ) -> CheckpointState | None:
        """Load checkpoint with fallback hierarchy.

        If no path is specified, searches for checkpoints in order:
        1. Memory snapshot (if available)
        2. Latest NVMe checkpoint
        3. Latest S3 checkpoint

        Args:
            path: Optional specific checkpoint path to load.
            tier: Optional specific tier to load from.

        Returns:
            Loaded CheckpointState, or None if no checkpoint found.
        """
        # If specific path and tier given, load directly
        if path and tier:
            return self._load_from_tier(path, tier)

        # Try fallback hierarchy
        start_time = time.time()

        # 1. Try memory snapshot first
        state = self.memory_backend.load()
        if state:
            elapsed = time.time() - start_time
            logger.info(
                f"Loaded from memory snapshot (step {state.step}) in {elapsed:.3f}s"
            )
            return state

        # 2. Try NVMe
        latest_nvme = self.nvme_backend.get_latest_checkpoint()
        if latest_nvme:
            state = self.nvme_backend.load(latest_nvme)
            elapsed = time.time() - start_time
            logger.info(
                f"Loaded from NVMe checkpoint ({latest_nvme}) in {elapsed:.3f}s"
            )
            return state

        # 3. Try S3
        if self.s3_backend:
            latest_s3 = self.s3_backend.get_latest_checkpoint()
            if latest_s3:
                state = self.s3_backend.load(latest_s3)
                elapsed = time.time() - start_time
                logger.info(
                    f"Loaded from S3 checkpoint ({latest_s3}) in {elapsed:.3f}s"
                )
                return state

        logger.info("No checkpoint found")
        return None

    def _load_from_tier(self, path: str, tier: CheckpointTier) -> CheckpointState | None:
        """Load from a specific tier and path.

        Args:
            path: Checkpoint path.
            tier: Storage tier.

        Returns:
            Loaded CheckpointState, or None if not found.
        """
        if tier == CheckpointTier.MEMORY:
            return self.memory_backend.load()
        elif tier == CheckpointTier.NVME:
            if self.nvme_backend.exists(path):
                return self.nvme_backend.load(path)
        elif tier == CheckpointTier.S3:
            if self.s3_backend and self.s3_backend.exists(path):
                return self.s3_backend.load(path)
        return None

    def wait_for_pending_saves(self, timeout: float | None = None) -> bool:
        """Wait for all pending async saves to complete.

        Handles failures according to the configured async_save_policy:
        - 'warn': Log warning and continue (default)
        - 'fail': Raise CheckpointSaveError on failure
        - 'retry': Retry failed saves with exponential backoff

        Args:
            timeout: Maximum time to wait (None for infinite).

        Returns:
            True if all saves completed successfully, False if timeout or failures.

        Raises:
            CheckpointSaveError: If policy is 'fail' and any save fails.
        """
        from concurrent.futures import wait, ALL_COMPLETED

        if not self._pending_saves:
            return True

        all_success = True

        # Use iterative approach instead of recursion to avoid stack overflow
        while self._pending_saves:
            # Extract futures from pending save metadata
            futures = [save_info["future"] for save_info in self._pending_saves]

            done, not_done = wait(futures, timeout=timeout, return_when=ALL_COMPLETED)

            # Process completed saves
            failed_saves = []

            for save_info in self._pending_saves:
                future = save_info["future"]
                if future in done:
                    exc = future.exception()
                    if exc:
                        failed_saves.append((save_info, exc))
                    # Successfully completed saves don't need tracking

            # Handle failures based on policy
            new_pending = []

            for save_info, exc in failed_saves:
                path = save_info["path"]
                retry_count = save_info["retry_count"]
                policy = self.config.async_save_policy

                if policy == AsyncSavePolicy.FAIL:
                    raise CheckpointSaveError(
                        f"Async checkpoint save failed for '{path}': {exc}",
                        path=path,
                        original_error=exc,
                    )

                elif policy == AsyncSavePolicy.RETRY:
                    if retry_count < self.config.async_save_retries:
                        # Retry with exponential backoff
                        delay = self.config.async_save_retry_delay * (2 ** retry_count)
                        logger.warning(
                            f"Async save failed for '{path}' (attempt {retry_count + 1}), "
                            f"retrying in {delay:.1f}s: {exc}"
                        )
                        time.sleep(delay)

                        # Resubmit the save
                        state = save_info["state"]
                        new_future = self.s3_backend.save_async(state, path)
                        new_pending.append({
                            "future": new_future,
                            "path": path,
                            "state": state,
                            "retry_count": retry_count + 1,
                        })
                    else:
                        logger.error(
                            f"Async save failed for '{path}' after {retry_count + 1} attempts. "
                            f"Checkpoint may be incomplete: {exc}"
                        )
                        all_success = False

                else:  # WARN (default)
                    logger.warning(
                        f"Async save failed for '{path}': {exc}. "
                        "Training continues but checkpoint may be incomplete."
                    )
                    all_success = False

            # Keep track of saves still in progress
            for save_info in self._pending_saves:
                if save_info["future"] in not_done:
                    new_pending.append(save_info)

            self._pending_saves = new_pending

            # If no retries pending or not using retry policy, exit the loop
            if not new_pending or self.config.async_save_policy != AsyncSavePolicy.RETRY:
                break

        return all_success and len(self._pending_saves) == 0

    def clear_memory_snapshots(self) -> None:
        """Clear in-memory snapshots to free RAM.

        Call this after a successful persistent checkpoint.
        """
        self.memory_backend.clear()

    def should_checkpoint(self, step: int, tier: CheckpointTier) -> bool:
        """Check if a checkpoint should be saved at this step.

        Args:
            step: Current training step.
            tier: Storage tier to check.

        Returns:
            True if checkpoint should be saved.
        """
        if tier == CheckpointTier.MEMORY:
            return step > 0 and step % self.config.memory_snapshot_interval == 0
        elif tier in (CheckpointTier.NVME, CheckpointTier.S3):
            return step > 0 and step % self.config.checkpoint_interval == 0
        return False


def create_checkpoint_state(
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any | None = None,
    dataset_state: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
) -> CheckpointState:
    """Create a CheckpointState from training components.

    This is a convenience function for creating checkpoint states
    during training.

    Args:
        step: Current training step.
        model: The model (or DDP-wrapped model).
        optimizer: The optimizer.
        lr_scheduler: Optional learning rate scheduler.
        dataset_state: Optional dataset/dataloader state.
        metrics: Optional training metrics.

    Returns:
        CheckpointState ready for saving.
    """
    # Handle DDP-wrapped models
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    # Capture RNG states
    rng_states = {
        "python": None,  # Would need import random
        "numpy": None,  # Would need import numpy
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    return CheckpointState(
        step=step,
        epoch=0,  # Set by caller if needed
        model_state_dict=model_state,
        optimizer_state_dict=optimizer.state_dict(),
        lr_scheduler_state_dict=lr_scheduler.state_dict() if lr_scheduler else {},
        dataset_state=dataset_state or {},
        rng_states=rng_states,
        metrics=metrics or {},
        world_size=dist.get_world_size() if dist.is_initialized() else 1,
        timestamp=time.time(),
    )


def load_checkpoint_to_model(
    state: CheckpointState,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any | None = None,
    strict: bool = True,
) -> None:
    """Load checkpoint state into training components.

    Args:
        state: The checkpoint state to load.
        model: The model (or DDP-wrapped model).
        optimizer: The optimizer.
        lr_scheduler: Optional learning rate scheduler.
        strict: Whether to strictly enforce state dict key matching.
    """
    # Handle DDP-wrapped models
    if hasattr(model, "module"):
        model.module.load_state_dict(state.model_state_dict, strict=strict)
    else:
        model.load_state_dict(state.model_state_dict, strict=strict)

    optimizer.load_state_dict(state.optimizer_state_dict)

    if lr_scheduler and state.lr_scheduler_state_dict:
        lr_scheduler.load_state_dict(state.lr_scheduler_state_dict)

    # Restore RNG states
    if state.rng_states.get("torch") is not None:
        torch.set_rng_state(state.rng_states["torch"])

    if state.rng_states.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.rng_states["cuda"])
