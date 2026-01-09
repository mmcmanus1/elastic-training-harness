"""In-memory checkpoint snapshots for fast recovery.

This module provides a memory-based checkpoint backend that stores
recent snapshots in RAM for near-instant recovery from non-fatal errors.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from elastic_harness.checkpoint.checkpointing import CheckpointState


@dataclass
class MemorySnapshot:
    """A single in-memory checkpoint snapshot.

    Attributes:
        state: The checkpoint state (CPU tensors).
        timestamp: Unix timestamp when snapshot was created.
        step: Training step when snapshot was created.
    """

    state: CheckpointState
    timestamp: float
    step: int


class MemorySnapshotBackend:
    """In-memory checkpointing for fast recovery from non-fatal errors.

    This backend maintains a circular buffer of recent checkpoint snapshots
    in CPU RAM. When a worker crashes and restarts on the same node, it can
    recover from memory in milliseconds instead of loading from disk.

    Key features:
    - Circular buffer keeps only the most recent N snapshots
    - Thread-safe for concurrent access
    - Automatic CPU tensor conversion
    - Memory usage tracking

    Example:
        >>> backend = MemorySnapshotBackend(max_snapshots=2)
        >>> backend.save(state)
        >>> # Later, on recovery...
        >>> restored = backend.load()
    """

    def __init__(self, max_snapshots: int = 2):
        """Initialize memory snapshot backend.

        Args:
            max_snapshots: Maximum number of snapshots to keep in memory.
        """
        self.max_snapshots = max_snapshots
        self._snapshots: deque[MemorySnapshot] = deque(maxlen=max_snapshots)
        self._lock = threading.RLock()

    def save(self, state: CheckpointState) -> None:
        """Save checkpoint snapshot to memory.

        Creates a deep copy of all tensors on CPU and stores in the
        circular buffer.

        Args:
            state: The checkpoint state to snapshot.
        """
        with self._lock:
            # Create CPU copy of state
            cpu_state = state.cpu_copy()

            snapshot = MemorySnapshot(
                state=cpu_state,
                timestamp=time.time(),
                step=state.step,
            )

            self._snapshots.append(snapshot)

    def load(self) -> CheckpointState | None:
        """Retrieve the most recent snapshot.

        Returns:
            The most recent CheckpointState, or None if no snapshots exist.
        """
        with self._lock:
            if not self._snapshots:
                return None

            # Return a copy of the most recent snapshot
            return self._snapshots[-1].state.cpu_copy()

    def load_by_step(self, step: int) -> CheckpointState | None:
        """Retrieve snapshot by training step.

        Args:
            step: The training step to look for.

        Returns:
            CheckpointState for that step, or None if not found.
        """
        with self._lock:
            for snapshot in reversed(self._snapshots):
                if snapshot.step == step:
                    return snapshot.state.cpu_copy()
            return None

    def clear(self) -> None:
        """Clear all snapshots from memory.

        Call this after a successful persistent checkpoint to free memory.
        """
        with self._lock:
            self._snapshots.clear()

    @property
    def num_snapshots(self) -> int:
        """Number of snapshots currently in memory."""
        with self._lock:
            return len(self._snapshots)

    @property
    def latest_step(self) -> int | None:
        """Training step of the most recent snapshot."""
        with self._lock:
            if not self._snapshots:
                return None
            return self._snapshots[-1].step

    @property
    def latest_timestamp(self) -> float | None:
        """Timestamp of the most recent snapshot."""
        with self._lock:
            if not self._snapshots:
                return None
            return self._snapshots[-1].timestamp

    def memory_usage_bytes(self) -> int:
        """Estimate current memory usage of all snapshots.

        Returns:
            Estimated memory usage in bytes.
        """
        with self._lock:
            total = 0
            for snapshot in self._snapshots:
                total += self._estimate_state_size(snapshot.state)
            return total

    def _estimate_state_size(self, state: CheckpointState) -> int:
        """Estimate memory size of a checkpoint state.

        Args:
            state: The checkpoint state to measure.

        Returns:
            Estimated size in bytes.
        """
        total = 0

        def count_dict(d: dict[str, Any]) -> int:
            size = 0
            for v in d.values():
                if isinstance(v, torch.Tensor):
                    size += v.numel() * v.element_size()
                elif isinstance(v, dict):
                    size += count_dict(v)
                elif isinstance(v, (list, tuple)):
                    for item in v:
                        if isinstance(item, torch.Tensor):
                            size += item.numel() * item.element_size()
            return size

        if state.model_state_dict:
            total += count_dict(state.model_state_dict)
        if state.optimizer_state_dict:
            total += count_dict(state.optimizer_state_dict)

        return total

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the snapshot buffer.

        Returns:
            Dictionary with snapshot statistics.
        """
        with self._lock:
            if not self._snapshots:
                return {
                    "num_snapshots": 0,
                    "memory_bytes": 0,
                    "oldest_step": None,
                    "latest_step": None,
                }

            return {
                "num_snapshots": len(self._snapshots),
                "memory_bytes": self.memory_usage_bytes(),
                "memory_mb": self.memory_usage_bytes() / (1024 * 1024),
                "oldest_step": self._snapshots[0].step,
                "latest_step": self._snapshots[-1].step,
                "oldest_timestamp": self._snapshots[0].timestamp,
                "latest_timestamp": self._snapshots[-1].timestamp,
            }


class SharedMemorySnapshotBackend(MemorySnapshotBackend):
    """Shared memory snapshot backend for multi-process recovery.

    This backend uses shared memory to allow recovery across process
    restarts on the same node. Useful when the training process crashes
    but the node remains available.

    Note: This is a more advanced feature and requires careful handling
    of shared memory resources.

    TODO: This is currently a placeholder that falls back to regular in-process
    memory snapshots. A full implementation would use torch.multiprocessing
    shared memory or POSIX shared memory to persist snapshots across process
    restarts. Key implementation considerations:
    - Use shared memory segments with proper cleanup on shutdown
    - Handle serialization of PyTorch tensors to shared memory
    - Implement proper locking for concurrent access
    - Add memory mapping for large checkpoints
    """

    def __init__(
        self,
        max_snapshots: int = 2,
        name: str = "elastic_training_snapshot",
    ):
        """Initialize shared memory snapshot backend.

        Args:
            max_snapshots: Maximum number of snapshots to keep.
            name: Name for the shared memory region.
        """
        super().__init__(max_snapshots)
        self.name = name
        # Note: Full shared memory implementation would require
        # torch.multiprocessing.shared_memory or similar.
        # This is a placeholder for the interface.

    def save(self, state: CheckpointState) -> None:
        """Save snapshot to shared memory.

        Args:
            state: The checkpoint state to snapshot.
        """
        # For now, fall back to regular memory snapshot
        # Full implementation would serialize to shared memory
        super().save(state)

    def load(self) -> CheckpointState | None:
        """Load snapshot from shared memory.

        Returns:
            The most recent CheckpointState, or None if not found.
        """
        # For now, fall back to regular memory snapshot
        return super().load()
