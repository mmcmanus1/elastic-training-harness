"""Training metrics collection and reporting.

This module provides utilities for collecting and aggregating training metrics,
including checkpoint timing, recovery events, and gradient statistics.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetrics:
    """Metrics related to checkpoint operations.

    Attributes:
        save_times: List of (tier, duration) tuples for saves.
        load_times: List of (tier, duration) tuples for loads.
        async_failures: Number of async save failures.
        async_retries: Number of async save retries.
    """

    save_times: list[tuple[str, float]] = field(default_factory=list)
    load_times: list[tuple[str, float]] = field(default_factory=list)
    async_failures: int = 0
    async_retries: int = 0

    def record_save(self, tier: str, duration: float) -> None:
        """Record a checkpoint save operation."""
        self.save_times.append((tier, duration))

    def record_load(self, tier: str, duration: float) -> None:
        """Record a checkpoint load operation."""
        self.load_times.append((tier, duration))

    def avg_save_time(self, tier: str | None = None) -> float:
        """Get average save time, optionally filtered by tier."""
        times = [d for t, d in self.save_times if tier is None or t == tier]
        return sum(times) / max(len(times), 1)

    def max_save_time(self) -> float:
        """Get maximum save time across all tiers."""
        if not self.save_times:
            return 0.0
        return max(d for _, d in self.save_times)


@dataclass
class RecoveryEvent:
    """A single topology change/recovery event.

    Attributes:
        timestamp: Unix timestamp when recovery started.
        old_world_size: World size before change.
        new_world_size: World size after change.
        recovery_time: Time to complete recovery in seconds.
        checkpoint_tier: Tier from which checkpoint was loaded.
        step_resumed: Training step after recovery.
    """

    timestamp: float
    old_world_size: int
    new_world_size: int
    recovery_time: float
    checkpoint_tier: str | None = None
    step_resumed: int = 0


@dataclass
class RecoveryMetrics:
    """Metrics related to topology changes and recovery.

    Attributes:
        events: List of recovery events.
    """

    events: list[RecoveryEvent] = field(default_factory=list)

    def record_recovery(
        self,
        old_world_size: int,
        new_world_size: int,
        recovery_time: float,
        checkpoint_tier: str | None = None,
        step_resumed: int = 0,
    ) -> None:
        """Record a recovery event."""
        event = RecoveryEvent(
            timestamp=time.time(),
            old_world_size=old_world_size,
            new_world_size=new_world_size,
            recovery_time=recovery_time,
            checkpoint_tier=checkpoint_tier,
            step_resumed=step_resumed,
        )
        self.events.append(event)
        logger.info(
            f"Recovery completed: {old_world_size} -> {new_world_size} workers, "
            f"time: {recovery_time:.2f}s, resumed at step {step_resumed}"
        )

    @property
    def total_recovery_time(self) -> float:
        """Total time spent in recovery."""
        return sum(e.recovery_time for e in self.events)

    @property
    def avg_recovery_time(self) -> float:
        """Average recovery time."""
        if not self.events:
            return 0.0
        return self.total_recovery_time / len(self.events)

    @property
    def worker_joins(self) -> int:
        """Number of events where workers joined."""
        return sum(1 for e in self.events if e.new_world_size > e.old_world_size)

    @property
    def worker_failures(self) -> int:
        """Number of events where workers failed/left."""
        return sum(1 for e in self.events if e.new_world_size < e.old_world_size)


@dataclass
class GradientMetrics:
    """Metrics related to gradients and training.

    Attributes:
        clip_count: Number of times gradients were clipped.
        total_steps: Total number of optimizer steps.
        clip_magnitudes: Recent gradient magnitudes before clipping (deque for efficiency).
    """

    clip_count: int = 0
    total_steps: int = 0
    _max_magnitudes: int = 100  # Keep last N magnitudes
    # Use deque with maxlen for O(1) append with automatic trimming
    clip_magnitudes: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def record_step(self, clipped: bool = False, grad_norm: float | None = None) -> None:
        """Record an optimizer step."""
        self.total_steps += 1
        if clipped:
            self.clip_count += 1
            if grad_norm is not None:
                # deque with maxlen automatically discards oldest items
                self.clip_magnitudes.append(grad_norm)

    @property
    def clip_frequency(self) -> float:
        """Fraction of steps where gradients were clipped."""
        if self.total_steps == 0:
            return 0.0
        return self.clip_count / self.total_steps

    @property
    def avg_clip_magnitude(self) -> float:
        """Average gradient magnitude when clipping occurred."""
        if not self.clip_magnitudes:
            return 0.0
        return sum(self.clip_magnitudes) / len(self.clip_magnitudes)


@dataclass
class LRMetrics:
    """Metrics related to learning rate adjustments.

    Attributes:
        adjustments: List of (timestamp, old_lr, new_lr, reason) tuples.
    """

    adjustments: list[tuple[float, float, float, str]] = field(default_factory=list)

    def record_adjustment(
        self,
        old_lr: float,
        new_lr: float,
        reason: str = "topology_change",
    ) -> None:
        """Record a learning rate adjustment."""
        self.adjustments.append((time.time(), old_lr, new_lr, reason))
        logger.info(f"LR adjusted: {old_lr:.2e} -> {new_lr:.2e} ({reason})")


@dataclass
class TrainingMetrics:
    """Aggregated training metrics.

    This is the main class for collecting all training metrics.

    Example:
        >>> metrics = TrainingMetrics()
        >>> metrics.checkpoint.record_save("nvme", 1.5)
        >>> metrics.gradient.record_step(clipped=True, grad_norm=2.5)
        >>> print(metrics.summary())
    """

    checkpoint: CheckpointMetrics = field(default_factory=CheckpointMetrics)
    recovery: RecoveryMetrics = field(default_factory=RecoveryMetrics)
    gradient: GradientMetrics = field(default_factory=GradientMetrics)
    lr: LRMetrics = field(default_factory=LRMetrics)

    # Training progress
    start_time: float = field(default_factory=time.time)
    current_step: int = 0
    total_tokens: int = 0

    def update_step(self, step: int, tokens_processed: int = 0) -> None:
        """Update current training progress."""
        self.current_step = step
        self.total_tokens += tokens_processed

    @property
    def training_time(self) -> float:
        """Total training wall time in seconds."""
        return time.time() - self.start_time

    @property
    def tokens_per_second(self) -> float:
        """Average throughput in tokens/second."""
        elapsed = self.training_time
        if elapsed == 0:
            return 0.0
        return self.total_tokens / elapsed

    @property
    def training_efficiency(self) -> float:
        """Fraction of time spent training (vs recovery)."""
        training_time = self.training_time
        recovery_time = self.recovery.total_recovery_time
        if training_time == 0:
            return 1.0
        return (training_time - recovery_time) / training_time

    def summary(self) -> dict[str, Any]:
        """Get a summary of all metrics.

        Returns:
            Dictionary with metric summaries suitable for logging or export.
        """
        return {
            "training": {
                "elapsed_seconds": self.training_time,
                "current_step": self.current_step,
                "total_tokens": self.total_tokens,
                "tokens_per_second": self.tokens_per_second,
                "efficiency": self.training_efficiency,
            },
            "checkpoint": {
                "total_saves": len(self.checkpoint.save_times),
                "total_loads": len(self.checkpoint.load_times),
                "avg_save_time": self.checkpoint.avg_save_time(),
                "max_save_time": self.checkpoint.max_save_time(),
                "async_failures": self.checkpoint.async_failures,
                "async_retries": self.checkpoint.async_retries,
            },
            "recovery": {
                "total_events": len(self.recovery.events),
                "worker_joins": self.recovery.worker_joins,
                "worker_failures": self.recovery.worker_failures,
                "total_recovery_time": self.recovery.total_recovery_time,
                "avg_recovery_time": self.recovery.avg_recovery_time,
            },
            "gradient": {
                "total_steps": self.gradient.total_steps,
                "clip_count": self.gradient.clip_count,
                "clip_frequency": self.gradient.clip_frequency,
                "avg_clip_magnitude": self.gradient.avg_clip_magnitude,
            },
            "lr_adjustments": len(self.lr.adjustments),
        }

    def log_summary(self) -> None:
        """Log a summary of metrics."""
        summary = self.summary()
        logger.info(
            f"Training Summary: "
            f"{summary['training']['current_step']} steps, "
            f"{summary['training']['tokens_per_second']:.0f} tokens/s, "
            f"{summary['training']['efficiency']*100:.1f}% efficiency"
        )
        logger.info(
            f"Checkpoint Stats: "
            f"{summary['checkpoint']['total_saves']} saves, "
            f"{summary['checkpoint']['avg_save_time']:.2f}s avg save time, "
            f"{summary['checkpoint']['async_failures']} failures"
        )
        if summary['recovery']['total_events'] > 0:
            logger.info(
                f"Recovery Stats: "
                f"{summary['recovery']['total_events']} events, "
                f"{summary['recovery']['avg_recovery_time']:.2f}s avg recovery"
            )
        if summary['gradient']['clip_frequency'] > 0:
            logger.info(
                f"Gradient Stats: "
                f"{summary['gradient']['clip_frequency']*100:.1f}% clipped"
            )
