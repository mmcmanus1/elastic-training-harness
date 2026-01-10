"""Learning rate scaling for elastic training topology changes.

This module provides utilities for adjusting learning rates when the
training topology changes (e.g., workers added or removed). The learning
rate must be scaled to maintain training stability when the effective
batch size changes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.optim as optim

logger = logging.getLogger(__name__)


class ScalingRule(Enum):
    """Learning rate scaling rules for batch size changes.

    LINEAR: Scale LR proportionally to batch size change.
            lr_new = lr_base * (batch_new / batch_base)
            Best for large batches where gradient noise is low.

    SQRT: Scale LR by square root of batch size change.
          lr_new = lr_base * sqrt(batch_new / batch_base)
          More conservative, often better for smaller batches.

    NONE: No scaling applied.
    """

    LINEAR = "linear"
    SQRT = "sqrt"
    NONE = "none"


@dataclass
class ScalingConfig:
    """Configuration for learning rate scaling.

    Attributes:
        base_lr: Base learning rate at base batch size.
        base_batch_size: Batch size per worker at base configuration.
        base_world_size: World size at base configuration.
        scaling_rule: Which scaling rule to apply.
        warmup_steps: Number of steps to warmup after topology change.
        min_lr: Minimum learning rate (prevents LR from going too low).
        max_lr: Maximum learning rate (prevents LR from going too high).
    """

    base_lr: float
    base_batch_size: int
    base_world_size: int
    scaling_rule: ScalingRule = ScalingRule.LINEAR
    warmup_steps: int = 100
    min_lr: float = 1e-7
    max_lr: float = 1e-2

    @property
    def base_effective_batch_size(self) -> int:
        """Effective batch size at base configuration."""
        return self.base_batch_size * self.base_world_size


class LRScalingManager:
    """Manages learning rate adjustment during topology changes.

    When the training world size changes (workers added/removed), the
    effective batch size changes. This manager adjusts the learning rate
    according to the configured scaling rule to maintain training stability.

    Example:
        >>> config = ScalingConfig(
        ...     base_lr=1e-4,
        ...     base_batch_size=8,
        ...     base_world_size=4,
        ...     scaling_rule=ScalingRule.LINEAR,
        ... )
        >>> manager = LRScalingManager(config, optimizer)
        >>> # On topology change...
        >>> new_lr = manager.on_topology_change(new_world_size=2, batch_size=8)
    """

    def __init__(
        self,
        config: ScalingConfig,
        optimizer: optim.Optimizer,
    ):
        """Initialize LR scaling manager.

        Args:
            config: Scaling configuration.
            optimizer: The optimizer whose LR will be adjusted.
        """
        self.config = config
        self.optimizer = optimizer

        # Track current state
        self._current_lr = config.base_lr
        self._current_world_size = config.base_world_size
        self._current_batch_size = config.base_batch_size
        self._warmup_scheduler: WarmupScheduler | None = None
        self._warmup_step = 0

    @property
    def current_lr(self) -> float:
        """Current learning rate."""
        return self._current_lr

    @property
    def current_effective_batch_size(self) -> int:
        """Current effective batch size."""
        return self._current_batch_size * self._current_world_size

    def on_topology_change(
        self,
        new_world_size: int,
        batch_size: int | None = None,
    ) -> float:
        """Handle topology change by adjusting learning rate.

        Args:
            new_world_size: New number of workers.
            batch_size: New batch size per worker (uses current if None).

        Returns:
            New learning rate after scaling.
        """
        old_world_size = self._current_world_size
        old_batch_size = self._current_batch_size

        if batch_size is None:
            batch_size = old_batch_size

        old_effective = old_batch_size * old_world_size
        new_effective = batch_size * new_world_size

        # Calculate scaled LR
        new_lr = self.get_scaled_lr(self._current_lr, old_effective, new_effective)

        # Clamp to min/max
        new_lr = max(self.config.min_lr, min(self.config.max_lr, new_lr))

        logger.info(
            f"Topology change: {old_world_size} -> {new_world_size} workers, "
            f"effective batch: {old_effective} -> {new_effective}, "
            f"LR: {self._current_lr:.2e} -> {new_lr:.2e}"
        )

        # Update state
        self._current_world_size = new_world_size
        self._current_batch_size = batch_size

        # Apply warmup if configured
        if self.config.warmup_steps > 0:
            self._setup_warmup(self._current_lr, new_lr)
        else:
            self._apply_lr(new_lr)
            self._current_lr = new_lr

        return new_lr

    def get_scaled_lr(
        self,
        base_lr: float,
        old_effective_batch: int,
        new_effective_batch: int,
    ) -> float:
        """Calculate scaled learning rate.

        Args:
            base_lr: Current/base learning rate.
            old_effective_batch: Old effective batch size.
            new_effective_batch: New effective batch size.

        Returns:
            Scaled learning rate.
        """
        if old_effective_batch == new_effective_batch:
            return base_lr

        ratio = new_effective_batch / old_effective_batch

        if self.config.scaling_rule == ScalingRule.LINEAR:
            return base_lr * ratio
        elif self.config.scaling_rule == ScalingRule.SQRT:
            return base_lr * math.sqrt(ratio)
        else:  # NONE
            return base_lr

    def _setup_warmup(self, start_lr: float, target_lr: float) -> None:
        """Setup warmup scheduler after topology change.

        Args:
            start_lr: Starting learning rate.
            target_lr: Target learning rate after warmup.
        """
        self._warmup_scheduler = WarmupScheduler(
            start_lr=start_lr,
            target_lr=target_lr,
            warmup_steps=self.config.warmup_steps,
        )
        self._warmup_step = 0
        self._apply_lr(start_lr)

    def step(self) -> float:
        """Step the warmup scheduler if active.

        Call this at each training step to progress warmup.

        Returns:
            Current learning rate after step.
        """
        if self._warmup_scheduler is not None:
            self._warmup_step += 1
            new_lr = self._warmup_scheduler.get_lr(self._warmup_step)
            self._apply_lr(new_lr)
            self._current_lr = new_lr

            # Check if warmup is complete
            if self._warmup_step >= self.config.warmup_steps:
                logger.info(f"Warmup complete at LR {new_lr:.2e}")
                self._warmup_scheduler = None

        return self._current_lr

    def _apply_lr(self, lr: float) -> None:
        """Apply learning rate to optimizer.

        Args:
            lr: Learning rate to set.
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @property
    def is_warming_up(self) -> bool:
        """Check if warmup is in progress."""
        return self._warmup_scheduler is not None

    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing.

        Returns:
            Dictionary with current state.
        """
        return {
            "current_lr": self._current_lr,
            "current_world_size": self._current_world_size,
            "current_batch_size": self._current_batch_size,
            "warmup_step": self._warmup_step,
            "warmup_active": self._warmup_scheduler is not None,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint.

        Args:
            state: State dictionary from state_dict().
        """
        self._current_lr = state.get("current_lr", self.config.base_lr)
        self._current_world_size = state.get("current_world_size", self.config.base_world_size)
        self._current_batch_size = state.get("current_batch_size", self.config.base_batch_size)
        self._warmup_step = state.get("warmup_step", 0)

        # Re-apply current LR
        self._apply_lr(self._current_lr)


class WarmupScheduler:
    """Linear warmup scheduler for learning rate.

    Linearly interpolates learning rate from start_lr to target_lr
    over warmup_steps.
    """

    def __init__(
        self,
        start_lr: float,
        target_lr: float,
        warmup_steps: int,
    ):
        """Initialize warmup scheduler.

        Args:
            start_lr: Initial learning rate.
            target_lr: Target learning rate after warmup.
            warmup_steps: Number of steps for warmup.
        """
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps

    def get_lr(self, step: int) -> float:
        """Get learning rate for given step.

        Args:
            step: Current warmup step (1-indexed).

        Returns:
            Learning rate for this step.
        """
        if step >= self.warmup_steps:
            return self.target_lr

        progress = step / self.warmup_steps
        return self.start_lr + progress * (self.target_lr - self.start_lr)


class RoundingMode:
    """Rounding modes for gradient accumulation step calculation."""

    CEIL = "ceil"  # Round up (never drop below target batch size)
    FLOOR = "floor"  # Round down (never exceed target batch size)
    NEAREST = "nearest"  # Round to nearest (Python's round())


@dataclass
class GradAccumulationConfig:
    """Configuration for dynamic gradient accumulation.

    Attributes:
        target_global_batch_size: Target global batch size to maintain.
        local_batch_size: Batch size per GPU.
        base_world_size: World size at base configuration.
        rounding_mode: How to round accumulation steps ('ceil', 'floor', 'nearest').
            - 'ceil': Round up, ensures effective batch >= target (default, conservative)
            - 'floor': Round down, ensures effective batch <= target
            - 'nearest': Round to nearest, may over or undershoot target
    """

    target_global_batch_size: int
    local_batch_size: int
    base_world_size: int
    rounding_mode: str = RoundingMode.CEIL

    @property
    def base_accumulation_steps(self) -> int:
        """Accumulation steps at base configuration."""
        per_step_batch = self.local_batch_size * self.base_world_size
        return max(1, self.target_global_batch_size // per_step_batch)


class GradientAccumulationManager:
    """Manages dynamic gradient accumulation to maintain constant global batch size.

    When the world size changes, this manager recalculates the number of
    gradient accumulation steps needed to maintain the target global batch size.

    The formula is:
        accumulation_steps = target_global_batch / (local_batch * world_size)

    Example:
        Target Global Batch: 1024
        Local Batch per GPU: 32

        Case 1 (4 GPUs): 4 × 32 = 128 per step → accumulation = 1024/128 = 8
        Case 2 (3 GPUs): 3 × 32 = 96 per step  → accumulation = 1024/96 ≈ 11

    Example:
        >>> config = GradAccumulationConfig(
        ...     target_global_batch_size=1024,
        ...     local_batch_size=32,
        ...     base_world_size=4,
        ... )
        >>> manager = GradientAccumulationManager(config)
        >>> manager.accumulation_steps  # With 4 GPUs
        8
        >>> manager.on_topology_change(new_world_size=3)
        >>> manager.accumulation_steps  # With 3 GPUs
        11
    """

    def __init__(self, config: GradAccumulationConfig):
        """Initialize gradient accumulation manager.

        Args:
            config: Gradient accumulation configuration.
        """
        self.config = config
        self._current_world_size = config.base_world_size
        self._accumulation_steps = config.base_accumulation_steps
        self._accumulated_count = 0

    @property
    def accumulation_steps(self) -> int:
        """Current number of gradient accumulation steps."""
        return self._accumulation_steps

    @property
    def current_world_size(self) -> int:
        """Current world size."""
        return self._current_world_size

    @property
    def effective_batch_size(self) -> int:
        """Current effective global batch size."""
        return (
            self.config.local_batch_size
            * self._current_world_size
            * self._accumulation_steps
        )

    def on_topology_change(self, new_world_size: int) -> int:
        """Handle topology change by recalculating accumulation steps.

        Args:
            new_world_size: New number of workers.

        Returns:
            New number of accumulation steps.
        """
        old_world_size = self._current_world_size
        old_accum = self._accumulation_steps

        # Calculate new accumulation steps with configurable rounding
        per_step_batch = self.config.local_batch_size * new_world_size
        raw_accum = self.config.target_global_batch_size / per_step_batch

        rounding_mode = self.config.rounding_mode
        if rounding_mode == RoundingMode.CEIL:
            new_accum = max(1, math.ceil(raw_accum))
        elif rounding_mode == RoundingMode.FLOOR:
            new_accum = max(1, math.floor(raw_accum))
        else:  # NEAREST (default fallback)
            new_accum = max(1, round(raw_accum))

        # Calculate actual effective batch sizes
        old_effective = self.config.local_batch_size * old_world_size * old_accum
        new_effective = self.config.local_batch_size * new_world_size * new_accum

        # Log warning if effective batch differs from target
        if new_effective != self.config.target_global_batch_size:
            logger.warning(
                f"Effective batch size ({new_effective}) differs from target "
                f"({self.config.target_global_batch_size}) due to rounding "
                f"(mode: {rounding_mode})"
            )

        logger.info(
            f"Gradient accumulation adjusted: "
            f"world_size {old_world_size} -> {new_world_size}, "
            f"accum_steps {old_accum} -> {new_accum}, "
            f"effective_batch {old_effective} -> {new_effective}"
        )

        # Update state
        self._current_world_size = new_world_size
        self._accumulation_steps = new_accum

        # Reset accumulation counter on topology change
        self._accumulated_count = 0

        return new_accum

    def should_step(self) -> bool:
        """Check if optimizer should step after this backward pass.

        Call this after each backward() to determine if you should
        call optimizer.step().

        Returns:
            True if accumulated enough gradients, False otherwise.
        """
        self._accumulated_count += 1

        if self._accumulated_count >= self._accumulation_steps:
            self._accumulated_count = 0
            return True

        return False

    def reset(self) -> None:
        """Reset accumulation counter."""
        self._accumulated_count = 0

    @property
    def current_accumulation_count(self) -> int:
        """Current number of accumulated gradients."""
        return self._accumulated_count

    @property
    def loss_scale_factor(self) -> float:
        """Factor to scale loss by for gradient accumulation.

        Divide loss by this factor before backward() to maintain
        correct gradient magnitudes.
        """
        return float(self._accumulation_steps)

    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing.

        Returns:
            Dictionary with current state.
        """
        return {
            "current_world_size": self._current_world_size,
            "accumulation_steps": self._accumulation_steps,
            "accumulated_count": self._accumulated_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint.

        Args:
            state: State dictionary from state_dict().
        """
        self._current_world_size = state.get("current_world_size", self.config.base_world_size)
        self._accumulation_steps = state.get("accumulation_steps", self.config.base_accumulation_steps)
        self._accumulated_count = state.get("accumulated_count", 0)


class ElasticScalingManager:
    """Combined manager for LR scaling and gradient accumulation.

    This manager coordinates both learning rate scaling and gradient
    accumulation adjustment when topology changes occur.

    There are two strategies for handling topology changes:

    Strategy A (Variable Batch): Keep accumulation constant, scale LR
        - Effective batch size changes with world size
        - LR is scaled to compensate

    Strategy B (Constant Batch): Adjust accumulation, keep LR constant
        - Accumulation steps adjusted to maintain global batch size
        - LR remains stable

    This manager supports both strategies via configuration.

    Example:
        >>> manager = ElasticScalingManager(
        ...     lr_config=ScalingConfig(...),
        ...     accum_config=GradAccumulationConfig(...),
        ...     optimizer=optimizer,
        ...     strategy="constant_batch",
        ... )
        >>> # On topology change...
        >>> manager.on_topology_change(new_world_size=3)
    """

    def __init__(
        self,
        lr_config: ScalingConfig,
        accum_config: GradAccumulationConfig | None,
        optimizer: optim.Optimizer,
        strategy: str = "constant_batch",
    ):
        """Initialize elastic scaling manager.

        Args:
            lr_config: Learning rate scaling configuration.
            accum_config: Gradient accumulation configuration (for constant_batch strategy).
            optimizer: The optimizer.
            strategy: "variable_batch" (scale LR) or "constant_batch" (adjust accumulation).
        """
        self.strategy = strategy
        self.lr_manager = LRScalingManager(lr_config, optimizer)

        self.accum_manager: GradientAccumulationManager | None = None
        if accum_config and strategy == "constant_batch":
            self.accum_manager = GradientAccumulationManager(accum_config)

    def on_topology_change(self, new_world_size: int) -> dict[str, Any]:
        """Handle topology change.

        Args:
            new_world_size: New number of workers.

        Returns:
            Dictionary with new LR and accumulation steps.
        """
        result = {}

        if self.strategy == "variable_batch":
            # Strategy A: Scale LR, keep accumulation constant
            new_lr = self.lr_manager.on_topology_change(new_world_size)
            result["lr"] = new_lr
            result["accumulation_steps"] = None  # Unchanged

        elif self.strategy == "constant_batch":
            # Strategy B: Adjust accumulation, minimal LR change
            if self.accum_manager:
                new_accum = self.accum_manager.on_topology_change(new_world_size)
                result["accumulation_steps"] = new_accum

            # Still update LR manager's world size tracking
            self.lr_manager._current_world_size = new_world_size
            result["lr"] = self.lr_manager.current_lr

        return result

    @property
    def accumulation_steps(self) -> int:
        """Current accumulation steps."""
        if self.accum_manager:
            return self.accum_manager.accumulation_steps
        return 1

    @property
    def current_lr(self) -> float:
        """Current learning rate."""
        return self.lr_manager.current_lr

    def should_step(self) -> bool:
        """Check if optimizer should step."""
        if self.accum_manager:
            return self.accum_manager.should_step()
        return True

    def step_lr(self) -> float:
        """Step LR warmup if active."""
        return self.lr_manager.step()

    def state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        state = {
            "lr_state": self.lr_manager.state_dict(),
            "strategy": self.strategy,
        }
        if self.accum_manager:
            state["accum_state"] = self.accum_manager.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.lr_manager.load_state_dict(state.get("lr_state", {}))
        if self.accum_manager and "accum_state" in state:
            self.accum_manager.load_state_dict(state["accum_state"])


def create_lr_scheduler_with_warmup(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> optim.lr_scheduler.LRScheduler:
    """Create a learning rate scheduler with warmup and cosine decay.

    This creates a standard scheduler that can be used alongside the
    LRScalingManager for initial warmup.

    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        min_lr: Minimum learning rate at end of cosine decay (absolute value).

    Returns:
        LR scheduler with warmup and cosine decay.
    """
    # Get base LR to calculate min multiplier (LambdaLR applies a multiplier to base_lr)
    base_lr = optimizer.param_groups[0]["lr"]
    min_multiplier = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / max(1, warmup_steps)
        else:
            # Cosine decay
            progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(min_multiplier, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
