"""Tests for learning rate scaling."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from elastic_harness.scaling.lr_scaling import (
    LRScalingManager,
    ScalingConfig,
    ScalingRule,
    WarmupScheduler,
    create_lr_scheduler_with_warmup,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def simple_optimizer(simple_model):
    """Create an optimizer for testing."""
    return torch.optim.Adam(simple_model.parameters(), lr=0.001)


@pytest.fixture
def scaling_config():
    """Create a scaling configuration for testing."""
    return ScalingConfig(
        base_lr=1e-4,
        base_batch_size=8,
        base_world_size=4,
        scaling_rule=ScalingRule.LINEAR,
        warmup_steps=100,
        min_lr=1e-7,
        max_lr=1e-2,
    )


class TestScalingRule:
    """Tests for ScalingRule enum."""

    def test_values(self):
        """Test enum values."""
        assert ScalingRule.LINEAR.value == "linear"
        assert ScalingRule.SQRT.value == "sqrt"
        assert ScalingRule.NONE.value == "none"


class TestScalingConfig:
    """Tests for ScalingConfig."""

    def test_base_effective_batch_size(self, scaling_config):
        """Test effective batch size calculation."""
        expected = 8 * 4  # batch_size * world_size
        assert scaling_config.base_effective_batch_size == expected

    def test_defaults(self):
        """Test default values."""
        config = ScalingConfig(
            base_lr=1e-4,
            base_batch_size=8,
            base_world_size=4,
        )

        assert config.scaling_rule == ScalingRule.LINEAR
        assert config.warmup_steps == 100


class TestWarmupScheduler:
    """Tests for WarmupScheduler."""

    def test_warmup_start(self):
        """Test LR at warmup start."""
        scheduler = WarmupScheduler(start_lr=1e-5, target_lr=1e-4, warmup_steps=100)

        lr = scheduler.get_lr(1)

        assert lr > 1e-5
        assert lr < 1e-4

    def test_warmup_end(self):
        """Test LR at warmup end."""
        scheduler = WarmupScheduler(start_lr=1e-5, target_lr=1e-4, warmup_steps=100)

        lr = scheduler.get_lr(100)

        assert lr == 1e-4

    def test_warmup_beyond(self):
        """Test LR beyond warmup steps."""
        scheduler = WarmupScheduler(start_lr=1e-5, target_lr=1e-4, warmup_steps=100)

        lr = scheduler.get_lr(150)

        assert lr == 1e-4  # Should stay at target

    def test_linear_interpolation(self):
        """Test linear interpolation during warmup."""
        scheduler = WarmupScheduler(start_lr=0.0, target_lr=1.0, warmup_steps=100)

        # At 50% warmup, should be 50% of target
        lr = scheduler.get_lr(50)

        assert abs(lr - 0.5) < 0.01


class TestLRScalingManager:
    """Tests for LRScalingManager."""

    def test_linear_scaling(self, scaling_config, simple_optimizer):
        """Test linear scaling rule."""
        manager = LRScalingManager(scaling_config, simple_optimizer)

        # Double the world size (double effective batch)
        new_lr = manager.get_scaled_lr(
            base_lr=1e-4,
            old_effective_batch=32,
            new_effective_batch=64,
        )

        assert new_lr == 2e-4  # Should double

    def test_sqrt_scaling(self, simple_optimizer):
        """Test sqrt scaling rule."""
        config = ScalingConfig(
            base_lr=1e-4,
            base_batch_size=8,
            base_world_size=4,
            scaling_rule=ScalingRule.SQRT,
        )
        manager = LRScalingManager(config, simple_optimizer)

        # Quadruple the effective batch
        new_lr = manager.get_scaled_lr(
            base_lr=1e-4,
            old_effective_batch=32,
            new_effective_batch=128,
        )

        expected = 1e-4 * math.sqrt(4)
        assert abs(new_lr - expected) < 1e-10

    def test_no_scaling(self, simple_optimizer):
        """Test no scaling rule."""
        config = ScalingConfig(
            base_lr=1e-4,
            base_batch_size=8,
            base_world_size=4,
            scaling_rule=ScalingRule.NONE,
        )
        manager = LRScalingManager(config, simple_optimizer)

        new_lr = manager.get_scaled_lr(
            base_lr=1e-4,
            old_effective_batch=32,
            new_effective_batch=128,
        )

        assert new_lr == 1e-4  # Should not change

    def test_on_topology_change(self, scaling_config, simple_optimizer):
        """Test handling topology change."""
        manager = LRScalingManager(scaling_config, simple_optimizer)

        # Scale from 4 to 2 workers
        new_lr = manager.on_topology_change(new_world_size=2)

        # Effective batch halved, so LR should halve
        assert new_lr == scaling_config.base_lr / 2

    def test_lr_clamping_min(self, simple_optimizer):
        """Test LR clamping at minimum."""
        config = ScalingConfig(
            base_lr=1e-6,
            base_batch_size=8,
            base_world_size=4,
            scaling_rule=ScalingRule.LINEAR,
            min_lr=1e-7,
        )
        manager = LRScalingManager(config, simple_optimizer)

        # Scale down dramatically
        new_lr = manager.on_topology_change(new_world_size=1)

        # Should be clamped to min_lr
        assert new_lr >= config.min_lr

    def test_lr_clamping_max(self, simple_optimizer):
        """Test LR clamping at maximum."""
        config = ScalingConfig(
            base_lr=1e-3,
            base_batch_size=8,
            base_world_size=4,
            scaling_rule=ScalingRule.LINEAR,
            max_lr=1e-2,
        )
        manager = LRScalingManager(config, simple_optimizer)

        # Scale up dramatically
        new_lr = manager.on_topology_change(new_world_size=100)

        # Should be clamped to max_lr
        assert new_lr <= config.max_lr

    def test_warmup_activation(self, simple_optimizer):
        """Test warmup is activated on topology change."""
        config = ScalingConfig(
            base_lr=1e-4,
            base_batch_size=8,
            base_world_size=4,
            warmup_steps=100,
        )
        manager = LRScalingManager(config, simple_optimizer)

        manager.on_topology_change(new_world_size=2)

        assert manager.is_warming_up

    def test_warmup_step(self, simple_optimizer):
        """Test warmup progression."""
        config = ScalingConfig(
            base_lr=1e-4,
            base_batch_size=8,
            base_world_size=4,
            warmup_steps=10,
        )
        manager = LRScalingManager(config, simple_optimizer)

        manager.on_topology_change(new_world_size=2)
        initial_lr = manager.current_lr

        # Step through warmup
        for _ in range(10):
            manager.step()

        final_lr = manager.current_lr

        assert not manager.is_warming_up
        assert final_lr != initial_lr

    def test_state_dict(self, scaling_config, simple_optimizer):
        """Test state serialization."""
        manager = LRScalingManager(scaling_config, simple_optimizer)
        manager.on_topology_change(new_world_size=2)

        state = manager.state_dict()

        assert "current_lr" in state
        assert "current_world_size" in state
        assert state["current_world_size"] == 2

    def test_load_state_dict(self, scaling_config, simple_optimizer):
        """Test state restoration."""
        manager = LRScalingManager(scaling_config, simple_optimizer)

        state = {
            "current_lr": 5e-5,
            "current_world_size": 2,
            "current_batch_size": 8,
            "warmup_step": 0,
        }
        manager.load_state_dict(state)

        assert manager.current_lr == 5e-5
        assert manager._current_world_size == 2

    def test_applies_lr_to_optimizer(self, simple_optimizer):
        """Test that LR changes are applied to optimizer."""
        # Use config without warmup so LR is applied immediately
        config = ScalingConfig(
            base_lr=1e-4,
            base_batch_size=8,
            base_world_size=4,
            scaling_rule=ScalingRule.LINEAR,
            warmup_steps=0,  # No warmup - LR change should be immediate
            min_lr=1e-7,
            max_lr=1e-2,
        )
        manager = LRScalingManager(config, simple_optimizer)

        manager.on_topology_change(new_world_size=2)

        # Check optimizer param groups - LR should have changed from base
        # With world_size halved (4->2), effective batch halved, so LR halved: 1e-4 / 2 = 5e-5
        for param_group in simple_optimizer.param_groups:
            assert param_group["lr"] == 5e-5


class TestCreateLRSchedulerWithWarmup:
    """Tests for create_lr_scheduler_with_warmup helper."""

    def test_warmup_phase(self, simple_optimizer):
        """Test warmup phase of scheduler."""
        scheduler = create_lr_scheduler_with_warmup(
            simple_optimizer,
            warmup_steps=100,
            total_steps=1000,
        )

        # Get LR at start
        initial_lr = scheduler.get_last_lr()[0]

        # Step through warmup
        for _ in range(50):
            scheduler.step()

        mid_lr = scheduler.get_last_lr()[0]

        # LR should have increased
        assert mid_lr > initial_lr

    def test_decay_phase(self, simple_optimizer):
        """Test cosine decay phase of scheduler."""
        scheduler = create_lr_scheduler_with_warmup(
            simple_optimizer,
            warmup_steps=10,
            total_steps=100,
        )

        # Step past warmup
        for _ in range(10):
            scheduler.step()

        post_warmup_lr = scheduler.get_last_lr()[0]

        # Step through decay
        for _ in range(50):
            scheduler.step()

        decayed_lr = scheduler.get_last_lr()[0]

        # LR should have decreased
        assert decayed_lr < post_warmup_lr

    def test_min_lr(self, simple_optimizer):
        """Test minimum LR at end of training."""
        min_lr = 0.0001
        scheduler = create_lr_scheduler_with_warmup(
            simple_optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=min_lr,
        )

        # Step to end
        for _ in range(100):
            scheduler.step()

        final_lr = scheduler.get_last_lr()[0]

        # Should be at or near min_lr
        assert final_lr >= min_lr
