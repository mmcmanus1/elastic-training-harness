"""Tests for checkpoint management."""

from __future__ import annotations

import copy
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from elastic_harness.checkpoint.checkpointing import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointState,
    CheckpointTier,
    create_checkpoint_state,
    load_checkpoint_to_model,
)
from elastic_harness.checkpoint.memory_snapshot import MemorySnapshotBackend
from elastic_harness.checkpoint.storage_backends import CheckpointLoadConfig, NVMeBackend


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
def checkpoint_state(simple_model, simple_optimizer):
    """Create a checkpoint state for testing."""
    return CheckpointState(
        step=100,
        epoch=2,
        model_state_dict=simple_model.state_dict(),
        optimizer_state_dict=simple_optimizer.state_dict(),
        lr_scheduler_state_dict={},
        dataset_state={"total_tokens_processed": 50000},
        rng_states={},
        metrics={"loss": 0.5, "accuracy": 0.9},
        world_size=4,
        timestamp=time.time(),
    )


class TestCheckpointState:
    """Tests for CheckpointState."""

    def test_to_dict(self, checkpoint_state):
        """Test serialization to dict."""
        data = checkpoint_state.to_dict()

        assert data["step"] == 100
        assert data["epoch"] == 2
        assert data["world_size"] == 4
        assert "model_state_dict" in data
        assert "optimizer_state_dict" in data

    def test_from_dict(self, checkpoint_state):
        """Test deserialization from dict."""
        data = checkpoint_state.to_dict()
        restored = CheckpointState.from_dict(data)

        assert restored.step == checkpoint_state.step
        assert restored.epoch == checkpoint_state.epoch
        assert restored.world_size == checkpoint_state.world_size

    def test_cpu_copy(self, simple_model, simple_optimizer):
        """Test CPU copy of state."""
        # Create state with GPU tensors (if available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = simple_model.to(device)

        state = CheckpointState(
            step=1,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=simple_optimizer.state_dict(),
        )

        cpu_state = state.cpu_copy()

        # Verify all tensors are on CPU
        for key, tensor in cpu_state.model_state_dict.items():
            assert tensor.device == torch.device("cpu")


class TestMemorySnapshotBackend:
    """Tests for MemorySnapshotBackend."""

    def test_save_and_load(self, checkpoint_state):
        """Test saving and loading snapshots."""
        backend = MemorySnapshotBackend(max_snapshots=2)

        backend.save(checkpoint_state)
        loaded = backend.load()

        assert loaded is not None
        assert loaded.step == checkpoint_state.step

    def test_circular_buffer(self, simple_model, simple_optimizer):
        """Test that circular buffer limits snapshots."""
        backend = MemorySnapshotBackend(max_snapshots=2)

        # Save 3 snapshots
        for i in range(3):
            state = CheckpointState(
                step=i,
                model_state_dict=simple_model.state_dict(),
                optimizer_state_dict=simple_optimizer.state_dict(),
            )
            backend.save(state)

        assert backend.num_snapshots == 2
        assert backend.latest_step == 2

    def test_clear(self, checkpoint_state):
        """Test clearing snapshots."""
        backend = MemorySnapshotBackend()

        backend.save(checkpoint_state)
        assert backend.num_snapshots == 1

        backend.clear()
        assert backend.num_snapshots == 0

    def test_load_empty(self):
        """Test loading from empty backend."""
        backend = MemorySnapshotBackend()

        result = backend.load()
        assert result is None

    def test_memory_usage(self, checkpoint_state):
        """Test memory usage estimation."""
        backend = MemorySnapshotBackend()

        backend.save(checkpoint_state)
        usage = backend.memory_usage_bytes()

        assert usage > 0

    def test_get_stats(self, checkpoint_state):
        """Test statistics retrieval."""
        backend = MemorySnapshotBackend()

        backend.save(checkpoint_state)
        stats = backend.get_stats()

        assert stats["num_snapshots"] == 1
        assert stats["latest_step"] == checkpoint_state.step


class TestNVMeBackend:
    """Tests for NVMeBackend."""

    def test_save_and_load(self, checkpoint_state, tmp_path):
        """Test saving and loading checkpoints."""
        backend = NVMeBackend(tmp_path)

        backend.save(checkpoint_state, "test_checkpoint.pt")
        loaded = backend.load("test_checkpoint.pt")

        assert loaded.step == checkpoint_state.step
        assert loaded.epoch == checkpoint_state.epoch

    def test_exists(self, checkpoint_state, tmp_path):
        """Test existence check."""
        backend = NVMeBackend(tmp_path)

        assert not backend.exists("nonexistent.pt")

        backend.save(checkpoint_state, "test.pt")
        assert backend.exists("test.pt")

    def test_list_checkpoints(self, checkpoint_state, tmp_path):
        """Test listing checkpoints."""
        backend = NVMeBackend(tmp_path)

        # Save multiple checkpoints
        for i in range(3):
            state = CheckpointState(step=i)
            backend.save(state, f"checkpoint_{i}.pt")

        checkpoints = backend.list_checkpoints()
        assert len(checkpoints) == 3

    def test_delete(self, checkpoint_state, tmp_path):
        """Test checkpoint deletion."""
        backend = NVMeBackend(tmp_path)

        backend.save(checkpoint_state, "to_delete.pt")
        assert backend.exists("to_delete.pt")

        backend.delete("to_delete.pt")
        assert not backend.exists("to_delete.pt")

    def test_get_latest_checkpoint(self, checkpoint_state, tmp_path):
        """Test getting latest checkpoint."""
        backend = NVMeBackend(tmp_path)

        # Save checkpoints with delays
        for i in range(3):
            state = CheckpointState(step=i)
            backend.save(state, f"checkpoint_{i}.pt")
            time.sleep(0.01)  # Small delay to ensure different mtimes

        latest = backend.get_latest_checkpoint()
        assert latest == "checkpoint_2.pt"

    def test_cleanup_old_checkpoints(self, checkpoint_state, tmp_path):
        """Test cleanup of old checkpoints."""
        backend = NVMeBackend(tmp_path)

        # Save 5 checkpoints
        for i in range(5):
            state = CheckpointState(step=i)
            backend.save(state, f"checkpoint_{i}.pt")
            time.sleep(0.01)

        # Keep only last 2
        deleted = backend.cleanup_old_checkpoints(keep_last=2)

        assert len(deleted) == 3
        assert len(backend.list_checkpoints()) == 2

    def test_atomic_save(self, checkpoint_state, tmp_path):
        """Test atomic save (no partial files on failure)."""
        backend = NVMeBackend(tmp_path, use_atomic_save=True)

        backend.save(checkpoint_state, "atomic_test.pt")

        # Verify no temp files left
        temp_files = list(tmp_path.glob(".*tmp"))
        assert len(temp_files) == 0


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_memory_tier(self, checkpoint_state, tmp_path):
        """Test saving to memory tier."""
        config = CheckpointConfig(nvme_path=tmp_path)
        manager = CheckpointManager(config)

        result = manager.save_checkpoint(checkpoint_state, tier=CheckpointTier.MEMORY)

        assert result is None  # Memory tier doesn't return path
        assert manager.memory_backend.num_snapshots == 1

    def test_save_nvme_tier(self, checkpoint_state, tmp_path):
        """Test saving to NVMe tier."""
        config = CheckpointConfig(nvme_path=tmp_path)
        manager = CheckpointManager(config)

        result = manager.save_checkpoint(checkpoint_state, tier=CheckpointTier.NVME)

        assert result is not None
        assert manager.nvme_backend.exists(result)

    def test_load_fallback_hierarchy(self, checkpoint_state, tmp_path):
        """Test loading with fallback hierarchy."""
        config = CheckpointConfig(nvme_path=tmp_path)
        manager = CheckpointManager(config)

        # Save to NVMe only
        manager.save_checkpoint(checkpoint_state, tier=CheckpointTier.NVME)

        # Load should find NVMe checkpoint
        loaded = manager.load_checkpoint()

        assert loaded is not None
        assert loaded.step == checkpoint_state.step

    def test_load_prefers_memory(self, simple_model, simple_optimizer, tmp_path):
        """Test that memory tier is preferred over NVMe."""
        config = CheckpointConfig(nvme_path=tmp_path)
        manager = CheckpointManager(config)

        # Save older checkpoint to NVMe
        old_state = CheckpointState(
            step=50,
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict=simple_optimizer.state_dict(),
        )
        manager.save_checkpoint(old_state, tier=CheckpointTier.NVME)

        # Save newer checkpoint to memory
        new_state = CheckpointState(
            step=100,
            model_state_dict=simple_model.state_dict(),
            optimizer_state_dict=simple_optimizer.state_dict(),
        )
        manager.save_checkpoint(new_state, tier=CheckpointTier.MEMORY)

        # Load should return memory checkpoint
        loaded = manager.load_checkpoint()

        assert loaded.step == 100

    def test_clear_memory_snapshots(self, checkpoint_state, tmp_path):
        """Test clearing memory snapshots."""
        config = CheckpointConfig(nvme_path=tmp_path)
        manager = CheckpointManager(config)

        manager.save_checkpoint(checkpoint_state, tier=CheckpointTier.MEMORY)
        assert manager.memory_backend.num_snapshots == 1

        manager.clear_memory_snapshots()
        assert manager.memory_backend.num_snapshots == 0

    def test_should_checkpoint(self, tmp_path):
        """Test checkpoint interval checking."""
        config = CheckpointConfig(
            nvme_path=tmp_path,
            checkpoint_interval=100,
            memory_snapshot_interval=10,
        )
        manager = CheckpointManager(config)

        # Memory tier
        assert not manager.should_checkpoint(0, CheckpointTier.MEMORY)
        assert manager.should_checkpoint(10, CheckpointTier.MEMORY)
        assert not manager.should_checkpoint(15, CheckpointTier.MEMORY)

        # NVMe tier
        assert not manager.should_checkpoint(0, CheckpointTier.NVME)
        assert not manager.should_checkpoint(50, CheckpointTier.NVME)
        assert manager.should_checkpoint(100, CheckpointTier.NVME)


class TestCheckpointHelpers:
    """Tests for checkpoint helper functions."""

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_create_checkpoint_state(self, mock_dist, simple_model, simple_optimizer):
        """Test creating checkpoint state from training components."""
        state = create_checkpoint_state(
            step=100,
            model=simple_model,
            optimizer=simple_optimizer,
            metrics={"loss": 0.5},
        )

        assert state.step == 100
        assert state.metrics["loss"] == 0.5
        assert "linear.weight" in state.model_state_dict

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_create_checkpoint_state_with_ddp(self, mock_dist, simple_model, simple_optimizer):
        """Test creating checkpoint state from DDP-wrapped model."""
        # Mock DDP wrapper
        class MockDDP:
            def __init__(self, module):
                self.module = module

        ddp_model = MockDDP(simple_model)

        state = create_checkpoint_state(
            step=100,
            model=ddp_model,
            optimizer=simple_optimizer,
        )

        # Should use module.state_dict()
        assert "linear.weight" in state.model_state_dict

    def test_load_checkpoint_to_model(self, simple_model, simple_optimizer):
        """Test loading checkpoint into model."""
        # Set known non-zero weights before creating checkpoint
        with torch.no_grad():
            simple_model.linear.weight.fill_(1.0)
            simple_model.linear.bias.fill_(0.5)

        # Create checkpoint with known weights
        # Note: deepcopy is needed because state_dict() returns references to tensors
        checkpoint_state = CheckpointState(
            step=100,
            epoch=2,
            model_state_dict=copy.deepcopy(simple_model.state_dict()),
            optimizer_state_dict=copy.deepcopy(simple_optimizer.state_dict()),
            lr_scheduler_state_dict={},
            dataset_state={},
            rng_states={},
            metrics={},
            world_size=1,
            timestamp=time.time(),
        )

        # Zero the model weights
        with torch.no_grad():
            simple_model.linear.weight.fill_(0.0)
            simple_model.linear.bias.fill_(0.0)

        # Load checkpoint
        load_checkpoint_to_model(checkpoint_state, simple_model, simple_optimizer)

        # Weights should be restored to known values
        assert torch.allclose(simple_model.linear.weight, torch.ones_like(simple_model.linear.weight))
        assert torch.allclose(simple_model.linear.bias, torch.full_like(simple_model.linear.bias, 0.5))


class TestCheckpointLoadConfig:
    """Tests for CheckpointLoadConfig and trusted_source functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CheckpointLoadConfig()

        assert config.safe_mode is True
        assert config.validate_structure is True
        assert config.warn_on_unsafe is True
        assert config.trusted_source is False

    def test_trusted_source_config(self):
        """Test trusted_source configuration."""
        config = CheckpointLoadConfig(trusted_source=True)

        assert config.trusted_source is True
        assert config.safe_mode is True  # Other defaults unchanged

    def test_load_with_trusted_source(self, checkpoint_state, tmp_path):
        """Test that trusted_source=True suppresses warnings during load."""
        backend = NVMeBackend(tmp_path)
        backend.save(checkpoint_state, "trusted_test.pt")

        # Load with trusted_source=True - should not log security warnings
        config = CheckpointLoadConfig(trusted_source=True)
        loaded = backend.load("trusted_test.pt", config=config)

        assert loaded.step == checkpoint_state.step

    def test_load_with_default_config(self, checkpoint_state, tmp_path):
        """Test loading with default config."""
        backend = NVMeBackend(tmp_path)
        backend.save(checkpoint_state, "default_test.pt")

        # Load with default config
        loaded = backend.load("default_test.pt")

        assert loaded.step == checkpoint_state.step

    def test_safe_globals_registration(self):
        """Test that safe globals registration is available on PyTorch 2.4+."""
        # This tests that the module loads without error
        # and that the safe globals function was called if available
        import elastic_harness.checkpoint.storage_backends as sb

        # The function should exist
        assert hasattr(sb, "_register_safe_checkpoint_types")

        # Check if add_safe_globals is available (PyTorch 2.4+)
        has_safe_globals = hasattr(torch.serialization, "add_safe_globals")

        if has_safe_globals:
            # On PyTorch 2.4+, OrderedDict should be registered
            # We can't directly check the registry, but we can verify
            # the function exists and was called without error
            import collections

            # Try to add it again - should not raise
            try:
                torch.serialization.add_safe_globals([collections.OrderedDict])
            except Exception:
                pytest.fail("add_safe_globals raised an exception")
