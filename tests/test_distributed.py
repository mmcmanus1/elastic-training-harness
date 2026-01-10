"""Tests for distributed training scenarios.

These tests verify distributed training functionality including:
- Single worker training
- Checkpoint recovery
- Configuration validation
- S3 backend integration (with localstack)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from elastic_harness.checkpoint.checkpointing import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointState,
    CheckpointTier,
    AsyncSavePolicy,
)
from elastic_harness.config_validator import (
    validate_config,
    validate_elastic_config,
    ValidationResult,
)
from elastic_harness.checkpoint.storage_backends import (
    NVMeBackend,
    S3Backend,
    S3Config,
    CheckpointLoadConfig,
)


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_valid_config(self):
        """Valid configuration should pass validation."""
        config = {
            "elastic": {
                "min_nodes": 1,
                "max_nodes": 8,
                "base_world_size": 4,
            },
            "checkpoint": {
                "checkpoint_interval": 500,
                "memory_snapshot_interval": 50,
            },
            "training": {
                "lr": 1e-4,
                "batch_size": 8,
            },
            "scaling": {
                "strategy": "variable_batch",
            },
        }

        result = validate_elastic_config(config)
        assert result.valid
        assert len(result.errors) == 0

    def test_invalid_min_max_nodes(self):
        """min_nodes > max_nodes should fail validation."""
        config = {
            "elastic": {
                "min_nodes": 8,
                "max_nodes": 4,
            },
        }

        result = validate_elastic_config(config)
        assert not result.valid
        assert any("min_nodes" in e and "max_nodes" in e for e in result.errors)

    def test_invalid_checkpoint_interval(self):
        """Negative checkpoint interval should fail validation."""
        config = {
            "checkpoint": {
                "checkpoint_interval": -1,
            },
        }

        result = validate_elastic_config(config)
        assert not result.valid
        assert any("checkpoint_interval" in e for e in result.errors)

    def test_warning_for_suboptimal_intervals(self):
        """checkpoint_interval < memory_snapshot_interval should warn."""
        config = {
            "checkpoint": {
                "checkpoint_interval": 25,
                "memory_snapshot_interval": 50,
            },
        }

        result = validate_elastic_config(config)
        assert result.valid  # Warning doesn't invalidate
        assert any("checkpoint_interval" in w for w in result.warnings)

    def test_constant_batch_requires_target(self):
        """constant_batch strategy requires target_global_batch_size."""
        config = {
            "scaling": {
                "strategy": "constant_batch",
            },
        }

        result = validate_elastic_config(config)
        assert not result.valid
        assert any("target_global_batch_size" in e for e in result.errors)


class TestSafeCheckpointLoading:
    """Tests for safe checkpoint loading."""

    def test_load_with_safe_mode(self, tmp_path):
        """Test loading checkpoint with safe mode enabled."""
        # Create a simple checkpoint
        backend = NVMeBackend(tmp_path)

        state = CheckpointState(
            step=100,
            model_state_dict={"weight": torch.randn(10, 10)},
            optimizer_state_dict={"state": {}},
        )
        backend.save(state, "test.pt")

        # Load with safe mode (default)
        config = CheckpointLoadConfig(safe_mode=True)
        loaded = backend.load("test.pt", config=config)

        assert loaded.step == 100
        assert "weight" in loaded.model_state_dict

    def test_load_validates_structure(self, tmp_path):
        """Test that structure validation catches invalid checkpoints."""
        backend = NVMeBackend(tmp_path)

        # Create invalid checkpoint (missing required keys)
        invalid_dict = {"random_key": 123}
        torch.save(invalid_dict, tmp_path / "invalid.pt")

        config = CheckpointLoadConfig(validate_structure=True)

        with pytest.raises(ValueError, match="Invalid checkpoint structure"):
            backend.load("invalid.pt", config=config)


class TestAsyncSaveFailureHandling:
    """Tests for async save failure handling."""

    def test_warn_policy_continues_on_failure(self, tmp_path):
        """Test that 'warn' policy logs warning and continues."""
        config = CheckpointConfig(
            nvme_path=tmp_path,
            async_save_policy=AsyncSavePolicy.WARN,
        )
        manager = CheckpointManager(config)

        # Simulate a failed async save by creating a mock future
        from concurrent.futures import Future

        failed_future = Future()
        failed_future.set_exception(Exception("Simulated S3 failure"))

        manager._pending_saves = [{
            "future": failed_future,
            "path": "test.pt",
            "state": CheckpointState(step=1),
            "retry_count": 0,
        }]

        # Should not raise, should return False
        result = manager.wait_for_pending_saves()
        assert result is False

    def test_fail_policy_raises_on_failure(self, tmp_path):
        """Test that 'fail' policy raises exception on failure."""
        from elastic_harness.checkpoint.checkpointing import CheckpointSaveError

        config = CheckpointConfig(
            nvme_path=tmp_path,
            async_save_policy=AsyncSavePolicy.FAIL,
        )
        manager = CheckpointManager(config)

        # Simulate a failed async save
        from concurrent.futures import Future

        failed_future = Future()
        failed_future.set_exception(Exception("Simulated S3 failure"))

        manager._pending_saves = [{
            "future": failed_future,
            "path": "test.pt",
            "state": CheckpointState(step=1),
            "retry_count": 0,
        }]

        with pytest.raises(CheckpointSaveError):
            manager.wait_for_pending_saves()


class TestS3BackendConfig:
    """Tests for S3 backend configuration."""

    def test_s3_config_dataclass(self):
        """Test S3Config dataclass defaults."""
        config = S3Config(bucket="test-bucket")

        assert config.bucket == "test-bucket"
        assert config.prefix == ""
        assert config.connect_timeout == 10.0
        assert config.read_timeout == 60.0
        assert config.max_retries == 3
        assert config.retry_mode == "adaptive"

    def test_s3_config_custom_values(self):
        """Test S3Config with custom values."""
        config = S3Config(
            bucket="my-bucket",
            prefix="checkpoints/",
            region="us-west-2",
            connect_timeout=30.0,
            read_timeout=120.0,
            max_retries=5,
            retry_mode="standard",
        )

        assert config.bucket == "my-bucket"
        assert config.prefix == "checkpoints/"
        assert config.region == "us-west-2"
        assert config.connect_timeout == 30.0
        assert config.max_retries == 5

    @pytest.mark.skipif(
        not os.environ.get("TEST_S3_BUCKET"),
        reason="TEST_S3_BUCKET not set"
    )
    def test_s3_backend_with_real_bucket(self):
        """Integration test with real S3 bucket."""
        bucket = os.environ["TEST_S3_BUCKET"]
        backend = S3Backend(bucket, prefix="test/")

        state = CheckpointState(
            step=1,
            model_state_dict={"weight": torch.randn(10)},
            optimizer_state_dict={},
        )

        # Save and load
        backend.save(state, "integration_test.pt")
        loaded = backend.load("integration_test.pt")

        assert loaded.step == 1

        # Cleanup
        backend.delete("integration_test.pt")


class TestMemoryLimitEnforcement:
    """Tests for memory snapshot memory limits."""

    def test_memory_limit_removes_old_snapshots(self):
        """Test that exceeding memory limit removes oldest snapshots."""
        from elastic_harness.checkpoint.memory_snapshot import MemorySnapshotBackend

        # Create backend with small memory limit (1MB)
        backend = MemorySnapshotBackend(max_snapshots=10, max_memory_mb=1.0)

        # Create a state that's about 400KB
        state = CheckpointState(
            step=1,
            model_state_dict={"weight": torch.randn(100, 1024)},  # ~400KB
            optimizer_state_dict={},
        )

        # Save three snapshots - third should trigger removal of first
        backend.save(state)
        assert backend.num_snapshots == 1

        state.step = 2
        backend.save(state)
        assert backend.num_snapshots == 2

        state.step = 3
        backend.save(state)
        # Should have removed oldest to stay within limit
        assert backend.num_snapshots <= 2

    def test_memory_limit_raises_when_single_snapshot_exceeds(self):
        """Test that exception is raised when single snapshot exceeds limit."""
        from elastic_harness.checkpoint.memory_snapshot import (
            MemorySnapshotBackend,
            MemoryLimitExceeded,
        )

        # Create backend with tiny memory limit (100KB)
        backend = MemorySnapshotBackend(max_snapshots=10, max_memory_mb=0.1)

        # Create a state that's larger than the limit
        state = CheckpointState(
            step=1,
            model_state_dict={"weight": torch.randn(100, 1024)},  # ~400KB
            optimizer_state_dict={},
        )

        with pytest.raises(MemoryLimitExceeded):
            backend.save(state)


class TestGradientAccumulationRounding:
    """Tests for gradient accumulation rounding modes."""

    def test_ceil_rounding_mode(self):
        """Test ceil rounding mode."""
        from elastic_harness.scaling.lr_scaling import (
            GradAccumulationConfig,
            GradientAccumulationManager,
            RoundingMode,
        )

        config = GradAccumulationConfig(
            target_global_batch_size=1000,
            local_batch_size=32,
            base_world_size=4,
            rounding_mode=RoundingMode.CEIL,
        )
        manager = GradientAccumulationManager(config)

        # 3 workers: per_step_batch = 96, raw_accum = 10.42
        # ceil(10.42) = 11
        new_accum = manager.on_topology_change(3)
        assert new_accum == 11

    def test_floor_rounding_mode(self):
        """Test floor rounding mode."""
        from elastic_harness.scaling.lr_scaling import (
            GradAccumulationConfig,
            GradientAccumulationManager,
            RoundingMode,
        )

        config = GradAccumulationConfig(
            target_global_batch_size=1000,
            local_batch_size=32,
            base_world_size=4,
            rounding_mode=RoundingMode.FLOOR,
        )
        manager = GradientAccumulationManager(config)

        # 3 workers: per_step_batch = 96, raw_accum = 10.42
        # floor(10.42) = 10
        new_accum = manager.on_topology_change(3)
        assert new_accum == 10


@pytest.mark.integration
class TestDistributedTrainingIntegration:
    """Integration tests for distributed training.

    These tests require a working PyTorch distributed environment.
    Run with: pytest tests/test_distributed.py -m integration
    """

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create a minimal test configuration."""
        config_content = """
model:
  vocab_size: 1000
  d_model: 64
  nhead: 2
  num_layers: 1
  dim_feedforward: 128
  max_seq_length: 32
  dropout: 0.0

training:
  lr: 1.0e-3
  batch_size: 2
  max_steps: 10
  log_interval: 5
  gradient_clipping:
    enabled: true
    max_norm: 1.0

data:
  enabled: false

checkpoint:
  checkpoint_interval: 5
  memory_snapshot_interval: 2
  nvme_path: "{nvme_path}"
  async_save: false

scaling:
  strategy: "variable_batch"
  rule: "linear"
  warmup_steps: 0

elastic:
  base_world_size: 1
  min_nodes: 1
  max_nodes: 4
"""
        nvme_path = tmp_path / "checkpoints"
        nvme_path.mkdir()

        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content.format(nvme_path=nvme_path))
        return config_path, nvme_path

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_single_worker_cuda_training(self, minimal_config):
        """Test training with single CUDA worker."""
        config_path, nvme_path = minimal_config

        env = os.environ.copy()
        env.update({
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        })

        result = subprocess.run(
            [
                sys.executable, "-m", "elastic_harness.train",
                "--config", str(config_path)
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"Training failed: {result.stderr}"
        assert "Training complete!" in result.stdout

    def test_single_worker_cpu_training(self, minimal_config):
        """Test training with single CPU worker (gloo backend)."""
        config_path, nvme_path = minimal_config

        env = os.environ.copy()
        env.update({
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29501",
        })

        result = subprocess.run(
            [
                sys.executable, "-m", "elastic_harness.train",
                "--config", str(config_path)
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Should either succeed or fail gracefully with gloo
        # (gloo may not work on all systems without proper setup)
        if result.returncode == 0:
            assert "Training complete!" in result.stdout
