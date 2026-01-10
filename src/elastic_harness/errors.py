"""Custom exceptions with actionable error messages.

This module provides specialized exceptions for elastic training that include
helpful suggestions and context to aid in debugging.
"""

from __future__ import annotations

from typing import Any


class ElasticTrainingError(Exception):
    """Base exception for elastic training errors.

    All elastic training specific exceptions inherit from this class.
    """

    pass


class ConfigurationError(ElasticTrainingError):
    """Configuration validation failed.

    Raised when training configuration is invalid or inconsistent.
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        suggestions: list[str] | None = None,
    ):
        self.config_key = config_key
        self.suggestions = suggestions or []

        full_message = message
        if config_key:
            full_message = f"Configuration error in '{config_key}': {message}"
        if self.suggestions:
            full_message += "\n\nSuggestions:\n" + "\n".join(
                f"  - {s}" for s in self.suggestions
            )
        super().__init__(full_message)


class DistributedSetupError(ElasticTrainingError):
    """Distributed environment setup failed.

    Raised when initializing distributed training fails, typically due to
    missing environment variables or network issues.
    """

    COMMON_ISSUES = {
        "RANK": (
            "Missing RANK environment variable. "
            "Ensure you are running with torchrun:\n"
            "  torchrun --nproc-per-node=N train.py --config config.yaml"
        ),
        "LOCAL_RANK": (
            "Missing LOCAL_RANK environment variable. "
            "This is set automatically by torchrun."
        ),
        "WORLD_SIZE": (
            "Missing WORLD_SIZE environment variable. "
            "This is set automatically by torchrun."
        ),
        "MASTER_ADDR": (
            "Missing MASTER_ADDR environment variable. "
            "For single-node training, set MASTER_ADDR=localhost. "
            "For multi-node, set to the IP of the rank 0 node."
        ),
        "MASTER_PORT": (
            "Missing MASTER_PORT environment variable. "
            "Set to an available port, e.g., MASTER_PORT=29500"
        ),
        "NCCL": (
            "NCCL error detected. Common causes:\n"
            "  - GPU not available: Check nvidia-smi\n"
            "  - Network issues: Verify nodes can communicate\n"
            "  - Version mismatch: Ensure consistent NCCL versions\n"
            "  - Try NCCL_DEBUG=INFO for more details"
        ),
        "timeout": (
            "Distributed initialization timed out. Common causes:\n"
            "  - Firewall blocking ports: Open MASTER_PORT\n"
            "  - Incorrect MASTER_ADDR: Verify rank 0 node IP\n"
            "  - Network latency: Increase timeout with NCCL_SOCKET_TIMEOUT"
        ),
    }

    def __init__(
        self,
        message: str,
        missing_vars: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.missing_vars = missing_vars or []
        self.context = context or {}

        # Build helpful message with suggestions
        full_message = message

        # Add suggestions based on error content
        suggestions = []
        for keyword, suggestion in self.COMMON_ISSUES.items():
            if keyword in message or keyword in self.missing_vars:
                suggestions.append(suggestion)

        if suggestions:
            full_message += "\n\nPossible solutions:\n" + "\n".join(
                f"  {s}" for s in suggestions
            )

        super().__init__(full_message)


class CheckpointError(ElasticTrainingError):
    """Checkpoint operation failed.

    Base class for checkpoint-related errors.
    """

    pass


class CheckpointLoadError(CheckpointError):
    """Failed to load checkpoint.

    Raised when checkpoint loading fails due to file issues, format
    problems, or validation errors.
    """

    def __init__(
        self,
        message: str,
        path: str | None = None,
        tier: str | None = None,
        suggestions: list[str] | None = None,
    ):
        self.path = path
        self.tier = tier
        self.suggestions = suggestions or []

        full_message = message
        if path:
            full_message = f"Failed to load checkpoint from '{path}': {message}"
        if tier:
            full_message = f"[{tier}] {full_message}"

        # Add default suggestions if none provided
        if not self.suggestions:
            self.suggestions = [
                "Verify the checkpoint file exists and is readable",
                "Check that the checkpoint was created with a compatible version",
                "Try loading from a different tier (NVMe/S3)",
                "Check disk space if checkpoint may be corrupted",
            ]

        full_message += "\n\nSuggestions:\n" + "\n".join(
            f"  - {s}" for s in self.suggestions
        )
        super().__init__(full_message)


class CheckpointSaveError(CheckpointError):
    """Failed to save checkpoint.

    Raised when checkpoint saving fails due to disk space, permissions,
    or network issues (for S3).
    """

    def __init__(
        self,
        message: str,
        path: str | None = None,
        tier: str | None = None,
        original_error: Exception | None = None,
    ):
        self.path = path
        self.tier = tier
        self.original_error = original_error

        full_message = message
        if path:
            full_message = f"Failed to save checkpoint to '{path}': {message}"
        if tier:
            full_message = f"[{tier}] {full_message}"

        # Add tier-specific suggestions
        suggestions = []
        if tier == "nvme" or tier == "NVME":
            suggestions = [
                "Check available disk space: df -h",
                "Verify write permissions to checkpoint directory",
                "Ensure checkpoint directory exists",
            ]
        elif tier == "s3" or tier == "S3":
            suggestions = [
                "Verify AWS credentials are configured",
                "Check S3 bucket permissions (s3:PutObject)",
                "Verify network connectivity to S3",
                "Check for S3 service outages",
            ]

        if suggestions:
            full_message += "\n\nSuggestions:\n" + "\n".join(
                f"  - {s}" for s in suggestions
            )

        super().__init__(full_message)


class CheckpointValidationError(CheckpointError):
    """Checkpoint validation failed.

    Raised when a checkpoint has invalid structure or missing required data.
    """

    def __init__(
        self,
        message: str,
        missing_keys: list[str] | None = None,
        invalid_keys: list[str] | None = None,
    ):
        self.missing_keys = missing_keys or []
        self.invalid_keys = invalid_keys or []

        full_message = f"Checkpoint validation failed: {message}"

        if self.missing_keys:
            full_message += f"\n  Missing keys: {self.missing_keys}"
        if self.invalid_keys:
            full_message += f"\n  Invalid keys: {self.invalid_keys}"

        full_message += "\n\nThis may indicate a corrupted checkpoint or version mismatch."
        super().__init__(full_message)


class TopologyChangeError(ElasticTrainingError):
    """Topology change handling failed.

    Raised when the system fails to handle a worker join/leave event.
    """

    def __init__(
        self,
        message: str,
        old_world_size: int | None = None,
        new_world_size: int | None = None,
    ):
        self.old_world_size = old_world_size
        self.new_world_size = new_world_size

        full_message = message
        if old_world_size is not None and new_world_size is not None:
            full_message = (
                f"Topology change ({old_world_size} -> {new_world_size} workers) "
                f"failed: {message}"
            )

        suggestions = [
            "Check if min_nodes constraint is still satisfied",
            "Verify workers can communicate over the network",
            "Check for checkpoint availability for recovery",
        ]
        full_message += "\n\nSuggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)

        super().__init__(full_message)


class DataLoadingError(ElasticTrainingError):
    """Data loading failed.

    Raised when data loading encounters errors, such as missing files,
    invalid formats, or tokenization issues.
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        suggestions: list[str] | None = None,
    ):
        self.file_path = file_path
        self.suggestions = suggestions or [
            "Verify data files exist and are readable",
            "Check data file format matches expected format",
            "Ensure tokenizer is compatible with the data",
        ]

        full_message = message
        if file_path:
            full_message = f"Failed to load data from '{file_path}': {message}"

        full_message += "\n\nSuggestions:\n" + "\n".join(
            f"  - {s}" for s in self.suggestions
        )
        super().__init__(full_message)
