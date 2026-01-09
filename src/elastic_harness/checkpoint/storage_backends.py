"""Storage backends for checkpoint persistence.

This module provides abstract and concrete implementations for storing
checkpoints to various backends: local NVMe, S3, etc.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from elastic_harness.checkpoint.checkpointing import CheckpointState


class StorageBackend(ABC):
    """Abstract base class for checkpoint storage backends.

    All storage backends must implement save, load, exists, and list_checkpoints
    methods to provide a consistent interface for checkpoint management.
    """

    @abstractmethod
    def save(self, state: CheckpointState, path: str) -> None:
        """Save checkpoint to storage.

        Args:
            state: The checkpoint state to save.
            path: Path/key where to save the checkpoint.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> CheckpointState:
        """Load checkpoint from storage.

        Args:
            path: Path/key of the checkpoint to load.

        Returns:
            The loaded CheckpointState.
        """
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if checkpoint exists.

        Args:
            path: Path/key to check.

        Returns:
            True if checkpoint exists, False otherwise.
        """
        pass

    @abstractmethod
    def list_checkpoints(self, prefix: str = "") -> list[str]:
        """List available checkpoints.

        Args:
            prefix: Optional prefix to filter checkpoints.

        Returns:
            List of checkpoint paths/keys.
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a checkpoint.

        Args:
            path: Path/key of the checkpoint to delete.
        """
        pass


class NVMeBackend(StorageBackend):
    """Local NVMe/SSD storage backend for fast checkpoint I/O.

    This backend saves checkpoints to local disk, optimized for NVMe drives
    with fast random access. It's the primary backend for quick recovery.

    Example:
        >>> backend = NVMeBackend("/nvme/checkpoints")
        >>> backend.save(state, "step_1000.pt")
        >>> loaded = backend.load("step_1000.pt")
    """

    def __init__(
        self,
        base_path: str | Path,
        use_atomic_save: bool = True,
    ):
        """Initialize NVMe storage backend.

        Args:
            base_path: Base directory for checkpoint storage.
            use_atomic_save: Whether to use atomic saves (write to temp, then rename).
        """
        self.base_path = Path(base_path)
        self.use_atomic_save = use_atomic_save
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nvme_ckpt")

        # Ensure base directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _full_path(self, path: str) -> Path:
        """Get full path for a checkpoint."""
        return self.base_path / path

    def save(self, state: CheckpointState, path: str) -> None:
        """Save checkpoint to local disk.

        Uses atomic save by default: writes to a temp file first, then
        renames to the target path. This prevents corrupted checkpoints
        if the process is killed during save.

        Args:
            state: The checkpoint state to save.
            path: Relative path within base_path.
        """
        full_path = self._full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert state to serializable dict
        state_dict = state.to_dict()

        if self.use_atomic_save:
            # Write to temp file, then rename (atomic on most filesystems)
            fd, temp_path = tempfile.mkstemp(
                dir=full_path.parent,
                prefix=f".{full_path.name}.",
                suffix=".tmp",
            )
            try:
                os.close(fd)
                torch.save(state_dict, temp_path, pickle_protocol=5)
                os.rename(temp_path, full_path)
            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        else:
            torch.save(state_dict, full_path, pickle_protocol=5)

    def save_async(self, state: CheckpointState, path: str) -> Future:
        """Asynchronously save checkpoint.

        Copies tensors to CPU and spawns a background thread for I/O.

        Args:
            state: The checkpoint state to save.
            path: Relative path within base_path.

        Returns:
            Future that completes when save is done.
        """
        # Create a CPU copy of the state for async save
        cpu_state = state.cpu_copy()
        return self._executor.submit(self.save, cpu_state, path)

    def load(self, path: str) -> CheckpointState:
        """Load checkpoint from local disk.

        Args:
            path: Relative path within base_path.

        Returns:
            The loaded CheckpointState.

        Note:
            Security: This uses weights_only=False to support arbitrary Python
            objects in checkpoints (optimizer state, RNG state, etc.). Only load
            checkpoints from trusted sources, as malicious checkpoints could
            execute arbitrary code during deserialization.
        """
        from elastic_harness.checkpoint.checkpointing import CheckpointState

        full_path = self._full_path(path)
        state_dict = torch.load(full_path, map_location="cpu", weights_only=False)
        return CheckpointState.from_dict(state_dict)

    def exists(self, path: str) -> bool:
        """Check if checkpoint exists on disk.

        Args:
            path: Relative path within base_path.

        Returns:
            True if file exists.
        """
        return self._full_path(path).exists()

    def list_checkpoints(self, prefix: str = "") -> list[str]:
        """List checkpoints matching prefix.

        Args:
            prefix: Path prefix to filter (supports glob patterns).

        Returns:
            List of checkpoint paths relative to base_path.
        """
        if prefix:
            pattern = f"{prefix}*.pt"
        else:
            pattern = "*.pt"

        paths = sorted(self.base_path.glob(pattern))
        return [str(p.relative_to(self.base_path)) for p in paths]

    def delete(self, path: str) -> None:
        """Delete a checkpoint from disk.

        Args:
            path: Relative path within base_path.
        """
        full_path = self._full_path(path)
        if full_path.exists():
            full_path.unlink()

    def get_latest_checkpoint(self) -> str | None:
        """Get the path to the most recent checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        # Sort by modification time
        def get_mtime(p: str) -> float:
            return self._full_path(p).stat().st_mtime

        return max(checkpoints, key=get_mtime)

    def cleanup_old_checkpoints(self, keep_last: int = 3) -> list[str]:
        """Remove old checkpoints, keeping only the N most recent.

        Args:
            keep_last: Number of recent checkpoints to keep.

        Returns:
            List of deleted checkpoint paths.
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_last:
            return []

        # Sort by modification time
        def get_mtime(p: str) -> float:
            return self._full_path(p).stat().st_mtime

        sorted_checkpoints = sorted(checkpoints, key=get_mtime, reverse=True)
        to_delete = sorted_checkpoints[keep_last:]

        for path in to_delete:
            self.delete(path)

        return to_delete


class S3Backend(StorageBackend):
    """S3 storage backend for durable checkpoint storage.

    This backend saves checkpoints to Amazon S3 for durability. It supports
    async uploads for non-blocking training and multipart uploads for large
    checkpoints.

    Example:
        >>> backend = S3Backend("my-bucket", prefix="training/run-001/")
        >>> backend.save(state, "step_1000.pt")
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
    ):
        """Initialize S3 storage backend.

        Args:
            bucket: S3 bucket name.
            prefix: Key prefix for all checkpoints.
            region: AWS region (uses default if not specified).
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self.region = region

        # Lazy import boto3
        try:
            import boto3
            self._s3 = boto3.client("s3", region_name=region)
        except ImportError:
            raise ImportError("boto3 is required for S3Backend. Install with: pip install boto3")

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3_ckpt")

    def _full_key(self, path: str) -> str:
        """Get full S3 key for a checkpoint."""
        return f"{self.prefix}{path}"

    def save(self, state: CheckpointState, path: str) -> None:
        """Save checkpoint to S3.

        Uses multipart upload for large checkpoints.

        Args:
            state: The checkpoint state to save.
            path: Key suffix within the prefix.
        """
        key = self._full_key(path)
        state_dict = state.to_dict()

        # Serialize to bytes
        buffer = io.BytesIO()
        torch.save(state_dict, buffer, pickle_protocol=5)
        buffer.seek(0)

        # Upload to S3
        self._s3.upload_fileobj(buffer, self.bucket, key)

    def save_async(self, state: CheckpointState, path: str) -> Future:
        """Asynchronously save checkpoint to S3.

        Args:
            state: The checkpoint state to save.
            path: Key suffix within the prefix.

        Returns:
            Future that completes when upload is done.
        """
        cpu_state = state.cpu_copy()
        return self._executor.submit(self.save, cpu_state, path)

    def load(self, path: str) -> CheckpointState:
        """Load checkpoint from S3.

        Args:
            path: Key suffix within the prefix.

        Returns:
            The loaded CheckpointState.

        Note:
            Security: This uses weights_only=False to support arbitrary Python
            objects in checkpoints (optimizer state, RNG state, etc.). Only load
            checkpoints from trusted sources, as malicious checkpoints could
            execute arbitrary code during deserialization.
        """
        from elastic_harness.checkpoint.checkpointing import CheckpointState

        key = self._full_key(path)

        buffer = io.BytesIO()
        self._s3.download_fileobj(self.bucket, key, buffer)
        buffer.seek(0)

        state_dict = torch.load(buffer, map_location="cpu", weights_only=False)
        return CheckpointState.from_dict(state_dict)

    def exists(self, path: str) -> bool:
        """Check if checkpoint exists in S3.

        Args:
            path: Key suffix within the prefix.

        Returns:
            True if object exists.
        """
        import botocore.exceptions

        key = self._full_key(path)
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except botocore.exceptions.ClientError:
            return False

    def list_checkpoints(self, prefix: str = "") -> list[str]:
        """List checkpoints in S3.

        Args:
            prefix: Additional prefix to filter.

        Returns:
            List of checkpoint keys (relative to configured prefix).
        """
        full_prefix = self._full_key(prefix)

        keys = []
        continuation_token = None

        # Paginate through all results (S3 returns max 1000 per request)
        while True:
            if continuation_token:
                response = self._s3.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=full_prefix,
                    ContinuationToken=continuation_token,
                )
            else:
                response = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=full_prefix)

            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj["Key"]
                    if key.endswith(".pt"):
                        # Remove the base prefix
                        relative_key = key[len(self.prefix) :]
                        keys.append(relative_key)

            # Check if there are more results
            if response.get("IsTruncated"):
                continuation_token = response.get("NextContinuationToken")
            else:
                break

        return sorted(keys)

    def delete(self, path: str) -> None:
        """Delete a checkpoint from S3.

        Args:
            path: Key suffix within the prefix.
        """
        key = self._full_key(path)
        self._s3.delete_object(Bucket=self.bucket, Key=key)

    def get_latest_checkpoint(self) -> str | None:
        """Get the path to the most recent checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
        """
        full_prefix = self.prefix

        response = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=full_prefix)

        if "Contents" not in response:
            return None

        # Find most recent by LastModified
        latest = None
        latest_time = None

        for obj in response["Contents"]:
            if obj["Key"].endswith(".pt"):
                if latest_time is None or obj["LastModified"] > latest_time:
                    latest = obj["Key"]
                    latest_time = obj["LastModified"]

        if latest:
            return latest[len(self.prefix) :]
        return None
