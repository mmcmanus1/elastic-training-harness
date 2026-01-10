"""Configuration validation utilities.

This module provides functions to validate training configuration at startup,
catching common misconfigurations before training begins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation.

    Attributes:
        valid: True if no errors were found.
        errors: List of error messages (configuration is invalid).
        warnings: List of warning messages (configuration is valid but suboptimal).
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_elastic_config(config: dict[str, Any]) -> ValidationResult:
    """Validate elastic training configuration.

    Args:
        config: Configuration dictionary (typically from OmegaConf.to_container()).

    Returns:
        ValidationResult with any errors or warnings found.

    Example:
        >>> config = OmegaConf.to_container(loaded_config)
        >>> result = validate_elastic_config(config)
        >>> if not result.valid:
        ...     for error in result.errors:
        ...         print(f"Error: {error}")
        ...     sys.exit(1)
    """
    result = ValidationResult()

    # Validate elastic node constraints
    elastic = config.get("elastic", {})
    min_nodes = elastic.get("min_nodes", 1)
    max_nodes = elastic.get("max_nodes", 8)

    if min_nodes < 1:
        result.errors.append(f"min_nodes ({min_nodes}) must be at least 1")

    if max_nodes < min_nodes:
        result.errors.append(
            f"min_nodes ({min_nodes}) cannot exceed max_nodes ({max_nodes})"
        )

    base_world_size = elastic.get("base_world_size", min_nodes)
    if base_world_size < min_nodes or base_world_size > max_nodes:
        result.warnings.append(
            f"base_world_size ({base_world_size}) is outside the "
            f"[min_nodes, max_nodes] range [{min_nodes}, {max_nodes}]"
        )

    # Validate checkpoint intervals
    checkpoint = config.get("checkpoint", {})
    ckpt_interval = checkpoint.get("checkpoint_interval", 500)
    mem_interval = checkpoint.get("memory_snapshot_interval", 50)

    if ckpt_interval <= 0:
        result.errors.append(f"checkpoint_interval ({ckpt_interval}) must be positive")

    if mem_interval <= 0:
        result.errors.append(f"memory_snapshot_interval ({mem_interval}) must be positive")

    if ckpt_interval < mem_interval:
        result.warnings.append(
            f"checkpoint_interval ({ckpt_interval}) < memory_snapshot_interval "
            f"({mem_interval}). Memory snapshots may be unnecessary."
        )

    if mem_interval > 0 and ckpt_interval % mem_interval != 0:
        result.warnings.append(
            f"checkpoint_interval ({ckpt_interval}) is not a multiple of "
            f"memory_snapshot_interval ({mem_interval}). This may cause suboptimal "
            "checkpoint behavior."
        )

    # Validate training configuration
    training = config.get("training", {})
    lr = training.get("lr", 1e-4)
    batch_size = training.get("batch_size", 8)

    if lr <= 0:
        result.errors.append(f"Learning rate ({lr}) must be positive")

    if batch_size <= 0:
        result.errors.append(f"Batch size ({batch_size}) must be positive")

    # Validate scaling configuration
    scaling = config.get("scaling", {})
    strategy = scaling.get("strategy", "variable_batch")
    valid_strategies = ["variable_batch", "constant_batch"]
    if strategy not in valid_strategies:
        result.errors.append(
            f"Unknown scaling strategy '{strategy}'. "
            f"Valid options: {valid_strategies}"
        )

    if strategy == "constant_batch":
        target_batch = scaling.get("target_global_batch_size")
        if target_batch is None:
            result.errors.append(
                "target_global_batch_size is required for 'constant_batch' strategy"
            )
        elif target_batch <= 0:
            result.errors.append(
                f"target_global_batch_size ({target_batch}) must be positive"
            )

    # Validate gradient clipping
    grad_clip = training.get("gradient_clipping", {})
    if grad_clip:
        max_norm = grad_clip.get("max_norm", 1.0)
        if max_norm <= 0:
            result.errors.append(f"gradient_clipping.max_norm ({max_norm}) must be positive")

    result.valid = len(result.errors) == 0
    return result


def validate_s3_access(bucket: str, region: str | None = None) -> ValidationResult:
    """Validate S3 bucket access.

    This performs a head_bucket call to verify the bucket exists and is accessible.

    Args:
        bucket: S3 bucket name.
        region: Optional AWS region.

    Returns:
        ValidationResult with any errors or warnings.

    Note:
        This requires boto3 to be installed and valid AWS credentials configured.
    """
    result = ValidationResult()

    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        s3 = boto3.client("s3", region_name=region)
        s3.head_bucket(Bucket=bucket)
        logger.debug(f"S3 bucket '{bucket}' is accessible")

    except ImportError:
        result.warnings.append(
            "boto3 not installed, cannot validate S3 bucket access. "
            "Install with: pip install boto3"
        )

    except NoCredentialsError:
        result.warnings.append(
            "AWS credentials not configured, cannot validate S3 bucket access. "
            "Configure credentials or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY"
        )

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "404":
            result.errors.append(f"S3 bucket '{bucket}' does not exist")
        elif error_code == "403":
            result.errors.append(
                f"Access denied to S3 bucket '{bucket}'. "
                "Check IAM permissions."
            )
        else:
            result.warnings.append(
                f"Could not verify S3 bucket '{bucket}': {error_code} - {e}"
            )

    result.valid = len(result.errors) == 0
    return result


def validate_config(config: dict[str, Any], validate_s3: bool = False) -> ValidationResult:
    """Comprehensive configuration validation.

    Combines all validation checks and optionally validates S3 access.

    Args:
        config: Configuration dictionary.
        validate_s3: If True, also validate S3 bucket access (may be slow).

    Returns:
        Combined ValidationResult.
    """
    result = validate_elastic_config(config)

    # Optionally validate S3 access
    if validate_s3:
        checkpoint = config.get("checkpoint", {})
        s3_bucket = checkpoint.get("s3_bucket")

        if s3_bucket:
            s3_result = validate_s3_access(s3_bucket)
            result.errors.extend(s3_result.errors)
            result.warnings.extend(s3_result.warnings)
            result.valid = result.valid and s3_result.valid

    return result
