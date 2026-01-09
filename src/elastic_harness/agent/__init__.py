"""Elastic agent module for distributed training coordination."""

from elastic_harness.agent.elastic_agent import (
    ElasticTrainingConfig,
    setup_distributed_environment,
    get_world_info,
)

__all__ = [
    "ElasticTrainingConfig",
    "setup_distributed_environment",
    "get_world_info",
]
