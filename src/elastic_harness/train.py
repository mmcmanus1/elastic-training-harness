"""Main training script for elastic distributed training.

This script serves as the entry point for torchrun and implements the
complete training loop with elastic fault tolerance.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf

from elastic_harness.agent import setup_distributed_environment, cleanup_distributed, get_world_info
from elastic_harness.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointState,
    CheckpointTier,
    create_checkpoint_state,
    load_checkpoint_to_model,
)
from elastic_harness.data import TokenIndexFile, ResumableDataset, create_dataloader
from elastic_harness.scaling import (
    ScalingConfig,
    ScalingRule,
    LRScalingManager,
    GradAccumulationConfig,
    GradientAccumulationManager,
    ElasticScalingManager,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Rank %(rank)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class RankFilter(logging.Filter):
    """Add rank to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = dist.get_rank() if dist.is_initialized() else 0
        return True


logger = logging.getLogger(__name__)
logger.addFilter(RankFilter())


class SimpleTransformerLM(nn.Module):
    """Simple transformer language model for demonstration.

    This is a minimal implementation for testing the training harness.
    Replace with your actual model in production.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_length: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)

        # Tie weights
        self.output_proj.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)
        x = self.transformer(x, mask=mask, is_causal=True)

        logits = self.output_proj(x)

        output = {"logits": logits}

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            output["loss"] = loss

        return output


def load_config(config_path: str) -> OmegaConf:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        OmegaConf configuration object.
    """
    return OmegaConf.load(config_path)


def create_model(config: OmegaConf, device: torch.device) -> nn.Module:
    """Create model from configuration.

    Args:
        config: Model configuration.
        device: Device to place model on.

    Returns:
        Initialized model.
    """
    model = SimpleTransformerLM(
        vocab_size=config.get("vocab_size", 50257),
        d_model=config.get("d_model", 512),
        nhead=config.get("nhead", 8),
        num_layers=config.get("num_layers", 6),
        dim_feedforward=config.get("dim_feedforward", 2048),
        max_seq_length=config.get("max_seq_length", 1024),
        dropout=config.get("dropout", 0.1),
    )
    return model.to(device)


def maybe_chaos_crash(step: int, rank: int, chaos_config: dict) -> None:
    """Randomly crash the process for chaos testing.

    This simulates real-world failures like CUDA OOM, hardware errors,
    or spot instance preemption.

    Args:
        step: Current training step.
        rank: Worker rank.
        chaos_config: Chaos testing configuration with keys:
            - enabled: Whether chaos mode is on
            - crash_probability: Probability of crash per step (e.g., 0.001)
            - crash_after_step: Only crash after this step
            - crash_ranks: List of ranks that can crash (default: all except 0)
    """
    if not chaos_config.get("enabled", False):
        return

    crash_after = chaos_config.get("crash_after_step", 50)
    if step < crash_after:
        return

    # Default: don't crash rank 0 (to keep training going)
    crash_ranks = chaos_config.get("crash_ranks", None)
    if crash_ranks is not None and rank not in crash_ranks:
        return
    elif crash_ranks is None and rank == 0:
        return

    probability = chaos_config.get("crash_probability", 0.001)

    if random.random() < probability:
        logger.warning(f"CHAOS: Worker {rank} self-destructing at step {step}!")
        # Give a moment for the log to flush
        time.sleep(0.1)
        # Exit with error code to simulate crash
        sys.exit(1)


def worker_main(args: argparse.Namespace) -> None:
    """Main training function executed by each worker.

    Args:
        args: Command line arguments.
    """
    # Setup distributed environment
    world_info = setup_distributed_environment(backend="nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{world_info.local_rank}" if torch.cuda.is_available() else "cpu")

    logger.info(f"Worker started: rank={world_info.rank}, world_size={world_info.world_size}")

    # Load configuration
    config = load_config(args.config)

    # Create model
    model = create_model(config.model, device)
    model = DDP(model, device_ids=[world_info.local_rank] if torch.cuda.is_available() else None)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.get("lr", 1e-4),
        weight_decay=config.training.get("weight_decay", 0.01),
    )

    # Setup scaling strategy
    scaling_strategy = config.scaling.get("strategy", "variable_batch")
    local_batch_size = config.training.get("batch_size", 8)
    target_global_batch = config.scaling.get("target_global_batch_size", None)

    # Setup LR scaling config
    lr_config = ScalingConfig(
        base_lr=config.training.get("lr", 1e-4),
        base_batch_size=local_batch_size,
        base_world_size=config.elastic.get("base_world_size", world_info.world_size),
        scaling_rule=ScalingRule(config.scaling.get("rule", "linear")),
        warmup_steps=config.scaling.get("warmup_steps", 100),
    )

    # Setup gradient accumulation config (for constant_batch strategy)
    accum_config = None
    if scaling_strategy == "constant_batch" and target_global_batch:
        accum_config = GradAccumulationConfig(
            target_global_batch_size=target_global_batch,
            local_batch_size=local_batch_size,
            base_world_size=config.elastic.get("base_world_size", world_info.world_size),
        )

    # Create unified scaling manager
    scaling_manager = ElasticScalingManager(
        lr_config=lr_config,
        accum_config=accum_config,
        optimizer=optimizer,
        strategy=scaling_strategy,
    )

    # Setup checkpoint manager
    checkpoint_config = CheckpointConfig(
        checkpoint_interval=config.checkpoint.get("checkpoint_interval", 500),
        memory_snapshot_interval=config.checkpoint.get("memory_snapshot_interval", 50),
        nvme_path=config.checkpoint.get("nvme_path", "/tmp/checkpoints"),
        s3_bucket=config.checkpoint.get("s3_bucket"),
        s3_prefix=config.checkpoint.get("s3_prefix", ""),
        async_save=config.checkpoint.get("async_save", True),
        keep_last_n=config.checkpoint.get("keep_last_n", 3),
    )
    checkpoint_manager = CheckpointManager(checkpoint_config)

    # Load checkpoint if exists
    start_step = 0
    start_tokens = 0
    checkpoint = checkpoint_manager.load_checkpoint()

    if checkpoint:
        load_checkpoint_to_model(checkpoint, model, optimizer)
        start_step = checkpoint.step

        # Check for topology change
        if checkpoint.world_size != world_info.world_size:
            result = scaling_manager.on_topology_change(world_info.world_size)
            logger.info(
                f"Topology changed: {checkpoint.world_size} -> {world_info.world_size}, "
                f"LR: {result.get('lr', 'unchanged')}, "
                f"Accum steps: {result.get('accumulation_steps', 'unchanged')}"
            )

        start_tokens = checkpoint.dataset_state.get("total_tokens_processed", 0)
        logger.info(f"Resumed from checkpoint at step {start_step}")
    else:
        logger.info("Starting fresh training run")

    # Setup data loading
    tokenizer = None
    dataloader = None

    if config.data.get("enabled", False):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.data.get("tokenizer", "gpt2"))

            index_file = TokenIndexFile(config.data.index_path)
            dataset = ResumableDataset(
                data_files=config.data.files,
                index_file=index_file,
                tokenizer=tokenizer,
                seq_length=config.model.get("max_seq_length", 1024),
                start_token_index=start_tokens,
            )

            if checkpoint and checkpoint.dataset_state:
                dataset.load_state_dict(checkpoint.dataset_state)

            dataloader = create_dataloader(
                dataset,
                batch_size=local_batch_size,
                num_workers=config.data.get("num_workers", 4),
            )
        except ImportError:
            logger.warning("transformers not installed, using synthetic data")

    # Create synthetic data generator for testing
    def synthetic_batch_generator():
        seq_length = config.model.get("max_seq_length", 1024)
        vocab_size = config.model.get("vocab_size", 50257)

        while True:
            input_ids = torch.randint(0, vocab_size, (local_batch_size, seq_length), device=device)
            labels = torch.randint(0, vocab_size, (local_batch_size, seq_length), device=device)
            yield {"input_ids": input_ids, "labels": labels}

    data_iter = iter(dataloader) if dataloader else synthetic_batch_generator()

    # Training parameters
    max_steps = config.training.get("max_steps", 10000)
    log_interval = config.training.get("log_interval", 10)

    # Chaos testing configuration
    chaos_config = {
        "enabled": config.get("chaos", {}).get("enabled", False),
        "crash_probability": config.get("chaos", {}).get("crash_probability", 0.001),
        "crash_after_step": config.get("chaos", {}).get("crash_after_step", 50),
        "crash_ranks": config.get("chaos", {}).get("crash_ranks", None),
    }

    if chaos_config["enabled"]:
        logger.warning(
            f"CHAOS MODE ENABLED: crash_probability={chaos_config['crash_probability']}, "
            f"crash_after_step={chaos_config['crash_after_step']}"
        )

    # Training loop
    model.train()
    accumulated_loss = 0.0
    micro_step = 0
    step_start_time = time.time()

    for step in range(start_step, max_steps):
        # Chaos testing: maybe crash
        maybe_chaos_crash(step, world_info.rank, chaos_config)

        # Memory snapshot (frequent)
        if checkpoint_manager.should_checkpoint(step, CheckpointTier.MEMORY):
            state = create_checkpoint_state(
                step=step,
                model=model,
                optimizer=optimizer,
                dataset_state=dataloader.dataset.state_dict() if dataloader else {},
            )
            checkpoint_manager.save_checkpoint(state, tier=CheckpointTier.MEMORY)

        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            if dataloader:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            else:
                data_iter = synthetic_batch_generator()
                batch = next(data_iter)

        # Move to device if needed
        if not batch["input_ids"].is_cuda and torch.cuda.is_available():
            batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(batch["input_ids"], labels=batch["labels"])

        # Scale loss by accumulation steps
        loss = outputs["loss"] / scaling_manager.accumulation_steps

        # Backward pass
        loss.backward()
        accumulated_loss += loss.item()
        micro_step += 1

        # Optimizer step (respects gradient accumulation)
        if scaling_manager.should_step():
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # Step LR warmup if active
            scaling_manager.step_lr()

            micro_step = 0

        # Logging
        if step % log_interval == 0 and world_info.rank == 0:
            elapsed = time.time() - step_start_time
            tokens_per_sec = (
                log_interval * local_batch_size *
                config.model.get("max_seq_length", 1024) * world_info.world_size
            ) / max(elapsed, 1e-6)

            logger.info(
                f"Step {step} | Loss: {accumulated_loss:.4f} | "
                f"LR: {scaling_manager.current_lr:.2e} | "
                f"Accum: {scaling_manager.accumulation_steps} | "
                f"Tokens/s: {tokens_per_sec:.0f}"
            )

            accumulated_loss = 0.0
            step_start_time = time.time()

        # NVMe checkpoint (less frequent)
        if checkpoint_manager.should_checkpoint(step, CheckpointTier.NVME):
            state = create_checkpoint_state(
                step=step,
                model=model,
                optimizer=optimizer,
                dataset_state=dataloader.dataset.state_dict() if dataloader else {},
                metrics={
                    "loss": accumulated_loss,
                    "world_size": world_info.world_size,
                    "accumulation_steps": scaling_manager.accumulation_steps,
                },
            )
            checkpoint_manager.save_checkpoint(state, tier=CheckpointTier.NVME)

            # Also save to S3 if configured
            if checkpoint_config.s3_bucket:
                checkpoint_manager.save_checkpoint(state, tier=CheckpointTier.S3)

            # Clear memory snapshots after persistent save
            checkpoint_manager.clear_memory_snapshots()

    # Final checkpoint
    state = create_checkpoint_state(
        step=max_steps,
        model=model,
        optimizer=optimizer,
        dataset_state=dataloader.dataset.state_dict() if dataloader else {},
    )
    checkpoint_manager.save_checkpoint(state, tier=CheckpointTier.NVME)

    # Wait for pending S3 saves
    checkpoint_manager.wait_for_pending_saves()

    logger.info("Training complete!")
    cleanup_distributed()


def main():
    """Entry point for torchrun."""
    parser = argparse.ArgumentParser(description="Elastic distributed training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file",
    )
    args = parser.parse_args()

    try:
        worker_main(args)
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
