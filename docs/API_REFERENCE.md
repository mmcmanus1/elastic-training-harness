# API Reference

This document provides a comprehensive reference for the elastic-training-harness public API.

## Table of Contents

- [elastic_harness.agent](#elastic_harnessagent)
- [elastic_harness.checkpoint](#elastic_harnesscheckpoint)
- [elastic_harness.scaling](#elastic_harnessscaling)
- [elastic_harness.metrics](#elastic_harnessmetrics)
- [elastic_harness.errors](#elastic_harnesserrors)
- [elastic_harness.config_validator](#elastic_harnessconfig_validator)
- [Usage Examples](#usage-examples)

---

## elastic_harness.agent

Distributed environment setup and coordination utilities.

### Classes

#### `ElasticTrainingConfig`

Configuration for elastic distributed training with PyTorch and etcd.

```python
from elastic_harness.agent import ElasticTrainingConfig

config = ElasticTrainingConfig(
    min_nodes=1,
    max_nodes=4,
    rdzv_backend="etcd-v2",
    rdzv_endpoint="localhost:2379",
    rdzv_id="elastic-training",
    heartbeat_interval=30,
    heartbeat_timeout=90,
    nproc_per_node=4,  # GPUs per node
    max_restarts=100,
)
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_nodes` | `int` | `1` | Minimum nodes required to start training |
| `max_nodes` | `int` | `4` | Maximum nodes allowed |
| `rdzv_backend` | `str` | `"etcd-v2"` | Rendezvous backend type |
| `rdzv_endpoint` | `str` | `"localhost:2379"` | Rendezvous endpoint address |
| `rdzv_id` | `str` | `"elastic-training"` | Unique job identifier |
| `heartbeat_interval` | `int` | `30` | Seconds between heartbeats |
| `heartbeat_timeout` | `int` | `90` | Seconds before missed heartbeat triggers failure |
| `nproc_per_node` | `int` | GPU count | Processes per node |
| `max_restarts` | `int` | `100` | Maximum worker restart attempts |

**Methods:**

- `to_torchrun_args() -> list[str]`: Convert config to torchrun CLI arguments.

---

#### `WorldInfo`

Information about the distributed world.

```python
from elastic_harness.agent import get_world_info

info = get_world_info()
print(f"Rank {info.rank} of {info.world_size}")
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `rank` | `int` | Global rank of this process |
| `local_rank` | `int` | Rank on this node (0 to nproc_per_node-1) |
| `world_size` | `int` | Total number of processes |
| `local_world_size` | `int` | Processes on this node |
| `master_addr` | `str` | Address of rank 0 |
| `master_port` | `int` | Port of rank 0 |

---

### Functions

#### `setup_distributed_environment`

```python
def setup_distributed_environment(
    backend: str = "nccl",
    init_method: str | None = None,
) -> WorldInfo
```

Initialize PyTorch distributed process group.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"nccl"` | Backend: `"nccl"` (GPU), `"gloo"` (CPU), or `"auto"` |
| `init_method` | `str` | `None` | Initialization method (default: `"env://"`) |

**Returns:** `WorldInfo` with distributed configuration.

**Raises:** `RuntimeError` if setup fails.

---

#### `cleanup_distributed`

```python
def cleanup_distributed() -> None
```

Clean up distributed environment. Call at end of training.

---

#### `is_main_process`

```python
def is_main_process() -> bool
```

Check if this is rank 0. Returns `True` if rank 0 or if distributed not initialized.

---

#### `get_rank`

```python
def get_rank() -> int
```

Get global rank (0 if distributed not initialized).

---

#### `get_world_size`

```python
def get_world_size() -> int
```

Get total processes (1 if distributed not initialized).

---

#### `barrier`

```python
def barrier() -> None
```

Synchronize all processes. Blocks until all processes reach this point.

---

#### `broadcast_object`

```python
def broadcast_object(obj: Any, src: int = 0) -> Any
```

Broadcast Python object from source rank to all ranks.

---

## elastic_harness.checkpoint

Multi-tier checkpoint management for fast recovery.

### Classes

#### `CheckpointConfig`

```python
from elastic_harness.checkpoint import CheckpointConfig

config = CheckpointConfig(
    checkpoint_interval=500,
    memory_snapshot_interval=50,
    nvme_path="/nvme/checkpoints",
    s3_bucket="my-bucket",
    s3_prefix="training/run-001/",
    async_save=True,
    keep_last_n=3,
    async_save_policy="warn",  # or "fail", "retry"
    async_save_retries=3,
    async_save_retry_delay=5.0,
)
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_interval` | `int` | `500` | Steps between NVMe/S3 checkpoints |
| `memory_snapshot_interval` | `int` | `50` | Steps between memory snapshots |
| `nvme_path` | `str\|Path` | `"/tmp/checkpoints"` | Local checkpoint directory |
| `s3_bucket` | `str\|None` | `None` | S3 bucket for durable storage |
| `s3_prefix` | `str` | `""` | S3 key prefix |
| `async_save` | `bool` | `True` | Use async S3 uploads |
| `keep_last_n` | `int` | `3` | Checkpoints to retain |
| `async_save_policy` | `str` | `"warn"` | Failure handling: `"warn"`, `"fail"`, `"retry"` |
| `async_save_retries` | `int` | `3` | Retry attempts for `"retry"` policy |
| `async_save_retry_delay` | `float` | `5.0` | Base delay between retries (seconds) |

---

#### `CheckpointState`

Complete training state for checkpointing.

```python
from elastic_harness.checkpoint import CheckpointState

state = CheckpointState(
    step=1000,
    epoch=2,
    model_state_dict=model.state_dict(),
    optimizer_state_dict=optimizer.state_dict(),
    lr_scheduler_state_dict=scheduler.state_dict(),
    dataset_state={"tokens_processed": 50000},
    rng_states={"torch": torch.get_rng_state()},
    metrics={"loss": 0.5},
    world_size=4,
)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `step` | `int` | Current training step |
| `epoch` | `int` | Current epoch |
| `model_state_dict` | `dict` | Model parameters |
| `optimizer_state_dict` | `dict` | Optimizer state |
| `lr_scheduler_state_dict` | `dict` | LR scheduler state |
| `dataset_state` | `dict` | Data loader position |
| `rng_states` | `dict` | RNG states (torch, cuda, etc.) |
| `metrics` | `dict` | Training metrics |
| `world_size` | `int` | World size when saved |
| `timestamp` | `float` | Unix timestamp |

**Methods:**

- `to_dict() -> dict`: Serialize to dictionary.
- `from_dict(data: dict) -> CheckpointState`: Deserialize from dictionary.
- `cpu_copy() -> CheckpointState`: Create copy with CPU tensors.

---

#### `CheckpointManager`

Manages checkpoint lifecycle across storage tiers.

```python
from elastic_harness.checkpoint import CheckpointManager, CheckpointConfig, CheckpointTier

config = CheckpointConfig(nvme_path="/nvme/ckpt", s3_bucket="my-bucket")
manager = CheckpointManager(config)

# Save checkpoint
manager.save_checkpoint(state, tier=CheckpointTier.NVME)
manager.save_checkpoint(state, tier=CheckpointTier.S3)
manager.save_checkpoint(state, tier=CheckpointTier.MEMORY)

# Load with fallback hierarchy (memory -> NVMe -> S3)
restored = manager.load_checkpoint()

# Or load from specific tier
restored = manager.load_checkpoint(path="checkpoint_00001000.pt", tier=CheckpointTier.NVME)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `save_checkpoint(state, tier)` | `str\|None` | Save to tier, returns path |
| `load_checkpoint(path?, tier?)` | `CheckpointState\|None` | Load with fallback |
| `wait_for_pending_saves(timeout?)` | `bool` | Wait for async S3 saves |
| `clear_memory_snapshots()` | `None` | Free memory after persistent save |
| `should_checkpoint(step, tier)` | `bool` | Check if checkpoint due |

---

#### `CheckpointTier`

```python
from elastic_harness.checkpoint import CheckpointTier

CheckpointTier.MEMORY  # In-memory (fastest, ~0ms)
CheckpointTier.NVME    # Local SSD (2-5s)
CheckpointTier.S3      # Cloud storage (minutes, durable)
```

---

#### `MemorySnapshotBackend`

In-memory checkpoint storage for instant recovery.

```python
from elastic_harness.checkpoint import MemorySnapshotBackend

backend = MemorySnapshotBackend(
    max_snapshots=2,
    max_memory_mb=1024,  # Optional memory limit
)

backend.save(state)
restored = backend.load()
backend.clear()
```

**Attributes/Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `num_snapshots` | `int` | Current snapshot count |
| `latest_step` | `int\|None` | Step of most recent snapshot |
| `memory_usage_bytes()` | `int` | Estimated memory usage |

---

#### `NVMeBackend`

Local disk storage backend.

```python
from elastic_harness.checkpoint import NVMeBackend

backend = NVMeBackend("/nvme/checkpoints", use_atomic_save=True)
backend.save(state, "step_1000.pt")
loaded = backend.load("step_1000.pt")
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `save(state, path)` | `None` | Save checkpoint |
| `save_async(state, path)` | `Future` | Async save |
| `load(path, config?)` | `CheckpointState` | Load checkpoint |
| `exists(path)` | `bool` | Check existence |
| `list_checkpoints(prefix?)` | `list[str]` | List checkpoints |
| `delete(path)` | `None` | Delete checkpoint |
| `get_latest_checkpoint()` | `str\|None` | Get most recent |
| `cleanup_old_checkpoints(keep_last)` | `list[str]` | Remove old checkpoints |

---

#### `S3Backend`

S3/cloud storage backend.

```python
from elastic_harness.checkpoint import S3Backend

backend = S3Backend(
    bucket="my-bucket",
    prefix="training/run-001/",
    region="us-east-1",
)
backend.save(state, "step_1000.pt")

# Async upload
future = backend.save_async(state, "step_1000.pt")
future.result()  # Wait for completion
```

Same methods as `NVMeBackend`.

---

### Helper Functions

#### `create_checkpoint_state`

```python
from elastic_harness.checkpoint import create_checkpoint_state

state = create_checkpoint_state(
    step=1000,
    model=model,  # or DDP-wrapped model
    optimizer=optimizer,
    lr_scheduler=scheduler,  # optional
    dataset_state={"position": 50000},  # optional
    metrics={"loss": 0.5},  # optional
)
```

Creates `CheckpointState` from training components. Handles DDP-wrapped models automatically.

---

#### `load_checkpoint_to_model`

```python
from elastic_harness.checkpoint import load_checkpoint_to_model

load_checkpoint_to_model(
    state=loaded_state,
    model=model,
    optimizer=optimizer,
    lr_scheduler=scheduler,  # optional
    strict=True,  # strict state dict matching
)
```

Loads checkpoint state into training components.

---

## elastic_harness.scaling

Learning rate scaling and gradient accumulation for topology changes.

### Classes

#### `ScalingRule`

```python
from elastic_harness.scaling import ScalingRule

ScalingRule.LINEAR  # lr_new = lr_base * (batch_new / batch_base)
ScalingRule.SQRT    # lr_new = lr_base * sqrt(batch_new / batch_base)
ScalingRule.NONE    # No scaling
```

---

#### `ScalingConfig`

```python
from elastic_harness.scaling import ScalingConfig, ScalingRule

config = ScalingConfig(
    base_lr=1e-4,
    base_batch_size=8,
    base_world_size=4,
    scaling_rule=ScalingRule.LINEAR,
    warmup_steps=100,
    min_lr=1e-7,
    max_lr=1e-2,
)
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_lr` | `float` | - | Base learning rate |
| `base_batch_size` | `int` | - | Batch size per worker at base config |
| `base_world_size` | `int` | - | World size at base config |
| `scaling_rule` | `ScalingRule` | `LINEAR` | Scaling rule to apply |
| `warmup_steps` | `int` | `100` | Warmup steps after topology change |
| `min_lr` | `float` | `1e-7` | Minimum allowed LR |
| `max_lr` | `float` | `1e-2` | Maximum allowed LR |

**Properties:**

- `base_effective_batch_size -> int`: `base_batch_size * base_world_size`

---

#### `LRScalingManager`

Manages learning rate adjustment during topology changes.

```python
from elastic_harness.scaling import LRScalingManager, ScalingConfig

config = ScalingConfig(base_lr=1e-4, base_batch_size=8, base_world_size=4)
manager = LRScalingManager(config, optimizer)

# On topology change (e.g., 4 -> 2 workers)
new_lr = manager.on_topology_change(new_world_size=2)
print(f"LR adjusted to {new_lr}")

# During training loop (for warmup)
for step in range(steps):
    manager.step()  # Progress warmup if active
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `current_lr` | `float` | Current learning rate |
| `current_effective_batch_size` | `int` | Current effective batch size |
| `is_warming_up` | `bool` | True if warmup in progress |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `on_topology_change(new_world_size, batch_size?)` | `float` | Handle change, returns new LR |
| `get_scaled_lr(base_lr, old_batch, new_batch)` | `float` | Calculate scaled LR |
| `step()` | `float` | Progress warmup scheduler |
| `state_dict()` | `dict` | State for checkpointing |
| `load_state_dict(state)` | `None` | Restore from checkpoint |

---

#### `GradAccumulationConfig`

```python
from elastic_harness.scaling import GradAccumulationConfig

config = GradAccumulationConfig(
    target_global_batch_size=1024,
    local_batch_size=32,
    base_world_size=4,
    rounding_mode="ceil",  # or "floor", "nearest"
)
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_global_batch_size` | `int` | - | Target global batch size |
| `local_batch_size` | `int` | - | Batch size per GPU |
| `base_world_size` | `int` | - | World size at base config |
| `rounding_mode` | `str` | `"ceil"` | How to round accumulation steps |

---

#### `GradientAccumulationManager`

Maintains constant global batch size across topology changes.

```python
from elastic_harness.scaling import GradientAccumulationManager, GradAccumulationConfig

config = GradAccumulationConfig(
    target_global_batch_size=1024,
    local_batch_size=32,
    base_world_size=4,
)
manager = GradientAccumulationManager(config)

# With 4 GPUs: 1024 / (32 * 4) = 8 accumulation steps
print(manager.accumulation_steps)  # 8

# On topology change (4 -> 3 workers)
manager.on_topology_change(new_world_size=3)
print(manager.accumulation_steps)  # 11 (1024 / (32 * 3) ≈ 10.67, ceil to 11)

# In training loop
for batch in dataloader:
    loss = model(batch)
    (loss / manager.loss_scale_factor).backward()

    if manager.should_step():
        optimizer.step()
        optimizer.zero_grad()
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `accumulation_steps` | `int` | Current accumulation steps |
| `effective_batch_size` | `int` | Actual global batch size |
| `loss_scale_factor` | `float` | Factor to scale loss (equals accumulation_steps) |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `on_topology_change(new_world_size)` | `int` | Recalculate, returns new steps |
| `should_step()` | `bool` | True if optimizer should step |
| `reset()` | `None` | Reset accumulation counter |
| `state_dict()` | `dict` | State for checkpointing |
| `load_state_dict(state)` | `None` | Restore from checkpoint |

---

#### `ElasticScalingManager`

Combined manager for both LR scaling and gradient accumulation.

```python
from elastic_harness.scaling import ElasticScalingManager, ScalingConfig, GradAccumulationConfig

manager = ElasticScalingManager(
    lr_config=ScalingConfig(base_lr=1e-4, base_batch_size=8, base_world_size=4),
    accum_config=GradAccumulationConfig(
        target_global_batch_size=1024,
        local_batch_size=32,
        base_world_size=4,
    ),
    optimizer=optimizer,
    strategy="constant_batch",  # or "variable_batch"
)

# On topology change
result = manager.on_topology_change(new_world_size=3)
print(f"New LR: {result['lr']}, Accumulation: {result['accumulation_steps']}")

# In training loop
if manager.should_step():
    optimizer.step()
    optimizer.zero_grad()

manager.step_lr()  # Progress warmup if active
```

**Strategies:**

| Strategy | LR Behavior | Batch Behavior |
|----------|-------------|----------------|
| `"variable_batch"` | Scales with batch size | Changes with world size |
| `"constant_batch"` | Stays stable | Accumulation adjusts to maintain target |

---

#### `WarmupScheduler`

Linear warmup scheduler.

```python
from elastic_harness.scaling import WarmupScheduler

scheduler = WarmupScheduler(
    start_lr=1e-6,
    target_lr=1e-4,
    warmup_steps=100,
)

lr = scheduler.get_lr(step=50)  # Returns interpolated LR
```

---

### Functions

#### `create_lr_scheduler_with_warmup`

```python
from elastic_harness.scaling import create_lr_scheduler_with_warmup

scheduler = create_lr_scheduler_with_warmup(
    optimizer=optimizer,
    warmup_steps=1000,
    total_steps=100000,
    min_lr=1e-6,
)
```

Creates a PyTorch LR scheduler with linear warmup and cosine decay.

---

## elastic_harness.metrics

Training metrics collection and reporting.

### Classes

#### `TrainingMetrics`

Main metrics aggregator.

```python
from elastic_harness.metrics import TrainingMetrics

metrics = TrainingMetrics()

# Record checkpoint operations
metrics.checkpoint.record_save("nvme", duration=1.5)
metrics.checkpoint.record_load("nvme", duration=0.8)

# Record recovery events
metrics.recovery.record_recovery(
    old_world_size=4,
    new_world_size=3,
    recovery_time=15.0,
    checkpoint_tier="nvme",
    step_resumed=1000,
)

# Record gradient stats
metrics.gradient.record_step(clipped=True, grad_norm=2.5)

# Update training progress
metrics.update_step(step=1000, tokens_processed=1024000)

# Get summary
print(metrics.summary())
metrics.log_summary()
```

**Nested Metrics:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `checkpoint` | `CheckpointMetrics` | Save/load timing |
| `recovery` | `RecoveryMetrics` | Topology change events |
| `gradient` | `GradientMetrics` | Gradient clipping stats |
| `lr` | `LRMetrics` | LR adjustment history |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `training_time` | `float` | Wall time in seconds |
| `tokens_per_second` | `float` | Throughput |
| `training_efficiency` | `float` | Fraction of time training (vs recovery) |

---

#### `CheckpointMetrics`

```python
metrics.checkpoint.record_save("nvme", 1.5)
metrics.checkpoint.record_load("s3", 45.0)
print(metrics.checkpoint.avg_save_time("nvme"))
print(metrics.checkpoint.max_save_time())
```

---

#### `RecoveryMetrics`

```python
metrics.recovery.record_recovery(
    old_world_size=4,
    new_world_size=3,
    recovery_time=15.0,
)

print(f"Total events: {len(metrics.recovery.events)}")
print(f"Worker joins: {metrics.recovery.worker_joins}")
print(f"Worker failures: {metrics.recovery.worker_failures}")
print(f"Avg recovery time: {metrics.recovery.avg_recovery_time}")
```

---

#### `RecoveryEvent`

Single recovery event record.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `timestamp` | `float` | Unix timestamp |
| `old_world_size` | `int` | Before change |
| `new_world_size` | `int` | After change |
| `recovery_time` | `float` | Seconds to recover |
| `checkpoint_tier` | `str\|None` | Tier loaded from |
| `step_resumed` | `int` | Step after recovery |

---

#### `GradientMetrics`

```python
metrics.gradient.record_step(clipped=True, grad_norm=2.5)
print(f"Clip frequency: {metrics.gradient.clip_frequency:.1%}")
print(f"Avg clip magnitude: {metrics.gradient.avg_clip_magnitude}")
```

---

#### `LRMetrics`

```python
metrics.lr.record_adjustment(old_lr=1e-4, new_lr=5e-5, reason="topology_change")
```

---

## elastic_harness.errors

Custom exceptions with actionable error messages.

### Exception Hierarchy

```
ElasticTrainingError (base)
├── ConfigurationError
├── DistributedSetupError
├── CheckpointError
│   ├── CheckpointLoadError
│   ├── CheckpointSaveError
│   └── CheckpointValidationError
├── TopologyChangeError
└── DataLoadingError
```

### Classes

#### `ElasticTrainingError`

Base exception for all elastic training errors.

---

#### `ConfigurationError`

```python
raise ConfigurationError(
    message="Invalid scaling strategy",
    config_key="scaling.strategy",
    suggestions=["Use 'variable_batch' or 'constant_batch'"],
)
```

**Attributes:**
- `config_key`: The problematic config key
- `suggestions`: List of suggestions

---

#### `DistributedSetupError`

```python
raise DistributedSetupError(
    message="Failed to initialize distributed",
    missing_vars=["RANK", "WORLD_SIZE"],
)
```

**Attributes:**
- `missing_vars`: List of missing environment variables
- `context`: Additional context dict

Has built-in `COMMON_ISSUES` dictionary with helpful messages for:
- `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`
- `NCCL` errors
- `timeout` errors

---

#### `CheckpointLoadError`

```python
raise CheckpointLoadError(
    message="File not found",
    path="/nvme/checkpoints/step_1000.pt",
    tier="NVME",
)
```

---

#### `CheckpointSaveError`

```python
raise CheckpointSaveError(
    message="Disk full",
    path="/nvme/checkpoints/step_1000.pt",
    tier="NVME",
    original_error=e,
)
```

---

#### `CheckpointValidationError`

```python
raise CheckpointValidationError(
    message="Invalid checkpoint structure",
    missing_keys=["model_state_dict"],
    invalid_keys=["step"],
)
```

---

#### `TopologyChangeError`

```python
raise TopologyChangeError(
    message="World size below min_nodes",
    old_world_size=4,
    new_world_size=0,
)
```

---

#### `DataLoadingError`

```python
raise DataLoadingError(
    message="File not found",
    file_path="data/train.jsonl",
)
```

---

## elastic_harness.config_validator

Configuration validation utilities.

### Classes

#### `ValidationResult`

```python
from elastic_harness.config_validator import ValidationResult

result = ValidationResult(
    valid=True,
    errors=[],
    warnings=["base_world_size outside range"],
)
```

**Attributes:**
- `valid`: `bool` - True if no errors
- `errors`: `list[str]` - Error messages
- `warnings`: `list[str]` - Warning messages

---

### Functions

#### `validate_elastic_config`

```python
from elastic_harness.config_validator import validate_elastic_config
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")
result = validate_elastic_config(OmegaConf.to_container(config))

if not result.valid:
    for error in result.errors:
        print(f"Error: {error}")
```

Validates:
- `min_nodes >= 1`
- `max_nodes >= min_nodes`
- `checkpoint_interval > 0`
- `memory_snapshot_interval > 0`
- Valid scaling strategy (`variable_batch` or `constant_batch`)
- `target_global_batch_size` required for `constant_batch`

---

#### `validate_s3_access`

```python
from elastic_harness.config_validator import validate_s3_access

result = validate_s3_access(bucket="my-bucket", region="us-east-1")
if not result.valid:
    print("S3 bucket not accessible")
```

Performs `head_bucket` call to verify bucket exists and is accessible.

---

#### `validate_config`

```python
from elastic_harness.config_validator import validate_config

result = validate_config(config_dict, validate_s3=True)
```

Comprehensive validation combining `validate_elastic_config` and optionally `validate_s3_access`.

---

## Usage Examples

### Basic Training Setup

```python
from elastic_harness.agent import setup_distributed_environment, cleanup_distributed
from elastic_harness.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointTier,
    create_checkpoint_state,
    load_checkpoint_to_model,
)
from elastic_harness.scaling import ElasticScalingManager, ScalingConfig, GradAccumulationConfig

# Initialize distributed
world_info = setup_distributed_environment(backend="nccl")
print(f"Rank {world_info.rank} of {world_info.world_size}")

# Create model and optimizer
model = MyModel().cuda()
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[world_info.local_rank])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Setup checkpoint manager
ckpt_config = CheckpointConfig(
    checkpoint_interval=500,
    memory_snapshot_interval=50,
    nvme_path="/nvme/checkpoints",
    s3_bucket="my-bucket",
)
ckpt_manager = CheckpointManager(ckpt_config)

# Setup scaling manager
scaling_manager = ElasticScalingManager(
    lr_config=ScalingConfig(base_lr=1e-4, base_batch_size=8, base_world_size=4),
    accum_config=GradAccumulationConfig(
        target_global_batch_size=256,
        local_batch_size=8,
        base_world_size=4,
    ),
    optimizer=optimizer,
    strategy="constant_batch",
)

# Try to resume from checkpoint
start_step = 0
checkpoint = ckpt_manager.load_checkpoint()
if checkpoint:
    load_checkpoint_to_model(checkpoint, model, optimizer)
    start_step = checkpoint.step + 1

    # Handle topology change
    if checkpoint.world_size != world_info.world_size:
        scaling_manager.on_topology_change(world_info.world_size)

# Training loop
for step in range(start_step, max_steps):
    # ... training code ...

    # Checkpoint
    if ckpt_manager.should_checkpoint(step, CheckpointTier.MEMORY):
        state = create_checkpoint_state(step, model, optimizer)
        ckpt_manager.save_checkpoint(state, tier=CheckpointTier.MEMORY)

    if ckpt_manager.should_checkpoint(step, CheckpointTier.NVME):
        state = create_checkpoint_state(step, model, optimizer)
        ckpt_manager.save_checkpoint(state, tier=CheckpointTier.NVME)
        ckpt_manager.save_checkpoint(state, tier=CheckpointTier.S3)
        ckpt_manager.clear_memory_snapshots()

# Cleanup
ckpt_manager.wait_for_pending_saves()
cleanup_distributed()
```

### Handling Topology Changes

```python
from elastic_harness.scaling import ElasticScalingManager

# When workers join or leave, PyTorch elastic will restart the process
# On restart, detect and handle the topology change:

checkpoint = ckpt_manager.load_checkpoint()
if checkpoint and checkpoint.world_size != world_info.world_size:
    result = scaling_manager.on_topology_change(world_info.world_size)
    print(f"Topology changed: {checkpoint.world_size} -> {world_info.world_size}")
    print(f"New LR: {result['lr']}")
    print(f"New accumulation steps: {result['accumulation_steps']}")
```

### Custom Checkpoint Loading

```python
from elastic_harness.checkpoint.storage_backends import CheckpointLoadConfig, NVMeBackend

# For trusted checkpoints (suppress security warnings)
config = CheckpointLoadConfig(trusted_source=True)
backend = NVMeBackend("/nvme/checkpoints")
state = backend.load("checkpoint.pt", config=config)

# For untrusted checkpoints (full validation)
config = CheckpointLoadConfig(
    safe_mode=True,
    validate_structure=True,
    warn_on_unsafe=True,
)
state = backend.load("checkpoint.pt", config=config)
```
