**Try it in Google Colab:**

| Notebook | Description |
|----------|-------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/elastic-training-harness/blob/main/notebooks/elastic_training_colab.ipynb) | **Basic Demo** — ~1M param model, synthetic data, 100 steps. Quick intro to checkpointing, LR scaling, and fault tolerance. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/elastic-training-harness/blob/main/notebooks/elastic_training_large.ipynb) | **Large-Scale Practice** — ~100M param model, WikiText-2 dataset, 500 steps. Full training run with validation tracking and text generation. |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/elastic-training-harness/blob/main/notebooks/elastic_training_gpt2_finetune.ipynb) | **GPT-2 Fine-Tuning** — 124M param pretrained GPT-2, WikiText-2 dataset, 1000 steps. Full pipeline producing coherent text output. |

# Elastic Training Harness

A fault-tolerant distributed training harness for LLM training that automatically handles node failures, re-balances workloads, and resumes training with minimal recovery time.

## Features

- **Elastic Fault Tolerance**: Automatically detects worker failures and continues training with remaining workers
- **Dynamic Re-Sharding**: Token-based data sharding that handles world size changes without data duplication or skipping
- **Multi-Tier Checkpointing**: In-memory snapshots (instant), local NVMe (fast), and S3 (durable) storage
- **Learning Rate Scaling**: Automatic LR adjustment when effective batch size changes due to topology changes
- **Sub-30s Recovery Time**: Designed to achieve RTO < 30 seconds after node failures

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Cluster                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ Worker 0│  │ Worker 1│  │ Worker 2│  │ Worker 3│           │
│  │ (Rank 0)│  │ (Rank 1)│  │ (Rank 2)│  │ (Rank 3)│           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
│       │            │            │            │                  │
│       └────────────┴─────┬──────┴────────────┘                  │
│                          │                                       │
│                    ┌─────▼─────┐                                │
│                    │   etcd    │  (Rendezvous Backend)          │
│                    │  Server   │                                │
│                    └───────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for etcd)
- NVIDIA GPUs (optional, works with CPU)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd elastic-training-harness

# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Start etcd

```bash
# Start etcd server
./scripts/start_etcd.sh start

# Check status
./scripts/start_etcd.sh status
```

### Run Training

```bash
# Single node with multiple GPUs
./scripts/launch_training.sh

# Multiple nodes
NNODES=4 MIN_NODES=2 MAX_NODES=8 ./scripts/launch_training.sh
```

## Configuration

Training is configured via YAML files. See `configs/training_config.yaml` for the full configuration reference.

```yaml
model:
  vocab_size: 50257
  d_model: 512
  num_layers: 6

training:
  lr: 1.0e-4
  batch_size: 8
  max_steps: 100000

checkpoint:
  checkpoint_interval: 500
  memory_snapshot_interval: 50
  nvme_path: "/nvme/checkpoints"

elastic:
  min_nodes: 1
  max_nodes: 8
```

## Key Components

### Resumable Dataset

Token-based sharding that survives topology changes:

```python
from elastic_harness.data import ResumableDataset, TokenIndexFile

# Build token index (one-time)
# python -m elastic_harness.data.index_builder data/*.jsonl -o index.bin

index = TokenIndexFile("index.bin")
dataset = ResumableDataset(
    data_files=["train.jsonl"],
    index_file=index,
    tokenizer=tokenizer,
    seq_length=1024,
)

# State is saved/restored automatically
state = dataset.state_dict()
dataset.load_state_dict(state)
```

### Checkpoint Manager

Multi-tier checkpointing for optimal recovery:

```python
from elastic_harness.checkpoint import CheckpointManager, CheckpointConfig, CheckpointTier

config = CheckpointConfig(
    checkpoint_interval=500,
    memory_snapshot_interval=50,
    nvme_path="/nvme/checkpoints",
    s3_bucket="my-bucket",
)

manager = CheckpointManager(config)

# Save to different tiers
manager.save_checkpoint(state, tier=CheckpointTier.MEMORY)   # Instant
manager.save_checkpoint(state, tier=CheckpointTier.NVME)     # 2-5 seconds
manager.save_checkpoint(state, tier=CheckpointTier.S3)       # Minutes (async)

# Load with automatic fallback
state = manager.load_checkpoint()  # Tries memory -> NVMe -> S3
```

### LR Scaling

Automatic learning rate adjustment for topology changes:

```python
from elastic_harness.scaling import LRScalingManager, ScalingConfig, ScalingRule

config = ScalingConfig(
    base_lr=1e-4,
    base_batch_size=8,
    base_world_size=4,
    scaling_rule=ScalingRule.LINEAR,
    warmup_steps=100,
)

manager = LRScalingManager(config, optimizer)

# Called when workers are added/removed
new_lr = manager.on_topology_change(new_world_size=2)
```

## Chaos Testing

Validate fault tolerance with the chaos testing script:

```bash
# Kill random workers every 5 minutes for 1 hour
python scripts/chaos_test.py --interval 300 --duration 3600

# Single kill test
python scripts/chaos_test.py --single-kill

# Custom RTO threshold
python scripts/chaos_test.py --rto-threshold 30 --output results.json
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=elastic_harness

# Run only unit tests (skip integration)
pytest -m "not integration"
```

## Documentation

- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** — Common issues and solutions for distributed training, etcd, checkpoints, and more.
- **[API Reference](docs/API_REFERENCE.md)** — Complete reference for all public modules, classes, and functions.
- **[Design Document](DESIGN.md)** — Architecture decisions, trade-offs, and implementation details.

## Success Metrics

The harness is designed to meet these targets:

| Metric | Target |
|--------|--------|
| Recovery Time (RTO) | < 30 seconds |
| Throughput Retention | > 95% after topology change |
| Loss Curve Accuracy | Within 1% of baseline |

## Project Structure

```
elastic-training-harness/
├── src/elastic_harness/
│   ├── agent/           # Elastic agent and distributed setup
│   ├── checkpoint/      # Multi-tier checkpoint management
│   ├── data/            # Resumable dataset and index builder
│   ├── scaling/         # Learning rate scaling
│   └── train.py         # Main training script
├── configs/             # Configuration files
├── docker/              # Docker compose for etcd
├── scripts/             # Launch and chaos testing scripts
└── tests/               # Unit and integration tests
```

## References

- [PyTorch Elastic](https://pytorch.org/docs/stable/elastic/index.html)
- [TorchData StatefulDataLoader](https://pytorch.org/data/)
- [Linear Scaling Rule](https://arxiv.org/abs/1706.02677)

## License

MIT
