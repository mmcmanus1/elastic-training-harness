# Troubleshooting Guide

This guide covers common issues when running elastic distributed training and how to resolve them.

## Quick Diagnostic Commands

```bash
# Check GPU availability
nvidia-smi

# Check etcd status
docker ps | grep etcd
curl -s http://localhost:2379/health

# Check NCCL connectivity (debug mode)
NCCL_DEBUG=INFO torchrun --nproc-per-node=2 your_script.py

# Verify AWS credentials for S3
aws s3 ls s3://your-bucket/

# Check disk space for NVMe checkpoints
df -h /nvme/checkpoints

# Check for port conflicts
lsof -i :29500

# View environment variables set by torchrun
env | grep -E "RANK|WORLD_SIZE|MASTER"
```

---

## 1. Distributed Setup Errors

### Missing Environment Variables

**Symptoms:**
```
RuntimeError: Missing required environment variables: ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']
```

**Cause:**
Running the training script directly with `python` instead of `torchrun`.

**Solution:**
Use `torchrun` to launch training:
```bash
# Wrong
python train.py --config config.yaml

# Correct
torchrun --nproc-per-node=4 train.py --config config.yaml
```

For multi-node training:
```bash
torchrun \
    --nnodes=2 \
    --nproc-per-node=4 \
    --rdzv-backend=etcd-v2 \
    --rdzv-endpoint=etcd-host:2379 \
    --rdzv-id=my-training-job \
    train.py --config config.yaml
```

### NCCL Initialization Failure

**Symptoms:**
```
RuntimeError: NCCL error in: ... unhandled system error
```
or
```
NCCL WARN Connect to ... failed : No route to host
```

**Cause:**
- GPU not available or driver issues
- Network connectivity problems between nodes
- NCCL version mismatch across nodes
- Firewall blocking communication

**Solution:**

1. Verify GPU availability:
   ```bash
   nvidia-smi
   ```

2. Enable NCCL debug logging:
   ```bash
   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL torchrun ...
   ```

3. Check network interface (useful in multi-NIC setups):
   ```bash
   # Force specific interface
   NCCL_SOCKET_IFNAME=eth0 torchrun ...
   ```

4. Verify nodes can communicate:
   ```bash
   # From worker node, ping master
   ping $MASTER_ADDR
   nc -zv $MASTER_ADDR $MASTER_PORT
   ```

5. Check firewall rules allow the required ports (default: 29500).

### Distributed Initialization Timeout

**Symptoms:**
```
RuntimeError: Distributed initialization timed out
```
or process hangs indefinitely at startup.

**Cause:**
- Workers cannot reach each other
- Firewall blocking MASTER_PORT
- Incorrect MASTER_ADDR
- Network latency too high

**Solution:**

1. Verify MASTER_ADDR is correct (should be the IP of rank 0 node):
   ```bash
   # On rank 0 node
   hostname -I
   ```

2. Increase timeout:
   ```bash
   NCCL_SOCKET_TIMEOUT=600 torchrun ...
   ```

3. For single-node testing, ensure MASTER_ADDR=localhost:
   ```bash
   MASTER_ADDR=localhost MASTER_PORT=29500 torchrun --nproc-per-node=2 ...
   ```

### Port Already in Use

**Symptoms:**
```
RuntimeError: Address already in use
```

**Cause:**
Previous training job didn't clean up, or another process is using the port.

**Solution:**

1. Find and kill the process:
   ```bash
   lsof -i :29500
   kill -9 <PID>
   ```

2. Use a different port:
   ```bash
   torchrun --master-port=29501 ...
   ```

---

## 2. etcd and Rendezvous Issues

### etcd Connection Refused

**Symptoms:**
```
ConnectionRefusedError: [Errno 111] Connection refused
```
or
```
Failed to connect to etcd at localhost:2379
```

**Cause:**
etcd server is not running or not accessible.

**Solution:**

1. Start etcd using the provided script:
   ```bash
   ./scripts/start_etcd.sh start
   ```

2. Or start with Docker directly:
   ```bash
   docker run -d --name etcd \
       -p 2379:2379 -p 2380:2380 \
       quay.io/coreos/etcd:v3.5.0 \
       /usr/local/bin/etcd \
       --listen-client-urls http://0.0.0.0:2379 \
       --advertise-client-urls http://localhost:2379
   ```

3. Verify etcd is healthy:
   ```bash
   curl http://localhost:2379/health
   # Should return {"health":"true"}
   ```

### Rendezvous Timeout

**Symptoms:**
```
RendezvousTimeoutError: Timed out waiting for rendezvous
```

**Cause:**
- Not all expected workers joined within timeout
- Workers using different rdzv_id values
- Network issues preventing worker discovery

**Solution:**

1. Ensure all workers use the same `--rdzv-id`:
   ```bash
   # All workers must use identical rdzv-id
   torchrun --rdzv-id=my-job-123 ...
   ```

2. Increase rendezvous timeout:
   ```bash
   torchrun --rdzv-conf=timeout=600 ...
   ```

3. Verify the expected number of nodes matches actual workers:
   ```bash
   # If expecting 4 nodes but only 2 are up
   torchrun --nnodes=2:4 ...  # min 2, max 4
   ```

### Stale Rendezvous State

**Symptoms:**
```
RendezvousStateError: Stale state detected
```
or workers from a previous run interfering with current run.

**Cause:**
Previous training job's rendezvous state wasn't cleaned up in etcd.

**Solution:**

1. Use a unique rdzv_id for each run:
   ```bash
   torchrun --rdzv-id="training-$(date +%s)" ...
   ```

2. Manually clean up etcd:
   ```bash
   # List keys
   docker exec etcd etcdctl get --prefix /torch/elastic

   # Delete stale keys
   docker exec etcd etcdctl del --prefix /torch/elastic/my-old-job
   ```

3. Restart etcd (nuclear option):
   ```bash
   ./scripts/start_etcd.sh stop
   ./scripts/start_etcd.sh start
   ```

---

## 3. Checkpoint Errors

### NVMe Checkpoint Save Failed

**Symptoms:**
```
CheckpointSaveError: [NVME] Failed to save checkpoint to '/nvme/checkpoints/...': No space left on device
```

**Cause:**
- Disk full
- Write permissions missing
- Directory doesn't exist

**Solution:**

1. Check disk space:
   ```bash
   df -h /nvme/checkpoints
   ```

2. Clean up old checkpoints:
   ```bash
   # Keep only last 3 checkpoints
   ls -t /nvme/checkpoints/*.pt | tail -n +4 | xargs rm -f
   ```

3. Check permissions:
   ```bash
   ls -la /nvme/checkpoints
   # Ensure write permission for your user
   ```

4. Create directory if missing:
   ```bash
   mkdir -p /nvme/checkpoints
   ```

### S3 Checkpoint Errors

**Symptoms:**
```
CheckpointSaveError: [S3] Failed to save checkpoint: NoCredentialsError
```
or
```
botocore.exceptions.ClientError: An error occurred (403) when calling PutObject
```

**Cause:**
- AWS credentials not configured
- Missing IAM permissions
- Bucket doesn't exist

**Solution:**

1. Configure AWS credentials:
   ```bash
   # Option 1: Environment variables
   export AWS_ACCESS_KEY_ID=your-key
   export AWS_SECRET_ACCESS_KEY=your-secret

   # Option 2: AWS CLI configuration
   aws configure

   # Option 3: IAM role (on EC2/EKS)
   # Ensure instance has appropriate IAM role attached
   ```

2. Verify bucket access:
   ```bash
   aws s3 ls s3://your-bucket/
   ```

3. Check required IAM permissions:
   ```json
   {
     "Effect": "Allow",
     "Action": [
       "s3:GetObject",
       "s3:PutObject",
       "s3:DeleteObject",
       "s3:ListBucket"
     ],
     "Resource": [
       "arn:aws:s3:::your-bucket",
       "arn:aws:s3:::your-bucket/*"
     ]
   }
   ```

4. Create bucket if it doesn't exist:
   ```bash
   aws s3 mb s3://your-bucket --region us-east-1
   ```

### Checkpoint Load Failed

**Symptoms:**
```
CheckpointLoadError: Failed to load checkpoint from '...': Missing required keys: {'model_state_dict'}
```
or
```
RuntimeError: Error(s) in loading state_dict for Model
```

**Cause:**
- Corrupted checkpoint file
- Model architecture changed since checkpoint was saved
- Version mismatch

**Solution:**

1. Try loading with `strict=False`:
   ```python
   from elastic_harness.checkpoint import load_checkpoint_to_model

   load_checkpoint_to_model(state, model, optimizer, strict=False)
   ```

2. For trusted checkpoints, use `trusted_source=True`:
   ```python
   from elastic_harness.checkpoint.storage_backends import CheckpointLoadConfig

   config = CheckpointLoadConfig(trusted_source=True)
   state = backend.load("checkpoint.pt", config=config)
   ```

3. Fall back to an earlier checkpoint tier:
   ```python
   # Skip memory, try NVMe then S3
   state = manager.load_checkpoint(tier=CheckpointTier.NVME)
   ```

4. If checkpoint is corrupted, delete and train from earlier checkpoint:
   ```bash
   rm /nvme/checkpoints/checkpoint_step_00010000.pt
   # Training will resume from next available checkpoint
   ```

### Checkpoint Security Warning

**Symptoms:**
```
SECURITY: Loading checkpoint with weights_only=False enables arbitrary code execution...
```

**Cause:**
The checkpoint contains complex objects (like optimizer state) that require full deserialization.

**Solution:**

For checkpoints from trusted sources (your own training runs):
```python
from elastic_harness.checkpoint.storage_backends import CheckpointLoadConfig

config = CheckpointLoadConfig(trusted_source=True)
state = backend.load("checkpoint.pt", config=config)
```

For untrusted checkpoints, this warning is intentional - exercise caution.

---

## 4. Memory and OOM Issues

### Memory Snapshot Exceeds Limit

**Symptoms:**
```
MemoryLimitExceeded: Cannot save snapshot: 2048.0MB required, but only 512.0MB available
```

**Cause:**
Model is too large for configured memory snapshot limit.

**Solution:**

1. Increase memory limit in config:
   ```python
   backend = MemorySnapshotBackend(
       max_snapshots=2,
       max_memory_mb=4096  # Increase limit
   )
   ```

2. Reduce number of snapshots kept:
   ```python
   backend = MemorySnapshotBackend(max_snapshots=1)
   ```

3. Disable memory snapshots entirely (rely on NVMe):
   ```yaml
   checkpoint:
     memory_snapshot_interval: 0  # Disable
     checkpoint_interval: 100      # More frequent NVMe saves
   ```

### CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Cause:**
- Batch size too large
- Model too large for GPU memory
- Memory fragmentation

**Solution:**

1. Reduce batch size:
   ```yaml
   training:
     batch_size: 4  # Reduce from 8
   ```

2. Use gradient accumulation to maintain effective batch size:
   ```yaml
   scaling:
     strategy: constant_batch
     target_global_batch_size: 64
   # With batch_size=4 and 4 GPUs: 64/(4*4) = 4 accumulation steps
   ```

3. Enable gradient checkpointing (if model supports it):
   ```python
   model.gradient_checkpointing_enable()
   ```

4. Clear CUDA cache periodically:
   ```python
   torch.cuda.empty_cache()
   ```

---

## 5. Topology Change Errors

### World Size Below Minimum

**Symptoms:**
```
TopologyChangeError: Topology change (4 -> 1 workers) failed: World size below min_nodes
```

**Cause:**
Too many workers failed, dropping below the configured minimum.

**Solution:**

1. Lower min_nodes if acceptable:
   ```yaml
   elastic:
     min_nodes: 1  # Allow single-node operation
     max_nodes: 8
   ```

2. Add more workers or restart failed workers.

3. Check why workers are failing (see other sections).

### Recovery Checkpoint Not Available

**Symptoms:**
```
TopologyChangeError: No checkpoint available for recovery
```

**Cause:**
All checkpoint tiers are empty when trying to recover.

**Solution:**

1. Ensure checkpointing is enabled:
   ```yaml
   checkpoint:
     checkpoint_interval: 500
     memory_snapshot_interval: 50
   ```

2. For new training, this is expected - training will start from scratch.

3. Check if S3 checkpoints exist:
   ```bash
   aws s3 ls s3://your-bucket/checkpoints/
   ```

---

## 6. Configuration Errors

### Invalid Configuration

**Symptoms:**
```
ConfigurationError: Configuration error in 'scaling': Unknown scaling strategy 'invalid'. Valid options: ['variable_batch', 'constant_batch']
```

**Cause:**
Configuration YAML has invalid values.

**Solution:**

Use the config validator before training:
```python
from elastic_harness.config_validator import validate_config
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")
result = validate_config(OmegaConf.to_container(config))

if not result.valid:
    for error in result.errors:
        print(f"Error: {error}")
    sys.exit(1)

for warning in result.warnings:
    print(f"Warning: {warning}")
```

### Common Configuration Mistakes

| Mistake | Error | Fix |
|---------|-------|-----|
| `min_nodes: 0` | min_nodes must be >= 1 | Set `min_nodes: 1` |
| `max_nodes < min_nodes` | min_nodes cannot exceed max_nodes | Ensure `max_nodes >= min_nodes` |
| `checkpoint_interval: 0` | must be positive | Set positive value like `500` |
| Missing `target_global_batch_size` with `constant_batch` | required for constant_batch strategy | Add the field or use `variable_batch` |

### YAML Syntax Errors

**Symptoms:**
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Common YAML Issues:**

```yaml
# Wrong - missing space after colon
training:
  lr:1e-4

# Correct
training:
  lr: 1e-4

# Wrong - tabs instead of spaces
training:
	lr: 1e-4

# Correct - use spaces
training:
  lr: 1e-4

# Wrong - unquoted special characters
rdzv_id: my-job:123

# Correct - quote strings with special chars
rdzv_id: "my-job:123"
```

---

## 7. Data Loading Errors

### Token Index Not Found

**Symptoms:**
```
DataLoadingError: Failed to load data from 'data/train.jsonl': Token index file not found
```

**Cause:**
The pre-computed token index file hasn't been built.

**Solution:**

Build the token index before training:
```bash
python -m elastic_harness.data.index_builder \
    --data-files data/train.jsonl \
    --tokenizer gpt2 \
    --output data/train.index
```

### Data Format Mismatch

**Symptoms:**
```
DataLoadingError: KeyError: 'text'
```

**Cause:**
Data file uses different key name than expected.

**Solution:**

Specify the correct text key:
```yaml
data:
  text_key: "content"  # If your JSONL uses {"content": "..."}
```

Or preprocess your data to use the expected format:
```python
# Expected format
{"text": "Your training text here"}
```

---

## 8. Getting Help

If you're still stuck:

1. **Check logs with debug level:**
   ```bash
   LOGLEVEL=DEBUG torchrun ...
   ```

2. **Enable NCCL debug output:**
   ```bash
   NCCL_DEBUG=INFO NCCL_DEBUG_FILE=/tmp/nccl.log torchrun ...
   ```

3. **Check PyTorch distributed debug:**
   ```bash
   TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun ...
   ```

4. **File an issue** with:
   - Full error message and stack trace
   - Configuration file (sanitize secrets)
   - Environment details (PyTorch version, CUDA version, OS)
   - Steps to reproduce
