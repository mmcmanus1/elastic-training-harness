#!/bin/bash
# Elastic Training Launch Script
#
# This script wraps torchrun with proper elastic training configuration
# using etcd-v2 as the rendezvous backend.
#
# Usage:
#   ./launch_training.sh [options] [-- additional torchrun args]
#
# Environment Variables:
#   NNODES          - Number of nodes (default: 1)
#   NPROC_PER_NODE  - Processes per node, typically GPUs (default: auto-detect)
#   RDZV_ENDPOINT   - etcd endpoint (default: localhost:2379)
#   RDZV_ID         - Unique job identifier (default: auto-generated)
#   MIN_NODES       - Minimum nodes to start training (default: 1)
#   MAX_NODES       - Maximum nodes allowed (default: 4)
#   CONFIG_FILE     - Training config path (default: configs/training_config.yaml)

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration with defaults
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
RDZV_BACKEND="etcd-v2"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:2379}"
RDZV_ID="${RDZV_ID:-elastic-training-$(date +%s)}"
MIN_NODES="${MIN_NODES:-1}"
MAX_NODES="${MAX_NODES:-4}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/training_config.yaml}"

# Heartbeat configuration (per design requirements)
# 30s interval, 90s timeout (3 missed heartbeats)
HEARTBEAT_TIMEOUT=90
LAST_CALL_TIMEOUT=30

# Auto-detect GPUs if not specified
if [[ -z "$NPROC_PER_NODE" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
        if [[ "$NPROC_PER_NODE" -eq 0 ]]; then
            NPROC_PER_NODE=1
        fi
    else
        NPROC_PER_NODE=1
    fi
    echo "Auto-detected $NPROC_PER_NODE GPU(s)"
fi

# Print configuration
echo "=========================================="
echo "Elastic Training Configuration"
echo "=========================================="
echo "Nodes:           ${MIN_NODES}:${MAX_NODES}"
echo "Procs per node:  ${NPROC_PER_NODE}"
echo "Rendezvous:      ${RDZV_BACKEND}://${RDZV_ENDPOINT}"
echo "Job ID:          ${RDZV_ID}"
echo "Config file:     ${CONFIG_FILE}"
echo "=========================================="

# Check if etcd is reachable
if ! curl -sf "http://${RDZV_ENDPOINT}/health" > /dev/null 2>&1; then
    echo "WARNING: etcd endpoint ${RDZV_ENDPOINT} may not be reachable"
    echo "Make sure etcd is running: docker compose -f docker/docker-compose.yml up -d"
fi

# Parse additional arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Launch training with torchrun
exec torchrun \
    --nnodes="${MIN_NODES}:${MAX_NODES}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --rdzv-backend="${RDZV_BACKEND}" \
    --rdzv-endpoint="${RDZV_ENDPOINT}" \
    --rdzv-id="${RDZV_ID}" \
    --rdzv-conf="timeout=${HEARTBEAT_TIMEOUT},last_call_timeout=${LAST_CALL_TIMEOUT}" \
    --max-restarts=100 \
    --monitor-interval=30 \
    "${PROJECT_ROOT}/src/elastic_harness/train.py" \
    --config "${CONFIG_FILE}" \
    "${EXTRA_ARGS[@]}"
