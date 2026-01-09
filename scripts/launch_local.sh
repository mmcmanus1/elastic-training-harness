#!/bin/bash
# Local Elastic Training Launch Script (No etcd required)
#
# This script uses the c10d FileStore backend for local testing
# without requiring Docker or etcd. Perfect for:
# - Google Colab
# - Single machine development
# - Quick testing
#
# Usage:
#   ./launch_local.sh [options]
#
# Environment Variables:
#   NPROC           - Number of processes to spawn (default: 2)
#   RDZV_ID         - Unique job identifier (default: auto-generated)
#   CONFIG_FILE     - Training config path (default: configs/training_config.yaml)

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration with defaults
NPROC="${NPROC:-2}"
RDZV_BACKEND="c10d"
RDZV_ID="${RDZV_ID:-local-training-$(date +%s)}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/training_config.yaml}"

# Create temp directory for FileStore
FILESTORE_DIR=$(mktemp -d)
trap "rm -rf $FILESTORE_DIR" EXIT

# Print configuration
echo "=========================================="
echo "Local Elastic Training (c10d Backend)"
echo "=========================================="
echo "Processes:       ${NPROC}"
echo "Rendezvous:      ${RDZV_BACKEND} (FileStore)"
echo "Job ID:          ${RDZV_ID}"
echo "Config file:     ${CONFIG_FILE}"
echo "FileStore dir:   ${FILESTORE_DIR}"
echo "=========================================="
echo ""
echo "NOTE: This uses CPU processes to simulate distributed training."
echo "      No GPU or etcd required!"
echo ""

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

# Launch training with torchrun using c10d backend
exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NPROC}" \
    --rdzv-backend="${RDZV_BACKEND}" \
    --rdzv-id="${RDZV_ID}" \
    --rdzv-endpoint="localhost:0" \
    --max-restarts=3 \
    "${PROJECT_ROOT}/src/elastic_harness/train.py" \
    --config "${CONFIG_FILE}" \
    "${EXTRA_ARGS[@]}"
