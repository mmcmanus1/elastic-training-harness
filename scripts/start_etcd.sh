#!/bin/bash
# Start etcd server for elastic training rendezvous
#
# This script starts the etcd Docker container with the v2 API enabled,
# which is required for PyTorch elastic training.
#
# Usage:
#   ./start_etcd.sh [start|stop|restart|status|logs]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.yml"

usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start   - Start etcd container"
    echo "  stop    - Stop etcd container"
    echo "  restart - Restart etcd container"
    echo "  status  - Show container status"
    echo "  logs    - Show container logs"
    echo "  clean   - Stop and remove all data"
    echo ""
    echo "If no command is provided, 'start' is assumed."
}

start_etcd() {
    echo "Starting etcd..."
    docker compose -f "$COMPOSE_FILE" up -d

    # Wait for etcd to be healthy
    echo "Waiting for etcd to be ready..."
    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if curl -sf "http://localhost:2379/health" > /dev/null 2>&1; then
            echo "etcd is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    echo "ERROR: etcd failed to start within ${max_attempts} seconds"
    return 1
}

stop_etcd() {
    echo "Stopping etcd..."
    docker compose -f "$COMPOSE_FILE" down
}

restart_etcd() {
    stop_etcd
    start_etcd
}

status_etcd() {
    docker compose -f "$COMPOSE_FILE" ps
    echo ""
    echo "Health check:"
    curl -sf "http://localhost:2379/health" && echo " - etcd is healthy" || echo " - etcd is not responding"
}

logs_etcd() {
    docker compose -f "$COMPOSE_FILE" logs -f
}

clean_etcd() {
    echo "Stopping and removing etcd data..."
    docker compose -f "$COMPOSE_FILE" down -v
}

# Main
case "${1:-start}" in
    start)
        start_etcd
        ;;
    stop)
        stop_etcd
        ;;
    restart)
        restart_etcd
        ;;
    status)
        status_etcd
        ;;
    logs)
        logs_etcd
        ;;
    clean)
        clean_etcd
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac
