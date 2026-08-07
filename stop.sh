#!/usr/bin/env bash
set -e

echo "=========================================="
echo "Stopping KR Stock Overnight Engine"
echo "=========================================="

if command -v podman-compose &> /dev/null; then
    COMPOSE_CMD="podman-compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "[ERROR] Neither podman-compose nor docker-compose was found!"
    exit 1
fi

$COMPOSE_CMD -f podman-compose.yml down
echo "[SUCCESS] Services stopped."
