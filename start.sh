#!/usr/bin/env bash
set -e

# Default environment to 'dev' if not specified
ENV_TYPE="${ENV_TYPE:-dev}"
ENV_FILE=".env.${ENV_TYPE}"

if [ ! -f "$ENV_FILE" ]; then
    echo "[ERROR] Environment file $ENV_FILE does not exist!"
    exit 1
fi

echo "=========================================="
echo "Starting KR Stock Overnight Engine [$ENV_TYPE]"
echo "Loading config from: $ENV_FILE"
echo "=========================================="

# Export variables from .env.{type}
set -o allexport
source "$ENV_FILE"
set +o allexport

# Run with podman-compose or docker-compose
if command -v podman-compose &> /dev/null; then
    COMPOSE_CMD="podman-compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    echo "[ERROR] Neither podman-compose nor docker-compose was found!"
    exit 1
fi

$COMPOSE_CMD -f podman-compose.yml up -d --build
echo "[SUCCESS] Services started successfully in $ENV_TYPE mode."
