#!/bin/bash

# Deepiri Clean Rebuild Script
# Removes old images, rebuilds fresh, and starts services
# Usage: ./rebuild.sh [docker-compose-file]

set -e

COMPOSE_FILE="${1:-docker-compose.dev.yml}"

npm pkg set BUILD_TIMESTAMP=$(date +%s) &>/dev/null || true

export BUILD_TIMESTAMP=$(date +%s)

echo "🧹 Stopping containers and removing old images..."
docker compose -f "$COMPOSE_FILE" down --rmi all --volumes --remove-orphans

echo "🔨 Rebuilding containers (no cache)..."
docker compose -f "$COMPOSE_FILE" build --no-cache --pull

echo "🚀 Starting services..."
docker compose -f "$COMPOSE_FILE" up -d

echo "✅ Rebuild complete!"
echo ""
echo "View logs: docker compose -f $COMPOSE_FILE logs -f"
echo "Check status: docker compose -f $COMPOSE_FILE ps"

