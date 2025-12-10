#!/bin/bash
# Frontend Team - Build script
# Builds frontend team services using docker-compose.dev.yml with service selection

set -e

cd "$(dirname "$0")/../.." || exit 1

# Enable BuildKit for better builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Frontend team services
SERVICES=(
  postgres redis influxdb
  api-gateway auth-service task-orchestrator
  engagement-service platform-analytics-service
  notification-service challenge-service
  realtime-gateway frontend-dev
)

echo "🔨 Building Frontend Team services..."
echo "   (Using docker-compose.dev.yml with service selection)"
echo "   Services: ${SERVICES[*]}"
echo ""

# Build services using docker-compose.dev.yml
docker compose -f docker-compose.dev.yml build "${SERVICES[@]}"

echo "✅ Frontend Team services built successfully!"
echo "   Built services: ${VALID_SERVICES[*]}"
