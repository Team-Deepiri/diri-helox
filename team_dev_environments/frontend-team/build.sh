#!/bin/bash
# Frontend Team - Build script
# Requirements: frontend-dev + all platform-services needed by api-gateway
# Dependencies: mongodb, influxdb, mongo-express (pulled as images, not built)

set -e

cd "$(dirname "$0")/../.." || exit 1

# Enable BuildKit for better builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

echo "🔨 Building Frontend Team services..."

# Build all platform-services that frontend needs (api-gateway depends on all of these)
# Services to build: frontend-dev, api-gateway, auth-service, task-orchestrator,
# engagement-service, platform-analytics-service, notification-service,
# challenge-service, realtime-gateway
# Note: external-bridge-service excluded - frontend team doesn't need integrations
SERVICES=()
for service in frontend-dev api-gateway auth-service task-orchestrator engagement-service platform-analytics-service notification-service challenge-service realtime-gateway; do
  case $service in
    api-gateway)
      if [ -f "platform-services/backend/deepiri-api-gateway/Dockerfile" ]; then
        SERVICES+=("$service")
      else
        echo "⚠️  Skipping $service (submodule not initialized)"
      fi
      ;;
    auth-service)
      if [ -f "platform-services/backend/deepiri-auth-service/Dockerfile" ]; then
        SERVICES+=("$service")
      else
        echo "⚠️  Skipping $service (submodule not initialized)"
      fi
      ;;
    frontend-dev)
      if [ -f "deepiri-web-frontend/Dockerfile.dev" ]; then
        SERVICES+=("$service")
      else
        echo "⚠️  Skipping $service (submodule not initialized)"
      fi
      ;;
    *)
      # For services without specific Dockerfile checks (task-orchestrator, engagement-service, etc.)
      SERVICES+=("$service")
      ;;
  esac
done

if [ ${#SERVICES[@]} -eq 0 ]; then
  echo "❌ No services to build!"
  exit 1
fi

echo "Building: ${SERVICES[*]} (and their dependencies)"

# Build services with their dependencies (mongodb, influxdb, mongo-express will be pulled as images, not built)
docker compose -f docker-compose.frontend-team.yml build "${SERVICES[@]}"

echo "✅ Frontend Team services built successfully!"

