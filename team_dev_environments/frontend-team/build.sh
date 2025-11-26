#!/bin/bash
# Frontend Team - Build script
# Builds: frontend-dev + backend services (excluding api-gateway and external-bridge-service)

set -e

cd "$(dirname "$0")/../.." || exit 1

# Enable BuildKit for better builds
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

echo "🔨 Building Frontend Team services..."

# Build services that exist (skip submodules if not initialized)
SERVICES=()
for service in frontend-dev auth-service task-orchestrator engagement-service platform-analytics-service notification-service challenge-service realtime-gateway; do
  case $service in
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
      SERVICES+=("$service")
      ;;
  esac
done

if [ ${#SERVICES[@]} -eq 0 ]; then
  echo "❌ No services to build!"
  exit 1
fi

echo "Building: ${SERVICES[*]}"

# Use --no-deps to prevent building dependencies we don't need (like cyrex)
docker compose -f docker-compose.dev.yml build --no-deps "${SERVICES[@]}"

echo "✅ Frontend Team services built successfully!"

