#!/bin/bash
# Frontend Team - Start script
# Services from SERVICE_COMMUNICATION_AND_TEAMS.md:
# - Frontend Service (Port 5173)
# - Realtime Gateway (Port 5008)
# - Backend Services: auth-service, task-orchestrator, engagement-service, platform-analytics-service, notification-service, challenge-service
# - Infrastructure: mongodb, redis, influxdb (optional)

set -e

cd "$(dirname "$0")/../.." || exit 1

echo "🚀 Starting Frontend Team services..."

# Start services that exist (skip submodules if not initialized)
# Based on SERVICE_COMMUNICATION_AND_TEAMS.md Frontend Team section
# Note: mongodb, redis, influxdb are optional for direct DB access
SERVICES=()
for service in mongodb redis influxdb auth-service task-orchestrator engagement-service platform-analytics-service notification-service challenge-service realtime-gateway frontend-dev; do
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
  echo "❌ No services to start!"
  exit 1
fi

echo "Starting: ${SERVICES[*]}"

# Use --no-build to prevent automatic building (images should already be built)
docker compose -f docker-compose.dev.yml up -d --no-build "${SERVICES[@]}"

echo "✅ Frontend Team services started!"
echo ""
echo "🎨 Frontend: http://localhost:5173"
echo "⚡ Realtime Gateway: http://localhost:5008"

