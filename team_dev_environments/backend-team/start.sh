#!/bin/bash
# Backend Team - Start script
# Services from SERVICE_COMMUNICATION_AND_TEAMS.md:
# - API Gateway (Port 5000), Auth Service (5001), Task Orchestrator (5002)
# - Engagement Service (5003), Analytics Service (5004), Notification Service (5005)
# - External Bridge Service (5006), Challenge Service (5007), Realtime Gateway (5008)
# - Infrastructure: mongodb, redis, influxdb

set -e

cd "$(dirname "$0")/../.." || exit 1

echo "🚀 Starting Backend Team services..."

# Start services that exist (skip submodules if not initialized)
# Based on SERVICE_COMMUNICATION_AND_TEAMS.md Backend Team section
SERVICES=()
for service in mongodb redis influxdb api-gateway auth-service task-orchestrator engagement-service platform-analytics-service notification-service external-bridge-service challenge-service realtime-gateway; do
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
    external-bridge-service)
      if [ -f "platform-services/backend/deepiri-external-bridge-service/Dockerfile" ]; then
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

echo "✅ Backend Team services started!"
echo ""
echo "🌐 API Gateway: http://localhost:5000"
echo "🔐 Auth Service: http://localhost:5001"
echo "📋 Task Orchestrator: http://localhost:5002"
echo "🎮 Engagement Service: http://localhost:5003"
echo "📈 Analytics Service: http://localhost:5004"
echo "🔔 Notification Service: http://localhost:5005"
echo "🌉 External Bridge: http://localhost:5006"
echo "🏆 Challenge Service: http://localhost:5007"
echo "⚡ Realtime Gateway: http://localhost:5008"

