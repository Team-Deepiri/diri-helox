#!/bin/bash
# Frontend Team - Start script
# Starts ONLY the services needed by the frontend using docker-compose.dev.yml with service selection

set -e

cd "$(dirname "$0")/../.." || exit 1

# Frontend team services
SERVICES=(
  postgres redis influxdb
  api-gateway auth-service task-orchestrator
  engagement-service platform-analytics-service
  notification-service challenge-service
  realtime-gateway frontend-dev
)

echo "🚀 Starting Frontend Team services..."
echo "   (Using docker-compose.dev.yml with service selection)"
echo "   Services: ${SERVICES[*]}"
echo ""

# Use --no-build to prevent automatic building (images should already be built)
# Dependencies will be started automatically
docker compose -f docker-compose.dev.yml up -d --no-build "${SERVICES[@]}"

# Get API Gateway port from environment or use default
API_GATEWAY_PORT=${API_GATEWAY_PORT:-5100}

echo "✅ Frontend Team services started!"
echo ""
echo "🎨 Frontend: http://localhost:5173"
echo "🌐 API Gateway: http://localhost:${API_GATEWAY_PORT}"
echo "🔐 Auth Service: http://localhost:5001"
echo "🗄️  pgAdmin: http://localhost:5050"
echo "🔍 Adminer: http://localhost:8080"

