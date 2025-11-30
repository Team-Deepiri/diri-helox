#!/bin/bash
# Frontend Team - Stop script
# Stops all frontend team services and their dependencies

set -e

cd "$(dirname "$0")/../.." || exit 1

echo "🛑 Stopping Frontend Team services..."

# Stop all services in the frontend-team compose file
# This stops all services defined in docker-compose.frontend-team.yml:
# - frontend-dev, api-gateway, auth-service
# - task-orchestrator, engagement-service, platform-analytics-service
# - notification-service, challenge-service
# - realtime-gateway
# - postgres, redis, influxdb, pgadmin
# Note: external-bridge-service excluded - frontend team doesn't need integrations
docker compose -f docker-compose.frontend-team.yml stop

echo "✅ Frontend Team services stopped!"

