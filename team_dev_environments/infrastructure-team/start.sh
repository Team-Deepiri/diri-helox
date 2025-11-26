#!/bin/bash
# Infrastructure Team - Start script
# Services from SERVICE_COMMUNICATION_AND_TEAMS.md:
# - All Infrastructure Services (MongoDB, Redis, InfluxDB, Mongo Express)
# - API Gateway and All Microservices for monitoring

set -e

cd "$(dirname "$0")/../.." || exit 1

echo "🚀 Starting Infrastructure Team services..."
echo "Services: All infrastructure + all microservices"

# Use --no-build to prevent automatic building (images should already be built)
docker compose -f docker-compose.dev.yml up -d --no-build

echo "✅ Infrastructure Team services started!"
echo ""
echo "🗄️  MongoDB: localhost:27017"
echo "🗄️  Mongo Express: http://localhost:8081"
echo "💾 Redis: localhost:6380"
echo "📊 InfluxDB: http://localhost:8086"
echo "🌐 API Gateway: http://localhost:5000"

