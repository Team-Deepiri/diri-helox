#!/bin/bash
# AI Team - Start script
# Services from SERVICE_COMMUNICATION_AND_TEAMS.md:
# - Cyrex AI Service (Port 8000), Jupyter, MLflow (Port 5500), Challenge Service (Port 5007)
# - Infrastructure: mongodb, influxdb, redis (optional), etcd, minio, milvus

set -e

cd "$(dirname "$0")/../.." || exit 1

echo "🚀 Starting AI Team services..."

# Start services that exist (skip submodules if not initialized)
# Based on SERVICE_COMMUNICATION_AND_TEAMS.md AI Team section
SERVICES=()
for service in mongodb influxdb redis etcd minio milvus cyrex cyrex-interface jupyter mlflow challenge-service; do
  case $service in
    cyrex|jupyter)
      if [ -f "diri-cyrex/Dockerfile" ] || [ -f "diri-cyrex/Dockerfile.jupyter" ]; then
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

echo "✅ AI Team services started!"
echo ""
echo "📊 MLflow: http://localhost:5500"
echo "📓 Jupyter: http://localhost:8888"
echo "🤖 Cyrex: http://localhost:8000"
echo "🖥️  Cyrex Interface: http://localhost:5175"
echo "🏆 Challenge Service: http://localhost:5007"

