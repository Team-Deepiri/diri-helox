# Docker Compose Refactoring Analysis

## Current Structure

### Problems:
1. **13+ docker-compose files** - Hard to maintain, easy to get out of sync
2. **Generated files** - `generate-team-compose-files.py` creates team-specific files
3. **Duplication** - Same service definitions in multiple files
4. **Maintenance burden** - Changes require regenerating all team files
5. **CI/CD complexity** - Multiple files to manage

### Current Files:
- `docker-compose.dev.yml` ✅ (KEEP - single source of truth)
- `docker-compose.ai-team.yml` ❌ (REMOVE)
- `docker-compose.backend-team.yml` ❌ (REMOVE)
- `docker-compose.frontend-team.yml` ❌ (REMOVE)
- `docker-compose.ml-team.yml` ❌ (REMOVE)
- `docker-compose.infrastructure-team.yml` ❌ (REMOVE)
- `docker-compose.platform-engineers.yml` ❌ (REMOVE)
- `docker-compose.qa-team.yml` ❌ (REMOVE)
- `docker-compose.microservices.yml` ❌ (REMOVE)
- `docker-compose.enhanced.yml` ❌ (REMOVE - or keep as staging?)
- `docker-compose.yml` ❓ (Check if needed)

## Proposed Structure

### Keep Only:
1. **`docker-compose.dev.yml`** - Development environment (all services)
2. **`docker-compose.staging.yml`** - Staging environment (if needed)
3. **`docker-compose.prod.yml`** - Production environment (if needed)

### How It Works:
Docker Compose natively supports selecting services:
```bash
# Start only specific services
docker compose -f docker-compose.dev.yml up -d postgres redis api-gateway

# Build only specific services
docker compose -f docker-compose.dev.yml build api-gateway auth-service

# Stop specific services
docker compose -f docker-compose.dev.yml stop api-gateway
```

## Benefits

### ✅ Advantages:
1. **Single source of truth** - One file to maintain
2. **No generation needed** - No script to run
3. **Always in sync** - Changes to docker-compose.dev.yml immediately available
4. **Simpler CI/CD** - One file to reference
5. **Less disk space** - No duplicate YAML
6. **Easier debugging** - One file to check
7. **Native Docker feature** - No custom tooling needed

### ⚠️ Considerations:
1. **Service dependencies** - Docker Compose handles this automatically
2. **Team isolation** - Can use different networks/volumes if needed (via env vars)
3. **Port conflicts** - Already handled in docker-compose.dev.yml

## Migration Plan

### Step 1: Update Team Scripts
Change from:
```bash
docker compose -f docker-compose.backend-team.yml up -d
```

To:
```bash
docker compose -f docker-compose.dev.yml up -d \
  postgres redis influxdb \
  api-gateway auth-service task-orchestrator \
  engagement-service platform-analytics-service \
  notification-service external-bridge-service \
  challenge-service realtime-gateway
```

### Step 2: Service Lists
Define service lists in each team's script:
```bash
# backend-team/start.sh
SERVICES=(
  postgres redis influxdb
  api-gateway auth-service task-orchestrator
  engagement-service platform-analytics-service
  notification-service external-bridge-service
  challenge-service realtime-gateway
)

docker compose -f docker-compose.dev.yml up -d "${SERVICES[@]}"
```

### Step 3: Remove Generated Files
- Delete all `docker-compose.*-team.yml` files
- Delete `generate-team-compose-files.py` (or keep for reference)
- Update `.gitignore` if needed

## Team Service Mappings

### Backend Team:
```bash
postgres redis influxdb \
api-gateway auth-service task-orchestrator \
engagement-service platform-analytics-service \
notification-service external-bridge-service \
challenge-service realtime-gateway
```

### Frontend Team:
```bash
postgres redis influxdb \
api-gateway auth-service task-orchestrator \
engagement-service platform-analytics-service \
notification-service external-bridge-service \
challenge-service realtime-gateway frontend-dev
```

### AI Team:
```bash
postgres redis influxdb etcd minio milvus \
cyrex jupyter mlflow \
challenge-service external-bridge-service \
ollama
```

### ML Team:
```bash
postgres redis influxdb \
cyrex jupyter mlflow \
platform-analytics-service
```

### Infrastructure Team:
```bash
postgres redis influxdb etcd minio milvus \
pgadmin adminer
```

### Platform Engineers / QA Team:
```bash
# All services (no filter)
docker compose -f docker-compose.dev.yml up -d
```

## Implementation

### Example: backend-team/start.sh
```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Backend team services
SERVICES=(
  postgres redis influxdb
  api-gateway auth-service task-orchestrator
  engagement-service platform-analytics-service
  notification-service external-bridge-service
  challenge-service realtime-gateway
)

echo "🚀 Starting Backend Team services..."
docker compose -f docker-compose.dev.yml up -d "${SERVICES[@]}"
```

### Example: backend-team/build.sh
```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Backend team services (same list)
SERVICES=(
  postgres redis influxdb
  api-gateway auth-service task-orchestrator
  engagement-service platform-analytics-service
  notification-service external-bridge-service
  challenge-service realtime-gateway
)

echo "🔨 Building Backend Team services..."
docker compose -f docker-compose.dev.yml build "${SERVICES[@]}"
```

## Conclusion

**✅ YES - This is a MUCH better approach!**

- Simpler
- More maintainable
- Uses native Docker Compose features
- Single source of truth
- No generation scripts needed
- Easier for new team members

