# Troubleshooting Guide

This guide addresses common issues when setting up and running Deepiri services.

## Table of Contents

1. [Jupyter Service Permission Issues](#jupyter-service-permission-issues)
2. [Missing Node.js Dependencies](#missing-nodejs-dependencies)
3. [Missing Python Dependencies](#missing-python-dependencies)
4. [Module Resolution Issues](#module-resolution-issues)
5. [Docker Build Issues](#docker-build-issues)
6. [Service Connection Issues](#service-connection-issues)

---

## Jupyter Service Permission Issues

### Problem
```
Permission denied when creating directories in /home/appuser
```

### Solution

The Jupyter service now uses a dedicated Dockerfile (`Dockerfile.jupyter`) that properly handles permissions:

1. **Proper User Setup**: The Dockerfile creates the `appuser` with a home directory and proper permissions
2. **Runtime Directory**: Jupyter runtime files are stored in `/tmp/jupyter_runtime` (writable by all users)
3. **Configuration**: Jupyter is configured to use writable directories

### Verification

```bash
# Check if Jupyter container is running
docker ps | grep jupyter

# Check Jupyter logs
docker logs deepiri-jupyter-dev

# Access Jupyter at http://localhost:8888
```

### Manual Fix (if needed)

If you still encounter permission issues:

```bash
# Rebuild the Jupyter container
docker-compose -f docker-compose.dev.yml build --no-cache jupyter
docker-compose -f docker-compose.dev.yml up -d jupyter
```

---

## Missing Node.js Dependencies

### Problem
```
Cannot find module 'axios'
Cannot find module 'express'
```

### Solution

All required dependencies are listed in each service's `package.json`. To install:

#### Option 1: Use the Fix Script (Recommended)

```bash
# Run the dependency fix script
bash scripts/fix-dependencies.sh
```

#### Option 2: Manual Installation

```bash
# Install dependencies for a specific service
cd services/user-service
npm install

# Or install all services
for service in services/*; do
  if [ -f "$service/package.json" ]; then
    echo "Installing $service..."
    cd "$service" && npm install && cd ../..
  fi
done
```

#### Option 3: Docker Rebuild

```bash
# Rebuild all services (will install dependencies)
docker-compose -f docker-compose.dev.yml build --no-cache
```

### Common Missing Dependencies

| Service | Common Dependencies |
|---------|-------------------|
| user-service | axios, express, mongoose, winston |
| task-service | axios, express, mongoose, winston |
| gamification-service | axios, express, mongoose, redis, winston |
| analytics-service | axios, express, mongoose, @influxdata/influxdb-client |
| integration-service | axios, express, passport, winston |
| notification-service | express, socket.io, firebase-admin, winston |

### Verification

```bash
# Check if dependencies are installed
cd services/user-service
ls node_modules/axios

# Or check package.json
cat services/user-service/package.json | grep axios
```

---

## Missing Python Dependencies

### Problem
```
No module named 'numpy'
No module named 'torch'
```

### Solution

All Python dependencies are listed in `python_backend/requirements.txt`. To install:

#### Option 1: Use the Fix Script

```bash
bash scripts/fix-dependencies.sh
```

#### Option 2: Manual Installation

```bash
cd python_backend
pip install -r requirements.txt
```

#### Option 3: Docker Rebuild

```bash
docker-compose -f docker-compose.dev.yml build --no-cache pyagent
```

### Key Dependencies

- **Core**: fastapi, uvicorn, pydantic
- **AI/ML**: torch, transformers, numpy, pandas, scikit-learn
- **Embeddings**: sentence-transformers
- **Analytics**: influxdb-client
- **Optional**: pinecone-client, weaviate-client

### Verification

```bash
# Check if numpy is installed
docker exec deepiri-pyagent-dev python -c "import numpy; print(numpy.__version__)"

# Or check requirements
cat python_backend/requirements.txt | grep numpy
```

---

## Module Resolution Issues

### Problem
```
Cannot find module '../../utils/logger'
Error: Cannot find module '../utils/logger'
```

### Solution

Each service has its own logger utility in `services/{service}/utils/logger.js`.

### File Structure

```
services/
├── user-service/
│   ├── utils/
│   │   └── logger.js          ✅ Logger exists here
│   └── src/
│       └── oauthService.js    ✅ Imports: require('../../utils/logger')
│
├── task-service/
│   ├── utils/
│   │   └── logger.js          ✅ Logger exists here
│   └── src/
│       └── taskVersioningService.js  ✅ Imports: require('../../utils/logger')
│
└── integration-service/
    ├── utils/
    │   └── logger.js          ✅ Logger exists here
    └── src/
        └── webhookService.js  ✅ Imports: require('../../utils/logger')
```

### Verification

```bash
# Check if logger files exist
find services -name "logger.js" -type f

# Should output:
# services/user-service/utils/logger.js
# services/task-service/utils/logger.js
# services/gamification-service/utils/logger.js
# services/analytics-service/utils/logger.js
# services/notification-service/utils/logger.js
# services/integration-service/utils/logger.js
```

### If Logger is Missing

The logger utility should be automatically created. If it's missing:

1. Check that the `utils/` directory exists in the service
2. Verify the logger.js file exists
3. Rebuild the Docker container

```bash
# Rebuild a specific service
docker-compose -f docker-compose.dev.yml build --no-cache user-service
```

---

## Docker Build Issues

### Problem
```
npm ci can only install with an existing package-lock.json
```

### Solution

The Dockerfiles have been updated to use `npm install` instead of `npm ci` when `package-lock.json` is missing.

### Current Dockerfile Pattern

```dockerfile
COPY package*.json ./
RUN npm install --omit=dev --legacy-peer-deps && npm cache clean --force
```

### If Build Still Fails

1. **Clear Docker cache**:
   ```bash
   docker system prune -a
   ```

2. **Rebuild without cache**:
   ```bash
   docker-compose -f docker-compose.dev.yml build --no-cache
   ```

3. **Check Dockerfile syntax**:
   ```bash
   docker build -t test -f services/user-service/Dockerfile services/user-service
   ```

---

## Service Connection Issues

### Problem
```
Cannot connect to MongoDB
Cannot connect to Redis
Service not responding
```

### Solution

1. **Check service health**:
   ```bash
   # Check all services
   docker-compose -f docker-compose.dev.yml ps
   
   # Check specific service logs
   docker logs deepiri-user-service-dev
   ```

2. **Verify environment variables**:
   ```bash
   # Check .env file exists
   ls -la .env
   
   # Check environment in container
   docker exec deepiri-user-service-dev env | grep MONGO
   ```

3. **Check network connectivity**:
   ```bash
   # Test MongoDB connection
   docker exec deepiri-user-service-dev curl http://mongodb:27017
   
   # Test Redis connection
   docker exec deepiri-gamification-service-dev redis-cli -h redis ping
   ```

4. **Restart services**:
   ```bash
   docker-compose -f docker-compose.dev.yml restart
   ```

---

## Quick Diagnostic Commands

```bash
# Check all services status
docker-compose -f docker-compose.dev.yml ps

# View logs for all services
docker-compose -f docker-compose.dev.yml logs

# Check service health endpoints
curl http://localhost:5001/health  # user-service
curl http://localhost:5002/health  # task-service
curl http://localhost:5000/health  # api-gateway
curl http://localhost:8000/health  # pyagent

# Check dependencies
cd services/user-service && npm list --depth=0
cd python_backend && pip list | grep numpy

# Rebuild everything
docker-compose -f docker-compose.dev.yml down
docker-compose -f docker-compose.dev.yml build --no-cache
docker-compose -f docker-compose.dev.yml up -d
```

---

## Getting Help

If you encounter issues not covered here:

1. Check service logs: `docker logs <container-name>`
2. Review the [Architecture Documentation](SHARED_UTILS_ARCHITECTURE.md)
3. Check the [Start Guide](START_EVERYTHING.md)
4. Review service-specific README files in each service directory

---

## Prevention

To avoid these issues in the future:

1. **Always run the dependency fix script** after pulling changes:
   ```bash
   bash scripts/fix-dependencies.sh
   ```

2. **Use Docker Compose** for consistent environments:
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

3. **Keep dependencies updated**:
   ```bash
   # Update Node.js dependencies
   npm update --workspace=services/*
   
   # Update Python dependencies
   pip install --upgrade -r python_backend/requirements.txt
   ```

4. **Generate package-lock.json files** for reproducible builds:
   ```bash
   for service in services/*; do
     if [ -f "$service/package.json" ]; then
       cd "$service" && npm install && cd ../..
     fi
   done
   ```

