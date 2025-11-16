# Docker Build Optimization Guide

## Problem
Building Docker images was freezing at step 113/120 due to downloading large CUDA packages (900MB+ PyTorch, 594MB NVIDIA libraries), overwhelming the network and causing WiFi disconnections.

## Solution
We've optimized the Dockerfiles to use prebuilt PyTorch images and better network handling.

## Changes Made

### 1. Use Prebuilt PyTorch Images
- **Before**: Installing PyTorch + CUDA from scratch (1.5GB+ downloads)
- **After**: Using `pytorch/pytorch:2.0.0-cuda12.1-cudnn8-runtime` (pre-downloaded, cached)

### 2. Split Package Installation
- Install packages in smaller chunks
- Optional packages can fail without breaking the build
- Better error handling and retries

### 3. Network Optimizations
- Added `PIP_DEFAULT_TIMEOUT=300` (5 minutes)
- Added `PIP_RETRIES=3` for automatic retries
- Packages installed with `--timeout=300 --retries=2/3`

### 4. CPU-Only Option
- Created `Dockerfile.cpu` for development (no CUDA downloads)
- Much faster builds for local development

## Usage

### GPU Build (Production)
```bash
docker compose -f docker-compose.dev.yml build cyrex
```

### CPU-Only Build (Development - Faster)
```bash
# Edit docker-compose.dev.yml to use Dockerfile.cpu
# Or build directly:
docker build -f diri-cyrex/Dockerfile.cpu -t deepiri-dev-cyrex:latest ./diri-cyrex
```

### Build with BuildKit (Recommended)
```bash
DOCKER_BUILDKIT=1 docker compose -f docker-compose.dev.yml build cyrex
```

## Additional Tips

1. **Use Build Cache**: The prebuilt images are cached, so subsequent builds are much faster
2. **Build in Stages**: Build cyrex separately first, then other services
3. **Network Issues**: If still having issues, try:
   - Build during off-peak hours
   - Use a wired connection
   - Build on a machine with better network

## Optional Packages

These packages are marked as optional and can fail without breaking the build:
- `deepspeed` (very large, for distributed training)
- `bitsandbytes` (GPU quantization, large)
- `mlflow`, `wandb` (MLOps tools)
- Vector DB clients (pinecone, weaviate)

If you need these, install them separately after the initial build.

