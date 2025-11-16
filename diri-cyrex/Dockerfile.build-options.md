# Docker Build Options - Hybrid Approach

## Overview

The Dockerfiles support two build approaches:
1. **Prebuilt** (default) - Fast, uses prebuilt PyTorch images
2. **From-Scratch** - Customizable, downloads everything separately with resume capability

## Build Arguments

```bash
# Prebuilt (default - fast, recommended)
docker build \
  --build-arg BUILD_TYPE=prebuilt \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.0.0-cuda12.1-cudnn8-runtime \
  -t cyrex:latest .

# From-Scratch (customizable, slower, resume-capable)
docker build \
  --build-arg BUILD_TYPE=from-scratch \
  --build-arg PYTORCH_VERSION=2.0.0 \
  --build-arg CUDA_VERSION=12.1 \
  --build-arg PYTHON_VERSION=3.11 \
  -t cyrex:latest .
```

## Build Types

### 1. Prebuilt (Default)

**Pros:**
- ✅ Fastest build time
- ✅ PyTorch + CUDA already included
- ✅ No large downloads during build
- ✅ Most reliable

**Cons:**
- ❌ Larger base image
- ❌ Less customizable

**Usage:**
```bash
docker build --build-arg BUILD_TYPE=prebuilt -t cyrex:latest .
```

### 2. From-Scratch

**Pros:**
- ✅ Smaller final image (if optimized)
- ✅ Full control over versions
- ✅ Staged downloads (can resume)
- ✅ Better for custom PyTorch versions

**Cons:**
- ❌ Slower build (downloads everything)
- ❌ Network-dependent
- ❌ More complex

**Usage:**
```bash
docker build --build-arg BUILD_TYPE=from-scratch -t cyrex:latest .
```

## Staged Downloads (From-Scratch Only)

When using `BUILD_TYPE=from-scratch`, heavy packages are downloaded in separate stages:

### Stage 1: Download Torch/CUDA Packages
- `torch` (900MB+)
- `torchvision`
- `torchaudio`

**Resume Capability:**
- Each package downloads separately
- If one fails, others continue
- Can resume from last successful stage

### Stage 2: Install Torch
- Installs from downloaded packages (offline, fast)
- Falls back to PyPI if downloads failed

### Stage 3: Download ML Packages
- `transformers` (large)
- `datasets`
- `accelerate`
- `sentence-transformers`
- `mlflow` (heavy)
- `wandb`

**Resume Capability:**
- Each package downloads separately
- Failed downloads don't block others
- Can resume from last successful download

### Stage 4: Install All Packages
- Tries to install from downloaded packages first
- Falls back to PyPI if needed
- Uses `--upgrade-strategy=only-if-needed` to prevent unnecessary upgrades

## Benefits of Staged Downloads

1. **Resume Capability**: If download fails at stage 3.2, you can resume from stage 3.3
2. **Better Caching**: Each stage is cached separately
3. **Faster Retries**: Only retry failed stages, not entire build
4. **Network Resilience**: Large downloads split into manageable chunks
5. **Offline Installation**: Once downloaded, can install offline

## Example: Resuming Failed Build

```bash
# Build fails at transformers download
docker build --build-arg BUILD_TYPE=from-scratch -t cyrex:latest .

# Resume from last successful stage (Docker caches stages)
# Just run the same command - Docker will skip completed stages
docker build --build-arg BUILD_TYPE=from-scratch -t cyrex:latest .
```

## CPU vs GPU Builds

Both build types support CPU fallback:

```bash
# CPU build (prebuilt)
docker build \
  --build-arg BUILD_TYPE=prebuilt \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.0.0-cpu \
  -t cyrex:latest .

# CPU build (from-scratch)
docker build \
  --build-arg BUILD_TYPE=from-scratch \
  --build-arg PYTORCH_VERSION=2.0.0 \
  -t cyrex:latest .
```

## Recommended Usage

**For Most Users:**
```bash
# Use prebuilt (fastest, most reliable)
docker build --build-arg BUILD_TYPE=prebuilt -t cyrex:latest .
```

**For Custom Requirements:**
```bash
# Use from-scratch with specific versions
docker build \
  --build-arg BUILD_TYPE=from-scratch \
  --build-arg PYTORCH_VERSION=2.1.0 \
  --build-arg CUDA_VERSION=12.1 \
  -t cyrex:latest .
```

**For Network Issues:**
```bash
# Use from-scratch with staged downloads (can resume)
docker build --build-arg BUILD_TYPE=from-scratch -t cyrex:latest .
# If it fails, just run again - Docker will resume from last stage
```

