# Cyrex Build Guide - GPU Detection & CPU Fallback

## Problem Solved
- **Before**: Builds froze at step 113/120 downloading 1.5GB+ CUDA packages, causing WiFi disconnections
- **After**: Automatic GPU detection with CPU fallback, using prebuilt PyTorch images

## Automatic GPU Detection

The build system automatically detects if you have a suitable GPU and chooses the appropriate base image:

- **GPU Detected (≥4GB VRAM)**: Uses `pytorch/pytorch:2.0.0-cuda12.1-cudnn8-runtime`
- **No GPU or Insufficient**: Uses `pytorch/pytorch:2.0.0-cpu` (much faster, no CUDA downloads)

## Quick Start

### Option 1: Use Build Scripts (Recommended)

**Linux/Mac:**
```bash
cd deepiri
./scripts/build-cyrex.sh
```

**Windows (PowerShell):**
```powershell
cd deepiri
.\scripts\build-cyrex.ps1
```

### Option 2: Manual Build with Detection

**Linux/Mac:**
```bash
cd deepiri/diri-cyrex
BASE_IMAGE=$(./detect_gpu.sh)
docker build --build-arg BASE_IMAGE="$BASE_IMAGE" -t deepiri-dev-cyrex:latest .
```

**Windows (PowerShell):**
```powershell
cd deepiri\diri-cyrex
$BaseImage = .\detect_gpu.ps1
docker build --build-arg BASE_IMAGE="$BaseImage" -t deepiri-dev-cyrex:latest .
```

### Option 3: Docker Compose (Auto-detects)

```bash
cd deepiri
# The build scripts set BASE_IMAGE automatically
./scripts/build-cyrex.sh

# Or manually set it:
BASE_IMAGE=$(./diri-cyrex/detect_gpu.sh) docker compose -f docker-compose.dev.yml build cyrex
```

## Force CPU Build

If you want to force CPU build (faster, no GPU downloads):

**Linux/Mac:**
```bash
BASE_IMAGE=pytorch/pytorch:2.0.0-cpu docker compose -f docker-compose.dev.yml build cyrex
```

**Windows (PowerShell):**
```powershell
$env:BASE_IMAGE = "pytorch/pytorch:2.0.0-cpu"
docker compose -f docker-compose.dev.yml build cyrex
```

## GPU Requirements

Default minimum requirements (configurable in `detect_gpu.sh`/`detect_gpu.ps1`):
- **Minimum GPU Memory**: 4GB VRAM
- **CUDA Support**: NVIDIA GPU with CUDA 11.8+

To adjust requirements, edit:
- `diri-cyrex/detect_gpu.sh` (Linux/Mac)
- `diri-cyrex/detect_gpu.ps1` (Windows)

## Build Optimizations

1. **Prebuilt Images**: PyTorch + CUDA already included (no 1.5GB downloads)
2. **Split Installation**: Packages installed in chunks to avoid timeouts
3. **Optional Packages**: GPU-specific packages (deepspeed, bitsandbytes) only install if CUDA available
4. **Network Timeouts**: 5-minute timeout with 3 retries
5. **Build Cache**: Better layer caching for faster rebuilds

## Troubleshooting

### Build Still Freezing?
1. Use CPU build: `BASE_IMAGE=pytorch/pytorch:2.0.0-cpu docker compose build cyrex`
2. Build during off-peak hours
3. Use wired connection instead of WiFi
4. Build on a machine with better network

### GPU Not Detected?
- Check `nvidia-smi` works: `nvidia-smi`
- Verify NVIDIA drivers are installed
- GPU may be below minimum requirements (check `detect_gpu.sh`)

### Want to Override Detection?
Set `BASE_IMAGE` environment variable:
```bash
export BASE_IMAGE=pytorch/pytorch:2.0.0-cpu  # Force CPU
export BASE_IMAGE=pytorch/pytorch:2.0.0-cuda12.1-cudnn8-runtime  # Force GPU
```

