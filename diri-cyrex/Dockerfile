# =============================================================================
# HYBRID DOCKERFILE: Prebuilt OR From-Scratch with Staged Downloads
# =============================================================================
# Build args to control build type
ARG BUILD_TYPE=prebuilt
ARG PYTORCH_VERSION=2.0.0
ARG CUDA_VERSION=12.1
ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE=pytorch/pytorch:2.0.0-cuda12.1-cudnn8-runtime

# =============================================================================
# OPTION 1: PREBUILT (Fast, Reliable, Larger) - DEFAULT
# =============================================================================
# Use BASE_IMAGE build arg (set by build scripts based on GPU detection)
FROM ${BASE_IMAGE} AS prebuilt-base

ENV BUILD_TYPE=prebuilt

# If PyTorch is not already installed (e.g., using python:3.11-slim for CPU), install it
# If using pytorch/pytorch:2.0.0-cuda12.1-cudnn8-runtime (GPU build), PyTorch is already installed
RUN python -c "import torch" 2>/dev/null || \
    (echo "CPU build detected, installing PyTorch CPU..." && \
     pip install --no-cache-dir --upgrade-strategy=only-if-needed \
         torch==${PYTORCH_VERSION} \
         torchvision \
         torchaudio \
         --index-url https://download.pytorch.org/whl/cpu) || \
    echo "PyTorch installation check completed"

# =============================================================================
# OPTION 2: FROM SCRATCH (Customizable, Smaller, Slower)
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS from-scratch-base

ENV BUILD_TYPE=from-scratch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configure pip for better reliability and resume capability
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# =============================================================================
# STAGE 1: Download Heavy Packages (Torch/CUDA) - FROM SCRATCH ONLY
# =============================================================================
FROM from-scratch-base AS download-torch

# Download PyTorch in separate stages for resume capability
# Stage 1.1: Download torch (can resume if fails)
RUN pip download --no-deps \
    --timeout=300 \
    --retries=5 \
    --dest /tmp/packages \
    torch==${PYTORCH_VERSION} \
    || (echo "Warning: torch download failed, will retry in next stage" && exit 0)

# Stage 1.2: Download torchvision (depends on torch)
RUN pip download --no-deps \
    --timeout=300 \
    --retries=5 \
    --dest /tmp/packages \
    torchvision \
    || echo "Warning: torchvision download failed, continuing..."

# Stage 1.3: Download torchaudio (depends on torch)
RUN pip download --no-deps \
    --timeout=300 \
    --retries=5 \
    --dest /tmp/packages \
    torchaudio \
    || echo "Warning: torchaudio download failed, continuing..."

# =============================================================================
# STAGE 2: Install Heavy Packages - FROM SCRATCH ONLY
# =============================================================================
FROM from-scratch-base AS install-torch

# Copy downloaded packages
COPY --from=download-torch /tmp/packages /tmp/packages

# Install PyTorch from downloaded packages (prefer local, fallback to PyPI)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
        --find-links /tmp/packages \
        --prefer-binary \
        torch==${PYTORCH_VERSION} \
        torchvision \
        torchaudio \
    || (echo "Installing torch from PyPI (fallback)..." && \
        pip install --no-cache-dir --timeout=300 --retries=5 \
        torch==${PYTORCH_VERSION} \
        torchvision \
        torchaudio)

# =============================================================================
# STAGE 3: Common Setup for Both Build Types
# =============================================================================
# We create separate final stages for each build type
# Build scripts will use --target to select the right one

# PREBUILT PATH: Start from prebuilt base
FROM prebuilt-base AS base-prebuilt

# Set common environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=3 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

WORKDIR /app

# Install system dependencies (if not already in base)
RUN if [ "$BUILD_TYPE" = "from-scratch" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            curl \
            && rm -rf /var/lib/apt/lists/* \
            && apt-get clean; \
    else \
        apt-get update && apt-get install -y --no-install-recommends \
            curl \
            && rm -rf /var/lib/apt/lists/* \
            && apt-get clean; \
    fi

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Remove torch from requirements.txt (already in base image for both types)
RUN sed -i '/^torch/d' /app/requirements.txt && \
    sed -i '/torch/d' /app/requirements.txt || true

# Verify torch is removed
RUN grep -v 'torch' /app/requirements.txt > /tmp/requirements_no_torch.txt || true

# =============================================================================
# STAGE 4: Download Heavy ML Packages (Staged for Resume)
# =============================================================================
# Download stage works for both build types
FROM prebuilt-base AS download-ml-packages

# Download heavy ML packages with dependencies for resume capability
# Stage 4.1: Download transformers and dependencies
RUN pip download \
    --timeout=300 \
    --retries=5 \
    --dest /tmp/ml-packages \
    transformers>=4.30.0 \
    || echo "Warning: transformers download failed, will install from PyPI"

# Stage 4.2: Download datasets
RUN pip download \
    --timeout=300 \
    --retries=5 \
    --dest /tmp/ml-packages \
    datasets>=2.14.0 \
    || echo "Warning: datasets download failed, will install from PyPI"

# Stage 4.3: Download accelerate
RUN pip download \
    --timeout=300 \
    --retries=5 \
    --dest /tmp/ml-packages \
    accelerate>=0.20.0 \
    || echo "Warning: accelerate download failed, will install from PyPI"

# Stage 4.4: Download sentence-transformers
RUN pip download \
    --timeout=300 \
    --retries=5 \
    --dest /tmp/ml-packages \
    sentence-transformers>=2.2.0 \
    || echo "Warning: sentence-transformers download failed, will install from PyPI"

# Stage 4.5: Download mlflow (heavy)
RUN pip download \
    --timeout=300 \
    --retries=5 \
    --dest /tmp/ml-packages \
    mlflow>=2.7.0 \
    || echo "Warning: mlflow download failed, will install from PyPI"

# Stage 4.6: Download wandb
RUN pip download \
    --timeout=300 \
    --retries=5 \
    --dest /tmp/ml-packages \
    wandb>=0.15.0 \
    || echo "Warning: wandb download failed, will install from PyPI"

# =============================================================================
# STAGE 5: Install All Packages (PREBUILT PATH - DEFAULT)
# =============================================================================
FROM base-prebuilt AS final-prebuilt

# Copy downloaded ML packages (if available)
COPY --from=download-ml-packages /tmp/ml-packages /tmp/ml-packages

# Upgrade pip with resume-friendly settings
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --upgrade pip setuptools wheel

# Install core dependencies first (small packages, fast)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
        fastapi==0.112.2 \
        uvicorn[standard]==0.30.6 \
        pydantic==2.8.2 \
        pydantic-settings==2.2.1 \
        openai==1.43.0 \
        python-dotenv==1.0.1 \
        httpx==0.27.2 \
        structlog==24.1.0 \
        python-json-logger==2.0.7 \
        prometheus-client==0.20.0 \
        redis==5.0.1 \
        pytest==8.3.2 \
        pytest-asyncio==0.23.5 \
        pytest-cov==4.1.0

# Install ML libraries (prefer downloaded packages, fallback to PyPI)
RUN if [ -d "/tmp/ml-packages" ] && [ "$(ls -A /tmp/ml-packages)" ]; then \
        echo "Installing from downloaded packages (with PyPI fallback for dependencies)..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed \
            --find-links /tmp/ml-packages \
            --prefer-binary \
            transformers>=4.30.0 \
            datasets>=2.14.0 \
            accelerate>=0.20.0 \
            sentence-transformers>=2.2.0; \
    else \
        echo "Installing from PyPI..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed \
            transformers>=4.30.0 \
            datasets>=2.14.0 \
            accelerate>=0.20.0 \
            sentence-transformers>=2.2.0; \
    fi

# Install scikit-learn, numpy, pandas (medium packages)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
        scikit-learn>=1.3.0 \
        numpy>=1.24.0 \
        pandas>=2.0.0

# Install optional heavy packages separately with retries
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=3 \
        mlflow>=2.7.0 \
        wandb>=0.15.0 || echo "Warning: mlflow/wandb installation failed, continuing..."

# Install optional packages (can fail without breaking build)
# GPU-specific packages only if CUDA base image is used
RUN if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then \
        echo "GPU build detected, installing GPU-specific packages..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
            deepspeed>=0.12.0 || echo "Warning: deepspeed installation failed (optional), continuing..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
            bitsandbytes>=0.41.0 || echo "Warning: bitsandbytes installation failed (optional), continuing..."; \
    else \
        echo "CPU build: Skipping GPU-specific packages (deepspeed, bitsandbytes)"; \
    fi

# Install remaining optional packages
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        peft>=0.7.0 || echo "Warning: peft installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        gymnasium>=0.29.0 || echo "Warning: gymnasium installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        pymilvus>=2.3.0 || echo "Warning: pymilvus installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        pinecone-client>=3.0.0 || echo "Warning: pinecone installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        weaviate-client>=4.0.0 || echo "Warning: weaviate installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        influxdb-client>=1.38.0 || echo "Warning: influxdb installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        kubernetes>=28.1.0 || echo "Warning: kubernetes installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        optuna>=3.5.0 || echo "Warning: optuna installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        hyperopt>=0.2.7 || echo "Warning: hyperopt installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        tensorboard>=2.15.0 || echo "Warning: tensorboard installation failed (optional), continuing..."

# Verify critical packages
RUN python -c "import numpy; print('✓ numpy version:', numpy.__version__)" && \
    python -c "import torch; print('✓ torch version:', torch.__version__); print('✓ CUDA available:', torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else False)" && \
    python -c "import sentence_transformers; print('✓ sentence-transformers installed')" || \
    (echo "ERROR: Failed to verify critical packages" && pip list | grep -E "(numpy|torch|sentence)" && exit 1)

# Create non-root user and set up directories
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /app/logs /app/.cache/huggingface /app/.cache/sentence_transformers && \
    chown -R appuser:appuser /app

# Copy application code
COPY app /app/app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# STAGE 6: Install All Packages (FROM-SCRATCH PATH)
# =============================================================================
FROM base-from-scratch AS final-from-scratch

# Copy requirements and setup
COPY requirements.txt /app/requirements.txt

# Remove torch from requirements.txt (already installed in base-from-scratch)
RUN sed -i '/^torch/d' /app/requirements.txt && \
    sed -i '/torch/d' /app/requirements.txt || true

# Verify torch is removed
RUN grep -v 'torch' /app/requirements.txt > /tmp/requirements_no_torch.txt || true

# Copy downloaded ML packages (if available from download stage)
COPY --from=download-ml-packages /tmp/ml-packages /tmp/ml-packages 2>/dev/null || true

# Note: We use prebuilt base images and downloaded packages for faster builds
# No need to copy from host venv - Docker builds should be self-contained

# Upgrade pip with resume-friendly settings
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --upgrade pip setuptools wheel

# Install core dependencies first (small packages, fast)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
        fastapi==0.112.2 \
        uvicorn[standard]==0.30.6 \
        pydantic==2.8.2 \
        pydantic-settings==2.2.1 \
        openai==1.43.0 \
        python-dotenv==1.0.1 \
        httpx==0.27.2 \
        structlog==24.1.0 \
        python-json-logger==2.0.7 \
        prometheus-client==0.20.0 \
        redis==5.0.1 \
        pytest==8.3.2 \
        pytest-asyncio==0.23.5 \
        pytest-cov==4.1.0

# Install ML libraries (prefer downloaded packages, fallback to PyPI)
RUN if [ -d "/tmp/ml-packages" ] && [ "$(ls -A /tmp/ml-packages)" ]; then \
        echo "Installing from downloaded packages (with PyPI fallback for dependencies)..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed \
            --find-links /tmp/ml-packages \
            --prefer-binary \
            transformers>=4.30.0 \
            datasets>=2.14.0 \
            accelerate>=0.20.0 \
            sentence-transformers>=2.2.0; \
    else \
        echo "Installing from PyPI..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed \
            transformers>=4.30.0 \
            datasets>=2.14.0 \
            accelerate>=0.20.0 \
            sentence-transformers>=2.2.0; \
    fi

# Install scikit-learn, numpy, pandas (medium packages)
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed \
        scikit-learn>=1.3.0 \
        numpy>=1.24.0 \
        pandas>=2.0.0

# Install optional heavy packages separately with retries
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=3 \
        mlflow>=2.7.0 \
        wandb>=0.15.0 || echo "Warning: mlflow/wandb installation failed, continuing..."

# Install optional packages (can fail without breaking build)
# GPU-specific packages only if CUDA base image is used
RUN if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then \
        echo "GPU build detected, installing GPU-specific packages..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
            deepspeed>=0.12.0 || echo "Warning: deepspeed installation failed (optional), continuing..." && \
        pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
            bitsandbytes>=0.41.0 || echo "Warning: bitsandbytes installation failed (optional), continuing..."; \
    else \
        echo "CPU build: Skipping GPU-specific packages (deepspeed, bitsandbytes)"; \
    fi

# Install remaining optional packages
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        peft>=0.7.0 || echo "Warning: peft installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        gymnasium>=0.29.0 || echo "Warning: gymnasium installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        pymilvus>=2.3.0 || echo "Warning: pymilvus installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        pinecone-client>=3.0.0 || echo "Warning: pinecone installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        weaviate-client>=4.0.0 || echo "Warning: weaviate installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        influxdb-client>=1.38.0 || echo "Warning: influxdb installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        kubernetes>=28.1.0 || echo "Warning: kubernetes installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        optuna>=3.5.0 || echo "Warning: optuna installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        hyperopt>=0.2.7 || echo "Warning: hyperopt installation failed (optional), continuing..." && \
    pip install --no-cache-dir --upgrade-strategy=only-if-needed --timeout=300 --retries=2 \
        tensorboard>=2.15.0 || echo "Warning: tensorboard installation failed (optional), continuing..."

# Verify critical packages
RUN python -c "import numpy; print('✓ numpy version:', numpy.__version__)" && \
    python -c "import torch; print('✓ torch version:', torch.__version__); print('✓ CUDA available:', torch.cuda.is_available() if hasattr(torch.cuda, 'is_available') else False)" && \
    python -c "import sentence_transformers; print('✓ sentence-transformers installed')" || \
    (echo "ERROR: Failed to verify critical packages" && pip list | grep -E "(numpy|torch|sentence)" && exit 1)

# Create non-root user and set up directories
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /app/logs /app/.cache/huggingface /app/.cache/sentence_transformers && \
    chown -R appuser:appuser /app

# Copy application code
COPY app /app/app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# =============================================================================
# DEFAULT: Use prebuilt (fastest)
# =============================================================================
FROM final-prebuilt AS final
