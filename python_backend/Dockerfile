# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Set working directory
WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies and verify in one layer, then clean up
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    python -c "import numpy; print('✓ numpy version:', numpy.__version__)" && \
    python -c "import sentence_transformers; print('✓ sentence-transformers installed')" || \
    (echo "ERROR: Failed to verify numpy or sentence-transformers" && pip list | grep -E "(numpy|sentence)" && exit 1) && \
    pip cache purge || true

# Create non-root user and set up directories
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    mkdir -p /app/logs /app/.cache/huggingface /app/.cache/sentence_transformers && \
    chown -R appuser:appuser /app

# Note: Some dependencies may require additional setup:
# - bitsandbytes requires CUDA (optional, for GPU)
# - pinecone-client and weaviate-client are optional
# - kubernetes requires kubectl (optional, for deployment)

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

