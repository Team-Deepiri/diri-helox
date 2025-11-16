#!/bin/bash
# GPU Detection Script for Docker Build
# Detects if a good enough GPU is present and returns appropriate base image

# Minimum GPU requirements (adjust as needed)
MIN_CUDA_VERSION=11.8
MIN_GPU_MEMORY_GB=4

# Check if nvidia-smi is available (indicates NVIDIA GPU)
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, checking capabilities..." >&2
    
    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    
    # Get GPU memory
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    GPU_MEMORY_GB=$((GPU_MEMORY / 1024))
    
    # Get GPU name
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    
    echo "GPU: $GPU_NAME" >&2
    echo "GPU Memory: ${GPU_MEMORY_GB}GB" >&2
    echo "Driver Version: $CUDA_VERSION" >&2
    
    # Check if GPU meets minimum requirements
    if [ "$GPU_MEMORY_GB" -ge "$MIN_GPU_MEMORY_GB" ]; then
        echo "GPU meets requirements, using CUDA image" >&2
        echo "pytorch/pytorch:2.0.0-cuda12.1-cudnn8-runtime"
        exit 0
    else
        echo "GPU memory (${GPU_MEMORY_GB}GB) below minimum (${MIN_GPU_MEMORY_GB}GB), using CPU image" >&2
        echo "python:3.11-slim"
        exit 0
    fi
else
    echo "No NVIDIA GPU detected, using CPU image" >&2
    echo "python:3.11-slim"
    exit 0
fi

