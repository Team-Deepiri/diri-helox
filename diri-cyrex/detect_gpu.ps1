# GPU Detection Script for Docker Build (PowerShell)
# Detects if a good enough GPU is present and returns appropriate base image

# Minimum GPU requirements (adjust as needed)
$MIN_GPU_MEMORY_GB = 4

# Check if nvidia-smi is available (indicates NVIDIA GPU)
$nvidiaSmiPath = Get-Command nvidia-smi -ErrorAction SilentlyContinue

if ($nvidiaSmiPath) {
    Write-Host "NVIDIA GPU detected, checking capabilities..." -ForegroundColor Yellow
    
    try {
        # Get GPU information
        $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | Select-Object -First 1
        
        if ($gpuInfo) {
            $parts = $gpuInfo -split ','
            $gpuName = $parts[0].Trim()
            $memoryStr = $parts[1].Trim()
            
            # Parse memory (format: "XXXX MiB" or "XXXX MB")
            $memoryValue = [regex]::Match($memoryStr, '(\d+)').Groups[1].Value
            $memoryGB = [math]::Round([int]$memoryValue / 1024, 1)
            
            Write-Host "GPU: $gpuName" -ForegroundColor Green
            Write-Host "GPU Memory: ${memoryGB}GB" -ForegroundColor Green
            
            # Check if GPU meets minimum requirements
            if ($memoryGB -ge $MIN_GPU_MEMORY_GB) {
                Write-Host "GPU meets requirements, using CUDA image" -ForegroundColor Green
                Write-Output "pytorch/pytorch:2.0.0-cuda12.1-cudnn8-runtime"
                exit 0
            } else {
                Write-Host "GPU memory (${memoryGB}GB) below minimum (${MIN_GPU_MEMORY_GB}GB), using CPU image" -ForegroundColor Yellow
                Write-Output "python:3.11-slim"
                exit 0
            }
        }
    } catch {
        Write-Host "Error checking GPU info: $_" -ForegroundColor Red
        Write-Host "Falling back to CPU image" -ForegroundColor Yellow
        Write-Output "python:3.11-slim"
        exit 0
    }
} else {
    Write-Host "No NVIDIA GPU detected, using CPU image" -ForegroundColor Yellow
    Write-Output "python:3.11-slim"
    exit 0
}

