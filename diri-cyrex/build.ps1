# Smart build script with GPU detection and CPU fallback (PowerShell)

param(
    [string]$Dockerfile = "Dockerfile",
    [string]$ImageName = "deepiri-dev-cyrex:latest"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "🔍 Detecting GPU capabilities..." -ForegroundColor Cyan

# Detect GPU and get base image
$BaseImage = & "$ScriptDir\detect_gpu.ps1"

Write-Host "📦 Using base image: $BaseImage" -ForegroundColor Green
Write-Host "🔨 Building Docker image..." -ForegroundColor Cyan

# Build with detected base image
docker build `
    --build-arg BASE_IMAGE="$BaseImage" `
    --file "$ScriptDir\$Dockerfile" `
    --tag "$ImageName" `
    "$ScriptDir"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build complete!" -ForegroundColor Green
    Write-Host "📊 Image info:" -ForegroundColor Cyan
    docker images "$ImageName" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
} else {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}

