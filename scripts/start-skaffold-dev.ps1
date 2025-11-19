# Start Deepiri development environment with Skaffold (PowerShell)
# This script ensures proper environment setup before running Skaffold

Write-Host "🚀 Starting Deepiri with Skaffold..." -ForegroundColor Cyan

# Check if minikube is running
$minikubeStatus = minikube status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Minikube is not running. Starting Minikube..." -ForegroundColor Yellow
    minikube start --driver=docker --cpus=4 --memory=8192
}

# Configure Docker to use Minikube's Docker daemon
Write-Host "🔧 Configuring Docker environment for Minikube..." -ForegroundColor Cyan
$dockerEnv = minikube docker-env
Invoke-Expression $dockerEnv

# Verify Docker is accessible
try {
    docker ps | Out-Null
} catch {
    Write-Host "❌ Docker is not accessible. Please check your Docker setup." -ForegroundColor Red
    exit 1
}

# Check if skaffold is installed
if (-not (Get-Command skaffold -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Skaffold is not installed. Please install it first." -ForegroundColor Red
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "skaffold.yaml")) {
    Write-Host "❌ skaffold.yaml not found. Please run this script from the project root." -ForegroundColor Red
    exit 1
}

# Check if k8s manifests exist
if (-not (Test-Path "ops/k8s")) {
    Write-Host "❌ Kubernetes manifests not found in ops/k8s/" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Environment ready. Starting Skaffold..." -ForegroundColor Green
Write-Host ""
Write-Host "📋 Skaffold will:" -ForegroundColor Cyan
Write-Host "   - Build Docker images using Minikube's Docker daemon"
Write-Host "   - Deploy to Kubernetes"
Write-Host "   - Auto-sync files for faster development"
Write-Host "   - Port-forward services automatically"
Write-Host "   - Stream logs"
Write-Host ""
Write-Host "Press Ctrl+C to stop and cleanup..." -ForegroundColor Yellow
Write-Host ""

# Run Skaffold in dev mode
skaffold dev `
    --port-forward `
    --trigger=notify `
    --watch-poll=1000 `
    --default-repo=localhost:5000

