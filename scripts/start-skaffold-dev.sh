#!/bin/bash
# Start Deepiri development environment with Skaffold
# This script ensures proper environment setup before running Skaffold

set -e

echo "🚀 Starting Deepiri with Skaffold..."

# Check if minikube is running
if ! minikube status &> /dev/null; then
    echo "❌ Minikube is not running. Starting Minikube..."
    minikube start --driver=docker --cpus=4 --memory=8192
fi

# Configure Docker to use Minikube's Docker daemon
echo "🔧 Configuring Docker environment for Minikube..."
eval $(minikube docker-env)

# Verify Docker is pointing to Minikube
if ! docker ps &> /dev/null; then
    echo "❌ Docker is not accessible. Please check your Docker setup."
    exit 1
fi

# Check if skaffold is installed
if ! command -v skaffold &> /dev/null; then
    echo "❌ Skaffold is not installed. Please install it first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "skaffold.yaml" ]; then
    echo "❌ skaffold.yaml not found. Please run this script from the project root."
    exit 1
fi

# Check if k8s manifests exist
if [ ! -d "ops/k8s" ]; then
    echo "❌ Kubernetes manifests not found in ops/k8s/"
    exit 1
fi

echo "✅ Environment ready. Starting Skaffold..."
echo ""
echo "📋 Skaffold will:"
echo "   - Build Docker images using Minikube's Docker daemon"
echo "   - Deploy to Kubernetes"
echo "   - Auto-sync files for faster development"
echo "   - Port-forward services automatically"
echo "   - Stream logs"
echo ""
echo "Press Ctrl+C to stop and cleanup..."
echo ""

# Run Skaffold in dev mode
skaffold dev \
    --port-forward \
    --trigger=notify \
    --watch-poll=1000 \
    --default-repo=localhost:5000

