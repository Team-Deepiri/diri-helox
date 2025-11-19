#!/bin/bash
# Setup script for Minikube on WSL2
# This script configures Minikube with Docker driver for local development

set -e

echo "🚀 Setting up Minikube for Deepiri on WSL2..."

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    echo "❌ Minikube is not installed. Please install it first:"
    echo "   curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64"
    echo "   sudo install minikube-linux-amd64 /usr/local/bin/minikube"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop or Docker daemon."
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Installing..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    sudo mv kubectl /usr/local/bin/
fi

# Check if skaffold is installed
if ! command -v skaffold &> /dev/null; then
    echo "⚠️  Skaffold is not installed. Installing..."
    curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64
    chmod +x skaffold
    sudo mv skaffold /usr/local/bin/
fi

# Stop existing minikube if running
if minikube status &> /dev/null; then
    echo "🛑 Stopping existing Minikube cluster..."
    minikube stop
fi

# Start Minikube with Docker driver
echo "🏗️  Starting Minikube cluster with Docker driver..."
minikube start \
    --driver=docker \
    --cpus=4 \
    --memory=8192 \
    --disk-size=20g \
    --addons=ingress \
    --addons=dashboard

# Configure Docker to use Minikube's Docker daemon
echo "🔧 Configuring Docker environment for Minikube..."
eval $(minikube docker-env)

# Verify setup
echo "✅ Verifying setup..."
minikube status
kubectl cluster-info

# Show helpful commands
echo ""
echo "✅ Minikube setup complete!"
echo ""
echo "📋 Useful commands:"
echo "   minikube dashboard          # Open Kubernetes dashboard"
echo "   minikube service list       # List all services"
echo "   minikube tunnel            # Expose LoadBalancer services"
echo "   eval \$(minikube docker-env) # Use Minikube's Docker daemon"
echo ""
echo "🚀 Next steps:"
echo "   1. Make sure you're in the Minikube Docker environment:"
echo "      eval \$(minikube docker-env)"
echo "   2. Run Skaffold:"
echo "      skaffold dev --port-forward"
echo ""

