# Skaffold Quick Start Guide

## 🚀 Quick Setup (WSL2)

### 1. Install Prerequisites

```bash
# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Install Skaffold
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64
chmod +x skaffold && sudo mv skaffold /usr/local/bin/
```

### 2. Start Minikube

```bash
minikube start --driver=docker --cpus=4 --memory=8192
eval $(minikube docker-env)  # Build images directly in k8s
```

### 3. Run Skaffold

```bash
skaffold dev --port-forward
```

That's it! Skaffold will:
- ✅ Build Docker images using Minikube's Docker daemon
- ✅ Deploy to Kubernetes
- ✅ Auto-sync files for instant updates
- ✅ Port-forward services automatically
- ✅ Stream logs from all services
- ✅ Cleanup on exit

## 📋 What Gets Deployed

- **Infrastructure**: MongoDB, Redis, LocalAI
- **Services**: Backend API (port 5000), Cyrex AI (port 8000)
- **Configuration**: ConfigMaps, Secrets, Ingress

## 🔧 Port Forwarding

Automatically forwarded to localhost:
- Backend: `http://localhost:5000`
- Cyrex: `http://localhost:8000`
- MongoDB: `localhost:27017`
- Redis: `localhost:6379`
- LocalAI: `localhost:8080`

## 🛠️ Common Commands

```bash
# View all services
kubectl get services

# View pods
kubectl get pods

# View logs
kubectl logs -f deployment/deepiri-core-api
kubectl logs -f deployment/deepiri-cyrex

# Open Kubernetes dashboard
minikube dashboard

# Stop everything
# Press Ctrl+C in Skaffold terminal (auto-cleanup)
# Or: skaffold delete
```

## 🐛 Troubleshooting

**Docker not building in Minikube?**
```bash
eval $(minikube docker-env)
docker ps  # Verify it works
```

**Ports already in use?**
Edit `skaffold.yaml` to change `localPort` values.

**Files not syncing?**
Check sync patterns in `skaffold.yaml` match your file types.

## 📚 Full Documentation

See [docs/SKAFFOLD_SETUP.md](docs/SKAFFOLD_SETUP.md) for detailed documentation.

