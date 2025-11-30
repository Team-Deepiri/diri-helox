#!/usr/bin/env python3
"""
Development Environment - Full Stack Runner
Runs ALL services from docker-compose.dev.yml
Mimics Kubernetes by injecting ConfigMaps and Secrets into Docker Compose
"""

import os
import sys
import subprocess
from pathlib import Path

# Add shared utilities to path
sys.path.insert(0, str(Path(__file__).parent / 'team_dev_environments' / 'shared'))

from k8s_env_loader import load_all_configmaps_and_secrets, GREEN, YELLOW, CYAN, GRAY, RESET


def run_docker_compose():
    """Run docker-compose with injected environment variables"""
    project_root = Path(__file__).parent
    
    print(f"{GREEN}🚀 Starting Development Environment (All Services)...{RESET}")
    print(f"{CYAN}   (Loading k8s ConfigMaps & Secrets from ops/k8s/){RESET}")
    print()
    
    # Load all k8s config
    env_vars = load_all_configmaps_and_secrets(project_root)
    
    print()
    print(f"{GREEN}📦 Loaded {len(env_vars)} environment variables{RESET}")
    print()
    
    # Inject into current environment
    os.environ.update(env_vars)
    
    # Run docker-compose
    compose_file = project_root / 'docker-compose.dev.yml'
    
    try:
        subprocess.run(
            ['docker', 'compose', '-f', str(compose_file), 'up', '-d'],
            cwd=project_root,
            check=True
        )
        
        print()
        print(f"{GREEN}✅ Development Environment Started!{RESET}")
        print()
        print(f"{YELLOW}Access your services:{RESET}")
        print()
        print("  🌐 Frontend & UI:")
        print("     - Frontend:        http://localhost:5173")
        print("     - Cyrex Interface: http://localhost:5175")
        print()
        print("  🔧 Backend Services:")
        print("     - API Gateway:     http://localhost:5100")
        print("     - Auth Service:    http://localhost:5001")
        print("     - Task Orchestrator: http://localhost:5002")
        print("     - Engagement:      http://localhost:5003")
        print("     - Analytics:       http://localhost:5004")
        print("     - Notifications:   http://localhost:5005")
        print("     - External Bridge: http://localhost:5006")
        print("     - Challenges:      http://localhost:5007")
        print("     - Realtime:        http://localhost:5008")
        print()
        print("  🤖 AI/ML Services:")
        print("     - Cyrex API:       http://localhost:8000")
        print("     - MLflow:          http://localhost:5500")
        print("     - Jupyter:         http://localhost:8888")
        print()
        print("  💾 Infrastructure:")
        print("     - pgAdmin:         http://localhost:5050")
        print("     - Adminer:         http://localhost:8080")
        print("     - InfluxDB:        http://localhost:8086")
        print("     - MinIO Console:   http://localhost:9001")
        print()
        print(f"{GRAY}View logs:{RESET}")
        print("  docker compose -f docker-compose.dev.yml logs -f")
        print()
        print(f"{GRAY}Stop services:{RESET}")
        print("  docker compose -f docker-compose.dev.yml down")
        print()
        
    except subprocess.CalledProcessError as e:
        print(f"{YELLOW}❌ Error starting services: {e}{RESET}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"{YELLOW}❌ Error: 'docker' command not found. Is Docker installed?{RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    run_docker_compose()

