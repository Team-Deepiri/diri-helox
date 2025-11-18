# Deepiri Documentation Index

Complete guide to all markdown documentation files in the Deepiri project. Use this as a navigation map to find the information you need.

---

## 📚 Core Documentation (Root Level)

### `README.md`
**Main project overview and quick start guide**
- Project introduction and core concept
- Architecture overview
- Quick start instructions
- API documentation links
- Deployment guide
- Common issues and troubleshooting

### `GETTING_STARTED.md`
**Complete setup guide for new developers**
- Prerequisites and installation
- Docker Compose setup (easiest method)
- Local services setup (no Docker)
- Kubernetes setup for local dev
- Environment configuration
- Troubleshooting common setup issues

### `CONTRIBUTING.md`
**Guidelines for contributing to the project**
- Code of conduct
- Development setup instructions
- Project structure overview
- Development workflow
- Coding standards and style guides
- Testing requirements
- Commit message conventions
- Pull request process

### `LICENSE.md`
**Project license information**
- MIT License details
- Usage rights and restrictions

---

## 🚀 Quick Start & Setup Guides

### `START_EVERYTHING.md`
**Complete guide to starting all services for testing**
- Prerequisites check
- Dependency installation
- Environment setup
- Starting all services
- Service health checks
- Access points and URLs
- Troubleshooting startup issues

### `QUICK-START-SCRIPTS.md`
**Quick reference for common scripts and commands**
- One-liner commands for common tasks
- Script shortcuts
- Development workflow shortcuts

### `ENVIRONMENT_SETUP.md`
**Detailed environment configuration guide**
- Environment variable reference
- Local vs Docker configuration
- API key setup
- Database configuration
- Service-specific settings

### `ENVIRONMENT_VARIABLES.md`
**Complete list of all environment variables**
- Required vs optional variables
- Default values
- Variable descriptions
- Configuration examples

---

## 🐳 Docker & Infrastructure

### `docs/DOCKER-IMAGE-CLEANSING-COMMANDS.md`
**Docker cleanup and rebuild commands**
- Clean rebuild process
- Using `--rmi local` flag
- Build cache management
- Disk usage monitoring
- Why explicit image tags matter

### `docs/README-REBUILD.md` (Archived)
**Guide to rebuilding Docker containers**
- Problem: Storage bloat from old images
- Solution: Clean rebuild scripts
- Manual cleanup commands
- Best practices for rebuilds

### `docs/README-CLEANUP.md` (Archived)
**Docker cleanup strategies and scripts**
- Automated cleanup scripts
- Manual cleanup procedures
- Storage management tips
- Preventing storage bloat

### `DEV_DEVICE_SPECS.md`
**Development device specifications and requirements**
- Minimum system requirements
- Recommended hardware
- Performance considerations

---

## 👥 Team Onboarding Guides

### `FIND_YOUR_TASKS.md`
**Role-based task finder for all team members**
- Quick navigation by role
- Task assignments by team
- File locations for each role
- Stack and technology per role
- Reporting structure

### `docs/AI_TEAM_ONBOARDING.md`
**Onboarding guide for AI team members**
- AI service architecture
- Model training workflows
- Challenge generation system
- RAG pipeline setup
- Experiment tracking

### `docs/AI_TEAM_STARTER.md`
**Quick start guide for AI engineers**
- Essential AI service commands
- Model training basics
- Testing AI features
- Common AI workflows

### `docs/BACKEND_TEAM_ONBOARDING.md`
**Onboarding for backend engineers**
- Microservices architecture
- API design patterns
- Database schemas
- Service communication
- Testing backend services

### `docs/FRONTEND_TEAM_ONBOARDING.md`
**Onboarding for frontend developers**
- React/Vite setup
- Component architecture
- State management
- API integration
- Styling with Tailwind
- HMR (Hot Module Replacement) guide

### `docs/ML_ENGINEER_COMPLETE_GUIDE.md`
**Complete guide for ML engineers**
- ML pipeline setup
- Model training workflows
- Experiment tracking with MLflow
- Model deployment
- Performance optimization

### `docs/MLOPS_TEAM_ONBOARDING.md`
**Onboarding for ML Ops engineers**
- ML Ops infrastructure
- Model registry and versioning
- Deployment automation
- Monitoring and alerting
- CI/CD for ML models

### `docs/PLATFORM_TEAM_ONBOARDING.md`
**Onboarding for platform/infrastructure engineers**
- Infrastructure architecture
- Docker and Kubernetes setup
- Monitoring and observability
- CI/CD pipelines
- Security and compliance

### `docs/SECURITY_QA_TEAM_ONBOARDING.md`
**Onboarding for security and QA teams**
- Security audit procedures
- Testing frameworks
- Quality assurance workflows
- Bug reporting process

---

## 📖 Team-Specific READMEs

### `docs/README_AI_TEAM.md`
**AI team documentation hub**
- AI service overview
- Model training guides
- Challenge generation system
- Links to AI-specific docs

### `docs/README_BACKEND_TEAM.md`
**Backend team documentation hub**
- Microservices overview
- API documentation
- Database schemas
- Service architecture

### `docs/README_FRONTEND_TEAM.md`
**Frontend team documentation hub**
- React component library
- State management patterns
- API integration guide
- UI/UX guidelines

### `docs/README_PLATFORM_TEAM.md`
**Platform team documentation hub**
- Infrastructure overview
- Deployment guides
- Monitoring setup
- DevOps workflows

### `docs/README_SECURITY_QA_TEAM.md`
**Security and QA team hub**
- Security protocols
- Testing procedures
- Quality standards

### `docs/README_DOCS.md`
**Documentation system overview**
- How documentation is organized
- Contributing to docs
- Documentation standards

---

## 🏗️ Architecture & System Design

### `docs/SYSTEM_ARCHITECTURE.md`
**Complete system architecture documentation**
- High-level architecture
- Service breakdown
- Data flow diagrams
- Technology stack
- Scalability considerations

### `docs/MICROSERVICES_ARCHITECTURE.md`
**Microservices architecture details**
- Service boundaries
- Inter-service communication
- API gateway patterns
- Service discovery
- Load balancing

### `docs/MICROSERVICES_SETUP.md`
**Microservices setup and configuration**
- Service deployment
- Network configuration
- Service dependencies
- Health checks
- Service URLs and ports

### `docs/SHARED_UTILS_ARCHITECTURE.md`
**Shared utilities architecture**
- Common libraries and utilities
- Shared code organization
- Utility service design
- Code reuse patterns

---

## 🤖 AI & ML Documentation

### `docs/AI_SERVICES_OVERVIEW.md`
**Overview of all AI services**
- AI service catalog
- Service capabilities
- Integration points
- Performance metrics

### `docs/AI_ENGINEER_ORCHESTRATION_GUIDE.md`
**Guide for orchestrating AI services**
- Service coordination
- Workflow management
- Error handling
- Performance optimization

### `diri-cyrex/app/train/README.md`
**ML training pipeline documentation**
- Training workflows
- Model architectures
- Data preparation
- Experiment tracking

### `diri-cyrex/mlops/README.md`
**ML Ops infrastructure documentation**
- Model deployment
- Model registry
- Monitoring ML models
- CI/CD for ML

### `diri-cyrex/inference/README.md`
**Model inference service documentation**
- Inference API
- Model loading
- Performance optimization
- Caching strategies

---

## 🔧 Development Guides

### `docs/TROUBLESHOOTING.md`
**Common issues and solutions**
- Setup problems
- Service startup issues
- Database connection problems
- API errors
- Performance issues

### `docs/LOG_INSPECTION_GUIDE.md`
**Guide to inspecting and analyzing logs**
- Log locations
- Log formats
- Log analysis tools
- Debugging with logs
- Common log patterns

### `docs/PRODUCT_CHECKLIST.md`
**Product development checklist**
- Feature development workflow
- Testing requirements
- Documentation requirements
- Release checklist

---

## 🎨 Frontend Documentation

### `frontend/README.md`
**Frontend project overview**
- React setup
- Component structure
- Build process
- Development workflow

### `frontend/HMR_GUIDE.md`
**Hot Module Replacement guide**
- HMR setup and configuration
- Troubleshooting HMR issues
- Development workflow with HMR

### `frontend/FRONTEND_DEBUG_GUIDE.md`
**Frontend debugging guide**
- Debugging tools
- Common issues
- Performance debugging
- Browser dev tools

### `frontend/PERFORMANCE_OPTIMIZATION.md`
**Frontend performance optimization**
- Optimization techniques
- Bundle size reduction
- Lazy loading
- Caching strategies

---

## 🔐 Security Documentation

### `api-server/SECURITY_AUDIT.md`
**Security audit documentation**
- Security checklist
- Vulnerability assessment
- Security best practices
- Compliance requirements

---

## 📦 Service Documentation

### `services/README.md`
**Microservices overview**
- Service catalog
- Service responsibilities
- Service communication patterns

### `services/deepiri-api-gateway/README.md`
**API Gateway service documentation**
- Gateway configuration
- Routing rules
- Rate limiting
- Authentication

### `services/deepiri-auth-service/README.md`
**User service documentation**
- User management
- Authentication flows
- Profile management

### `services/deepiri-task-orchestrator/README.md`
**Task service documentation**
- Task CRUD operations
- Task metadata
- Task filtering

### `services/deepiri-challenge-service/README.md`
**Challenge service documentation**
- Challenge generation
- Challenge state management
- AI integration

### `services/deepiri-engagement-service/README.md`
**Gamification service documentation**
- Points system
- Badge management
- Leaderboards
- Streak tracking

### `services/deepiri-platform-analytics-service/README.md`
**Analytics service documentation**
- Performance tracking
- Analytics collection
- Insights generation

### `services/deepiri-notification-service/README.md`
**Notification service documentation**
- Notification delivery
- Notification preferences
- Push notifications

### `services/deepiri-external-bridge-service/README.md`
**Integration service documentation**
- External API integrations
- OAuth flows
- Webhook management

### `services/deepiri-realtime-gateway/README.md`
**WebSocket service documentation**
- Real-time communication
- WebSocket connections
- Event broadcasting

### `services/deepiri-shared-utils/README.md`
**Shared utilities documentation**
- Common utilities
- Shared libraries
- Utility functions

---

## 🐍 Python Backend Documentation

### `diri-cyrex/app/train/infrastructure/README.md`
**Training infrastructure documentation**
- LoRA/QLoRA training
- Experiment tracking
- Model registry
- RAG pipeline

---

## 🔄 Git & Version Control

### `SUBMODULE_COMMANDS.md`
**Git submodule command reference**
- Submodule setup
- Submodule updates
- Submodule workflows

### `submodule_docs/SUBMODULE_COMMANDS.md`
**Detailed submodule documentation**
- Advanced submodule operations
- Submodule troubleshooting

### `submodule_docs/GIT_SUBMODULE_MIGRATION.md`
**Submodule migration guide**
- Migrating to submodules
- Best practices
- Common issues

---

## 🛠️ Scripts Documentation

### `scripts/README-SCRIPTS.md`
**Scripts overview and usage**
- Available scripts
- Script descriptions
- Usage examples

### `scripts/README-FOR-PLATFORM-ENGINEERS.md`
**Platform engineering scripts guide**
- Infrastructure scripts
- Deployment scripts
- Maintenance scripts

### `scripts/README-DOCKER-CLEANUP.md`
**Docker cleanup scripts documentation**
- Cleanup procedures
- Script usage
- Storage management

---

## 🚀 Python Environment Startup Scripts

### `py_environment_startup_scripts/README.md`
**Python-based Docker startup scripts**
- Role-based startup scripts
- Service management
- Docker SDK usage
- Team-specific configurations

---

## 📝 Quick Reference

### Rebuild Commands
- **`rebuild.sh`** / **`rebuild.ps1`**: Clean rebuild (removes old images, rebuilds, starts)
- **`rebuild-clean.sh`** / **`rebuild-clean.ps1`**: Advanced clean rebuild with options
- **`rebuild-fresh.sh`** / **`rebuild-fresh.bat`**: Complete fresh rebuild

### Docker Commands
- **`docker-compose down --rmi local`**: Remove images created by compose file
- **`docker builder prune -af`**: Clean build cache
- **`docker system df`**: Check disk usage

### Makefile (Optional)
- **`make rebuild`**: Full clean rebuild
- **`make rebuild-service SERVICE=cyrex`**: Rebuild one service
- **`make clean`**: Clean everything
- **`make build`**: Normal build with cache
- **`make up`**: Start services
- **`make down`**: Stop services
- **`make logs`**: View logs
- **`make df`**: Show disk usage

---

## 🗺️ Navigation Tips

1. **New to the project?** Start with `README.md` → `GETTING_STARTED.md`
2. **Setting up for first time?** Follow `START_EVERYTHING.md`
3. **Team member?** Check `FIND_YOUR_TASKS.md` for your role
4. **Having issues?** See `docs/TROUBLESHOOTING.md`
5. **Rebuilding containers?** Use `rebuild.sh` or see `docs/DOCKER-IMAGE-CLEANSING-COMMANDS.md`
6. **Need architecture info?** Check `docs/SYSTEM_ARCHITECTURE.md`
7. **Contributing?** Read `CONTRIBUTING.md`

---

## 📍 File Locations by Category

### Root Level (Main Docs)
- `README.md`, `GETTING_STARTED.md`, `CONTRIBUTING.md`, `LICENSE.md`
- `START_EVERYTHING.md`, `FIND_YOUR_TASKS.md`
- `ENVIRONMENT_SETUP.md`, `ENVIRONMENT_VARIABLES.md`
- `docs/DOCKER-IMAGE-CLEANSING-COMMANDS.md`, `docs/README-REBUILD.md`, `docs/README-CLEANUP.md`

### Team Onboarding (`docs/`)
- `AI_TEAM_ONBOARDING.md`, `BACKEND_TEAM_ONBOARDING.md`, `FRONTEND_TEAM_ONBOARDING.md`
- `ML_ENGINEER_COMPLETE_GUIDE.md`, `MLOPS_TEAM_ONBOARDING.md`, `PLATFORM_TEAM_ONBOARDING.md`
- `SECURITY_QA_TEAM_ONBOARDING.md`

### Architecture (`docs/`)
- `SYSTEM_ARCHITECTURE.md`, `MICROSERVICES_ARCHITECTURE.md`, `MICROSERVICES_SETUP.md`
- `SHARED_UTILS_ARCHITECTURE.md`

### Service Docs (`services/*/README.md`)
- Individual service documentation in each service directory

### Python Backend (`diri-cyrex/`)
- `app/train/README.md`, `mlops/README.md`, `inference/README.md`

### Frontend (`frontend/`)
- `README.md`, `HMR_GUIDE.md`, `FRONTEND_DEBUG_GUIDE.md`, `PERFORMANCE_OPTIMIZATION.md`

---

**Last Updated:** 2025-11-16
**Total Documentation Files:** 63 markdown files



