# Diri-Helox: ML Training & Research

**Purpose**: ML training pipelines, model development, and research

## Structure

```
diri-helox/
├── pipelines/          # Training pipelines
├── experiments/        # Research notebooks
├── data/              # Data management
├── models/            # Model checkpoints
├── mlops/             # MLOps tools
├── scripts/           # Training scripts
└── utils/             # Utilities
```

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites

Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Or on Windows (PowerShell):
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### Installation

```bash
# Install all dependencies (production + dev)
poetry install

# Install only production dependencies
poetry install --no-dev

# Install with optional groups
poetry install --with visualization,optional
```

### Using Poetry

```bash
# Activate the virtual environment
poetry shell

# Run commands within the Poetry environment
poetry run python scripts/train_task_classifier.py

# Add a new dependency
poetry add package-name

# Add a dev dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Export to requirements.txt (for compatibility)
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## Integration with Cyrex

Models trained in Helox are:
1. Exported to model registry (MLflow/S3)
2. Published via streaming service (`model-ready` event)
3. Auto-loaded by Cyrex runtime

## Usage

```bash
# Train a model (using Poetry)
poetry run python scripts/train_task_classifier.py

# Or activate the shell first
poetry shell
python scripts/train_task_classifier.py

# Model automatically:
# - Exported to registry
# - Published to streaming service
# - Available in Cyrex
```

## Related

- `diri-cyrex`: Runtime AI services (consumes models)
- `deepiri-modelkit`: Shared contracts and utilities
- `deepiri-synapse`: Streaming service

