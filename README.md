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

## Integration with Cyrex

Models trained in Helox are:
1. Exported to model registry (MLflow/S3)
2. Published via streaming service (`model-ready` event)
3. Auto-loaded by Cyrex runtime

## Usage

```bash
# Train a model
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

