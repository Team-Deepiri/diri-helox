# Dataset Versioning System Guide

## Overview

The Dataset Versioning System provides reproducible, auditable dataset management for language intelligence training pipelines. It ensures that training runs can be exactly reproduced and dataset evolution is tracked.

## Key Features

- **Semantic Versioning**: MAJOR.MINOR.PATCH versioning with automatic change detection
- **Data Integrity**: SHA256 checksums ensure datasets haven't been modified
- **Version Lineage**: Track parent-child relationships between dataset versions
- **Rich Metadata**: Store quality scores, tags, sources, and custom metadata
- **Multiple Storage Backends**: Local, S3, MinIO support
- **CLI & API**: Both command-line and programmatic interfaces
- **Training Integration**: Seamless integration with training pipelines

## How It Works (Walkthrough)

The dataset versioning system does the following when you create or use a version:

1. **Create version**
   - You call `create_version(dataset_name, dataset_type, data_path, ...)` (or the CLI `create` command).
   - The manager **computes statistics** for the folder: total samples (from `.json`/`.jsonl`), file count, total size.
   - It **hashes the data** (SHA256 over all files in sorted order) to get a `data_checksum` for integrity.
   - It **hashes metadata** (name, version, stats, custom metadata) to get `metadata_checksum`.
   - If you didn’t pass a version string, it **auto-increments** from the latest version (e.g. `1.0.0` → `1.0.1`). Version strings can be `1.0.0`, `1.0`, or `v1.0.0`.
   - It **classifies the change** (MAJOR / MINOR / PATCH) using the parent version’s sample count: e.g. &gt;50% → MAJOR, &gt;10% → MINOR, else PATCH.
   - With **local** storage, it **copies** the dataset to `./datasets/{dataset_name}/{version}/` and stores that path.
   - It **writes one row** into the `dataset_versions` table (SQLite or PostgreSQL) with all of the above plus tags, change summary, and optional quality/validation fields.

2. **List / get version**
   - `list_versions(dataset_name=...)` and `get_version(dataset_name, version)` read from the same database and return `DatasetVersionMetadata` (or a list of them).
   - `get_latest_version(dataset_name)` is the most recent by `created_at`.

3. **Compare versions**
   - `compare_versions(dataset_name, version1, version2)` loads both version records and returns sample/file/size differences and the second version’s change type and summary.

4. **Validate version**
   - `validate_version(dataset_name, version)` re-reads the data from the stored path, recomputes the data checksum, and compares it to the stored `data_checksum`. So you can detect if files were changed on disk after registration. **Note:** Validation that re-reads data only works with **local** storage; S3 retrieve is not implemented yet.

5. **Use in training**
   - Training code calls `get_version()` or `get_latest_version()` to get `DatasetVersionMetadata`, then uses `storage_path` to load data. For local backend, `storage_path` is the path to the copied folder; the manager does not modify your original directory.

**Storage:** Only the **local** backend is fully implemented (copy to `./datasets/...`). S3/MinIO are not implemented yet: `create_version` with `storage_backend="s3"` and `validate_version` for S3-stored data will raise `NotImplementedError` until S3 upload/download is added.

## Quick Start

### 1. Install Dependencies (use a virtual environment)

On Linux (e.g. Ubuntu/WSL), the system Python is often "externally managed", so install into a **virtual environment** instead of system-wide:

```bash
cd diri-helox

# Create a venv (one-time)
python3 -m venv .venv

# Activate it (do this in every new terminal where you work on this project)
source .venv/bin/activate   # Linux/macOS
# On Windows:  .venv\Scripts\activate

# Install dependencies (choose one):

# Option A – Full project (slower; can timeout on slow networks)
pip install -r requirements.txt

# Option B – Dataset versioning only (small, fast; enough for tests and CLI)
pip install -r requirements-dataset-versioning.txt
```

If `pip install` times out (e.g. "Read timed out" from files.pythonhosted.org), see **Troubleshooting → pip install times out** below.

After activation, your prompt usually shows `(.venv)` and `python` / `pip` use the venv. To leave the venv later, run `deactivate`.

### 2. Create Dataset Versions
```bash
# Create first version
python scripts/dataset_versioning_cli.py create \
  --name lease_abstraction_training \
  --type lease_abstraction \
  --path data/samples/lease_abstraction_v1 \
  --summary "Initial dataset with 5 lease documents"

# Create updated version
python scripts/dataset_versioning_cli.py create \
  --name lease_abstraction_training \
  --type lease_abstraction \
  --path data/samples/lease_abstraction_v2 \
  --parent 1.0.0 \
  --summary "Added 3 more lease documents"
```

### 3. Test That It Works

From the `diri-helox` directory with the venv **activated** (see step 1):

**Option A – Automated tests (recommended)**

No pytest needed — run the test script directly:

```bash
cd diri-helox
source .venv/bin/activate   # if not already activated
python tests/test_dataset_versioning.py
```

Or, if you have pytest installed (`pip install pytest`):

```bash
pytest tests/test_dataset_versioning.py -v
```

The tests use temporary directories and minimal fake data; they cover create version, list, get, compare, validate, and auto-increment. No existing data folders are required.

**Option B – Quick CLI smoke test**

Create a small folder and run the CLI (venv activated):

```bash
mkdir -p /tmp/test_ds_v1
echo '{"id":1,"text":"sample"}' > /tmp/test_ds_v1/sample.jsonl

python scripts/dataset_versioning_cli.py create \
  --name smoke_test \
  --type lease_abstraction \
  --path /tmp/test_ds_v1 \
  --summary "Smoke test"

python scripts/dataset_versioning_cli.py list --name smoke_test
```

You should see version `1.0.0` and a storage path under `./datasets/smoke_test/1.0.0/`.

### 4. List and Compare Versions
```bash
# List all versions
python scripts/dataset_versioning_cli.py list --name lease_abstraction_training

# Compare versions
python scripts/dataset_versioning_cli.py compare \
  --name lease_abstraction_training \
  --version1 1.0.0 \
  --version2 1.0.1
```

### 5. Use in Training
```bash
# Train with specific dataset version
python pipelines/training/versioned_training_pipeline.py \
  --config configs/versioned_training_config.json
```

## CLI Reference

### Create Version
```bash
python scripts/dataset_versioning_cli.py create \
  --name DATASET_NAME \
  --type DATASET_TYPE \
  --path DATA_PATH \
  [--version VERSION] \
  [--parent PARENT_VERSION] \
  [--summary SUMMARY] \
  [--tags TAG1,TAG2]
```

**Options:**
- `--name`: Dataset name (required)
- `--type`: Dataset type (lease_abstraction, contract_intelligence, etc.) (required)
- `--path`: Path to dataset files (required)
- `--version`: Specific version string (auto-increments if not provided)
- `--parent`: Parent version for lineage tracking
- `--summary`: Change summary description
- `--tags`: Comma-separated tags

### List Versions
```bash
python scripts/dataset_versioning_cli.py list \
  --name DATASET_NAME \
  [--version VERSION]
```

### Compare Versions
```bash
python scripts/dataset_versioning_cli.py compare \
  --name DATASET_NAME \
  --version1 VERSION1 \
  --version2 VERSION2
```

## Python API

### Basic Usage

```python
from utils.dataset_versioning import DatasetVersionManager, DatasetType

# Initialize manager
manager = DatasetVersionManager(
    db_url="sqlite:///dataset_versions.db",
    storage_backend="local"
)

# Create version
version = manager.create_version(
    dataset_name="my_dataset",
    dataset_type=DatasetType.LEASE_ABSTRACTION,
    data_path=Path("./data/my_dataset"),
    change_summary="Updated training data",
    tags=["production", "lease_abstraction"],
    metadata={
        "source": "SEC filings",
        "quality_score": 0.95
    }
)

# Get version
version = manager.get_version("my_dataset", "1.0.0")

# Get latest version
latest = manager.get_latest_version("my_dataset")

# List all versions
versions = manager.list_versions(dataset_name="my_dataset")

# Compare versions
comparison = manager.compare_versions("my_dataset", "1.0.0", "1.0.1")
```

### Training Integration

```python
from pipelines.training.versioned_training_pipeline import VersionedTrainingPipeline

# Configuration with versioned dataset
config = {
    "dataset_spec": "lease_abstraction_training@latest",  # Use latest version
    "dataset_type": "lease_abstraction",
    "version_db_url": "sqlite:///dataset_versions.db",
    # ... other training config
}

# Alternative: use specific version
config["dataset_spec"] = "lease_abstraction_training@1.0.1"

pipeline = VersionedTrainingPipeline(config)
output_dir, metadata = pipeline.run()
```

## Dataset Specification Formats

The system supports several ways to specify datasets:

### 1. Versioned Dataset
```
"dataset_name@version"
"dataset_name@latest"
```

### 2. Direct Path (with optional auto-versioning)
```
"/path/to/dataset"
```

### 3. Auto-versioning
Set `auto_version: true` in config to automatically create versions for unversioned datasets.

## Configuration

### Training Pipeline Config

```json
{
  "experiment_name": "lease_abstraction_training",
  "base_model": "mistralai/Mistral-7B-v0.1",

  "dataset_spec": "lease_abstraction_training@latest",
  "dataset_type": "lease_abstraction",

  "version_db_url": "sqlite:///dataset_versions.db",
  "storage_backend": "local",

  "auto_version": false,
  "auto_dataset_name": "auto_versioned",

  "output_dir": "./models/versioned_training",
  "num_epochs": 3,
  "batch_size": 4,
  "learning_rate": 0.0002
}
```

## Dataset Types

- `lease_abstraction`: Lease document analysis datasets
- `contract_intelligence`: Contract analysis datasets
- `obligation_dependency`: Obligation dependency datasets
- `regulatory_language`: Regulatory language datasets
- `clause_evolution`: Clause evolution tracking datasets

## Versioning Rules

### Semantic Versioning
Versions follow MAJOR.MINOR.PATCH format:

- **MAJOR**: Significant changes (>50% samples changed, schema changes)
- **MINOR**: Moderate changes (10-50% samples added/modified)
- **PATCH**: Small changes (<10% samples, bug fixes)

### Automatic Change Detection
The system automatically determines version changes based on:
- Sample count differences
- File count changes
- Parent version relationships

## Storage Backends

### Local Storage (Development)
```python
manager = DatasetVersionManager(
    db_url="sqlite:///dataset_versions.db",
    storage_backend="local"
)
```

### S3 Storage (Production)
**Note:** S3 upload and download are not implemented yet. Use `storage_backend="local"` until S3 support is added.

```python
manager = DatasetVersionManager(
    db_url="postgresql://user:pass@host:port/db",
    storage_backend="s3",
    storage_config={
        "bucket": "my-dataset-bucket",
        "region": "us-west-2"
    }
)
```

## Best Practices

### 1. Meaningful Change Summaries
```python
version = manager.create_version(
    dataset_name="lease_abstraction_training",
    data_path=Path("./data/v2"),
    change_summary="Added 500 retail lease documents from Q4 2024, improved annotation quality by 15%",
    parent_version="1.1.0"
)
```

### 2. Rich Metadata
```python
metadata = {
    "source": "SEC filings Q3 2024",
    "annotation_method": "manual",
    "annotators": ["team_a", "team_b"],
    "quality_score": 0.94,
    "inter_annotator_agreement": 0.91,
    "regulations_applicable": ["ASC 842", "ASC 606"]
}
```

### 3. Consistent Tagging
```python
tags = ["production", "lease_abstraction", "customer_data", "q4_2024"]
```

### 4. Regular Validation
```python
# Validate dataset integrity
validation = manager.validate_version("dataset_name", "1.0.0")
if not validation["is_valid"]:
    print(f"Dataset corrupted! Expected: {validation['expected_checksum'][:16]}...")
```

## Troubleshooting

### Database Locked Error
This is common in development environments. Solutions:
1. Use a different database file name
2. Switch to PostgreSQL for production
3. Ensure proper file permissions

### Version Not Found
```python
# Check available versions
versions = manager.list_versions(dataset_name="my_dataset")
print("Available versions:", [v.version for v in versions])
```

### Storage Issues
```python
# Verify storage path exists and is accessible
version = manager.get_version("dataset", "1.0.0")
print(f"Storage path: {version.storage_path}")
assert Path(version.storage_path).exists()
```

## Migration from Unversioned Training

### Before (Unversioned)
```python
config = {
    "train_dataset_path": "./data/lease_docs.jsonl",
    # ... other config
}
```

### After (Versioned)
```python
config = {
    "dataset_spec": "lease_abstraction_training@latest",
    "dataset_type": "lease_abstraction",
    "version_db_url": "sqlite:///dataset_versions.db",
    # ... other config
}
```

## Monitoring and Auditing

The system provides comprehensive audit trails:

- **Version History**: Complete lineage of dataset changes
- **Training Correlation**: Which dataset version trained each model
- **Metadata Tracking**: Quality scores, sources, and annotations
- **Integrity Checks**: SHA256 checksums for data validation

## Production Deployment

For production use:

1. **Use PostgreSQL** instead of SQLite
2. **Configure S3/MinIO** for storage (S3/MinIO upload and download are not implemented yet; use local storage until then)
3. **Set up monitoring** and alerts
4. **Implement access controls** for dataset versions
5. **Regular backups** of version metadata

## API Reference

### DatasetVersionManager

#### Methods
- `create_version()`: Create new dataset version
- `get_version()`: Retrieve specific version
- `get_latest_version()`: Get most recent version
- `list_versions()`: List all versions with filtering
- `compare_versions()`: Compare two versions
- `validate_version()`: Check data integrity

#### Constructor Parameters
- `db_url`: Database connection string
- `storage_backend`: "local", "s3", or "minio"
- `storage_config`: Backend-specific configuration

## Support

For issues or questions:
1. Check this guide first
2. Review error messages and logs
3. Ensure database connectivity
4. Verify file permissions and paths

The dataset versioning system ensures your ML training is reproducible, auditable, and production-ready! 🚀
