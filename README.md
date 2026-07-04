# Diri-Helox

Helox is Deepiri's **training-side ML repository**. It owns model training, fine-tuning, dataset versioning, experiment tracking, and model export. **Cyrex** (`diri-cyrex`) is the runtime: inference, RAG, and agent services. Helox is the factory; Cyrex is the runtime.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Deepiri ML Platform                              │
├──────────────────────────────┬──────────────────────────────────────────┤
│  Helox (diri-helox)          │  Cyrex (diri-cyrex)                       │
│  • Training pipelines        │  • Inference & RAG                        │
│  • Dataset versioning        │  • Agent runtime                          │
│  • Experiment tracking       │  • Publishes training samples             │
│  • Model export & registry   │  • Consumes trained models                │
│  • Reproducibility controls  │  • Submits training jobs back to Helox    │
└──────────────────────────────┴──────────────────────────────────────────┘
         │                                      ▲
         │  model-ready events + MLflow/S3      │
         └──────────────────────────────────────┘
                    deepiri-modelkit + deepiri-synapse
```

---

## Table of Contents

- [What Problem Does Helox Solve?](#what-problem-does-helox-solve)
- [Repository Layout](#repository-layout)
- [End-to-End Training Pipeline](#end-to-end-training-pipeline)
- [Data Ingestion](#data-ingestion)
- [Job Submission & Workers](#job-submission--workers)
- [Training Pipelines](#training-pipelines)
- [Dataset Versioning](#dataset-versioning)
- [Model Deployment (Training → Production)](#model-deployment-training--production)
- [How Cyrex Consumes Trained Models](#how-cyrex-consumes-trained-models)
- [Reproducibility](#reproducibility)
- [Evaluation & Quality Gates](#evaluation--quality-gates)
- [Integration with Cyrex RAG](#integration-with-cyrex-rag)
- [Synapse & Event Streams](#synapse--event-streams)
- [Environment Variables](#environment-variables)
- [Setup & Usage](#setup--usage)
- [Related Repositories & Docs](#related-repositories--docs)

---

## What Problem Does Helox Solve?

Helox exists because training and inference have different lifecycles, dependencies, and failure modes. Splitting them solves four concrete problems:

| Problem | How Helox addresses it |
|---------|------------------------|
| **Training/runtime coupling** | Heavy training deps (HF, PEFT, DeepSpeed, notebooks) stay in Helox; Cyrex stays lean for low-latency inference |
| **Closed-loop learning** | Cyrex and Language Intelligence emit live training samples; Helox turns them into fine-tuned models and pushes them back |
| **Reproducible language-intelligence training** | Lease abstraction, contract intelligence, obligation graphs, and regulatory language datasets are versioned, checksummed, and tied to MLflow runs |
| **Production handoff without manual deploys** | Trained artifacts register in MLflow/S3 and publish a `model-ready` event; Cyrex auto-downloads and hot-reloads |

### Primary use cases

- **Lease abstraction** — version lease document datasets as new leases are added
- **Contract intelligence** — track clause-extraction datasets across versions
- **Obligation dependency** — version dependency-graph training data
- **Regulatory language** — track regulation document datasets as regulations evolve
- **Intent / task classification** — DeBERTa and LLM fine-tuning for routing and command understanding

---

## Repository Layout

```
diri-helox/
├── pipelines/
│   ├── data_preprocessing/     # Data cleaning and transformation
│   ├── data_processing/        # Data collection and dataset building
│   └── training/               # Full, versioned, RAG, distributed, dynamic pipelines
├── training/                   # Trainers and unified orchestrator
│   └── unified_training_orchestrator.py
├── data_sources/               # Redis, Postgres, Milvus, synthetic, composite sources
├── data_management/            # Dataset lifecycle utilities
├── utils/
│   └── dataset_versioning.py   # Re-exports deepiri-dataset-processor versioning
├── mlops/
│   ├── training_bridge.py      # Single integration surface (modelkit + orchestrator)
│   ├── training_job_worker.py  # Consumes training-jobs Redis stream
│   ├── model_registry/         # Model registration
│   └── infrastructure/         # LoRA, experiment tracking, model adaptation layers
├── integrations/
│   ├── cyrex_rag_bridge.py     # RAG-aware training from Cyrex
│   └── synapse_event_publisher.py
├── helox_http/
│   ├── app.py                  # FastAPI training API surface
│   └── training_api.py         # POST /training/runs (deepiri-jobs helox.train)
├── model_export/               # PyTorch, ONNX export
├── helox_sdk/                  # Post-training eval harness (deepiri-helox-sdk)
├── deepiri-training-orchestrator/  # Reproducibility, HF adapter, experiment tracking
├── deepiri-dataset-processor/      # Dataset versioning database and validation
├── deepiri-gpu-utils/              # GPU detection, Ollama tiers, torch device helpers
├── scripts/                    # Training entrypoints and CLI tools
├── experiments/                # Research notebooks and configs
├── data/                       # Raw, processed, and synthetic datasets
├── models/                     # Checkpoints and exports
└── docs/                       # Detailed guides (see Related Repositories & Docs)
```

---

## End-to-End Training Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION                                                         │
│    Language Intelligence / Cyrex → Redis streams + Postgres mirror        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. JOB TRIGGER                                                            │
│    • Cyrex HeloxJobClient → training-jobs Redis stream                    │
│    • deepiri-jobs helox.train → POST /training/runs                       │
│    • TrainingJobWorker polls stream (LIVE jobs before BATCH)              │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. DATASET PREPARATION (training_bridge.prepare_training_dataset)          │
│    • Quality validation                                                   │
│    • Dataset manifest (checksums, provenance)                             │
│    • Optional version pin: dataset_name@1.0.0                             │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 4. TRAINING EXECUTION                                                     │
│    ReproducibilityController → TrainingOrchestrator → HF Trainer / custom   │
│    MLflow/W&B tracking, LoRA/QLoRA, distributed training                  │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 5. EXPORT & REGISTRATION                                                  │
│    Checkpoint → MLflow + S3/MinIO → model-ready event on model-events      │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 6. CYREX RUNTIME                                                          │
│    AutoModelLoader subscribes → downloads → caches → inference/hot-reload │
└──────────────────────────────────────────────────────────────────────────┘
```

### Lifecycle events published during a run

| Event | Stream | When |
|-------|--------|------|
| `training.started` | `training-events` | Run begins |
| `training.progress` | `training-events` | Periodic step updates |
| `training.checkpoint` | `training-events` | Checkpoint saved |
| `training.completed` | `training-events` | Run finishes successfully |
| `training.failed` | `training-events` | Run errors out |
| `model-ready` | `model-events` | Model registered and available for Cyrex |

---

## Data Ingestion

Helox is designed around **live data** from the language intelligence layer. Cyrex publishes samples; Helox consumes them for training and backfill.

### Redis streams (live, low-latency)

| Stream | Content |
|--------|---------|
| `pipeline.helox-training.raw` | Raw text payloads from document processing |
| `pipeline.helox-training.structured` | Instruction/input/output structured training pairs |

Defined in `deepiri_modelkit.streaming.topics.StreamTopics`. Consumed by `StreamDataSource`.

### Postgres mirror (durable replay)

Cyrex upserts the same logical records into `cyrex.helox_training_samples` whenever it publishes to the Redis streams. This gives Helox a durable audit and backfill source.

See `docs/HELOX_POSTGRES_MIRROR_CONTRACT.md` for the full schema. `PostgresDataSource` defaults:

- Table: `cyrex.helox_training_samples`
- Min quality: `0.4`
- Filters: `stream_type` (`raw` / `structured`), `producer`

### Other data sources

| Source | Module | Purpose |
|--------|--------|---------|
| `MilvusDataSource` | `data_sources/milvus_source.py` | Vector-indexed documents |
| `SyntheticDataSource` | `data_sources/synthetic_source.py` | Template-based generation when live data unavailable |
| `SelfFeedbackDataSource` | `data_sources/self_feedback_source.py` | High-confidence model predictions as training signal |
| `CompositeDataSource` | `data_sources/composite_source.py` | Weighted mix of any sources above |

Instantiate via `create_data_sources_from_config()` from a config dict.

---

## Job Submission & Workers

### Path 1: Cyrex → Redis job queue

Cyrex enqueues `TrainingRunRequest` messages on the `training-jobs` stream via `HeloxJobClient` (`diri-cyrex/app/training/helox_job_client.py`).

`TrainingJobWorker` (`mlops/training_job_worker.py`):

- Polls the `training-jobs` stream
- Classifies jobs by priority: **LIVE** jobs are processed before **BATCH**
- Dispatches each job through `training_bridge.consume_training_job()`
- Publishes lifecycle events on completion or failure

Run the worker:

```bash
poetry run python scripts/run_training_worker.py
# or
HELOX_WORKER_NAME=helox-worker-1 poetry run python -m mlops.training_job_worker
```

### Path 2: HTTP trigger (deepiri-jobs)

`POST /training/runs` on the Helox FastAPI surface (`helox_http/training_api.py`):

- Used by `deepiri-jobs` `helox.train` task
- Runs `TrainingOrchestrator` in-process
- Publishes Synapse `training.*` events

### Path 3: Direct script / pipeline invocation

Run any pipeline script directly for local or batch training:

```bash
poetry run python scripts/train_instruction_finetuning.py
poetry run python pipelines/training/full_training_pipeline.py --config configs/training.json
```

---

## Training Pipelines

All pipelines integrate through `mlops/training_bridge.py`, which consolidates `deepiri-training-orchestrator` and `deepiri-modelkit` into one surface.

### Key bridge functions

| Function | Purpose |
|----------|---------|
| `make_run_context()` | Create a `TrainingRunContext` tagged `source="helox"` |
| `prepare_training_dataset()` | Validate, manifest, and provenance-wrap a dataset |
| `persist_manifest()` | Write manifest JSON alongside run artifacts |
| `build_run_config()` | Build `TrainingRunConfig` with seed, hyperparameters, tracking |
| `create_orchestrator()` | Wire reproducibility, callbacks, and experiment tracker |
| `register_trained_model()` | Register in MLflow and publish `model-ready` |
| `consume_training_job()` | Dispatch a job payload by priority |

### Available pipelines

| Pipeline | File | Description |
|----------|------|-------------|
| **FullTrainingPipeline** | `pipelines/training/full_training_pipeline.py` | LoRA/QLoRA, HF Trainer, MLflow/W&B, distributed |
| **VersionedTrainingPipeline** | `pipelines/training/versioned_training_pipeline.py` | Pins dataset to a specific version via `DatasetVersionManager` |
| **UnifiedTrainingOrchestrator** | `training/unified_training_orchestrator.py` | Full feature loop: monitoring, checkpointing, Synapse events, optional RAG |
| **RAGTrainingPipeline** | `pipelines/training/rag_training_pipeline.py` | Context-aware training with retrieved documents |
| **DistributedTraining** | `pipelines/training/distributed_training.py` | Multi-GPU / DeepSpeed distributed runs |
| **DynamicTrainingPipeline** | `pipelines/training/dynamic_training_pipeline.py` | Config-driven pipeline with graceful MLflow/model-ready fallback |

### Unified orchestrator quick start

```python
import asyncio
from pathlib import Path
from core.training_config import TrainingConfig, ModelConfig, DataConfig
from training.unified_training_orchestrator import UnifiedTrainingOrchestrator
from integrations.cyrex_rag_bridge import create_cyrex_rag_bridge

async def train():
    model_config = ModelConfig.from_file("configs/model_config.json")
    data_config = DataConfig.from_file("configs/data_config.json")
    training_config = TrainingConfig.from_file("configs/training_config.json")

    rag_bridge = create_cyrex_rag_bridge(auto_discover=True)
    rag_pipeline = rag_bridge if rag_bridge.is_available() else None

    orchestrator = UnifiedTrainingOrchestrator(
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        rag_pipeline=rag_pipeline,
        seed=1337,
    )
    await orchestrator.initialize()
    orchestrator.create_model()
    orchestrator.create_optimizer_and_scheduler()
    train_loader, val_loader = orchestrator.create_data_loaders(
        train_dataset_path=Path("data/tokenized/pretraining_dataset/train"),
        val_dataset_path=Path("data/tokenized/pretraining_dataset/validation"),
    )
    await orchestrator.train(train_loader, val_loader)
    await orchestrator.cleanup()

asyncio.run(train())
```

Or via CLI:

```bash
python scripts/train_with_full_features.py \
    --config-dir configs \
    --train-data data/tokenized/pretraining_dataset/train \
    --val-data data/tokenized/pretraining_dataset/validation \
    --seed 1337 \
    --enable-rag
```

---

## Dataset Versioning

Dataset versioning lives in **`deepiri-dataset-processor`** and is re-exported from `utils/dataset_versioning.py`.

### Core type: `DatasetVersionManager`

Backed by SQLAlchemy (SQLite for local dev, Postgres in production).

| Capability | Method | Details |
|------------|--------|---------|
| Create version | `create_version()` | Semver, parent lineage, change summary, tags |
| Integrity | checksums | `data_checksum` + `metadata_checksum` over file contents |
| Storage | versioned paths | Local filesystem or S3/MinIO |
| Query | `get_version()`, `list_versions()`, `get_latest_version()` | Filter by name and type |
| Compare | `compare_versions()` | Diff statistics between two versions |
| Lineage | `parent_version` | Track how datasets evolved |

### Dataset types

```python
class DatasetType(str, Enum):
    LEASE_ABSTRACTION = "lease_abstraction"
    CONTRACT_INTELLIGENCE = "contract_intelligence"
    OBLIGATION_DEPENDENCY = "obligation_dependency"
    REGULATORY_LANGUAGE = "regulatory_language"
    CLAUSE_EVOLUTION = "clause_evolution"
```

### Version spec syntax

Used by `VersionedTrainingPipeline`:

```
lease_corpus@1.0.0      # Pin to exact version
lease_corpus@latest     # Use latest registered version
/path/to/data           # Auto-version from path
```

MLflow params logged per run: `dataset_name`, `dataset_version`, `dataset_checksum`, `total_samples`, `dataset_type`.

### CLI

```bash
poetry run python scripts/dataset_versioning_cli.py --help
```

See also: `docs/DATASET_VERSIONING_SYSTEM.md`, `docs/DATASET_VERSIONING_GUIDE.md`.

---

## Model Deployment (Training → Production)

```
Train → Export checkpoint → Register (MLflow + S3) → model-ready event → Cyrex download + cache → Inference
```

### Step 1: Artifact export

On training completion, pipelines export model weights:

- **PyTorch** — primary format, includes optimizer state when configured
- **ONNX** — optional, for faster inference runtimes

Handled by `model_export/format_exporter.py` and `UnifiedTrainingOrchestrator.export_model()`.

### Step 2: Registry write

`register_trained_model()` (via `training_bridge`) or `ModelRegistrar` (`mlops/model_registry/model_registrar.py`):

1. Calls `ModelRegistryClient.register_model()` → MLflow tracking URI + S3/MinIO bucket
2. Stores metadata: model type, accuracy, training fingerprint, dataset version

Environment:

```bash
MODEL_REGISTRY_TYPE=mlflow
MLFLOW_TRACKING_URI=http://mlflow:5000
S3_ENDPOINT_URL=http://minio:9000
S3_BUCKET=mlflow-artifacts
```

### Step 3: Event publish

`register_model_ready()` in `deepiri-modelkit` publishes a typed `ModelReadyEvent`:

```json
{
  "event": "model-ready",
  "source": "helox",
  "model_name": "lease-abstraction-v2",
  "version": "1.0.0",
  "registry_path": "s3://mlflow-artifacts/models/lease-abstraction-v2/1.0.0",
  "metadata": { "dataset_version": "1.0.0", "fingerprint": "abc123..." }
}
```

Published to the `model-events` Redis stream (or via Synapse sidecar).

### Step 4: Cyrex pickup

See [How Cyrex Consumes Trained Models](#how-cyrex-consumes-trained-models) below.

---

## How Cyrex Consumes Trained Models

Cyrex has three integration paths for models trained in Helox.

### 1. Event-driven auto-load (primary)

`diri-cyrex/app/integrations/model_loader.py` — `AutoModelLoader`:

```
Helox trains → registers in MLflow + S3
           → publishes ModelReadyEvent to model-events
           → Cyrex AutoModelLoader receives event
           → registry.download_model() → local MODEL_CACHE_DIR
           → model cached and available for inference
```

Singleton access:

```python
from app.integrations.model_loader import get_auto_loader

loader = await get_auto_loader()
model_info = loader.get_model("lease-abstraction-v2", version="1.0.0")
```

### 2. PEFT hot reload

`diri-cyrex/app/training/model_reload_listener.py`:

- Subscribes to `model-events`
- On `model-ready`, calls `DynamicLoRAService.reload_adapter(model_name, version, path)`
- Allows LoRA adapter swap without full service restart

### 3. Closed loop (Cyrex → Helox → Cyrex)

```
Cyrex observes usage / collects feedback
    → HeloxJobClient submits TrainingRunRequest (LIVE or BATCH priority)
    → Helox trains on new data
    → model-ready event published
    → Cyrex AutoModelLoader picks up new weights
```

### Cyrex job submission (Cyrex side)

```python
from app.training.helox_job_client import HeloxJobClient
from deepiri_modelkit.contracts.training import TrainingRunRequest

client = HeloxJobClient()
message_id = client.submit(TrainingRunRequest(
    experiment_id="feedback-loop-001",
    model_name="intent-classifier",
    priority=TrainingPriority.LIVE,
    ...
))
```

---

## Reproducibility

Reproducibility is enforced at multiple layers via `deepiri-training-orchestrator` (`ReproducibilityController`).

### Layer 1: Randomness control

```python
ReproducibilityController(seed=1337, deterministic=True)
controller.set_seeds()
```

Sets seeds for:

- Python `random`
- NumPy
- PyTorch CPU and CUDA
- `torch.use_deterministic_algorithms(True)`
- cuDNN deterministic mode
- DataLoader `worker_init_fn` for multi-worker determinism

### Layer 2: Training fingerprint

SHA-256 hash of `{ seed, sorted config JSON, code_hash }` → 16-char fingerprint saved to `training_fingerprint.json` alongside checkpoints.

```python
fingerprint = controller.generate_training_fingerprint(config, code_hash=git_hash)
controller.save_fingerprint(output_path / "training_fingerprint.json")
controller.verify_reproducibility(checkpoint_path, expected_fingerprint=fingerprint)
```

### Layer 3: Dataset pinning

- Dataset manifests written by `prepare_training_dataset()` include checksums and provenance
- `validate_manifest_against_path()` verifies data hasn't drifted since manifest creation
- Versioned pipelines pin `dataset_name@version` so runs are tied to exact data snapshots

### Layer 4: Experiment audit trail

MLflow/W&B logs per run:

- Dataset version and checksum
- Git commit info
- Hyperparameters and seed
- Training fingerprint

### What "reproducible" means in practice

> Same seed + same config fingerprint + same dataset version/manifest → same training run identity, with full artifact and lineage trace in MLflow.

---

## Evaluation & Quality Gates

Post-training validation lives in **`helox_sdk`** (`deepiri-helox-sdk`):

| Tool | Module | Purpose |
|------|--------|---------|
| Post-training eval harness | `evaluation/harness.py` | Run eval suites after training |
| Inference parity tester | `evaluation/parity.py` | Verify training vs inference output match |
| Regression tracker | `evaluation/regression.py` | Compare metrics against prior model versions |
| Model comparison | `evaluation/comparison.py` | Side-by-side eval across model directories |
| Inference benchmark | `evaluation/benchmark.py` | Latency and throughput profiling |

CLI:

```bash
poetry run helox-eval --help
```

Run eval after training:

```bash
poetry run python scripts/evaluation/evaluate_trained_model.py \
    --model-path models/exports/my-model \
    --suite helox_sdk/tests/fixtures/generation_suite.jsonl
```

---

## Integration with Cyrex RAG

When RAG-aware training is enabled (`--enable-rag` or `rag_pipeline` passed to orchestrator):

1. `create_cyrex_rag_bridge(auto_discover=True)` connects to Cyrex RAG service
2. Retrieved context is packed into training sequences
3. Retrieval-conditioned prompts are created with chunk boundary marking
4. Falls back gracefully when RAG is unavailable

Files:

- `integrations/cyrex_rag_bridge.py` — auto-discovery and connection
- `integrations/rag_aware_training_integration.py` — context window packing

This creates a bidirectional loop: Cyrex knowledge informs Helox training; Helox-trained models improve Cyrex inference.

---

## Synapse & Event Streams

Helox publishes events through Redis Streams directly or via the Go Synapse sidecar.

### Transport toggle

Helox defaults to direct Redis. To route through the sidecar:

```bash
export SYNAPSE_TRANSPORT=sidecar
export SYNAPSE_SIDECAR_URL=http://localhost:8081
export SYNAPSE_SIDECAR_TIMEOUT_SEC=5
export SYNAPSE_SIDECAR_SENDER=helox
```

To revert:

```bash
export SYNAPSE_TRANSPORT=redis
```

### Canonical stream names (deepiri-modelkit)

| Stream | Direction | Content |
|--------|-----------|---------|
| `training-jobs` | Cyrex → Helox | `TrainingRunRequest`, `AgentTrainingJob` |
| `training-events` | Helox → platform | Lifecycle: started, progress, checkpoint, completed, failed |
| `model-events` | Helox → Cyrex | `model-ready` events |
| `pipeline.helox-training.raw` | Cyrex → Helox | Raw training samples |
| `pipeline.helox-training.structured` | Cyrex → Helox | Structured instruction pairs |

Publisher: `integrations/synapse_event_publisher.py`

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `REDIS_URL` | — | Redis connection for streams and job queue |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | MLflow experiment tracking |
| `MODEL_REGISTRY_TYPE` | `mlflow` | Registry backend |
| `S3_ENDPOINT_URL` | `http://minio:9000` | MinIO/S3 for artifacts |
| `S3_BUCKET` | `mlflow-artifacts` | Artifact bucket |
| `MINIO_ACCESS_KEY` | — | S3 access key |
| `MINIO_SECRET_KEY` | — | S3 secret key |
| `HELOX_WORKER_NAME` | `helox-worker-1` | Training job worker consumer name |
| `SYNAPSE_TRANSPORT` | `redis` | `redis` or `sidecar` |
| `SYNAPSE_SIDECAR_URL` | — | Go sidecar URL when using sidecar transport |
| `MODEL_CACHE_DIR` | — | Cyrex-side local model cache (runtime) |

Postgres mirror (for `PostgresDataSource`):

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` / `POSTGRES_URL` | Connection to `cyrex.helox_training_samples` |

---

## Setup & Usage

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Installation

```bash
# All dependencies (production + dev)
poetry install

# Production only
poetry install --no-dev

# With optional groups
poetry install --with visualization,optional
```

### Common commands

```bash
# Activate virtual environment
poetry shell

# Train a classifier
poetry run python scripts/train_task_classifier.py

# Run the training job worker
poetry run python scripts/run_training_worker.py

# Dataset versioning CLI
poetry run python scripts/dataset_versioning_cli.py --help

# Post-training evaluation
poetry run helox-eval --help

# Export requirements.txt for non-Poetry environments
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

After training completes, the model is automatically:

1. Exported to the model registry (MLflow/S3)
2. Published via `model-ready` event on `model-events`
3. Available in Cyrex via `AutoModelLoader`

---

## Related Repositories & Docs

### Repositories

| Repo | Role |
|------|------|
| `diri-cyrex` | Runtime AI services — inference, RAG, consumes models |
| `deepiri-modelkit` | Shared contracts, events, registry client, streaming topics |
| `deepiri-synapse` | Go sidecar for Redis stream routing |
| `deepiri-jobs` | Job scheduler; triggers `helox.train` → `POST /training/runs` |
| `deepiri-api-gateway` | Routes `/api/agent` to Cyrex |

### In-repo docs

| Doc | Topic |
|-----|-------|
| `docs/INTEGRATION_GUIDE.md` | Full Cyrex + Synapse integration walkthrough |
| `docs/HELOX_POSTGRES_MIRROR_CONTRACT.md` | Postgres training sample mirror schema |
| `docs/DATASET_VERSIONING_SYSTEM.md` | Dataset versioning architecture |
| `docs/DATASET_VERSIONING_GUIDE.md` | Versioning usage guide |
| `docs/README_TRAINING_PIPELINE.md` | Quick-start training pipeline |
| `docs/README_LLM_TRAINING.md` | LLM training guide |
| `docs/FULLY_IMPLEMENTED_AND_INTEGRATED.md` | Feature integration status |
| `helox_sdk/README.md` | Post-training evaluation SDK |
