# deepiri-helox-training

**Abstraction layer for training infrastructure.** Helox defines the *generic pipeline
primitives* — interfaces and base classes — and downstream services (Mudspeed, Cyrex, …)
**compose and specialize** them. This keeps the architecture layered:

- **Helox** → defines primitives and contracts
- **Service** → composes and specializes them

The package is **stdlib-only** so the contracts stay framework-agnostic (PyTorch, JAX,
sklearn, …). Concrete behavior is supplied by subclasses.

## Primitives

| Concept | Primitive |
|---------|-----------|
| Hyperparameters | `BaseTrainingConfig` |
| Data pipeline | `DataPipeline` (Protocol), `StaticDataPipeline` |
| Batching | `BatchPolicy` |
| Metrics | `StepResult`, `MetricAggregator` |
| Checkpointing | `CheckpointStrategy`, `InMemoryCheckpointStrategy` |
| Orchestration | `TrainerCallback`, `CallbackList` |
| Training loop | `BaseTrainer`, `TrainerState` |

## Extending (service side)

```python
from deepiri_helox_training import BaseTrainer, BaseTrainingConfig, StepResult

class MyTrainer(BaseTrainer):
    def set_train_mode(self): self.model.train()
    def set_eval_mode(self): self.model.eval()
    def train_step(self, batch) -> StepResult: ...   # optimize, return loss
    def eval_step(self, batch) -> StepResult: ...     # no-grad eval
    def collect_state(self) -> dict: ...              # for checkpoints
    def restore_state(self, state: dict): ...
```

`BaseTrainer.fit(data_pipeline)` then drives the epoch loop, evaluation cadence,
best-model tracking, early stopping, checkpoint cadence, and callback dispatch.

## Install from git (Mudspeed and others)

```toml
deepiri-helox-training = { git = "https://github.com/Team-Deepiri/diri-helox.git", rev = "<GIT_SHA>", subdirectory = "helox_training_core" }
```

## Local development

```bash
cd helox_training_core
poetry install
poetry run pytest
```
