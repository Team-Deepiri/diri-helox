"""Generic training-infrastructure primitives shared across Deepiri services.

This package defines **contracts and base classes** — not concrete, framework-specific
implementations. Downstream systems (e.g. Mudspeed, Cyrex) import these primitives and
**specialize** them for their own models and runtimes.

Layers:

- ``BaseTrainingConfig``           — common training hyperparameters.
- ``DataPipeline`` / ``BatchPolicy`` — data + batching contracts.
- ``CheckpointStrategy``           — save/load/cleanup contract.
- ``TrainerCallback`` / ``CallbackList`` — orchestration hooks.
- ``BaseTrainer``                  — the generic training loop skeleton.

The base layer depends only on the Python standard library so it stays reusable across
frameworks (PyTorch, JAX, sklearn, ...). Concrete behavior is supplied by subclasses.
"""

from .callbacks import CallbackList, TrainerCallback
from .checkpoint import CheckpointStrategy, InMemoryCheckpointStrategy
from .config import BaseTrainingConfig
from .data import BatchPolicy, DataPipeline
from .metrics import MetricAggregator, StepResult
from .trainer import BaseTrainer, TrainerState

__all__ = [
    "BaseTrainingConfig",
    "DataPipeline",
    "BatchPolicy",
    "CheckpointStrategy",
    "InMemoryCheckpointStrategy",
    "TrainerCallback",
    "CallbackList",
    "StepResult",
    "MetricAggregator",
    "BaseTrainer",
    "TrainerState",
]

__version__ = "0.1.0"
