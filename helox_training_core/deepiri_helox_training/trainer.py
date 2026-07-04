"""The generic training-loop skeleton.

``BaseTrainer`` implements the *orchestration* of training (epoch loop, evaluation
cadence, best-model tracking, early stopping, checkpoint cadence, callback dispatch)
while delegating all *framework-specific* work to abstract hooks that subclasses
implement:

- :meth:`set_train_mode` / :meth:`set_eval_mode`
- :meth:`train_step` / :meth:`eval_step`
- :meth:`collect_state` / :meth:`restore_state`

This keeps the loop reusable across PyTorch, JAX, sklearn, etc.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Iterable, List, Optional, TypeVar

from .callbacks import CallbackList, TrainerCallback
from .checkpoint import CheckpointStrategy
from .config import BaseTrainingConfig
from .data import DataPipeline
from .metrics import MetricAggregator, StepResult

logger = logging.getLogger(__name__)

TBatch = TypeVar("TBatch")


@dataclass
class TrainerState:
    """Mutable bookkeeping shared with callbacks during a run."""

    epoch: int = 0
    global_step: int = 0
    best_metric: Optional[float] = None
    epochs_without_improvement: int = 0
    should_stop: bool = False
    history: Dict[str, List[float]] = field(default_factory=dict)

    def record(self, metrics: Dict[str, float]) -> None:
        for name, value in metrics.items():
            self.history.setdefault(name, []).append(value)


class BaseTrainer(ABC, Generic[TBatch]):
    """Framework-agnostic training loop. Subclass and implement the abstract hooks."""

    def __init__(
        self,
        config: BaseTrainingConfig,
        *,
        checkpoint_strategy: Optional[CheckpointStrategy] = None,
        callbacks: Optional[Iterable[TrainerCallback]] = None,
    ) -> None:
        self.config = config
        self.checkpoint_strategy = checkpoint_strategy
        self.callbacks = CallbackList(callbacks or [])
        self.state = TrainerState()

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def set_train_mode(self) -> None:
        """Put the underlying model into training mode."""

    @abstractmethod
    def set_eval_mode(self) -> None:
        """Put the underlying model into evaluation mode."""

    @abstractmethod
    def train_step(self, batch: TBatch) -> StepResult:
        """Run one optimization step and return its :class:`StepResult`."""

    @abstractmethod
    def eval_step(self, batch: TBatch) -> StepResult:
        """Run one evaluation step (no grad) and return its :class:`StepResult`."""

    @abstractmethod
    def collect_state(self) -> Dict[str, Any]:
        """Return a serializable mapping capturing model/optimizer/loop state."""

    @abstractmethod
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore from a mapping produced by :meth:`collect_state`."""

    # ------------------------------------------------------------- loop pieces
    def train_epoch(self, loader: Iterable[TBatch]) -> Dict[str, float]:
        self.set_train_mode()
        agg = MetricAggregator()
        for batch in loader:
            result = self.train_step(batch)
            agg.update(result)
            self.state.global_step += 1
            self.callbacks.on_step_end(self.state, {"loss": result.loss, **result.metrics})
        return {f"train_{k}": v for k, v in agg.compute().items()}

    def evaluate(self, loader: Iterable[TBatch]) -> Dict[str, float]:
        self.set_eval_mode()
        agg = MetricAggregator()
        for batch in loader:
            agg.update(self.eval_step(batch))
        return {f"val_{k}": v for k, v in agg.compute().items()}

    # --------------------------------------------------------------- full loop
    def fit(self, data: DataPipeline) -> Dict[str, List[float]]:
        """Run the full training loop and return the metric history."""
        cfg = self.config
        self.callbacks.on_train_begin(self.state)
        logger.info("Starting training for %d epoch(s)", cfg.epochs)

        for epoch in range(self.state.epoch, cfg.epochs):
            self.state.epoch = epoch
            self.callbacks.on_epoch_begin(self.state)

            metrics = self.train_epoch(data.train_loader())

            val_loader = data.val_loader()
            if val_loader is not None and (epoch + 1) % max(cfg.eval_interval, 1) == 0:
                metrics.update(self.evaluate(val_loader))
                self._update_best_and_early_stop(metrics)

            if (
                self.checkpoint_strategy is not None
                and (epoch + 1) % max(cfg.checkpoint_interval, 1) == 0
            ):
                self.save_checkpoint(f"epoch_{epoch + 1}")

            self.state.record(metrics)
            self.callbacks.on_epoch_end(self.state, metrics)

            if self.state.should_stop:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        self.callbacks.on_train_end(self.state)
        return self.state.history

    def _update_best_and_early_stop(self, metrics: Dict[str, float]) -> None:
        cfg = self.config
        if cfg.monitor_metric not in metrics:
            return
        current = metrics[cfg.monitor_metric]
        improved = (
            self.state.best_metric is None
            or (cfg.monitor_mode == "min" and current < self.state.best_metric)
            or (cfg.monitor_mode == "max" and current > self.state.best_metric)
        )
        if improved:
            self.state.best_metric = current
            self.state.epochs_without_improvement = 0
            if self.checkpoint_strategy is not None:
                self.save_checkpoint("best")
        else:
            self.state.epochs_without_improvement += 1
            if (
                cfg.early_stopping_patience > 0
                and self.state.epochs_without_improvement >= cfg.early_stopping_patience
            ):
                self.state.should_stop = True

    # ------------------------------------------------------------- checkpoints
    def save_checkpoint(self, tag: str) -> Optional[str]:
        if self.checkpoint_strategy is None:
            return None
        ref = self.checkpoint_strategy.save(self.collect_state(), tag)
        self.checkpoint_strategy.cleanup(self.config.keep_last_checkpoints)
        return ref

    def load_checkpoint(self, tag: str) -> None:
        if self.checkpoint_strategy is None:
            raise RuntimeError("No checkpoint_strategy configured")
        self.restore_state(self.checkpoint_strategy.load(tag))


__all__ = ["BaseTrainer", "TrainerState"]
