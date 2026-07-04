"""Orchestration hooks.

``TrainerCallback`` lets services inject behavior (logging, experiment tracking,
distributed barriers, profiling) at well-defined points in the loop without subclassing
the trainer. ``CallbackList`` fans events out to many callbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List

if TYPE_CHECKING:  # avoid import cycle at runtime
    from .trainer import TrainerState


class TrainerCallback:
    """Base no-op callback. Override the hooks you care about."""

    def on_train_begin(self, state: "TrainerState") -> None:
        ...

    def on_train_end(self, state: "TrainerState") -> None:
        ...

    def on_epoch_begin(self, state: "TrainerState") -> None:
        ...

    def on_epoch_end(self, state: "TrainerState", metrics: Dict[str, float]) -> None:
        ...

    def on_step_end(self, state: "TrainerState", metrics: Dict[str, float]) -> None:
        ...


class CallbackList(TrainerCallback):
    """Dispatches each event to a list of callbacks in order."""

    def __init__(self, callbacks: Iterable[TrainerCallback] = ()) -> None:
        self._callbacks: List[TrainerCallback] = list(callbacks)

    def add(self, callback: TrainerCallback) -> None:
        self._callbacks.append(callback)

    def on_train_begin(self, state: "TrainerState") -> None:
        for cb in self._callbacks:
            cb.on_train_begin(state)

    def on_train_end(self, state: "TrainerState") -> None:
        for cb in self._callbacks:
            cb.on_train_end(state)

    def on_epoch_begin(self, state: "TrainerState") -> None:
        for cb in self._callbacks:
            cb.on_epoch_begin(state)

    def on_epoch_end(self, state: "TrainerState", metrics: Dict[str, float]) -> None:
        for cb in self._callbacks:
            cb.on_epoch_end(state, metrics)

    def on_step_end(self, state: "TrainerState", metrics: Dict[str, float]) -> None:
        for cb in self._callbacks:
            cb.on_step_end(state, metrics)


__all__ = ["TrainerCallback", "CallbackList"]
