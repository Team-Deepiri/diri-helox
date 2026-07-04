"""Checkpointing contract.

``CheckpointStrategy`` decouples *what* to persist (a state mapping produced by the
trainer) from *how/where* it is stored. Services provide concrete strategies (e.g. a
torch ``torch.save`` strategy, an object-store strategy) without changing the loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class CheckpointStrategy(ABC):
    """Persist and restore opaque trainer state under string tags."""

    @abstractmethod
    def save(self, state: Dict[str, Any], tag: str) -> str:
        """Persist ``state`` under ``tag``; return a reference (path/uri/key)."""
        raise NotImplementedError

    @abstractmethod
    def load(self, tag: str) -> Dict[str, Any]:
        """Restore the state previously saved under ``tag``."""
        raise NotImplementedError

    @abstractmethod
    def list_tags(self) -> List[str]:
        """Return saved tags, oldest first."""
        raise NotImplementedError

    def cleanup(self, keep_last: int) -> None:
        """Drop all but the ``keep_last`` most recent checkpoints. Optional to override.

        Tags containing ``"best"`` are always retained.
        """
        if keep_last <= 0:
            return
        tags = [t for t in self.list_tags() if "best" not in t]
        for tag in tags[:-keep_last]:
            self.delete(tag)

    def delete(self, tag: str) -> None:  # pragma: no cover - optional
        """Remove a single checkpoint. Override if cleanup is supported."""
        raise NotImplementedError


class InMemoryCheckpointStrategy(CheckpointStrategy):
    """Reference implementation that keeps checkpoints in a dict (handy for tests)."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []

    def save(self, state: Dict[str, Any], tag: str) -> str:
        if tag not in self._store:
            self._order.append(tag)
        self._store[tag] = dict(state)
        return tag

    def load(self, tag: str) -> Dict[str, Any]:
        return dict(self._store[tag])

    def list_tags(self) -> List[str]:
        return list(self._order)

    def delete(self, tag: str) -> None:
        self._store.pop(tag, None)
        if tag in self._order:
            self._order.remove(tag)


__all__ = ["CheckpointStrategy", "InMemoryCheckpointStrategy"]
