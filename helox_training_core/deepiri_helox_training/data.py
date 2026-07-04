"""Data + batching contracts.

``DataPipeline`` standardizes how a training loop obtains train/val/test iterables, so
the trainer never needs to know how data is loaded. ``BatchPolicy`` standardizes how raw
samples are collated into batches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Iterable, List, Optional, Sequence, TypeVar, runtime_checkable

try:  # Protocol is stdlib on 3.8+, but guard for very old runtimes.
    from typing import Protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol  # type: ignore

TBatch = TypeVar("TBatch")
TSample = TypeVar("TSample")


@runtime_checkable
class DataPipeline(Protocol[TBatch]):
    """Contract for supplying batched data to a trainer.

    Implementations return any iterable of batches (e.g. a torch ``DataLoader``). A
    ``None`` validation/test loader means that split is unavailable.
    """

    def train_loader(self) -> Iterable[TBatch]: ...

    def val_loader(self) -> Optional[Iterable[TBatch]]: ...

    def test_loader(self) -> Optional[Iterable[TBatch]]: ...


class BatchPolicy(ABC, Generic[TSample, TBatch]):
    """Contract for turning a list of samples into a single batch."""

    @abstractmethod
    def collate(self, samples: Sequence[TSample]) -> TBatch:
        """Combine ``samples`` into one batch."""
        raise NotImplementedError


class StaticDataPipeline(Generic[TBatch]):
    """Trivial :class:`DataPipeline` wrapping already-built iterables.

    Useful for tests and for adapting pre-constructed loaders into the contract.
    """

    def __init__(
        self,
        train: Iterable[TBatch],
        val: Optional[Iterable[TBatch]] = None,
        test: Optional[Iterable[TBatch]] = None,
    ) -> None:
        self._train = train
        self._val = val
        self._test = test

    def train_loader(self) -> Iterable[TBatch]:
        return self._train

    def val_loader(self) -> Optional[Iterable[TBatch]]:
        return self._val

    def test_loader(self) -> Optional[Iterable[TBatch]]:
        return self._test


__all__ = ["DataPipeline", "BatchPolicy", "StaticDataPipeline"]
