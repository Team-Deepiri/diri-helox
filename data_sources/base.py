"""
Base classes for the data source abstraction layer.
All data sources implement the DataSource ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass(slots=True)
class DataSample:
    """
    Universal data sample across all sources.

    Uses __slots__ (via slots=True, Python 3.10+) to reduce per-instance
    memory overhead — important when thousands of samples are held in memory
    during preprocessing and splitting.
    """

    text: str
    label: Optional[int] = None
    label_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"


@dataclass(slots=True)
class DataSourceConfig:
    """Configuration for a single data source."""

    source_type: str  # "stream", "postgres", "milvus", "synthetic", "self_feedback", "composite"
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


class DataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, config: DataSourceConfig) -> None:
        self.config = config

    @abstractmethod
    def stream(self) -> Iterator[DataSample]:
        """
        Yield data samples one at a time without buffering all in memory.

        This is the primary method. Subclasses implement a true lazy generator
        (Redis XREAD, Postgres server-side cursor, Milvus paginated query, etc.).
        """

    def load(self) -> List[DataSample]:
        """
        Load all samples into a list.

        Default collects from stream() so subclasses only need to implement
        stream() to get both interfaces for free. Override when bulk loading
        is meaningfully faster than streaming (e.g. small static files).
        """
        return list(self.stream())

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return metadata about this source (count, categories, etc.)."""

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def source_type(self) -> str:
        return self.config.source_type
