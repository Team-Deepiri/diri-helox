"""
Base classes for the data source abstraction layer.
All data sources implement the DataSource ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class DataSample:
    """Universal data sample across all sources."""

    text: str
    label: Optional[int] = None
    label_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"


@dataclass
class DataSourceConfig:
    """Configuration for a single data source."""

    source_type: str  # "static", "stream", "synthetic", "self_feedback", "composite"
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0


class DataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, config: DataSourceConfig) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> List[DataSample]:
        """Load all data samples at once."""
        pass

    @abstractmethod
    def stream(self) -> Iterator[DataSample]:
        """Yield data samples one at a time."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return metadata about this source (count, categories, etc.)."""
        pass

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def source_type(self) -> str:
        return self.config.source_type
