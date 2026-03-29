"""
CompositeDataSource: combines multiple data sources with configurable weights.
Performs weighted sampling across all sources and shuffles the result.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, Iterator, List

from .base import DataSample, DataSource, DataSourceConfig


class CompositeDataSource(DataSource):
    """
    Mixes multiple DataSource instances by weight.

    Config params:
        sources (list[dict]): list of {source_type, name, params, weight} dicts
        seed    (int | None): random seed for reproducible shuffling (default: 42)

    The weight of each source determines its proportional contribution.
    Example: two sources with weights 0.7 and 0.3 means the first contributes
    70% of the final samples.
    """

    def __init__(self, config: DataSourceConfig, sources: List[DataSource]) -> None:
        super().__init__(config)
        self._sources = sources
        self._seed = config.params.get("seed", 42)

    def load(self) -> List[DataSample]:
        if not self._sources:
            return []

        total_weight = sum(s.config.weight for s in self._sources)
        all_per_source: List[List[DataSample]] = []
        total_count = 0

        for source in self._sources:
            samples = source.load()
            all_per_source.append(samples)
            total_count += len(samples)

        if total_count == 0:
            return []

        # Weighted interleaving: take proportional slices
        combined: List[DataSample] = []
        for source, samples in zip(self._sources, all_per_source):
            proportion = source.config.weight / total_weight
            take = math.ceil(proportion * total_count)
            combined.extend(samples[:take])

        rng = random.Random(self._seed)
        rng.shuffle(combined)
        return combined

    def stream(self) -> Iterator[DataSample]:
        yield from self.load()

    def get_info(self) -> Dict[str, Any]:
        return {
            "source_type": "composite",
            "name": self.name,
            "num_sources": len(self._sources),
            "sources": [
                {"name": s.name, "type": s.source_type, "weight": s.config.weight}
                for s in self._sources
            ],
        }
