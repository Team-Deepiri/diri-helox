"""Metric primitives: per-step results and batch-size-weighted aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StepResult:
    """Outcome of a single train/eval step.

    ``loss`` is the scalar optimized this step. ``batch_size`` weights the contribution
    when aggregating across a loader. ``metrics`` holds any extra scalars to track.
    """

    loss: float
    batch_size: int = 1
    metrics: Dict[str, float] = field(default_factory=dict)


class MetricAggregator:
    """Accumulates :class:`StepResult` values into batch-size-weighted means."""

    def __init__(self) -> None:
        self._weighted_sums: Dict[str, float] = {}
        self._total_weight: float = 0.0

    def update(self, result: StepResult) -> None:
        weight = max(result.batch_size, 1)
        self._total_weight += weight
        self._weighted_sums["loss"] = self._weighted_sums.get("loss", 0.0) + result.loss * weight
        for name, value in result.metrics.items():
            self._weighted_sums[name] = self._weighted_sums.get(name, 0.0) + value * weight

    def compute(self) -> Dict[str, float]:
        if self._total_weight == 0:
            return {}
        return {name: total / self._total_weight for name, total in self._weighted_sums.items()}
