"""Backward-compatible re-export shim for data quality checks.

The canonical implementations live in ``deepiri_dataset_processor.quality``.
This file exists so legacy imports from ``pipelines.data_preprocessing.quality`` keep working.
"""

from deepiri_dataset_processor.quality import (  # noqa: F401
    QualityCheckStage,
    QualityChecker,
    QualityConfig,
    QualityMetric,
    QualityReport,
    check_data_quality,
)

__all__ = [
    "QualityCheckStage",
    "QualityChecker",
    "QualityConfig",
    "QualityMetric",
    "QualityReport",
    "check_data_quality",
]

