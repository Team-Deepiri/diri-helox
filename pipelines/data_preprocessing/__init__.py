"""Backward-compatible preprocessing module namespace.

This package is a thin shim over ``deepiri_dataset_processor`` so old imports like
``from pipelines.data_preprocessing import DataLoadingStage`` keep working.

When ``deepiri_dataset_processor`` is not installed (e.g. CI without the sibling
repo), imports are skipped so pytest can collect ``test_quality`` after
``importorskip`` without loading this package's implementation modules.
"""

from __future__ import annotations

import importlib.util

__all__ = [
    "DEFAULT_MIN_LABEL_ID",
    "DEFAULT_MAX_LABEL_ID",
    "PreprocessingStage",
    "ProcessedData",
    "StageResult",
    "ValidationResult",
    "DataPreprocessor",
    "DatasetPipeline",
    "PipelineOrchestrator",
    "DataLoadingStage",
    "DataCleaningStage",
    "DataValidationStage",
    "DataRoutingStage",
    "LabelValidationStage",
    "DataTransformationStage",
    "QualityChecker",
    "QualityConfig",
    "QualityMetric",
    "QualityReport",
    "QualityCheckStage",
    "check_data_quality",
]

if importlib.util.find_spec("deepiri_dataset_processor") is not None:
    from .base import (  # noqa: F401
        DEFAULT_MAX_LABEL_ID,
        DEFAULT_MIN_LABEL_ID,
        DataPreprocessor,
        PreprocessingStage,
        ProcessedData,
        StageResult,
        ValidationResult,
    )
    from .orchestrator import DatasetPipeline, PipelineOrchestrator  # noqa: F401
    from .stages import (  # noqa: F401
        DataCleaningStage,
        DataLoadingStage,
        DataRoutingStage,
        DataTransformationStage,
        DataValidationStage,
        LabelValidationStage,
    )
    from .quality import (  # noqa: F401
        QualityCheckStage,
        QualityChecker,
        QualityConfig,
        QualityMetric,
        QualityReport,
        check_data_quality,
    )
