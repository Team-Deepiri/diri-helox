"""Backward-compatible preprocessing module namespace.

This package is a thin shim over ``deepiri_dataset_processor`` so old imports like
``from pipelines.data_preprocessing import DataLoadingStage`` keep working.
"""

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

