"""
Data preprocessing pipeline framework.

This module provides a production-grade data preprocessing framework with:
- Pipeline orchestration
- Stage-based processing
- Data quality validation
- Schema management
- Data versioning
"""

from .base import (
    PreprocessingStage,
    ProcessedData,
    StageResult,
    ValidationResult,
)

from .orchestrator import PipelineOrchestrator

from .stages import (
    DataLoadingStage,
    DataCleaningStage,
    DataValidationStage,
    DataRoutingStage,
    LabelValidationStage,
    DataTransformationStage,
)

from .quality import QualityChecker, QualityReport, QualityMetric

__all__ = [
    # Base classes
    "PreprocessingStage",
    "ProcessedData",
    "StageResult",
    "ValidationResult",
    # Orchestrator
    "PipelineOrchestrator",
    # Stage implementations
    "DataLoadingStage",
    "DataCleaningStage",
    "DataValidationStage",
    "DataRoutingStage",
    "LabelValidationStage",
    "DataTransformationStage",
    # Quality framework
    "QualityChecker",
    "QualityReport",
    "QualityMetric",
]