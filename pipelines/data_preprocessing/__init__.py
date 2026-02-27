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

from .quality import (
    QualityMetric,
    QualityReport,
    QualityConfig,
    QualityChecker,
    QualityCheckStage,
    StatisticalValidator,
    check_data_quality,
)

# Import orchestrator (may fail if networkx is not installed)
try:
    from .orchestrator import (
        PipelineOrchestrator,
    )
except ImportError:
    # networkx might not be installed
    PipelineOrchestrator = None  # type: ignore

from .stages import (
    DataLoadingStage,
    DataCleaningStage,
    DataValidationStage,
    DataRoutingStage,
    LabelValidationStage,
    DataTransformationStage,
)

__all__ = [
    # Base classes
    "PreprocessingStage",
    "ProcessedData",
    "StageResult",
    "ValidationResult",
    # Quality framework
    "QualityMetric",
    "QualityReport",
    "QualityConfig",
    "QualityChecker",
    "QualityCheckStage",
    "StatisticalValidator",
    "check_data_quality",
    # Stages
    "DataLoadingStage",
    "DataCleaningStage",
    "DataValidationStage",
    "DataRoutingStage",
    "LabelValidationStage",
    "DataTransformationStage",
]

# Conditionally add PipelineOrchestrator if it was imported successfully
if PipelineOrchestrator is not None:
    __all__.append("PipelineOrchestrator")

