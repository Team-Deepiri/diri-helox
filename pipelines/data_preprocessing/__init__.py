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

__all__ = [
    # Base classes
    "PreprocessingStage",
    "ProcessedData",
    "StageResult",
    "ValidationResult",
]

