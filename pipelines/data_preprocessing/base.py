"""Backward-compatible re-export shim for pipeline base types.

The canonical implementations live in ``deepiri_dataset_processor.pipeline.base``.
This module also provides a minimal ``DataPreprocessor`` compatibility class for
older training pipelines that import it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from deepiri_dataset_processor.pipeline.base import (  # noqa: F401
    DEFAULT_MAX_LABEL_ID,
    DEFAULT_MIN_LABEL_ID,
    PreprocessingStage,
    ProcessedData,
    StageResult,
    ValidationResult,
)

from deepiri_dataset_processor.pipeline.orchestrator import PipelineOrchestrator
from deepiri_dataset_processor.pipeline.stages import (
    DataCleaningStage,
    DataLoadingStage,
    DataRoutingStage,
    DataTransformationStage,
    DataValidationStage,
    LabelValidationStage,
)


class DataPreprocessor:
    """Legacy facade retained for backward compatibility.

    New code should build pipelines directly from ``deepiri_dataset_processor.pipeline``.
    """

    def __init__(
        self,
        *,
        label_mapping: Optional[Dict[str, int]] = None,
        stages: Optional[List[PreprocessingStage]] = None,
    ):
        if stages is None:
            stages = [
                DataLoadingStage(),
                DataCleaningStage(),
                DataValidationStage(),
                DataRoutingStage(config={"label_mapping": label_mapping or {}}),
                LabelValidationStage(),
                DataTransformationStage(),
            ]

        self.orchestrator = PipelineOrchestrator(stages=stages)
        self.orchestrator.build_dag()

    def preprocess(self, data: Any) -> StageResult:
        """Run the default preprocessing pipeline on input data."""
        return self.orchestrator.execute(initial_data=data)


__all__ = [
    # base re-exports
    "DEFAULT_MIN_LABEL_ID",
    "DEFAULT_MAX_LABEL_ID",
    "PreprocessingStage",
    "ProcessedData",
    "StageResult",
    "ValidationResult",
    # legacy facade
    "DataPreprocessor",
]

