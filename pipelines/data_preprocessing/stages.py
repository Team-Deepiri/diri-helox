"""Backward-compatible re-export shim for preprocessing stages.

The canonical implementations live in ``deepiri_dataset_processor``.
This file exists to keep legacy imports from ``pipelines.data_preprocessing`` working.
"""

from deepiri_dataset_processor.pipeline.stages import (  # noqa: F401
    DataCleaningStage,
    DataLoadingStage,
    DataRoutingStage,
    DataTransformationStage,
    DataValidationStage,
    LabelValidationStage,
)

__all__ = [
    "DataLoadingStage",
    "DataCleaningStage",
    "DataValidationStage",
    "DataRoutingStage",
    "LabelValidationStage",
    "DataTransformationStage",
]

