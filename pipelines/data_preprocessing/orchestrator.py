"""Backward-compatible re-export shim for pipeline orchestration.

The canonical implementations live in ``deepiri_dataset_processor.pipeline``.
"""

from deepiri_dataset_processor.pipeline import (  # noqa: F401
    DatasetPipeline,
    PipelineOrchestrator,
)

__all__ = ["DatasetPipeline", "PipelineOrchestrator"]
