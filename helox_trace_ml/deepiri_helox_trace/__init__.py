"""Reproducible trace ML pipeline: JSON ingest → features → CSV splits → optional sklearn adapter."""

from .features import FEATURE_COLUMNS, operator_rows_to_arrays, row_to_features
from .ingest import (
    default_data_roots,
    list_trace_json_paths,
    load_pytorch_trace_json,
    merge_operator_stats,
)
from .pipeline import TraceDatasetPipeline
from .training import fit_trace_runtime_adapter, load_trace_split_csv

__all__ = [
    "FEATURE_COLUMNS",
    "TraceDatasetPipeline",
    "default_data_roots",
    "fit_trace_runtime_adapter",
    "list_trace_json_paths",
    "load_pytorch_trace_json",
    "load_trace_split_csv",
    "merge_operator_stats",
    "operator_rows_to_arrays",
    "row_to_features",
]
