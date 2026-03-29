"""
Data source abstraction layer for the Helox training pipeline.

All data sources implement the DataSource ABC and return List[DataSample].
Use create_data_sources_from_config() to instantiate sources from a config dict.
"""
from .base import DataSample, DataSource, DataSourceConfig
from .static_source import StaticDataSource
from .stream_source import StreamDataSource
from .synthetic_source import SyntheticDataSource
from .self_feedback_source import SelfFeedbackDataSource
from .composite_source import CompositeDataSource
from .factory import create_data_source, create_data_sources_from_config

__all__ = [
    "DataSample",
    "DataSource",
    "DataSourceConfig",
    "StaticDataSource",
    "StreamDataSource",
    "SyntheticDataSource",
    "SelfFeedbackDataSource",
    "CompositeDataSource",
    "create_data_source",
    "create_data_sources_from_config",
]
