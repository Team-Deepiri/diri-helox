"""
Data source abstraction layer for the Helox training pipeline.

The pipeline is designed around live data produced by the language intelligence
service, which processes documents → stores in Postgres/Milvus → publishes to
Redis streams. Subscribe to those streams here.

Primary sources (live data):
  - StreamDataSource   : Redis stream subscription (pipeline.helox-training.*)
  - PostgresDataSource : Structured data from the language intelligence Postgres tables
  - MilvusDataSource   : Vector-indexed documents from Milvus collections

Supplementary sources (when live data isn't available):
  - SyntheticDataSource  : Template-based synthetic generation
  - SelfFeedbackDataSource: High-confidence model predictions as training signal
  - CompositeDataSource   : Weighted mix of any sources above

Use create_data_sources_from_config() to instantiate from a config dict.
"""

from .base import DataSample, DataSource, DataSourceConfig
from .stream_source import StreamDataSource
from .postgres_source import PostgresDataSource
from .milvus_source import MilvusDataSource
from .synthetic_source import SyntheticDataSource
from .self_feedback_source import SelfFeedbackDataSource
from .composite_source import CompositeDataSource
from .factory import create_data_source, create_data_sources_from_config

__all__ = [
    # Contracts
    "DataSample",
    "DataSource",
    "DataSourceConfig",
    # Live data sources
    "StreamDataSource",
    "PostgresDataSource",
    "MilvusDataSource",
    # Supplementary sources
    "SyntheticDataSource",
    "SelfFeedbackDataSource",
    "CompositeDataSource",
    # Factory
    "create_data_source",
    "create_data_sources_from_config",
]
