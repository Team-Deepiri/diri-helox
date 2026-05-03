"""
Factory function for creating DataSource instances from config dicts.
"""

from __future__ import annotations

from typing import Dict

from .base import DataSource, DataSourceConfig
from .stream_source import StreamDataSource
from .postgres_source import PostgresDataSource
from .milvus_source import MilvusDataSource
from .synthetic_source import SyntheticDataSource
from .self_feedback_source import SelfFeedbackDataSource
from .composite_source import CompositeDataSource
from .static_source import StaticDataSource

_REGISTRY: Dict[str, type[DataSource]] = {
    # Live data sources (primary)
    "stream": StreamDataSource,
    "postgres": PostgresDataSource,
    "milvus": MilvusDataSource,
    # Supplementary / fallback
    "synthetic": SyntheticDataSource,
    "self_feedback": SelfFeedbackDataSource,
    # Legacy file-based (for local testing / offline runs)
    "static": StaticDataSource,
}


def create_data_source(config: DataSourceConfig, child_sources=None) -> DataSource:
    """
    Create a DataSource from a DataSourceConfig.

    For composite sources, pass child_sources as a list of DataSource objects.
    All other types are constructed directly from config.params.
    """
    source_type = config.source_type

    if source_type == "composite":
        if child_sources is None:
            raise ValueError("CompositeDataSource requires child_sources to be provided")
        return CompositeDataSource(config, child_sources)

    cls = _REGISTRY.get(source_type)
    if cls is None:
        raise ValueError(
            f"Unknown source_type '{source_type}'. "
            f"Available: {list(_REGISTRY.keys()) + ['composite']}"
        )
    return cls(config)


def create_data_sources_from_config(source_configs: list) -> list:
    """
    Build a list of DataSource instances from a list of config dicts.
    Handles nested composite sources by recursively creating children.
    """
    sources = []
    for cfg in source_configs:
        config = DataSourceConfig(
            source_type=cfg["source_type"],
            name=cfg.get("name", cfg["source_type"]),
            params=cfg.get("params", {}),
            weight=cfg.get("weight", 1.0),
        )

        source: DataSource
        if config.source_type == "composite":
            child_cfgs = cfg.get("sources", [])
            children = create_data_sources_from_config(child_cfgs)
            source = CompositeDataSource(config, children)
        else:
            source = create_data_source(config)

        sources.append(source)
    return sources
