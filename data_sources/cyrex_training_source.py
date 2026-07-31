"""
Cyrex-owned training sources for the Helox pipeline.

These classes are intentionally thin wrappers around the generic stream and
Postgres primitives. The primitive names stay available for reusable plumbing,
while these aliases make the production Cyrex -> Helox data lane explicit in
configs, docs, and tests.
"""

from __future__ import annotations

from typing import Any, Dict

from .base import DataSourceConfig
from .postgres_source import DEFAULT_TABLE, PostgresDataSource
from .stream_source import (
    MIN_QUALITY_SCORE,
    RAW_STREAM,
    STRUCTURED_STREAM,
    StreamDataSource,
)

CYREX_REALTIME_PIPELINE_PRODUCER = "cyrex_realtime_pipeline"
CYREX_HELOX_CONTRACT = "cyrex.helox-training.v1"


def _with_default_params(
    config: DataSourceConfig,
    defaults: Dict[str, Any],
) -> DataSourceConfig:
    params = {**defaults, **config.params}
    return DataSourceConfig(
        source_type=config.source_type,
        name=config.name,
        params=params,
        weight=config.weight,
    )


class CyrexTrainingStreamSource(StreamDataSource):
    """Live Redis source for Cyrex-produced Helox training samples."""

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(
            _with_default_params(
                config,
                {
                    "mode": "subscribe",
                    "stream_type": "both",
                    "min_quality": MIN_QUALITY_SCORE,
                },
            )
        )

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "source_type": self.config.source_type,
                "producer": CYREX_REALTIME_PIPELINE_PRODUCER,
                "contract": CYREX_HELOX_CONTRACT,
                "raw_stream": RAW_STREAM,
                "structured_stream": STRUCTURED_STREAM,
            }
        )
        return info


class CyrexTrainingPostgresSource(PostgresDataSource):
    """Durable replay source for Cyrex-owned Helox training samples."""

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(
            _with_default_params(
                config,
                {
                    "table": DEFAULT_TABLE,
                    "producer": CYREX_REALTIME_PIPELINE_PRODUCER,
                    "min_quality": MIN_QUALITY_SCORE,
                },
            )
        )

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update(
            {
                "source_type": self.config.source_type,
                "producer": self._producer,
                "contract": CYREX_HELOX_CONTRACT,
                "durable_table": DEFAULT_TABLE,
            }
        )
        return info
