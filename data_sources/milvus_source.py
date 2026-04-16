"""
MilvusDataSource: loads training data from Milvus vector collections.

The language intelligence service indexes processed documents into Milvus.
This source queries a collection and yields DataSamples — useful for
similarity-based data selection or retrieval-augmented training.

Config params:
    host          (str):  Milvus host (default: "localhost")
    port          (int):  Milvus port (default: 19530)
    collection    (str):  Collection name (default: "training_data")
    text_field    (str):  Field containing raw text (default: "text")
    label_field   (str):  Field containing label/category (default: "category")
    quality_field (str):  Field containing quality score (default: "quality_score")
    min_quality   (float): Minimum quality score filter (default: 0.4)
    max_samples   (int):  Max records to load (default: 1000)
    query_filter  (str):  Optional Milvus boolean expression filter,
                          e.g. "quality_score >= 0.5 and status == 'approved'"
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

from .base import DataSample, DataSource, DataSourceConfig


class MilvusDataSource(DataSource):
    """
    Reads training samples from a Milvus vector collection populated by the
    language intelligence service.
    """

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        self._host: str = config.params.get("host", os.environ.get("MILVUS_HOST", "localhost"))
        self._port: int = int(config.params.get("port", 19530))
        self._collection_name: str = config.params.get("collection", "training_data")
        self._text_field: str = config.params.get("text_field", "text")
        self._label_field: str = config.params.get("label_field", "category")
        self._quality_field: str = config.params.get("quality_field", "quality_score")
        self._min_quality: float = float(config.params.get("min_quality", 0.4))
        self._max_samples: int = int(config.params.get("max_samples", 1000))
        self._query_filter: Optional[str] = config.params.get("query_filter", None)

    def _build_filter(self) -> str:
        if self._query_filter:
            return self._query_filter
        return f"{self._quality_field} >= {self._min_quality}"

    def _entity_to_sample(self, entity: Any) -> Optional[DataSample]:
        text = entity.get(self._text_field, "")
        if not text or len(str(text).strip()) < 3:
            return None
        quality = float(entity.get(self._quality_field, 1.0))
        if quality < self._min_quality:
            return None
        return DataSample(
            text=str(text).strip(),
            label=None,
            label_name=(
                str(entity.get(self._label_field)) if entity.get(self._label_field) else None
            ),
            metadata={
                "source_stream": "milvus",
                "quality_score": quality,
                "collection": self._collection_name,
            },
            source=f"milvus:{self.name}",
        )

    def load(self) -> List[DataSample]:
        try:
            from pymilvus import Collection, connections
        except ImportError:
            raise ImportError("MilvusDataSource requires pymilvus: pip install pymilvus")

        samples: List[DataSample] = []
        try:
            connections.connect(host=self._host, port=self._port)
            collection = Collection(self._collection_name)
            collection.load()

            output_fields = [self._text_field, self._label_field, self._quality_field]
            results = collection.query(
                expr=self._build_filter(),
                output_fields=output_fields,
                limit=self._max_samples,
            )
            for entity in results:
                sample = self._entity_to_sample(entity)
                if sample:
                    samples.append(sample)
        except Exception as exc:
            print(f"  Warning: MilvusDataSource failed ({exc}) — returning empty")
        return samples

    def stream(self) -> Iterator[DataSample]:
        """Stream entities from Milvus in batches to avoid loading all at once."""
        try:
            from pymilvus import Collection, connections
        except ImportError:
            raise ImportError("MilvusDataSource requires pymilvus: pip install pymilvus")

        batch_size = 100
        offset = 0
        try:
            connections.connect(host=self._host, port=self._port)
            collection = Collection(self._collection_name)
            collection.load()
            output_fields = [self._text_field, self._label_field, self._quality_field]

            while offset < self._max_samples:
                fetch = min(batch_size, self._max_samples - offset)
                results = collection.query(
                    expr=self._build_filter(),
                    output_fields=output_fields,
                    limit=fetch,
                    offset=offset,
                )
                if not results:
                    break
                for entity in results:
                    sample = self._entity_to_sample(entity)
                    if sample:
                        yield sample
                offset += len(results)
                if len(results) < fetch:
                    break
        except Exception as exc:
            print(f"  Warning: MilvusDataSource stream failed ({exc})")

    def get_info(self) -> Dict[str, Any]:
        return {
            "source_type": "milvus",
            "name": self.name,
            "host": self._host,
            "port": self._port,
            "collection": self._collection_name,
            "min_quality": self._min_quality,
            "max_samples": self._max_samples,
        }
