"""
PostgresDataSource: loads cleaned/structured training data from PostgreSQL.

The language intelligence service processes documents and stores them in Postgres
after cleaning. This source reads from those tables and yields DataSamples.

Config params:
    dsn           (str):  PostgreSQL DSN, e.g. "postgresql://user:pass@host:5432/db"
                          Falls back to POSTGRES_DSN env var if not set.
    table         (str):  Table to read from (default: "training_samples")
    label_column  (str):  Column containing the label/category (default: "category")
    text_column   (str):  Column containing the text (default: "text")
    min_quality   (float): Minimum quality score (default: 0.4)
    max_samples   (int):  Cap on rows to load (default: None = no cap)
    where         (str):  Optional extra WHERE clause, e.g. "status = 'approved'"
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

from .base import DataSample, DataSource, DataSourceConfig

DEFAULT_TABLE = "training_samples"


class PostgresDataSource(DataSource):
    """
    Reads training samples from a PostgreSQL table populated by the
    language intelligence service pipeline.
    """

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        self._dsn: str = config.params.get(
            "dsn", os.environ.get("POSTGRES_DSN", "postgresql://localhost:5432/deepiri")
        )
        self._table: str = config.params.get("table", DEFAULT_TABLE)
        self._text_col: str = config.params.get("text_column", "text")
        self._label_col: str = config.params.get("label_column", "category")
        self._min_quality: float = float(config.params.get("min_quality", 0.4))
        self._max_samples: Optional[int] = config.params.get("max_samples", None)
        self._where: Optional[str] = config.params.get("where", None)

    def _build_query(self) -> str:
        where_parts = [f"quality_score >= {self._min_quality}"]
        if self._where:
            where_parts.append(f"({self._where})")
        where_clause = " AND ".join(where_parts)
        limit_clause = f" LIMIT {self._max_samples}" if self._max_samples else ""
        return (
            f"SELECT {self._text_col}, {self._label_col}, quality_score, id "
            f"FROM {self._table} WHERE {where_clause}{limit_clause}"
        )

    def _row_to_sample(self, row: Any) -> Optional[DataSample]:
        text = row[0] if row[0] else ""
        if not text or len(text.strip()) < 3:
            return None
        return DataSample(
            text=text.strip(),
            label=None,
            label_name=str(row[1]) if row[1] else None,
            metadata={
                "source_stream": "postgres",
                "quality_score": float(row[2]) if row[2] is not None else 1.0,
                "id": str(row[3]) if row[3] is not None else None,
            },
            source=f"postgres:{self.name}",
        )

    def load(self) -> List[DataSample]:
        try:
            import psycopg2
        except ImportError:
            raise ImportError("PostgresDataSource requires psycopg2: pip install psycopg2-binary")

        samples: List[DataSample] = []
        try:
            conn = psycopg2.connect(self._dsn)
            with conn.cursor() as cur:
                cur.execute(self._build_query())
                for row in cur:
                    sample = self._row_to_sample(row)
                    if sample:
                        samples.append(sample)
        except Exception as exc:
            print(f"  Warning: PostgresDataSource failed ({exc}) — returning empty")
        finally:
            try:
                conn.close()  # type: ignore[possibly-undefined]
            except Exception:
                pass
        return samples

    def stream(self) -> Iterator[DataSample]:
        """Stream rows directly from Postgres cursor without buffering all in memory."""
        try:
            import psycopg2
        except ImportError:
            raise ImportError("PostgresDataSource requires psycopg2: pip install psycopg2-binary")

        try:
            conn = psycopg2.connect(self._dsn)
            with conn.cursor("helox_stream_cursor") as cur:  # server-side cursor
                cur.execute(self._build_query())
                for row in cur:
                    sample = self._row_to_sample(row)
                    if sample:
                        yield sample
        except Exception as exc:
            print(f"  Warning: PostgresDataSource stream failed ({exc})")
        finally:
            try:
                conn.close()  # type: ignore[possibly-undefined]
            except Exception:
                pass

    def get_info(self) -> Dict[str, Any]:
        return {
            "source_type": "postgres",
            "name": self.name,
            "table": self._table,
            "dsn": self._dsn.split("@")[-1] if "@" in self._dsn else self._dsn,
            "min_quality": self._min_quality,
            "max_samples": self._max_samples,
        }
