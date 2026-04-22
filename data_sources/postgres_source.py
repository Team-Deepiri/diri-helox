"""
PostgresDataSource: loads training data from the durable Postgres table.

This source is intentionally schema-aware and defensive:
  - Uses parameterized SQL for all value filters.
  - Validates SQL identifiers (table/column names) early.
  - Adapts when optional columns are missing (schema drift).
  - Keeps the "actual" durable table path (no mirror concept in Helox).

Config params:
    dsn              (str): PostgreSQL DSN. Falls back to POSTGRES_DSN env var.
    table            (str): Durable table (default: "cyrex.helox_training_samples")
    label_column     (str): Category/label column (default: "category")
    text_column      (str): Training text column (default: "text")
    min_quality      (float): Minimum quality threshold (default: 0.4)
    stream_type      (str): Optional filter: "raw" or "structured"
    producer         (str): Optional producer filter (e.g. "language_intelligence")
    max_samples      (int): Optional LIMIT cap (must be > 0)
    where            (str): Legacy raw SQL fragment (ignored unless allow_unsafe_where=true)
    allow_unsafe_where (bool): If true, appends raw WHERE fragment directly (default: false)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterator, List, Optional

from .base import DataSample, DataSource, DataSourceConfig

DEFAULT_TABLE = "cyrex.helox_training_samples"

# SQL identifiers (table/column names) cannot be parameterized — they must be
# interpolated directly into the query string. This regex is the guard: only
# plain identifiers (letters, digits, underscores, no quotes or special chars)
# are allowed, which prevents SQL injection through config-supplied names.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PostgresDataSource(DataSource):
    """
    Reads training samples from a PostgreSQL table populated by the
    language intelligence service pipeline.
    """

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        self._dsn: str = config.params.get(
            "dsn", os.environ.get("POSTGRES_DSN", "postgresql://localhost:5434/cyrex_db")
        )
        self._table: str = config.params.get("table", DEFAULT_TABLE)
        self._text_col: str = config.params.get("text_column", "text")
        self._label_col: str = config.params.get("label_column", "category")
        self._min_quality: float = float(config.params.get("min_quality", 0.4))
        self._stream_type: Optional[str] = config.params.get("stream_type", None)
        self._producer: Optional[str] = config.params.get("producer", None)
        self._max_samples: Optional[int] = config.params.get("max_samples", None)
        self._where: Optional[str] = config.params.get("where", None)
        # allow_unsafe_where is an opt-in escape hatch for trusted operators who need
        # complex WHERE logic that can't be expressed with the parameterized filters
        # above (e.g. date ranges, JSON operators). Off by default; callers must
        # explicitly set it to true and are responsible for sanitizing the fragment.
        self._allow_unsafe_where: bool = bool(config.params.get("allow_unsafe_where", False))

        self._validate_config()

    @staticmethod
    def _validate_identifier(identifier: str, kind: str) -> None:
        """Reject any table/column/schema name that contains SQL-special chars.

        Because identifiers are interpolated (not parameterized) in the SELECT
        and FROM clauses, we must validate them before use.
        """
        if not _IDENTIFIER_RE.match(identifier):
            raise ValueError(f"Invalid {kind} identifier: {identifier!r}")

    def _validate_config(self) -> None:
        if "." in self._table:
            parts = self._table.split(".")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid table format {self._table!r}. Expected 'schema.table' or 'table'."
                )
            self._validate_identifier(parts[0], "schema")
            self._validate_identifier(parts[1], "table")
        else:
            self._validate_identifier(self._table, "table")

        self._validate_identifier(self._text_col, "column")
        self._validate_identifier(self._label_col, "column")

        if self._max_samples is not None and int(self._max_samples) <= 0:
            raise ValueError("max_samples must be > 0 when provided")

    def _split_table(self) -> tuple[str, str]:
        if "." not in self._table:
            return "public", self._table
        schema, table = self._table.split(".")
        return schema, table

    def _resolve_table_columns(self, conn: Any) -> set[str]:
        schema, table = self._split_table()
        query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
        """
        with conn.cursor() as cur:
            cur.execute(query, (schema, table))
            return {row[0] for row in cur.fetchall()}

    def _build_query_and_params(self, available_columns: set[str]) -> tuple[str, tuple[Any, ...]]:
        required = [self._text_col, self._label_col]
        missing_required = [col for col in required if col not in available_columns]
        if missing_required:
            raise RuntimeError(
                f"Postgres source missing required columns {missing_required} in table {self._table}"
            )

        select_cols = [self._text_col, self._label_col]
        optional_cols = ["quality_score", "record_id", "stream_type", "producer"]
        for col in optional_cols:
            if col in available_columns:
                select_cols.append(col)

        where_parts: List[str] = []
        params: List[Any] = []

        if "quality_score" in available_columns:
            where_parts.append("quality_score >= %s")
            params.append(self._min_quality)

        if self._stream_type:
            if "stream_type" in available_columns:
                where_parts.append("stream_type = %s")
                params.append(self._stream_type)
            else:
                print("  Warning: stream_type filter requested but column missing; ignoring filter")

        if self._producer:
            if "producer" in available_columns:
                where_parts.append("producer = %s")
                params.append(self._producer)
            else:
                print("  Warning: producer filter requested but column missing; ignoring filter")

        if self._where:
            if self._allow_unsafe_where:
                # Escape hatch for trusted operators only.
                where_parts.append(f"({self._where})")
                print("  Warning: using allow_unsafe_where raw SQL fragment")
            else:
                print(
                    "  Warning: ignoring raw 'where' filter for safety (set allow_unsafe_where=true)"
                )

        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""

        order_clause = ""
        if "created_at" in available_columns:
            order_clause = " ORDER BY created_at DESC"
        elif "record_id" in available_columns:
            order_clause = " ORDER BY record_id DESC"

        limit_clause = f" LIMIT {int(self._max_samples)}" if self._max_samples else ""

        query = f"SELECT {', '.join(select_cols)} FROM {self._table}{where_clause}{order_clause}{limit_clause}"
        return query, tuple(params)

    def _build_query(self) -> str:
        """
        Backward-compatible debug helper used in unit tests.
        Uses a representative schema shape; execution always uses _build_query_and_params().
        """
        default_columns = {
            self._text_col,
            self._label_col,
            "quality_score",
            "record_id",
            "stream_type",
            "producer",
            "created_at",
        }
        query, _ = self._build_query_and_params(default_columns)
        return query

    @staticmethod
    def _row_value(row_map: Dict[str, Any], key: str, default: Any = None) -> Any:
        return row_map[key] if key in row_map else default

    def _row_to_sample(self, row_map: Dict[str, Any]) -> Optional[DataSample]:
        text_val = self._row_value(row_map, self._text_col, "")
        text = str(text_val) if text_val is not None else ""
        if not text or len(text.strip()) < 3:
            return None
        return DataSample(
            text=text.strip(),
            label=None,
            label_name=(
                str(self._row_value(row_map, self._label_col))
                if self._row_value(row_map, self._label_col) is not None
                else None
            ),
            metadata={
                "source_stream": "postgres",
                "quality_score": (
                    float(self._row_value(row_map, "quality_score"))
                    if self._row_value(row_map, "quality_score") is not None
                    else 1.0
                ),
                "record_id": (
                    str(self._row_value(row_map, "record_id"))
                    if self._row_value(row_map, "record_id") is not None
                    else None
                ),
                "stream_type": (
                    str(self._row_value(row_map, "stream_type"))
                    if self._row_value(row_map, "stream_type") is not None
                    else None
                ),
                "producer": (
                    str(self._row_value(row_map, "producer"))
                    if self._row_value(row_map, "producer") is not None
                    else None
                ),
            },
            source=f"postgres:{self.name}",
        )

    def load(self) -> List[DataSample]:
        try:
            import psycopg2
        except ImportError:
            raise ImportError("PostgresDataSource requires psycopg2: pip install psycopg2-binary")

        samples: List[DataSample] = []
        conn = None
        try:
            conn = psycopg2.connect(self._dsn)
            available_columns = self._resolve_table_columns(conn)
            query, params = self._build_query_and_params(available_columns)
            with conn.cursor() as cur:
                cur.execute(query, params)
                column_names = [desc[0] for desc in cur.description]
                for row in cur:
                    row_map = {column_names[i]: row[i] for i in range(len(column_names))}
                    sample = self._row_to_sample(row_map)
                    if sample:
                        samples.append(sample)
        except Exception as exc:
            print(f"  Warning: PostgresDataSource failed ({exc}) — returning empty")
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception as exc:
                print(f"  Warning: failed to close Postgres connection ({exc})")
        return samples

    def stream(self) -> Iterator[DataSample]:
        """Stream rows directly from Postgres cursor without buffering all in memory."""
        try:
            import psycopg2
        except ImportError:
            raise ImportError("PostgresDataSource requires psycopg2: pip install psycopg2-binary")

        conn = None
        try:
            conn = psycopg2.connect(self._dsn)
            available_columns = self._resolve_table_columns(conn)
            query, params = self._build_query_and_params(available_columns)
            with conn.cursor("helox_stream_cursor") as cur:  # server-side cursor
                cur.execute(query, params)
                column_names = [desc[0] for desc in cur.description]
                for row in cur:
                    row_map = {column_names[i]: row[i] for i in range(len(column_names))}
                    sample = self._row_to_sample(row_map)
                    if sample:
                        yield sample
        except Exception as exc:
            print(f"  Warning: PostgresDataSource stream failed ({exc})")
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception as exc:
                print(f"  Warning: failed to close Postgres connection ({exc})")

    def get_info(self) -> Dict[str, Any]:
        return {
            "source_type": "postgres",
            "name": self.name,
            "table": self._table,
            "dsn": self._dsn.split("@")[-1] if "@" in self._dsn else self._dsn,
            "min_quality": self._min_quality,
            "stream_type": self._stream_type,
            "producer": self._producer,
            "max_samples": self._max_samples,
        }
