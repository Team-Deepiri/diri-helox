"""
Tests for the data source abstraction layer.
Run: pytest tests/test_data_sources.py -v
"""

import json
import sys
from pathlib import Path

import pytest

_HELOX_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_HELOX_ROOT))

from data_sources.base import DataSample, DataSourceConfig
from data_sources.composite_source import CompositeDataSource
from data_sources.postgres_source import PostgresDataSource
from data_sources.self_feedback_source import SelfFeedbackDataSource
from data_sources.static_source import StaticDataSource
from data_sources.stream_source import StreamDataSource
from data_sources.factory import create_data_source, create_data_sources_from_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# StaticDataSource
# ---------------------------------------------------------------------------


class TestStaticDataSource:
    def test_load_basic(self, tmp_path):
        data = [{"text": f"sample {i}", "label": i % 3} for i in range(10)]
        p = tmp_path / "test.jsonl"
        _write_jsonl(p, data)

        cfg = DataSourceConfig("static", "test", {"file_paths": [str(p)]})
        src = StaticDataSource(cfg)
        samples = src.load()

        assert len(samples) == 10
        assert all(isinstance(s, DataSample) for s in samples)
        assert samples[0].text == "sample 0"
        assert samples[0].label == 0

    def test_stream_yields_same_as_load(self, tmp_path):
        data = [{"text": f"t{i}", "label": i} for i in range(5)]
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, data)

        cfg = DataSourceConfig("static", "s", {"file_paths": [str(p)]})
        src = StaticDataSource(cfg)
        assert list(src.stream()) == src.load()

    def test_max_samples(self, tmp_path):
        data = [{"text": f"x{i}", "label": 0} for i in range(20)]
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, data)

        cfg = DataSourceConfig("static", "s", {"file_paths": [str(p)], "max_samples": 7})
        src = StaticDataSource(cfg)
        assert len(src.load()) == 7

    def test_missing_file_raises(self):
        cfg = DataSourceConfig("static", "s", {"file_paths": ["/nonexistent/file.jsonl"]})
        src = StaticDataSource(cfg)
        with pytest.raises(FileNotFoundError):
            src.load()

    def test_get_info(self, tmp_path):
        p = tmp_path / "x.jsonl"
        _write_jsonl(p, [{"text": "hi", "label": 0}])
        cfg = DataSourceConfig("static", "myname", {"file_paths": [str(p)]})
        src = StaticDataSource(cfg)
        info = src.get_info()
        assert info["source_type"] == "static"
        assert info["name"] == "myname"
        assert info["existing_files"] == 1


# ---------------------------------------------------------------------------
# StreamDataSource (file mode)
# ---------------------------------------------------------------------------


class TestStreamDataSource:
    def test_load_raw_file(self, tmp_path):
        records = [{"text": f"agent said {i}", "quality_score": 0.8} for i in range(5)]
        _write_jsonl(tmp_path / "raw_data.jsonl", records)

        cfg = DataSourceConfig(
            "stream",
            "s",
            {
                "mode": "file",
                "pipeline_dir": str(tmp_path),
                "stream_type": "raw",
            },
        )
        src = StreamDataSource(cfg)
        samples = src.load()
        assert len(samples) == 5
        assert all(s.source.startswith("stream:") for s in samples)

    def test_quality_gate_filters_low_quality(self, tmp_path):
        records = [
            {"text": "good data", "quality_score": 0.9},
            {"text": "bad data", "quality_score": 0.1},
        ]
        _write_jsonl(tmp_path / "raw_x.jsonl", records)

        cfg = DataSourceConfig(
            "stream",
            "s",
            {
                "mode": "file",
                "pipeline_dir": str(tmp_path),
                "stream_type": "raw",
                "min_quality": 0.4,
            },
        )
        samples = StreamDataSource(cfg).load()
        assert len(samples) == 1
        assert samples[0].text == "good data"

    def test_structured_format(self, tmp_path):
        records = [
            {
                "instruction": "Classify task",
                "input": "write unit tests",
                "output": "testing",
                "quality_score": 0.7,
            }
        ]
        _write_jsonl(tmp_path / "structured_data.jsonl", records)

        cfg = DataSourceConfig(
            "stream",
            "s",
            {
                "mode": "file",
                "pipeline_dir": str(tmp_path),
                "stream_type": "structured",
            },
        )
        samples = StreamDataSource(cfg).load()
        assert len(samples) == 1
        assert "write unit tests" in samples[0].text

    def test_empty_pipeline_dir_returns_empty(self, tmp_path):
        cfg = DataSourceConfig(
            "stream",
            "s",
            {
                "mode": "file",
                "pipeline_dir": str(tmp_path / "nonexistent"),
            },
        )
        samples = StreamDataSource(cfg).load()
        assert samples == []


# ---------------------------------------------------------------------------
# SelfFeedbackDataSource
# ---------------------------------------------------------------------------


class TestSelfFeedbackDataSource:
    def test_filters_by_confidence(self, tmp_path):
        log = [
            {"text": "high conf", "predicted_label": 5, "confidence": 0.95},
            {"text": "medium conf", "predicted_label": 3, "confidence": 0.70},
            {"text": "low conf", "predicted_label": 1, "confidence": 0.50},
        ]
        p = tmp_path / "inference_log.jsonl"
        _write_jsonl(p, log)

        cfg = DataSourceConfig(
            "self_feedback",
            "sf",
            {
                "inference_log_path": str(p),
                "confidence_threshold": 0.90,
            },
        )
        samples = SelfFeedbackDataSource(cfg).load()
        assert len(samples) == 1
        assert samples[0].text == "high conf"
        assert samples[0].label == 5
        assert samples[0].metadata["confidence"] == 0.95

    def test_missing_log_returns_empty(self, tmp_path):
        cfg = DataSourceConfig(
            "self_feedback",
            "sf",
            {
                "inference_log_path": str(tmp_path / "no_file.jsonl"),
            },
        )
        assert SelfFeedbackDataSource(cfg).load() == []

    def test_max_samples(self, tmp_path):
        log = [{"text": f"t{i}", "predicted_label": 0, "confidence": 0.99} for i in range(10)]
        p = tmp_path / "log.jsonl"
        _write_jsonl(p, log)

        cfg = DataSourceConfig(
            "self_feedback",
            "sf",
            {
                "inference_log_path": str(p),
                "confidence_threshold": 0.0,
                "max_samples": 3,
            },
        )
        assert len(SelfFeedbackDataSource(cfg).load()) == 3


# ---------------------------------------------------------------------------
# PostgresDataSource
# ---------------------------------------------------------------------------


class TestPostgresDataSource:
    def test_build_query_defaults_to_durable_table(self):
        cfg = DataSourceConfig("postgres", "pg", {})
        src = PostgresDataSource(cfg)
        query = src._build_query()
        assert "FROM cyrex.helox_training_samples" in query
        assert "quality_score >= 0.4" in query
        assert "ORDER BY created_at DESC" in query

    def test_build_query_with_stream_and_producer_filters(self):
        cfg = DataSourceConfig(
            "postgres",
            "pg",
            {
                "stream_type": "structured",
                "producer": "language_intelligence",
                "max_samples": 50,
            },
        )
        src = PostgresDataSource(cfg)
        query = src._build_query()
        assert "stream_type = 'structured'" in query
        assert "producer = 'language_intelligence'" in query
        assert "LIMIT 50" in query

    def test_row_to_sample_maps_metadata(self):
        cfg = DataSourceConfig("postgres", "pg", {})
        src = PostgresDataSource(cfg)
        row = (
            "Write unit tests for stream ingestion",
            "testing",
            0.91,
            "rec-123",
            "structured",
            "language_intelligence",
        )
        sample = src._row_to_sample(row)
        assert sample is not None
        assert sample.text.startswith("Write unit tests")
        assert sample.label_name == "testing"
        assert sample.metadata["record_id"] == "rec-123"
        assert sample.metadata["stream_type"] == "structured"
        assert sample.metadata["producer"] == "language_intelligence"


# ---------------------------------------------------------------------------
# CompositeDataSource
# ---------------------------------------------------------------------------


class TestCompositeDataSource:
    def _make_static(self, tmp_path: Path, n: int, label: int, weight: float) -> StaticDataSource:
        data = [{"text": f"source{label}-{i}", "label": label} for i in range(n)]
        p = tmp_path / f"data_{label}.jsonl"
        _write_jsonl(p, data)
        cfg = DataSourceConfig("static", f"src_{label}", {"file_paths": [str(p)]}, weight=weight)
        return StaticDataSource(cfg)

    def test_combines_sources(self, tmp_path):
        src_a = self._make_static(tmp_path, 10, label=0, weight=1.0)
        src_b = self._make_static(tmp_path, 10, label=1, weight=1.0)

        cfg = DataSourceConfig("composite", "combo", {})
        composite = CompositeDataSource(cfg, [src_a, src_b])
        samples = composite.load()

        assert len(samples) > 0
        labels = {s.label for s in samples}
        assert 0 in labels
        assert 1 in labels

    def test_empty_sources_returns_empty(self):
        cfg = DataSourceConfig("composite", "empty", {})
        assert CompositeDataSource(cfg, []).load() == []

    def test_stream_same_as_load(self, tmp_path):
        src = self._make_static(tmp_path, 5, 0, 1.0)
        cfg = DataSourceConfig("composite", "c", {})
        composite = CompositeDataSource(cfg, [src])
        assert composite.load() == list(composite.stream())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_static(self, tmp_path):
        p = tmp_path / "x.jsonl"
        _write_jsonl(p, [{"text": "hi", "label": 0}])
        cfg = DataSourceConfig("static", "s", {"file_paths": [str(p)]})
        src = create_data_source(cfg)
        assert isinstance(src, StaticDataSource)

    def test_create_stream(self, tmp_path):
        cfg = DataSourceConfig("stream", "s", {"mode": "file", "pipeline_dir": str(tmp_path)})
        src = create_data_source(cfg)
        assert isinstance(src, StreamDataSource)

    def test_invalid_type_raises(self):
        cfg = DataSourceConfig("nonexistent_type", "s", {})
        with pytest.raises(ValueError, match="Unknown source_type"):
            create_data_source(cfg)

    def test_create_from_config_list(self, tmp_path):
        p = tmp_path / "f.jsonl"
        _write_jsonl(p, [{"text": "a", "label": 0}])
        config_list = [
            {"source_type": "static", "name": "s", "params": {"file_paths": [str(p)]}},
        ]
        sources = create_data_sources_from_config(config_list)
        assert len(sources) == 1
        assert isinstance(sources[0], StaticDataSource)
