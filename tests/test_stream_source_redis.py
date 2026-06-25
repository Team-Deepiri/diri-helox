"""
Tests for StreamDataSource Redis integration using fakeredis.

Tests cover:
  - live mode (xrange snapshot)
  - subscribe mode (blocking xread loop)
  - quality filtering
  - max_samples cap
  - idle timeout in subscribe mode
  - fallback on Redis connection failure
  - raw and structured record parsing
  - payload JSON decoding
"""

import json
import sys
from pathlib import Path

import pytest

_HELOX_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_HELOX_ROOT))

try:
    import fakeredis

    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _FAKEREDIS_AVAILABLE,
    reason="fakeredis not installed",
)

from data_sources.base import DataSourceConfig  # noqa: E402
from data_sources.stream_source import (  # noqa: E402
    RAW_STREAM,
    STRUCTURED_STREAM,
    StreamDataSource,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RAW_RECORDS = [
    {"text": f"Fix the bug in module {i}", "source": "test", "quality_score": 0.9, "id": f"r{i}"}
    for i in range(5)
]

STRUCTURED_RECORDS = [
    {
        "instruction": f"Refactor function {i}",
        "input": "clean up the code",
        "output": "refactored code",
        "category": "refactoring",
        "quality_score": 0.85,
        "id": f"s{i}",
    }
    for i in range(5)
]

LOW_QUALITY_RECORD = {"text": "bad sample", "quality_score": 0.1, "id": "lq0"}


def _make_config(params: dict) -> DataSourceConfig:
    return DataSourceConfig(source_type="stream", name="test_stream", params=params)


def _seed_stream(r, stream_name: str, records: list) -> None:
    """Push records to a fakeredis stream as JSON payloads."""
    for rec in records:
        r.xadd(stream_name, {"payload": json.dumps(rec)})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_redis():
    r = fakeredis.FakeRedis()
    yield r
    r.close()


@pytest.fixture()
def seeded_redis(fake_redis):
    """fakeredis pre-seeded with raw + structured records."""
    _seed_stream(fake_redis, RAW_STREAM, RAW_RECORDS)
    _seed_stream(fake_redis, STRUCTURED_STREAM, STRUCTURED_RECORDS)
    return fake_redis


# ---------------------------------------------------------------------------
# Live mode (xrange snapshot)
# ---------------------------------------------------------------------------


class TestLiveMode:
    def test_loads_raw_records(self, seeded_redis):
        src = StreamDataSource(
            _make_config({"mode": "live", "stream_type": "raw"}),
            _redis_client=seeded_redis,
        )
        samples = src.load()
        assert len(samples) == 5
        assert all(s.metadata["source_stream"] == "raw" for s in samples)

    def test_loads_structured_records(self, seeded_redis):
        src = StreamDataSource(
            _make_config({"mode": "live", "stream_type": "structured"}),
            _redis_client=seeded_redis,
        )
        samples = src.load()
        assert len(samples) == 5
        assert all(s.metadata["source_stream"] == "structured" for s in samples)
        assert all(s.label_name == "refactoring" for s in samples)

    def test_loads_both_stream_types(self, seeded_redis):
        src = StreamDataSource(
            _make_config({"mode": "live", "stream_type": "both"}),
            _redis_client=seeded_redis,
        )
        samples = src.load()
        assert len(samples) == 10

    def test_quality_gate_filters_low_quality(self, fake_redis):
        _seed_stream(fake_redis, RAW_STREAM, [*RAW_RECORDS, LOW_QUALITY_RECORD])
        src = StreamDataSource(
            _make_config({"mode": "live", "stream_type": "raw", "min_quality": 0.4}),
            _redis_client=fake_redis,
        )
        samples = src.load()
        assert len(samples) == 5  # low-quality filtered out
        assert all(s.metadata["quality_score"] >= 0.4 for s in samples)

    def test_max_samples_cap(self, seeded_redis):
        src = StreamDataSource(
            _make_config({"mode": "live", "stream_type": "both", "max_samples": 3}),
            _redis_client=seeded_redis,
        )
        samples = src.load()
        assert len(samples) == 3

    def test_empty_stream_returns_empty(self, fake_redis):
        src = StreamDataSource(
            _make_config({"mode": "live", "stream_type": "raw"}),
            _redis_client=fake_redis,
        )
        samples = src.load()
        assert samples == []

    def test_payload_json_decoding(self, fake_redis):
        record = {"text": "debug the pipeline", "quality_score": 0.95}
        fake_redis.xadd(RAW_STREAM, {"payload": json.dumps(record)})
        src = StreamDataSource(
            _make_config({"mode": "live", "stream_type": "raw"}),
            _redis_client=fake_redis,
        )
        samples = src.load()
        assert len(samples) == 1
        assert samples[0].text == "debug the pipeline"

    def test_datasample_source_field(self, seeded_redis):
        src = StreamDataSource(
            _make_config({"mode": "live", "stream_type": "raw"}),
            _redis_client=seeded_redis,
        )
        samples = src.load()
        assert all(s.source == "stream:test_stream" for s in samples)


# ---------------------------------------------------------------------------
# Subscribe mode (blocking xread)
# ---------------------------------------------------------------------------


class TestSubscribeMode:
    def test_yields_existing_messages_with_last_id_zero(self, seeded_redis):
        """last_id='0' replays all existing messages in the stream."""
        src = StreamDataSource(
            _make_config(
                {
                    "mode": "subscribe",
                    "stream_type": "raw",
                    "last_id": "0",
                    "idle_timeout_s": 1.0,
                    "block_ms": 100,
                }
            ),
            _redis_client=seeded_redis,
        )
        samples = src.load()
        assert len(samples) == 5

    def test_subscribe_max_samples_stops_early(self, seeded_redis):
        src = StreamDataSource(
            _make_config(
                {
                    "mode": "subscribe",
                    "stream_type": "both",
                    "last_id": "0",
                    "max_samples": 4,
                    "idle_timeout_s": 5.0,
                    "block_ms": 100,
                }
            ),
            _redis_client=seeded_redis,
        )
        samples = src.load()
        assert len(samples) == 4

    def test_subscribe_idle_timeout_stops_cleanly(self, fake_redis):
        """With no messages and a short timeout, subscribe should return quickly."""
        src = StreamDataSource(
            _make_config(
                {
                    "mode": "subscribe",
                    "stream_type": "raw",
                    "last_id": "0",
                    "idle_timeout_s": 0.5,
                    "block_ms": 100,
                }
            ),
            _redis_client=fake_redis,
        )
        samples = src.load()
        assert samples == []

    def test_subscribe_stream_generator(self, seeded_redis):
        """stream() yields DataSamples one by one."""
        src = StreamDataSource(
            _make_config(
                {
                    "mode": "subscribe",
                    "stream_type": "raw",
                    "last_id": "0",
                    "idle_timeout_s": 1.0,
                    "block_ms": 100,
                }
            ),
            _redis_client=seeded_redis,
        )
        yielded = list(src.stream())
        assert len(yielded) == 5

    def test_subscribe_quality_filtering(self, fake_redis):
        records = [*RAW_RECORDS, LOW_QUALITY_RECORD]
        _seed_stream(fake_redis, RAW_STREAM, records)
        src = StreamDataSource(
            _make_config(
                {
                    "mode": "subscribe",
                    "stream_type": "raw",
                    "last_id": "0",
                    "idle_timeout_s": 1.0,
                    "block_ms": 100,
                    "min_quality": 0.4,
                }
            ),
            _redis_client=fake_redis,
        )
        samples = src.load()
        assert all(s.metadata["quality_score"] >= 0.4 for s in samples)
        assert len(samples) == 5  # low quality filtered

    def test_new_messages_during_subscribe(self, fake_redis):
        """Messages added to stream while subscribing (last_id='0') are consumed."""
        _seed_stream(fake_redis, RAW_STREAM, RAW_RECORDS[:3])
        src = StreamDataSource(
            _make_config(
                {
                    "mode": "subscribe",
                    "stream_type": "raw",
                    "last_id": "0",
                    "idle_timeout_s": 1.0,
                    "block_ms": 100,
                }
            ),
            _redis_client=fake_redis,
        )
        samples = src.load()
        assert len(samples) == 3


# ---------------------------------------------------------------------------
# Fallback on Redis unavailable
# ---------------------------------------------------------------------------


class TestFallback:
    def test_live_mode_falls_back_to_file_on_bad_url(self, tmp_path):
        """Live mode with an unreachable Redis URL falls back to file mode silently."""
        # Create a pipeline JSONL file for file fallback
        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()
        record = {"text": "fallback task from file", "quality_score": 0.9}
        (pipeline_dir / "raw_data.jsonl").write_text(json.dumps(record) + "\n")

        src = StreamDataSource(
            _make_config(
                {
                    "mode": "live",
                    "stream_type": "raw",
                    "redis_url": "redis://localhost:19999",  # unreachable
                    "fallback_mode": "file",
                    "pipeline_dir": str(pipeline_dir),
                }
            )
        )
        samples = src.load()
        assert len(samples) == 1
        assert samples[0].text == "fallback task from file"

    def test_subscribe_mode_falls_back_on_bad_url(self, tmp_path):
        """Subscribe mode with unreachable Redis falls back to file mode."""
        pipeline_dir = tmp_path / "pipeline"
        pipeline_dir.mkdir()
        record = {"text": "fallback subscribe task", "quality_score": 0.9}
        (pipeline_dir / "raw_data.jsonl").write_text(json.dumps(record) + "\n")

        src = StreamDataSource(
            _make_config(
                {
                    "mode": "subscribe",
                    "stream_type": "raw",
                    "redis_url": "redis://localhost:19999",
                    "fallback_mode": "file",
                    "pipeline_dir": str(pipeline_dir),
                    "idle_timeout_s": 1.0,
                }
            )
        )
        samples = src.load()
        assert len(samples) == 1


# ---------------------------------------------------------------------------
# get_info
# ---------------------------------------------------------------------------


class TestGetInfo:
    def test_get_info_live_mode(self, fake_redis):
        src = StreamDataSource(
            _make_config({"mode": "live", "redis_url": "redis://localhost:6380"}),
            _redis_client=fake_redis,
        )
        info = src.get_info()
        assert info["mode"] == "live"
        assert info["redis_url"] == "redis://localhost:6380"
        assert "fallback_mode" in info

    def test_get_info_subscribe_mode(self, fake_redis):
        src = StreamDataSource(
            _make_config(
                {
                    "mode": "subscribe",
                    "block_ms": 500,
                    "idle_timeout_s": 10.0,
                    "last_id": "0",
                }
            ),
            _redis_client=fake_redis,
        )
        info = src.get_info()
        assert info["mode"] == "subscribe"
        assert info["block_ms"] == 500
        assert info["idle_timeout_s"] == 10.0
        assert info["last_id"] == "0"

    def test_get_info_file_mode(self):
        src = StreamDataSource(_make_config({"mode": "file"}))
        info = src.get_info()
        assert info["mode"] == "file"
        assert "pipeline_dir" in info
