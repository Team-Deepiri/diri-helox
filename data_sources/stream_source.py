"""
StreamDataSource: reads data published by Cyrex via Redis Streams.

Two modes:
  - "file"  (default): reads pre-ingested JSONL files from data/datasets/pipeline/
  - "live": consumes Redis streams directly (requires Redis connection)

Cyrex publishes to two streams:
  - pipeline.helox-training.raw        -> {id, text, source, quality_score, timestamp}
  - pipeline.helox-training.structured -> {id, instruction, input, output, category, ...}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .base import DataSample, DataSource, DataSourceConfig

# Minimum quality score required to include a sample (matches Cyrex gate)
MIN_QUALITY_SCORE = 0.4


class StreamDataSource(DataSource):
    """
    Reads training data published by the Cyrex realtime data pipeline.

    Config params (file mode):
        mode          (str): "file" or "live" (default: "file")
        pipeline_dir  (str): path to pipeline JSONL files (default: "data/datasets/pipeline")
        stream_type   (str): "raw", "structured", or "both" (default: "both")
        min_quality   (float): quality gate threshold (default: 0.4)
        max_samples   (int | None): cap on samples loaded

    Config params (live mode, additional):
        redis_url     (str): Redis connection URL (default: "redis://localhost:6379")
        batch_size    (int): number of messages to consume per batch (default: 100)
    """

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        self._mode = config.params.get("mode", "file")
        self._pipeline_dir = Path(config.params.get("pipeline_dir", "data/datasets/pipeline"))
        self._stream_type = config.params.get("stream_type", "both")
        self._min_quality = config.params.get("min_quality", MIN_QUALITY_SCORE)
        self._max_samples = config.params.get("max_samples", None)
        self._redis_url = config.params.get("redis_url", "redis://localhost:6379")
        self._batch_size = config.params.get("batch_size", 100)

    # ------------------------------------------------------------------
    # File mode helpers
    # ------------------------------------------------------------------

    def _iter_pipeline_files(self) -> Iterator[Path]:
        """Yield JSONL files from the pipeline directory."""
        if not self._pipeline_dir.exists():
            return
        patterns = []
        if self._stream_type in ("raw", "both"):
            patterns.append("*raw*.jsonl")
        if self._stream_type in ("structured", "both"):
            patterns.append("*structured*.jsonl")
        # Also yield any generic .jsonl files if no match
        for pattern in patterns:
            yield from sorted(self._pipeline_dir.glob(pattern))
        if not patterns:
            yield from sorted(self._pipeline_dir.glob("*.jsonl"))

    def _parse_raw_record(self, item: Dict) -> Optional[DataSample]:
        quality = item.get("quality_score", 1.0)
        if quality is not None and quality < self._min_quality:
            return None
        text = item.get("text", "")
        if not text:
            return None
        return DataSample(
            text=text,
            label=None,
            label_name=None,
            metadata={"source_stream": "raw", "quality_score": quality},
            source=f"stream:{self.name}",
        )

    def _parse_structured_record(self, item: Dict) -> Optional[DataSample]:
        quality = item.get("quality_score", 1.0)
        if quality is not None and quality < self._min_quality:
            return None
        # Combine instruction + input as text; output becomes a metadata field
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")
        text = f"{instruction} {inp}".strip() or output
        if not text:
            return None
        return DataSample(
            text=text,
            label=None,
            label_name=item.get("category"),
            metadata={
                "source_stream": "structured",
                "output": output,
                "quality_score": quality,
                "category": item.get("category"),
            },
            source=f"stream:{self.name}",
        )

    def _load_file_mode(self) -> List[DataSample]:
        samples: List[DataSample] = []
        for path in self._iter_pipeline_files():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    # Detect format by presence of 'instruction' key
                    if "instruction" in item or "input" in item:
                        sample = self._parse_structured_record(item)
                    else:
                        sample = self._parse_raw_record(item)
                    if sample:
                        samples.append(sample)
                    if self._max_samples and len(samples) >= self._max_samples:
                        return samples
        return samples

    # ------------------------------------------------------------------
    # Live mode helpers
    # ------------------------------------------------------------------

    def _load_live_mode(self) -> List[DataSample]:
        try:
            import redis
        except ImportError:
            raise ImportError("StreamDataSource live mode requires redis-py: pip install redis")

        r = redis.from_url(self._redis_url)
        streams = []
        if self._stream_type in ("raw", "both"):
            streams.append("pipeline.helox-training.raw")
        if self._stream_type in ("structured", "both"):
            streams.append("pipeline.helox-training.structured")

        samples: List[DataSample] = []
        for stream_name in streams:
            try:
                messages = r.xrange(stream_name, count=self._batch_size)
            except Exception:
                continue
            for _msg_id, data in messages:  # type: ignore[union-attr]
                # Redis returns bytes; decode
                decoded = {
                    k.decode() if isinstance(k, bytes) else k: (
                        v.decode() if isinstance(v, bytes) else v
                    )
                    for k, v in data.items()
                }
                # The payload may be JSON-encoded inside a 'payload' key
                payload_str = decoded.get("payload", "")
                if payload_str:
                    try:
                        item = json.loads(payload_str)
                    except json.JSONDecodeError:
                        item = decoded
                else:
                    item = decoded

                if "instruction" in item or "input" in item:
                    sample = self._parse_structured_record(item)
                else:
                    sample = self._parse_raw_record(item)
                if sample:
                    samples.append(sample)
                if self._max_samples and len(samples) >= self._max_samples:
                    return samples
        return samples

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def load(self) -> List[DataSample]:
        if self._mode == "live":
            return self._load_live_mode()
        return self._load_file_mode()

    def stream(self) -> Iterator[DataSample]:
        yield from self.load()

    def get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "source_type": "stream",
            "name": self.name,
            "mode": self._mode,
            "stream_type": self._stream_type,
            "min_quality": self._min_quality,
        }
        if self._mode == "file":
            info["pipeline_dir"] = str(self._pipeline_dir)
            info["pipeline_dir_exists"] = self._pipeline_dir.exists()
        else:
            info["redis_url"] = self._redis_url
        return info
