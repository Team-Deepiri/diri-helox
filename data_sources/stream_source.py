"""
StreamDataSource: reads training data published by the Language Intelligence Service.

Data flow:
  Language Intelligence Service (Cyrex)
    → parses/cleans documents via deepiri-dataset-processor
    → stores structured data in Postgres / Milvus
    → publishes cleaned samples to Redis Streams (realtime_data_pipeline.py)
  Helox (this source) subscribes to those Redis channels for training.

Two Redis Streams (published by Cyrex's HeloxRealtimeIngestion pipeline):
  - pipeline.helox-training.raw        → {id, text, source, quality_score, timestamp}
  - pipeline.helox-training.structured → {id, instruction, input, output, category, ...}

Three reading modes:
  - "file"      (default): reads pre-ingested JSONL snapshots from data/datasets/pipeline/
  - "live":     one-shot snapshot via xrange (reads current messages in stream)
  - "subscribe": blocking xread loop — continuously listens for new messages

Fallback: if Redis is unavailable in live/subscribe mode, automatically falls back
to "file" mode (or "synthetic" if configured).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .base import DataSample, DataSource, DataSourceConfig

# Minimum quality score required to include a sample (matches Cyrex gate)
MIN_QUALITY_SCORE = 0.4

RAW_STREAM = "pipeline.helox-training.raw"
STRUCTURED_STREAM = "pipeline.helox-training.structured"


class StreamDataSource(DataSource):
    """
    Reads training data published by the Cyrex realtime data pipeline.

    Config params (all modes):
        mode              (str):   "file", "live", or "subscribe" (default: "file")
        stream_type       (str):   "raw", "structured", or "both" (default: "both")
        min_quality       (float): quality gate threshold (default: 0.4)
        max_samples       (int):   cap on samples loaded (default: None = no cap)

    Config params (file mode):
        pipeline_dir      (str):   path to pipeline JSONL files (default: "data/datasets/pipeline")

    Config params (live/subscribe mode):
        redis_url         (str):   Redis connection URL (default: "redis://localhost:6379")
        batch_size        (int):   messages per xrange/xread batch (default: 100)
        fallback_mode     (str):   "file" or "synthetic" if Redis unavailable (default: "file")
        fallback_synthetic_examples (int): samples to generate on synthetic fallback (default: 50)

    Config params (subscribe mode only):
        block_ms          (int):   xread blocking timeout in ms (default: 2000)
        idle_timeout_s    (float): stop after N seconds with no new messages (default: 30)
        last_id           (str):   starting stream ID — "$" = new only, "0" = replay all
                                   (default: "0")
    """

    def __init__(
        self,
        config: DataSourceConfig,
        _redis_client: Any = None,  # injectable for testing
    ) -> None:
        super().__init__(config)
        self._mode = config.params.get("mode", "file")
        self._pipeline_dir = Path(config.params.get("pipeline_dir", "data/datasets/pipeline"))
        self._stream_type = config.params.get("stream_type", "both")
        self._min_quality = float(config.params.get("min_quality", MIN_QUALITY_SCORE))
        self._max_samples: Optional[int] = config.params.get("max_samples", None)
        self._redis_url: str = config.params.get("redis_url", "redis://localhost:6379")
        self._batch_size: int = config.params.get("batch_size", 100)
        self._fallback_mode: str = config.params.get("fallback_mode", "file")
        self._fallback_synthetic_examples: int = config.params.get(
            "fallback_synthetic_examples", 50
        )
        # Subscribe-mode params
        self._block_ms: int = config.params.get("block_ms", 2000)
        self._idle_timeout_s: float = config.params.get("idle_timeout_s", 30.0)
        self._last_id: str = config.params.get("last_id", "0")

        # Injectable Redis client (used in tests via fakeredis)
        self._injected_client: Any = _redis_client

    # ------------------------------------------------------------------
    # Redis connection helper
    # ------------------------------------------------------------------

    def _connect_redis(self) -> Any:
        """Return a Redis client. Raises ConnectionError if unavailable."""
        if self._injected_client is not None:
            return self._injected_client
        try:
            import redis

            client = redis.from_url(self._redis_url, socket_connect_timeout=2)
            client.ping()
            return client
        except Exception as exc:
            raise ConnectionError(f"Redis unavailable at {self._redis_url}: {exc}") from exc

    def _streams_for_type(self) -> List[str]:
        streams = []
        if self._stream_type in ("raw", "both"):
            streams.append(RAW_STREAM)
        if self._stream_type in ("structured", "both"):
            streams.append(STRUCTURED_STREAM)
        return streams

    # ------------------------------------------------------------------
    # Record parsers
    # ------------------------------------------------------------------

    def _decode_message(self, data: Dict) -> Dict:
        """Decode bytes keys/values from Redis and unwrap JSON payload if present."""
        decoded = {
            k.decode() if isinstance(k, bytes) else k: (v.decode() if isinstance(v, bytes) else v)
            for k, v in data.items()
        }
        payload_str = decoded.get("payload", "")
        if payload_str:
            try:
                return json.loads(payload_str)  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass
        return decoded

    def _parse_raw_record(self, item: Dict) -> Optional[DataSample]:
        try:
            quality = float(item.get("quality_score", 1.0))
        except (TypeError, ValueError):
            quality = 0.0
        if quality < self._min_quality:
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
        try:
            quality = float(item.get("quality_score", 1.0))
        except (TypeError, ValueError):
            quality = 0.0
        if quality < self._min_quality:
            return None
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
                "record_id": item.get("id"),
                # Keep both canonical instruction-tuning keys and the original fields.
                "instruction": instruction,
                "input": inp,
                "response": output,
                "output": output,
                "quality_score": quality,
                "category": item.get("category"),
            },
            source=f"stream:{self.name}",
        )

    def _unwrap_payload(self, item: Dict) -> Dict:
        """
        Unwrap payload records from file fallbacks and bridge formats.

        Cyrex fallback exports may contain envelope fields like:
          {"event_type": "...", "payload": {...}}
        or payload as a JSON-encoded string.
        """
        payload = item.get("payload")
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            try:
                decoded = json.loads(payload)
                if isinstance(decoded, dict):
                    return decoded
            except json.JSONDecodeError:
                # Payload strings from fallback/export files are not guaranteed to be
                # JSON; on decode failure we intentionally keep the original envelope.
                return item
        return item

    def _parse_item(self, item: Dict) -> Optional[DataSample]:
        item = self._unwrap_payload(item)
        if "instruction" in item or "input" in item:
            return self._parse_structured_record(item)
        return self._parse_raw_record(item)

    # ------------------------------------------------------------------
    # File mode
    # ------------------------------------------------------------------

    def _iter_pipeline_files(self) -> Iterator[Path]:
        if not self._pipeline_dir.exists():
            return
        patterns = []
        if self._stream_type in ("raw", "both"):
            patterns.append("*raw*.jsonl")
        if self._stream_type in ("structured", "both"):
            patterns.append("*structured*.jsonl")

        matched: List[Path] = []
        for pattern in patterns:
            matched.extend(sorted(self._pipeline_dir.glob(pattern)))
        matched = sorted(set(matched))

        # If stream-specific filenames do not exist, gracefully fall back to any JSONL file.
        # This helps interop with fallback exporters that use names like events_training.jsonl.
        if matched:
            yield from matched
            return

        yield from sorted(self._pipeline_dir.glob("*.jsonl"))

    def _load_file_mode(self) -> List[DataSample]:
        samples: List[DataSample] = []
        for path in self._iter_pipeline_files():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sample = self._parse_item(item)
                    if sample:
                        samples.append(sample)
                    if self._max_samples and len(samples) >= self._max_samples:
                        return samples
        return samples

    # ------------------------------------------------------------------
    # Live mode (one-shot xrange snapshot)
    # ------------------------------------------------------------------

    def _load_live_mode(self) -> List[DataSample]:
        try:
            r = self._connect_redis()
        except ConnectionError as exc:
            print(f"  Warning: {exc} — falling back to {self._fallback_mode} mode")
            return self._load_fallback()

        samples: List[DataSample] = []
        for stream_name in self._streams_for_type():
            try:
                messages = r.xrange(stream_name, count=self._batch_size)
            except Exception:
                continue
            for _msg_id, data in messages:  # type: ignore[union-attr]
                item = self._decode_message(data)
                sample = self._parse_item(item)
                if sample:
                    samples.append(sample)
                if self._max_samples and len(samples) >= self._max_samples:
                    return samples
        return samples

    # ------------------------------------------------------------------
    # Subscribe mode (blocking xread loop)
    # ------------------------------------------------------------------

    def _stream_live(self) -> Iterator[DataSample]:
        """
        Blocking generator that continuously yields DataSamples from Redis streams.
        Uses xread with blocking to wait for new messages.
        Stops after idle_timeout_s seconds with no new messages.
        """
        try:
            r = self._connect_redis()
        except ConnectionError as exc:
            print(f"  Warning: {exc} — falling back to {self._fallback_mode} mode")
            yield from self._load_fallback()
            return

        streams = self._streams_for_type()
        # Track the last seen ID per stream; "0" replays all, "$" = new only
        last_ids: Dict[str, str] = {s: self._last_id for s in streams}
        last_message_time = time.monotonic()
        count = 0

        print(
            f"  [subscribe] Listening on {streams} "
            f"(idle_timeout={self._idle_timeout_s}s, block={self._block_ms}ms)"
        )

        while True:
            # Check idle timeout
            idle_s = time.monotonic() - last_message_time
            if idle_s >= self._idle_timeout_s:
                print(f"  [subscribe] Idle timeout reached ({idle_s:.1f}s) — stopping")
                break

            # Check sample cap
            if self._max_samples and count >= self._max_samples:
                print(f"  [subscribe] max_samples={self._max_samples} reached — stopping")
                break

            try:
                results = r.xread(last_ids, block=self._block_ms, count=self._batch_size)
            except Exception as exc:
                print(f"  Warning: xread error ({exc}) — stopping subscribe loop")
                break

            if not results:
                continue  # timed out with no messages, loop again (idle check will catch timeout)

            for stream_name, messages in results:
                stream_key = stream_name.decode() if isinstance(stream_name, bytes) else stream_name
                for msg_id, data in messages:
                    last_ids[stream_key] = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                    item = self._decode_message(data)
                    sample = self._parse_item(item)
                    if sample:
                        count += 1
                        last_message_time = time.monotonic()
                        yield sample
                        if self._max_samples and count >= self._max_samples:
                            return

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _load_fallback(self) -> List[DataSample]:
        if self._fallback_mode == "synthetic":
            return self._load_synthetic_fallback()
        return self._load_file_mode()

    def _load_synthetic_fallback(self) -> List[DataSample]:
        try:
            from .synthetic_source import SyntheticDataSource
            from .base import DataSourceConfig as DSC

            cfg = DSC(
                source_type="synthetic",
                name=f"{self.name}_fallback",
                params={
                    "total_examples": self._fallback_synthetic_examples,
                    "use_ollama": False,
                },
            )
            return SyntheticDataSource(cfg).load()
        except Exception as exc:
            print(f"  Warning: synthetic fallback failed ({exc}) — returning empty")
            return []

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def load(self) -> List[DataSample]:
        if self._mode == "subscribe":
            # Drain the subscribe generator up to max_samples
            samples: List[DataSample] = []
            for sample in self._stream_live():
                samples.append(sample)
                if self._max_samples and len(samples) >= self._max_samples:
                    break
            return samples
        if self._mode == "live":
            return self._load_live_mode()
        return self._load_file_mode()

    def stream(self) -> Iterator[DataSample]:
        if self._mode == "subscribe":
            yield from self._stream_live()
        else:
            yield from self.load()

    def get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "source_type": "stream",
            "name": self.name,
            "mode": self._mode,
            "stream_type": self._stream_type,
            "min_quality": self._min_quality,
            "max_samples": self._max_samples,
        }
        if self._mode == "file":
            info["pipeline_dir"] = str(self._pipeline_dir)
            info["pipeline_dir_exists"] = self._pipeline_dir.exists()
        else:
            info["redis_url"] = self._redis_url
            info["fallback_mode"] = self._fallback_mode
        if self._mode == "subscribe":
            info["block_ms"] = self._block_ms
            info["idle_timeout_s"] = self._idle_timeout_s
            info["last_id"] = self._last_id
        return info
