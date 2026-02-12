"""
Helox Real-Time Ingestion Consumer
====================================

Listens to two Redis Streams channels pushed by the Cyrex
RealtimeDataPipeline and writes records into Helox training data:

  pipeline.helox-training.raw        → raw text pairs for pre-training
  pipeline.helox-training.structured → instruction-tuning JSONL with typed fields

Data lands as JSONL files ready for the Helox training pipelines to consume.

Output structure:
    data/datasets/pipeline/
    ├── raw/
    │   ├── raw_2026-02-09.jsonl
    │   └── ...
    ├── structured/
    │   ├── agent_interaction/
    │   │   └── agent_interaction_2026-02-09.jsonl
    │   ├── tool_execution/
    │   │   └── tool_execution_2026-02-09.jsonl
    │   └── ...
    ├── all/
    │   └── all_2026-02-09.jsonl      ← combined stream
    └── metrics/
        └── ingestion_stats.json

Usage (standalone):
    python -m integrations.realtime_ingestion

Usage (from training scripts):
    from integrations.realtime_ingestion import HeloxRealtimeIngestion
    ingestion = HeloxRealtimeIngestion()
    await ingestion.start()
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

REQUIRED_RAW_FIELDS = {"id", "text", "source"}
REQUIRED_STRUCTURED_FIELDS = {"id", "instruction", "input", "output"}
MIN_QUALITY_FOR_TRAINING = 0.3  # records below this are logged but not used


def _validate_raw_record(record: Dict[str, Any]) -> List[str]:
    """Validate a raw training record"""
    errors = []
    missing = REQUIRED_RAW_FIELDS - set(record.keys())
    if missing:
        errors.append(f"Missing required fields: {missing}")
    text = record.get("text", "")
    if isinstance(text, str) and len(text.strip()) < 10:
        errors.append(f"Text too short ({len(text.strip())} chars)")
    return errors


def _validate_structured_record(record: Dict[str, Any]) -> List[str]:
    """Validate a structured (instruction-tuning) training record"""
    errors = []
    missing = REQUIRED_STRUCTURED_FIELDS - set(record.keys())
    if missing:
        errors.append(f"Missing required fields: {missing}")
    # At least instruction+input or instruction+output should be present
    has_input = bool(record.get("input", ""))
    has_output = bool(record.get("output", ""))
    if not has_input and not has_output:
        errors.append("Structured record needs at least input or output text")
    return errors


def _passes_quality_gate(record: Dict[str, Any]) -> bool:
    """Check if record meets minimum quality threshold for training"""
    score = record.get("quality_score")
    if score is None:
        return True  # no score = assume okay
    try:
        return float(score) >= MIN_QUALITY_FOR_TRAINING
    except (ValueError, TypeError):
        return True


# ---------------------------------------------------------------------------
# Ingestion consumer
# ---------------------------------------------------------------------------

class HeloxRealtimeIngestion:
    """
    Consumes records from the Cyrex pipeline Redis streams and
    writes them to Helox training-ready JSONL files.

    Handles two stream types:
      - pipeline.helox-training.raw        (raw text pairs)
      - pipeline.helox-training.structured (instruction-tuning format)
    """

    RAW_STREAM = "pipeline.helox-training.raw"
    STRUCTURED_STREAM = "pipeline.helox-training.structured"
    CONSUMER_GROUP = "helox-training-consumers"
    CONSUMER_NAME_PREFIX = "helox-consumer"

    def __init__(
        self,
        redis_url: Optional[str] = None,
        output_dir: Optional[Path] = None,
        batch_size: int = 50,
        flush_interval_seconds: float = 5.0,
        consumer_id: int = 1,
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379")
        self.output_dir = output_dir or Path("data/datasets/pipeline")
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self.consumer_name = f"{self.CONSUMER_NAME_PREFIX}-{consumer_id}"
        self._redis: Optional[aioredis.Redis] = None
        self._running = False

        # Separate buffers for raw and structured
        self._raw_buffer: List[Dict[str, Any]] = []
        self._structured_buffer: List[Dict[str, Any]] = []

        self._stats = {
            "raw_consumed": 0,
            "raw_written": 0,
            "structured_consumed": 0,
            "structured_written": 0,
            "validation_failures": 0,
            "quality_filtered": 0,
            "errors": 0,
            "started_at": None,
            "last_flush_at": None,
        }

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "raw").mkdir(exist_ok=True)
        (self.output_dir / "structured").mkdir(exist_ok=True)
        (self.output_dir / "all").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to Redis and ensure consumer groups exist"""
        if aioredis is None:
            logger.error("redis.asyncio not installed – cannot start ingestion")
            return False

        try:
            self._redis = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=10.0,
                retry_on_timeout=True,
            )
            await self._redis.ping()
            logger.info(f"Helox ingestion connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

        # Create consumer groups for both streams (idempotent)
        for stream in (self.RAW_STREAM, self.STRUCTURED_STREAM):
            try:
                await self._redis.xgroup_create(
                    stream, self.CONSUMER_GROUP, id="0", mkstream=True,
                )
                logger.info(f"Created consumer group '{self.CONSUMER_GROUP}' on '{stream}'")
            except Exception:
                pass  # Group already exists

        return True

    async def disconnect(self):
        """Close Redis connection"""
        self._running = False
        if self._redis:
            await self._redis.close()
            self._redis = None

    # ------------------------------------------------------------------
    # Main consumer loop
    # ------------------------------------------------------------------

    async def start(self):
        """Start consuming from both streams"""
        if not await self.connect():
            logger.error("Cannot start ingestion – Redis unavailable")
            return

        self._running = True
        self._stats["started_at"] = datetime.utcnow().isoformat()
        logger.info(
            "Helox real-time ingestion started – listening on "
            f"{self.RAW_STREAM} + {self.STRUCTURED_STREAM}"
        )

        flush_task = asyncio.create_task(self._periodic_flush())
        stats_task = asyncio.create_task(self._periodic_stats_save())

        try:
            while self._running:
                try:
                    # Read from both streams simultaneously
                    messages = await self._redis.xreadgroup(
                        groupname=self.CONSUMER_GROUP,
                        consumername=self.consumer_name,
                        streams={
                            self.RAW_STREAM: ">",
                            self.STRUCTURED_STREAM: ">",
                        },
                        count=self.batch_size,
                        block=int(self.flush_interval * 1000),
                    )
                except Exception as e:
                    logger.error(f"Stream read error: {e}")
                    await asyncio.sleep(2)
                    continue

                if not messages:
                    continue

                for stream_name, entries in messages:
                    for entry_id, fields in entries:
                        try:
                            record = self._parse_record(fields)
                            is_raw = stream_name == self.RAW_STREAM

                            # Validate
                            if is_raw:
                                errors = _validate_raw_record(record)
                            else:
                                errors = _validate_structured_record(record)

                            if errors:
                                self._stats["validation_failures"] += 1
                                logger.warning(
                                    f"Validation failed for {entry_id} on {stream_name}: {errors}"
                                )
                                # Ack even invalid records so they don't block
                                await self._redis.xack(stream_name, self.CONSUMER_GROUP, entry_id)
                                continue

                            # Quality gate
                            if not _passes_quality_gate(record):
                                self._stats["quality_filtered"] += 1
                                await self._redis.xack(stream_name, self.CONSUMER_GROUP, entry_id)
                                continue

                            # Buffer
                            if is_raw:
                                self._raw_buffer.append(record)
                                self._stats["raw_consumed"] += 1
                            else:
                                self._structured_buffer.append(record)
                                self._stats["structured_consumed"] += 1

                            # Acknowledge
                            await self._redis.xack(stream_name, self.CONSUMER_GROUP, entry_id)

                        except Exception as e:
                            logger.warning(f"Failed to parse record {entry_id}: {e}")
                            self._stats["errors"] += 1

                # Flush if buffers are large enough
                if len(self._raw_buffer) >= self.batch_size:
                    await self._flush_raw()
                if len(self._structured_buffer) >= self.batch_size:
                    await self._flush_structured()

        except asyncio.CancelledError:
            pass
        finally:
            flush_task.cancel()
            stats_task.cancel()
            await self._flush_raw()
            await self._flush_structured()
            await self._save_stats()
            await self.disconnect()
            logger.info(f"Helox ingestion stopped – stats: {json.dumps(self._stats)}")

    async def stop(self):
        """Signal the consumer to stop"""
        self._running = False

    # ------------------------------------------------------------------
    # Record parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_record(fields: Dict[str, str]) -> Dict[str, Any]:
        """Parse a Redis stream entry into a training record dict"""
        parsed = {}
        for key, value in fields.items():
            try:
                parsed[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed[key] = value
        return parsed

    # ------------------------------------------------------------------
    # Buffer flushing
    # ------------------------------------------------------------------

    async def _periodic_flush(self):
        """Periodically flush both buffers to disk"""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            if self._raw_buffer:
                await self._flush_raw()
            if self._structured_buffer:
                await self._flush_structured()

    async def _periodic_stats_save(self):
        """Save stats to disk periodically"""
        while self._running:
            await asyncio.sleep(60)
            await self._save_stats()

    async def _flush_raw(self):
        """Write raw buffered records to JSONL"""
        if not self._raw_buffer:
            return

        records = self._raw_buffer[:]
        self._raw_buffer.clear()
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

        for record in records:
            try:
                # Raw file
                raw_file = self.output_dir / "raw" / f"raw_{date_str}.jsonl"
                with open(raw_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")

                # Combined "all" file
                all_file = self.output_dir / "all" / f"all_{date_str}.jsonl"
                with open(all_file, "a", encoding="utf-8") as f:
                    tagged = {**record, "_stream": "raw"}
                    f.write(json.dumps(tagged, default=str) + "\n")

                self._stats["raw_written"] += 1
            except Exception as e:
                logger.error(f"Failed to write raw record: {e}")
                self._stats["errors"] += 1

        self._stats["last_flush_at"] = datetime.utcnow().isoformat()
        logger.info(
            f"Flushed {len(records)} raw records "
            f"(total raw written: {self._stats['raw_written']})"
        )

    async def _flush_structured(self):
        """Write structured buffered records to JSONL, organized by category"""
        if not self._structured_buffer:
            return

        records = self._structured_buffer[:]
        self._structured_buffer.clear()
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

        for record in records:
            try:
                category = record.get("category", "unknown")

                # Category-specific file under structured/
                cat_dir = self.output_dir / "structured" / category
                cat_dir.mkdir(exist_ok=True)
                cat_file = cat_dir / f"{category}_{date_str}.jsonl"
                with open(cat_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")

                # Combined "all" file
                all_file = self.output_dir / "all" / f"all_{date_str}.jsonl"
                with open(all_file, "a", encoding="utf-8") as f:
                    tagged = {**record, "_stream": "structured"}
                    f.write(json.dumps(tagged, default=str) + "\n")

                self._stats["structured_written"] += 1
            except Exception as e:
                logger.error(f"Failed to write structured record: {e}")
                self._stats["errors"] += 1

        self._stats["last_flush_at"] = datetime.utcnow().isoformat()
        logger.info(
            f"Flushed {len(records)} structured records "
            f"(total structured written: {self._stats['structured_written']})"
        )

    async def _save_stats(self):
        """Persist stats to disk for monitoring"""
        try:
            stats_file = self.output_dir / "metrics" / "ingestion_stats.json"
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(
                    {**self._stats, "saved_at": datetime.utcnow().isoformat()},
                    f, indent=2,
                )
        except Exception as e:
            logger.debug(f"Failed to save stats: {e}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "raw_buffer_size": len(self._raw_buffer),
            "structured_buffer_size": len(self._structured_buffer),
        }

    def list_available_data(self) -> Dict[str, List[str]]:
        """List available JSONL files by category"""
        result = {}
        if self.output_dir.exists():
            for subdir in sorted(self.output_dir.iterdir()):
                if subdir.is_dir():
                    files = sorted(str(f.name) for f in subdir.rglob("*.jsonl"))
                    if files:
                        result[subdir.name] = files
        return result

    def count_records(self) -> Dict[str, int]:
        """Count total records per category"""
        counts = {}
        if self.output_dir.exists():
            for jsonl_file in self.output_dir.rglob("*.jsonl"):
                category = jsonl_file.parent.name
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        line_count = sum(1 for _ in f)
                    counts[category] = counts.get(category, 0) + line_count
                except Exception:
                    pass
        return counts


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

async def _main():
    """Run ingestion consumer as a standalone process"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    ingestion = HeloxRealtimeIngestion()
    try:
        await ingestion.start()
    except KeyboardInterrupt:
        await ingestion.stop()


if __name__ == "__main__":
    asyncio.run(_main())
