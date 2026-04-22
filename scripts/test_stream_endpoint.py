#!/usr/bin/env python3
"""
Stream Endpoint Diagnostic — tests the full Redis ingest path end-to-end.

Runs three checks:
  1. Redis connectivity + stream lengths
  2. Live mode snapshot load (xrange)
  3. Subscribe mode with short idle timeout
  4. Dry-run through the full pipeline (ingest -> preprocess -> split)

Use this to verify Redis integration before committing to a full training run.

Usage:
    python scripts/test_stream_endpoint.py
    python scripts/test_stream_endpoint.py --redis-url redis://localhost:6380
    python scripts/test_stream_endpoint.py --subscribe-timeout 10
"""

import argparse
import sys
from pathlib import Path

_HELOX_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_HELOX_ROOT))

from data_sources.base import DataSourceConfig  # noqa: E402
from data_sources.stream_source import RAW_STREAM, STRUCTURED_STREAM, StreamDataSource  # noqa: E402
from pipelines.training.dynamic_training_pipeline import DynamicTrainingPipeline  # noqa: E402

SEPARATOR = "=" * 60


def check_redis_connection(redis_url: str) -> bool:
    print(f"\n{SEPARATOR}")
    print("CHECK 1: Redis Connectivity")
    print(SEPARATOR)
    try:
        import redis

        r = redis.from_url(redis_url, socket_connect_timeout=3)
        r.ping()
        print(f"  Connected to {redis_url}")

        for stream_name in (RAW_STREAM, STRUCTURED_STREAM):
            try:
                length = r.xlen(stream_name)
                print(f"  {stream_name}: {length} messages")
            except Exception:
                print(f"  {stream_name}: (does not exist yet)")
        return True
    except Exception as exc:
        print(f"  FAILED: {exc}")
        return False


def check_live_mode(redis_url: str) -> int:
    print(f"\n{SEPARATOR}")
    print("CHECK 2: Live Mode (xrange snapshot)")
    print(SEPARATOR)
    src = StreamDataSource(
        DataSourceConfig(
            source_type="stream",
            name="diagnostic_live",
            params={
                "mode": "live",
                "redis_url": redis_url,
                "stream_type": "both",
                "max_samples": 100,
            },
        )
    )
    samples = src.load()
    raw_count = sum(1 for s in samples if s.metadata.get("source_stream") == "raw")
    structured_count = sum(1 for s in samples if s.metadata.get("source_stream") == "structured")
    print(f"  Loaded {len(samples)} samples  [raw: {raw_count} | structured: {structured_count}]")

    if samples:
        print(f'  Sample preview: "{samples[0].text[:80]}"')
        avg_quality = sum(s.metadata.get("quality_score", 0) for s in samples) / len(samples)
        print(f"  Avg quality score: {avg_quality:.2f}")
    return len(samples)


def check_subscribe_mode(redis_url: str, timeout_s: float) -> int:
    print(f"\n{SEPARATOR}")
    print(f"CHECK 3: Subscribe Mode (blocking xread, {timeout_s}s timeout)")
    print(SEPARATOR)
    src = StreamDataSource(
        DataSourceConfig(
            source_type="stream",
            name="diagnostic_subscribe",
            params={
                "mode": "subscribe",
                "redis_url": redis_url,
                "stream_type": "both",
                "last_id": "0",
                "idle_timeout_s": timeout_s,
                "block_ms": 500,
                "max_samples": 200,
            },
        )
    )
    samples = src.load()
    print(f"  Received {len(samples)} samples during {timeout_s}s window")
    return len(samples)


def check_pipeline_dry_run(redis_url: str) -> None:
    print(f"\n{SEPARATOR}")
    print("CHECK 4: Pipeline Dry Run (ingest -> preprocess -> split)")
    print(SEPARATOR)
    config = {
        "pipeline_name": "stream_diagnostic",
        "data_sources": [
            {
                "source_type": "stream",
                "name": "redis_stream",
                "params": {
                    "mode": "live",
                    "redis_url": redis_url,
                    "stream_type": "both",
                    "max_samples": 100,
                    "fallback_mode": "synthetic",
                    "fallback_synthetic_examples": 20,
                },
            }
        ],
        "preprocessing": {
            "use_text_cleaner": False,
            "use_deduplication": False,
            "min_text_length": 5,
        },
        "split": {"train_ratio": 0.70, "val_ratio": 0.15, "seed": 42},
    }

    pipeline = DynamicTrainingPipeline(config)
    pipeline.setup_data_sources()
    samples = pipeline.load_data()
    samples = pipeline.preprocess(samples)
    train, val, test = pipeline.split_data(samples)

    print("\n  Results:")
    print(f"  Loaded    : {len(samples)} samples")
    print(f"  Train     : {len(train)}")
    print(f"  Val       : {len(val)}")
    print(f"  Test      : {len(test)}")

    # Label distribution
    label_counts: dict = {}
    for s in samples:
        key = s.label_name or "unlabeled"
        label_counts[key] = label_counts.get(key, 0) + 1
    print(f"  Unique labels: {len(label_counts)}")
    if label_counts:
        top = sorted(label_counts.items(), key=lambda x: -x[1])[:5]
        for label, cnt in top:
            print(f"    {label}: {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Redis stream ingest endpoint")
    parser.add_argument(
        "--redis-url", default="redis://localhost:6379", help="Redis connection URL"
    )
    parser.add_argument(
        "--subscribe-timeout",
        type=float,
        default=5.0,
        help="Seconds to listen in subscribe mode (default: 5)",
    )
    parser.add_argument(
        "--skip-subscribe", action="store_true", help="Skip the subscribe mode check"
    )
    args = parser.parse_args()

    print("\nStream Endpoint Diagnostic")
    print(f"Redis: {args.redis_url}")

    connected = check_redis_connection(args.redis_url)
    live_count = check_live_mode(args.redis_url)

    if not args.skip_subscribe:
        check_subscribe_mode(args.redis_url, args.subscribe_timeout)

    check_pipeline_dry_run(args.redis_url)

    print(f"\n{SEPARATOR}")
    print("SUMMARY")
    print(SEPARATOR)
    print(f"  Redis connected : {'YES' if connected else 'NO (fallback active)'}")
    print(f"  Messages in stream : {live_count}")
    print(
        "\nTo populate the stream with test data, run:"
        "\n  python scripts/redis_mock_producer.py --count 50\n"
    )


if __name__ == "__main__":
    main()
