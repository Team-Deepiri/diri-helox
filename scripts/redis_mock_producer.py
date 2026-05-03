#!/usr/bin/env python3
"""
Redis Mock Producer — publishes synthetic task requests to helox training streams.

Use this to test the full ingestion pipeline without needing a live Cyrex instance.
Messages are published in the same format Cyrex uses, so StreamDataSource consumes
them identically whether they come from this script or from the real system.

Usage:
    # Publish 20 messages to both streams
    python scripts/redis_mock_producer.py --count 20

    # Publish only to raw stream, with 0.2s delay between messages
    python scripts/redis_mock_producer.py --stream raw --count 10 --delay 0.2

    # Custom Redis URL
    python scripts/redis_mock_producer.py --redis-url redis://localhost:6380 --count 50
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

_HELOX_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_HELOX_ROOT))

RAW_STREAM = "pipeline.helox-training.raw"
STRUCTURED_STREAM = "pipeline.helox-training.structured"

CATEGORIES = [
    "debugging",
    "refactoring",
    "writing_code",
    "programming",
    "running_code",
    "inspecting",
    "writing",
    "learning_research",
    "learning_study",
    "learning_training",
    "learning_practice",
    "creative",
    "administrative",
    "team_organization",
    "team_collaboration",
    "team_planning",
    "research",
    "planning",
    "communication",
    "big_data_analytics",
    "data_processing",
    "design",
    "qa",
    "testing",
    "validation",
    "reporting",
    "documentation",
    "system_admin",
    "ux_ui",
    "security",
    "data_privacy",
]

RAW_TEMPLATES = [
    "Fix the {issue} in the {module} module",
    "Debug the failing {component} tests",
    "Review and clean up the {module} codebase",
    "Investigate the {issue} error in production",
    "Write unit tests for the {component} service",
    "Set up monitoring for {component} metrics",
    "Refactor the {module} to improve readability",
    "Document the {component} API endpoints",
    "Optimize the {module} performance bottleneck",
    "Deploy the latest {component} changes to staging",
]

STRUCTURED_TEMPLATES = [
    (
        "Implement a {feature} for the {component}",
        "Follow existing patterns",
        "Implementation complete",
    ),
    ("Review the PR for {module}", "Check for edge cases and style", "Review done with comments"),
    ("Write documentation for {feature}", "Include examples", "Docs updated in README"),
    ("Debug {issue} in {component}", "Check logs and traces", "Root cause identified and fixed"),
    ("Optimize {module} query performance", "Profile first", "Query time reduced by 40%"),
]

ISSUES = ["NullPointerException", "memory leak", "race condition", "timeout", "auth failure"]
MODULES = ["auth", "data_pipeline", "training", "inference", "streaming", "registry"]
COMPONENTS = ["API gateway", "tokenizer", "evaluator", "scheduler", "cache layer"]
FEATURES = ["rate limiting", "batch processing", "caching", "retry logic", "health checks"]


def _make_raw_record(idx: int) -> dict:
    template = random.choice(RAW_TEMPLATES)
    text = template.format(
        issue=random.choice(ISSUES),
        module=random.choice(MODULES),
        component=random.choice(COMPONENTS),
    )
    return {
        "id": f"mock-raw-{idx:04d}",
        "text": text,
        "source": "mock_producer",
        "quality_score": round(random.uniform(0.5, 1.0), 2),
        "timestamp": time.time(),
    }


def _make_structured_record(idx: int) -> dict:
    instruction_tpl, input_tpl, output_tpl = random.choice(STRUCTURED_TEMPLATES)
    category = random.choice(CATEGORIES)
    instruction = instruction_tpl.format(
        feature=random.choice(FEATURES),
        module=random.choice(MODULES),
        component=random.choice(COMPONENTS),
        issue=random.choice(ISSUES),
    )
    return {
        "id": f"mock-structured-{idx:04d}",
        "instruction": instruction,
        "input": input_tpl,
        "output": output_tpl,
        "category": category,
        "quality_score": round(random.uniform(0.5, 1.0), 2),
        "timestamp": time.time(),
    }


def publish(redis_url: str, stream: str, count: int, delay: float) -> None:
    try:
        import redis as redis_lib
    except ImportError:
        print("Error: redis-py not installed. Run: pip install redis")
        sys.exit(1)

    try:
        r = redis_lib.from_url(redis_url, socket_connect_timeout=3)
        r.ping()
    except Exception as exc:
        print(f"Error: cannot connect to Redis at {redis_url}: {exc}")
        sys.exit(1)

    streams_to_publish = []
    if stream in ("raw", "both"):
        streams_to_publish.append(("raw", RAW_STREAM))
    if stream in ("structured", "both"):
        streams_to_publish.append(("structured", STRUCTURED_STREAM))

    total_published = 0
    for i in range(count):
        for record_type, stream_name in streams_to_publish:
            record = _make_raw_record(i) if record_type == "raw" else _make_structured_record(i)
            msg_id = r.xadd(stream_name, {"payload": json.dumps(record)})
            total_published += 1
            print(
                f"  [{total_published:04d}] -> {stream_name} "
                f"(id={msg_id.decode() if isinstance(msg_id, bytes) else msg_id}, "
                f"quality={record['quality_score']})"
            )
        if delay > 0 and i < count - 1:
            time.sleep(delay)

    print(f"\nDone. Published {total_published} messages to Redis at {redis_url}")
    for _, stream_name in streams_to_publish:
        length = r.xlen(stream_name)
        print(f"  {stream_name}: {length} total messages in stream")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish mock task requests to Redis streams")
    parser.add_argument(
        "--redis-url", default="redis://localhost:6379", help="Redis connection URL"
    )
    parser.add_argument(
        "--stream",
        choices=["raw", "structured", "both"],
        default="both",
        help="Which stream(s) to publish to",
    )
    parser.add_argument("--count", type=int, default=20, help="Number of messages to publish")
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Seconds between messages (default: 0)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Publishing {args.count} mock messages to {args.redis_url}")
    print(f"  Stream: {args.stream}")
    print(f"  Delay:  {args.delay}s\n")

    publish(args.redis_url, args.stream, args.count, args.delay)


if __name__ == "__main__":
    main()
