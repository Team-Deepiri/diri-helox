#!/usr/bin/env python3
"""
CLI entry point for the DynamicTrainingPipeline.

Usage:
    # Full pipeline with 100 synthetic samples (default config)
    python scripts/pipelines/run_dynamic_pipeline.py \
        --config configs/dynamic_pipeline_config.json

    # Dry run: load and preprocess data without training
    python scripts/pipelines/run_dynamic_pipeline.py \
        --config configs/dynamic_pipeline_config.json --dry-run

    # Stream-based pipeline
    python scripts/pipelines/run_dynamic_pipeline.py \
        --config configs/dynamic_pipeline_stream_config.json

Run from the diri-helox root directory.
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure helox root is on the path
_HELOX_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_HELOX_ROOT))

from pipelines.training.dynamic_training_pipeline import DynamicTrainingPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Deepiri dynamic training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to pipeline JSON config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and preprocess data only — skip training and evaluation",
    )
    parser.add_argument(
        "--output-metrics",
        help="Optional path to write final metrics JSON (e.g. metrics.json)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        return 1

    with open(config_path) as f:
        config = json.load(f)

    pipeline = DynamicTrainingPipeline(config)

    if args.dry_run:
        print(f"\nDry run: loading data from config '{config_path.name}'")
        pipeline.setup_data_sources()
        samples = pipeline.load_data()
        samples = pipeline.preprocess(samples)
        train, val, test = pipeline.split_data(samples)
        print(f"\nDry run complete.")
        print(f"  Total samples : {len(samples)}")
        print(f"  Train / Val / Test : {len(train)} / {len(val)} / {len(test)}")
        label_counts = {}
        for s in samples:
            if s.label_name:
                label_counts[s.label_name] = label_counts.get(s.label_name, 0) + 1
        print(f"  Unique labels : {len(label_counts)}")
        return 0

    metrics = pipeline.run()

    if args.output_metrics:
        out = Path(args.output_metrics)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics written to: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
