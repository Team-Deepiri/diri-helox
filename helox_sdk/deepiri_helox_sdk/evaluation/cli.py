"""CLI entry point for post-training evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .comparison import compare_model_directories
from .harness import PostTrainingEvalHarness
from .schemas import EvalRunConfig, EvalThresholds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="helox-eval",
        description="Post-training evaluation harness for Helox models",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run evaluation against a model checkpoint")
    run.add_argument("--model-path", required=True, type=Path)
    run.add_argument("--suite-path", required=True, type=Path)
    run.add_argument("--suite-name", default="default")
    run.add_argument("--output-dir", type=Path, default=Path("evaluation_runs"))
    run.add_argument("--mode", choices=["auto", "classifier", "generation"], default="auto")
    run.add_argument("--min-accuracy", type=float, default=0.0)
    run.add_argument("--min-f1", type=float, default=0.0)
    run.add_argument("--min-pass-rate", type=float, default=0.0)
    run.add_argument("--benchmark", action="store_true")
    run.add_argument("--fail-on-threshold", action="store_true")

    compare = sub.add_parser("compare", help="Compare model directories")
    compare.add_argument("model_paths", nargs="+", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config = EvalRunConfig(
            model_path=args.model_path,
            output_dir=args.output_dir,
            suite_name=args.suite_name,
            run_benchmark=args.benchmark,
            thresholds=EvalThresholds(
                min_accuracy=args.min_accuracy,
                min_f1=args.min_f1,
                min_pass_rate=args.min_pass_rate,
            ),
        )
        harness = PostTrainingEvalHarness(config)
        harness.load_suite(args.suite_name, args.suite_path)
        result = harness.run(mode=args.mode)
        print(json.dumps(result.to_dict(), indent=2))
        if args.fail_on_threshold and not result.passed:
            return 1
        return 0

    if args.command == "compare":
        report = compare_model_directories(args.model_paths)
        print(json.dumps(report.to_dict(), indent=2))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
