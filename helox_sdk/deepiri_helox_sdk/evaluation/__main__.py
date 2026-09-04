"""
Command-line entry point for the automatic evaluation harness.

Usage::

    python -m deepiri_helox_sdk.evaluation run --suite gen --model-path path/to/model
    python -m deepiri_helox_sdk.evaluation classify --suite clf --model-path path/to/classifier
    python -m deepiri_helox_sdk.evaluation benchmark --model-path path/to/model
    python -m deepiri_helox_sdk.evaluation validate --suite gen
    python -m deepiri_helox_sdk.evaluation summary [--eval-dir DIR]
    python -m deepiri_helox_sdk.evaluation history [--eval-dir DIR] [--suite gen]

``run`` loads the named suite (JSONL), wraps the HF causal LM in a
:class:`~deepiri_helox_sdk.evaluation.subjects.HFModelGenerator`, and evaluates
it. ``classify`` wraps an HF sequence classifier in an
:class:`~deepiri_helox_sdk.evaluation.subjects.HFClassifierPredictor`. Results
are printed as JSON and persisted to ``--eval-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

_PACKAGE_SUITE_DIR = Path(__file__).parent / "suites"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deepiri_helox_sdk.evaluation",
        description="Automatic evaluation harness CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_common_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--eval-dir", type=Path, default=Path("evaluation"))
        subparser.add_argument("--suite-dir", type=Path, default=_PACKAGE_SUITE_DIR)

    run = subparsers.add_parser("run", help="Evaluate a model against a suite")
    _add_common_args(run)
    run.add_argument("--model-path", type=str, required=True)
    run.add_argument("--suite", type=str, required=True)
    run.add_argument("--max-new-tokens", type=int, default=100)
    run.add_argument("--min-pass-rate", type=float, default=0.5)
    run.add_argument("--min-avg-score", type=float, default=0.0)
    run.add_argument("--quiet", action="store_true")

    classify = subparsers.add_parser("classify", help="Evaluate a classifier suite")
    _add_common_args(classify)
    classify.add_argument("--model-path", type=str, required=True)
    classify.add_argument("--suite", type=str, required=True)
    classify.add_argument("--batch-size", type=int, default=32)
    classify.add_argument("--max-length", type=int, default=128)
    classify.add_argument("--min-accuracy", type=float, default=0.0)
    classify.add_argument("--min-f1", type=float, default=0.0)
    classify.add_argument("--quiet", action="store_true")

    bench = subparsers.add_parser("benchmark", help="Benchmark model latency")
    _add_common_args(bench)
    bench.add_argument("--model-path", type=str, required=True)
    bench.add_argument("--prompt", type=str, default="Write a hello world function.")
    bench.add_argument("--max-new-tokens", type=int, default=50)
    bench.add_argument("--runs", type=int, default=5)

    validate = subparsers.add_parser("validate", help="Validate a suite without running it")
    _add_common_args(validate)
    validate.add_argument("--suite", type=str, required=True)

    summary = subparsers.add_parser("summary", help="Show aggregate evaluation history")
    _add_common_args(summary)

    hist = subparsers.add_parser("history", help="Show evaluation history")
    _add_common_args(hist)
    hist.add_argument("--suite", type=str, default=None)

    return parser


def _load_suite_names(suite_dir: Path) -> List[str]:
    if not suite_dir.exists():
        return []
    return sorted(p.stem for p in suite_dir.glob("*.jsonl"))


def _load_suites(harness, suite_dir: Path) -> None:
    if suite_dir.exists() and suite_dir.is_dir():
        harness.load_suites_from_dir(suite_dir)
    elif suite_dir.exists() and suite_dir.is_file():
        harness.load_test_suite(suite_dir.stem, suite_dir)
    else:
        print(f"error: suite dir/file not found: {suite_dir}", file=sys.stderr)
        raise SystemExit(1)


def _cmd_run(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .automatic_evaluation_harness import AutomaticEvaluationHarness
    from .subjects import HFModelGenerator

    harness = AutomaticEvaluationHarness(
        eval_dir=args.eval_dir,
        min_pass_rate=args.min_pass_rate,
        min_avg_score=args.min_avg_score,
    )
    _load_suites(harness, args.suite_dir)
    if args.suite not in harness.list_suites():
        print(
            f"error: suite {args.suite!r} not loaded. available: {_load_suite_names(args.suite_dir)}",
            file=sys.stderr,
        )
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    subject = HFModelGenerator(model, tokenizer)

    result = harness.evaluate_subject(subject, args.suite, max_new_tokens=args.max_new_tokens)
    if args.quiet:
        summary = {k: result[k] for k in ("suite_name", "mode", "passed") if k in result}
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    from .automatic_evaluation_harness import AutomaticEvaluationHarness
    from .subjects import HFClassifierPredictor

    harness = AutomaticEvaluationHarness(
        eval_dir=args.eval_dir,
        min_accuracy=args.min_accuracy,
        min_f1=args.min_f1,
    )
    _load_suites(harness, args.suite_dir)
    if args.suite not in harness.list_suites():
        print(
            f"error: suite {args.suite!r} not loaded. available: {_load_suite_names(args.suite_dir)}",
            file=sys.stderr,
        )
        return 1

    predictor = HFClassifierPredictor(
        args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    result = harness.evaluate_predictor(predictor, args.suite)
    if args.quiet:
        summary = {k: result[k] for k in ("suite_name", "mode", "passed") if k in result}
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .automatic_evaluation_harness import AutomaticEvaluationHarness
    from .subjects import HFModelGenerator

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    subject = HFModelGenerator(model, tokenizer)
    harness = AutomaticEvaluationHarness(eval_dir=args.eval_dir)
    stats = harness.benchmark_subject(
        subject,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_runs=args.runs,
    )
    print(json.dumps(stats, indent=2, default=str))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from .automatic_evaluation_harness import AutomaticEvaluationHarness

    harness = AutomaticEvaluationHarness(eval_dir=args.eval_dir)
    _load_suites(harness, args.suite_dir)
    if args.suite not in harness.list_suites():
        print(
            f"error: suite {args.suite!r} not loaded. available: {_load_suite_names(args.suite_dir)}",
            file=sys.stderr,
        )
        return 1
    report = harness.validate_suite(args.suite)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("valid") else 1


def _cmd_summary(args: argparse.Namespace) -> int:
    from .automatic_evaluation_harness import AutomaticEvaluationHarness

    harness = AutomaticEvaluationHarness(eval_dir=args.eval_dir)
    print(json.dumps(harness.get_evaluation_summary(), indent=2, default=str))
    return 0


def _cmd_history(args: argparse.Namespace) -> int:
    from .automatic_evaluation_harness import AutomaticEvaluationHarness

    harness = AutomaticEvaluationHarness(eval_dir=args.eval_dir)
    records = harness.get_history(args.suite)
    print(json.dumps(records, indent=2, default=str))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "classify": _cmd_classify,
        "benchmark": _cmd_benchmark,
        "validate": _cmd_validate,
        "summary": _cmd_summary,
        "history": _cmd_history,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
