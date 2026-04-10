#!/usr/bin/env python3
"""Model comparison and efficiency benchmarking utility."""

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    """Load JSON file if it exists, otherwise return an empty dict."""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"⚠️  Failed to parse {path}: {exc}")
        return {}


def load_training_info(model_path: Path) -> Dict[str, Any]:
    """Load training info from model directory."""
    return load_json_if_exists(model_path / "training_info.json")


def load_evaluation_report(model_path: Path) -> Dict[str, Any]:
    """Load evaluation report from model directory."""
    return load_json_if_exists(model_path / "evaluation_report.json")


def pick_number(*values: Any) -> Optional[float]:
    """Return the first value that can be parsed as float."""
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def extract_metrics(training_info: Dict[str, Any], eval_report: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metrics from training/evaluation files into one structure."""
    overall = eval_report.get("metrics", {}).get("overall", {})
    benchmark = eval_report.get("benchmark", {})

    accuracy = pick_number(overall.get("accuracy"), training_info.get("eval_accuracy"))
    f1 = pick_number(overall.get("f1"), training_info.get("eval_f1"))
    precision = pick_number(overall.get("precision"))
    recall = pick_number(overall.get("recall"))
    avg_confidence = pick_number(overall.get("avg_confidence"))

    avg_latency_ms = pick_number(benchmark.get("avg_latency_ms"), eval_report.get("avg_latency_ms"))
    p95_latency_ms = pick_number(benchmark.get("p95_latency_ms"), eval_report.get("p95_latency_ms"))
    throughput_per_sec = pick_number(
        benchmark.get("throughput_per_sec"), eval_report.get("throughput_per_sec")
    )

    quality = pick_number(f1, accuracy)

    efficiency_score: Optional[float] = None
    if quality is not None and throughput_per_sec is not None and throughput_per_sec > 0:
        efficiency_score = quality * throughput_per_sec
    elif quality is not None and avg_latency_ms is not None and avg_latency_ms > 0:
        efficiency_score = quality / avg_latency_ms

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "avg_confidence": avg_confidence,
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "throughput_per_sec": throughput_per_sec,
        "efficiency_score": efficiency_score,
        "quality_score": quality,
        "num_epochs": training_info.get("num_epochs"),
        "train_examples": training_info.get("train_examples"),
    }


def format_metric(value: Optional[float], digits: int = 4) -> str:
    """Format numeric metric for console output."""
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def build_model_record(model_path: Path) -> Dict[str, Any]:
    """Build full model record from on-disk metadata."""
    training_info = load_training_info(model_path)
    eval_report = load_evaluation_report(model_path)
    metrics = extract_metrics(training_info, eval_report)

    return {
        "name": model_path.name,
        "path": str(model_path),
        "training_info": training_info,
        "eval_report": eval_report,
        "metrics": metrics,
    }


def rank_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank models by efficiency first, then quality metrics."""

    def score(record: Dict[str, Any]) -> Any:
        metrics = record["metrics"]
        return (
            (
                metrics.get("efficiency_score")
                if metrics.get("efficiency_score") is not None
                else -1.0
            ),
            metrics.get("f1") if metrics.get("f1") is not None else -1.0,
            metrics.get("accuracy") if metrics.get("accuracy") is not None else -1.0,
            record["name"],
        )

    return sorted(models, key=score, reverse=True)


def print_comparison_table(models: List[Dict[str, Any]]) -> None:
    """Print compact, repeatable comparison table."""
    print("=" * 120)
    print("MODEL COMPARISON")
    print("=" * 120)
    print()
    print(
        f"{'Model':<28} {'Acc':<8} {'F1':<8} {'Latency(ms)':<12} "
        f"{'Throughput/s':<13} {'Eff.Score':<10} {'Epochs':<8} {'Examples':<10}"
    )
    print("-" * 120)

    for record in models:
        metrics = record["metrics"]
        print(
            f"{record['name'][:28]:<28} "
            f"{format_metric(metrics.get('accuracy')):<8} "
            f"{format_metric(metrics.get('f1')):<8} "
            f"{format_metric(metrics.get('avg_latency_ms'), 2):<12} "
            f"{format_metric(metrics.get('throughput_per_sec'), 2):<13} "
            f"{format_metric(metrics.get('efficiency_score'), 6):<10} "
            f"{str(metrics.get('num_epochs', 'N/A')):<8} "
            f"{str(metrics.get('train_examples', 'N/A')):<10}"
        )


def print_detailed_metrics(models: List[Dict[str, Any]]) -> None:
    """Print per-model detailed metrics."""
    print("\n" + "=" * 120)
    print("DETAILED METRICS")
    print("=" * 120)

    for idx, record in enumerate(models, start=1):
        metrics = record["metrics"]
        print(f"\n{idx}. {record['name']}")
        print(f"   Path: {record['path']}")
        print(f"   Accuracy:       {format_metric(metrics.get('accuracy'))}")
        print(f"   F1 Score:       {format_metric(metrics.get('f1'))}")
        print(f"   Precision:      {format_metric(metrics.get('precision'))}")
        print(f"   Recall:         {format_metric(metrics.get('recall'))}")
        print(f"   Avg Confidence: {format_metric(metrics.get('avg_confidence'))}")
        print(f"   Avg Latency ms: {format_metric(metrics.get('avg_latency_ms'), 2)}")
        print(f"   P95 Latency ms: {format_metric(metrics.get('p95_latency_ms'), 2)}")
        print(f"   Throughput/s:   {format_metric(metrics.get('throughput_per_sec'), 2)}")
        print(f"   Efficiency:     {format_metric(metrics.get('efficiency_score'), 6)}")


def save_report(models: List[Dict[str, Any]], output_path: Path) -> None:
    """Save repeatable comparison output to JSON."""
    report = {
        "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "scoring": {
            "efficiency_score": "quality_score * throughput_per_sec when available; otherwise quality_score / avg_latency_ms",
            "quality_score": "f1 when available, else accuracy",
        },
        "models": models,
        "ranking": [record["name"] for record in models],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\n📝 Saved comparison report: {output_path}")


def compare_models(model_paths: List[str], output_report: Optional[str], emit_json: bool) -> None:
    """Compare multiple models and optionally persist a report."""
    records: List[Dict[str, Any]] = []

    for raw_path in model_paths:
        model_path = Path(raw_path)
        if not model_path.exists():
            print(f"⚠️  Model not found: {raw_path}")
            continue
        records.append(build_model_record(model_path))

    if not records:
        print("❌ No valid models found")
        return

    # Stable sort for repeatability before ranking.
    records = sorted(records, key=lambda item: item["name"])
    ranked = rank_models(records)

    print_comparison_table(ranked)
    print_detailed_metrics(ranked)

    best = ranked[0]
    print("\n" + "=" * 120)
    print("🏆 TOP RECOMMENDATION")
    print("=" * 120)
    print(f"Name: {best['name']}")
    print(f"Path: {best['path']}")
    print(f"Efficiency Score: {format_metric(best['metrics'].get('efficiency_score'), 6)}")
    print(f"F1 Score: {format_metric(best['metrics'].get('f1'))}")
    print(f"Accuracy: {format_metric(best['metrics'].get('accuracy'))}")

    if output_report:
        save_report(ranked, Path(output_report))

    if emit_json:
        print("\nJSON_OUTPUT_START")
        print(json.dumps(ranked, indent=2))
        print("JSON_OUTPUT_END")


def create_model_snapshot(model_path: str, version: Optional[str] = None) -> bool:
    """Create a versioned snapshot of the model."""
    source = Path(model_path)
    if not source.exists():
        print(f"❌ Model not found: {source}")
        return False

    if version is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v_{timestamp}"

    snapshot_dir = source.parent / f"{source.name}_{version}"
    print(f"Creating model snapshot: {snapshot_dir}")

    shutil.copytree(source, snapshot_dir, dirs_exist_ok=True)

    snapshot_info = {
        "original_path": str(source),
        "snapshot_version": version,
        "created_at": datetime.datetime.now().isoformat(),
    }

    with open(snapshot_dir / "snapshot_info.json", "w", encoding="utf-8") as handle:
        json.dump(snapshot_info, handle, indent=2)

    print(f"✓ Snapshot created: {snapshot_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["models/intent_classifier"],
        help="Paths to model directories to compare",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Create a versioned snapshot of each model path",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version name used when --snapshot is enabled",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="scripts/evaluation/model_comparison_report.json",
        help="JSON path for comparison output (set empty string to skip)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit model ranking JSON to stdout",
    )

    args = parser.parse_args()

    if args.snapshot:
        for path in args.models:
            create_model_snapshot(path, args.version)
        return

    report_path = args.output_report.strip() if args.output_report else ""
    compare_models(
        model_paths=args.models,
        output_report=report_path if report_path else None,
        emit_json=args.json,
    )


if __name__ == "__main__":
    main()
