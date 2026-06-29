"""Compare multiple trained model directories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import efficiency_score, pick_number


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


@dataclass(slots=True)
class ModelComparisonReport:
    """Normalized comparison across one or more model artifact directories."""

    models: List[Dict[str, Any]] = field(default_factory=list)
    winner: Optional[str] = None
    ranking: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "models": self.models,
            "winner": self.winner,
            "ranking": self.ranking,
        }


def extract_model_metrics(model_path: Path) -> Dict[str, Any]:
    """Load training and evaluation artifacts from a model directory."""
    training_info = _load_json_if_exists(model_path / "training_info.json")
    eval_report = _load_json_if_exists(model_path / "evaluation_report.json")
    overall = eval_report.get("metrics", {}).get("overall", {})
    benchmark = eval_report.get("benchmark", {})

    accuracy = pick_number(overall.get("accuracy"), training_info.get("eval_accuracy"))
    f1 = pick_number(overall.get("f1"), training_info.get("eval_f1"))
    precision = pick_number(overall.get("precision"))
    recall = pick_number(overall.get("recall"))
    avg_confidence = pick_number(overall.get("avg_confidence"))
    avg_latency_ms = pick_number(benchmark.get("avg_latency_ms"), eval_report.get("avg_latency_ms"))
    throughput = pick_number(
        benchmark.get("throughput_per_sec"),
        eval_report.get("throughput_per_sec"),
    )
    quality = pick_number(f1, accuracy)

    return {
        "model_path": str(model_path),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "avg_confidence": avg_confidence,
        "avg_latency_ms": avg_latency_ms,
        "throughput_per_sec": throughput,
        "efficiency_score": efficiency_score(quality, throughput, avg_latency_ms),
    }


def compare_model_directories(model_paths: List[Path]) -> ModelComparisonReport:
    """Build a ranked comparison report from on-disk model directories."""
    rows = [extract_model_metrics(Path(path)) for path in model_paths]
    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("efficiency_score") is not None,
            row.get("efficiency_score") or 0.0,
            row.get("f1") or 0.0,
            row.get("accuracy") or 0.0,
        ),
        reverse=True,
    )
    ranking = [str(row["model_path"]) for row in ranked]
    winner = ranking[0] if ranking else None
    return ModelComparisonReport(models=ranked, winner=winner, ranking=ranking)
