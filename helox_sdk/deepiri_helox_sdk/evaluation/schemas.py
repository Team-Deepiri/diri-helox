"""Configuration and result schemas for post-training evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class EvalThresholds:
    """Pass/fail gates applied after an evaluation run."""

    min_accuracy: float = 0.0
    min_f1: float = 0.0
    min_pass_rate: float = 0.0
    max_regression_drop: float = 0.05
    min_avg_score: float = 0.0


@dataclass(slots=True)
class EvalRunConfig:
    """Single post-training evaluation run."""

    model_path: Path
    output_dir: Path = Path("evaluation_runs")
    suite_name: str = "default"
    batch_size: int = 32
    max_length: int = 128
    num_labels: int = 31
    max_new_tokens: int = 100
    regression_threshold: float = 0.05
    thresholds: EvalThresholds = field(default_factory=EvalThresholds)
    run_parity: bool = False
    run_benchmark: bool = False
    benchmark_samples: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRunResult:
    """Aggregated output from a post-training evaluation harness run."""

    run_id: str
    model_path: str
    suite_name: str
    timestamp: str = field(default_factory=_utc_now_iso)
    classification: Optional[Dict[str, Any]] = None
    generation: Optional[Dict[str, Any]] = None
    parity: Optional[Dict[str, Any]] = None
    benchmark: Optional[Dict[str, Any]] = None
    regression: Optional[Dict[str, Any]] = None
    passed: bool = True
    failures: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_path": self.model_path,
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "classification": self.classification,
            "generation": self.generation,
            "parity": self.parity,
            "benchmark": self.benchmark,
            "regression": self.regression,
            "passed": self.passed,
            "failures": self.failures,
            "metadata": self.metadata,
        }
