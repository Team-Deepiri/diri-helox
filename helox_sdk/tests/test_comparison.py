from __future__ import annotations

import json
from pathlib import Path

from deepiri_helox_sdk.evaluation.comparison import compare_model_directories


def _write_model_dir(path: Path, f1: float, throughput: float) -> None:
    path.mkdir(parents=True)
    (path / "evaluation_report.json").write_text(
        json.dumps(
            {
                "metrics": {"overall": {"f1": f1, "accuracy": f1}},
                "benchmark": {"throughput_per_sec": throughput, "avg_latency_ms": 10.0},
            }
        ),
        encoding="utf-8",
    )


def test_compare_model_directories_ranks_by_efficiency(tmp_path: Path):
    model_a = tmp_path / "model_a"
    model_b = tmp_path / "model_b"
    _write_model_dir(model_a, f1=0.9, throughput=50)
    _write_model_dir(model_b, f1=0.8, throughput=200)

    report = compare_model_directories([model_a, model_b])
    assert report.winner == str(model_b)
    assert report.ranking[0] == str(model_b)
    assert report.models[0]["efficiency_score"] == 160.0
