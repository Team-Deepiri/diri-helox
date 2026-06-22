from __future__ import annotations

import json
from pathlib import Path

from deepiri_helox_sdk.evaluation.cli import main
from deepiri_helox_sdk.evaluation.report import load_eval_report, save_eval_report
from deepiri_helox_sdk.evaluation.schemas import EvalRunResult


def test_report_roundtrip(tmp_path: Path):
    result = EvalRunResult(
        run_id="abc",
        model_path="/tmp/model",
        suite_name="suite",
        passed=True,
    )
    path = save_eval_report(result, tmp_path / "report.json")
    loaded = load_eval_report(path)
    assert loaded["run_id"] == "abc"
    assert loaded["passed"] is True


def test_cli_compare_command(tmp_path: Path, capsys):
    model = tmp_path / "m1"
    model.mkdir()
    (model / "evaluation_report.json").write_text(
        json.dumps({"metrics": {"overall": {"f1": 0.9}}}),
        encoding="utf-8",
    )
    code = main(["compare", str(model)])
    assert code == 0
    captured = capsys.readouterr()
    assert "m1" in captured.out
