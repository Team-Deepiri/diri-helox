from __future__ import annotations

from pathlib import Path

import pytest

from deepiri_helox_sdk.evaluation.regression import RegressionTracker


def test_regression_tracker_detects_drop(tmp_path: Path):
    tracker = RegressionTracker(tmp_path, regression_threshold=0.05)
    tracker.record({"suite_name": "suite_a", "avg_score": 0.90})
    tracker.record({"suite_name": "suite_a", "avg_score": 0.80})

    regression = tracker.check_score_regression("suite_a", "avg_score", 0.70)
    assert regression is not None
    assert regression["detected"] is True
    assert regression["previous_best"] == 0.90
    assert regression["score_drop"] == pytest.approx(0.20)


def test_regression_tracker_no_regression(tmp_path: Path):
    tracker = RegressionTracker(tmp_path, regression_threshold=0.05)
    tracker.record({"suite_name": "suite_a", "f1": 0.80})

    regression = tracker.check_score_regression("suite_a", "f1", 0.79)
    assert regression is None


def test_regression_summary(tmp_path: Path):
    tracker = RegressionTracker(tmp_path)
    tracker.record({"suite_name": "suite_a", "avg_score": 0.5})
    tracker.record({"suite_name": "suite_a", "avg_score": 0.7})
    summary = tracker.summary()
    assert summary["total_evaluations"] == 2
    assert summary["suites"]["suite_a"]["count"] == 2
    assert summary["suites"]["suite_a"]["max_score"] == 0.7
