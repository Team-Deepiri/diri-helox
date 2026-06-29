from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from deepiri_helox_sdk.evaluation.harness import PostTrainingEvalHarness
from deepiri_helox_sdk.evaluation.parity import InferenceParityTester
from deepiri_helox_sdk.evaluation.schemas import EvalRunConfig, EvalThresholds
from deepiri_helox_sdk.evaluation.samples import EvaluationSample


class _DummyModel(nn.Module):
    def forward(self, input_ids=None, **kwargs):
        batch = input_ids.shape[0]
        vocab = 8
        return {"logits": torch.zeros(batch, vocab)}


def test_parity_suite_passes():
    model = _DummyModel()
    input_ids = torch.tensor([[1, 2, 3]])
    tester = InferenceParityTester()
    result = tester.run_full_parity_suite(model, input_ids)
    assert result["train_inference"]["parity_passed"] is True
    assert result["batch_size"]["parity_passed"] is True


def test_harness_classifier_mode_with_mock(tmp_path: Path):
    config = EvalRunConfig(
        model_path=tmp_path / "model",
        output_dir=tmp_path / "runs",
        suite_name="test_suite",
        thresholds=EvalThresholds(min_f1=0.5),
    )
    harness = PostTrainingEvalHarness(
        config,
        suites={
            "test_suite": [
                EvaluationSample(text="debug auth", label=0, label_name="debugging"),
                EvaluationSample(text="write docs", label=26, label_name="documentation"),
            ]
        },
    )

    fake_metrics = {
        "overall": {"accuracy": 1.0, "f1": 1.0},
        "per_class": {},
        "confusion_matrix": [[1, 0], [0, 1]],
        "_num_examples": 2,
    }

    with patch.object(harness.classifier, "evaluate", return_value=fake_metrics):
        result = harness.run(mode="classifier")

    assert result.passed is True
    assert result.classification["overall"]["f1"] == 1.0
    report_files = list((tmp_path / "runs").glob("eval_test_suite_*.json"))
    assert len(report_files) == 1


def test_harness_generation_mode_with_mock(tmp_path: Path):
    config = EvalRunConfig(
        model_path=tmp_path / "model",
        output_dir=tmp_path / "runs",
        suite_name="gen_suite",
        thresholds=EvalThresholds(min_pass_rate=0.5, min_avg_score=0.5),
    )
    harness = PostTrainingEvalHarness(
        config,
        suites={
            "gen_suite": [
                EvaluationSample(
                    prompt="status",
                    expected="ok",
                    test_type="exact_match",
                    threshold=1.0,
                )
            ]
        },
    )

    fake_generation = {
        "total_tests": 1,
        "passed_tests": 1,
        "pass_rate": 1.0,
        "avg_score": 1.0,
        "results": [],
    }

    with patch.object(harness, "_run_generation", return_value=fake_generation):
        result = harness.run(mode="generation")

    assert result.passed is True
    assert result.generation["pass_rate"] == 1.0


def test_harness_threshold_failure(tmp_path: Path):
    config = EvalRunConfig(
        model_path=tmp_path / "model",
        output_dir=tmp_path / "runs",
        thresholds=EvalThresholds(min_f1=0.99),
    )
    harness = PostTrainingEvalHarness(
        config,
        suites={"default": [EvaluationSample(text="x", label=0)]},
    )
    fake_metrics = {
        "overall": {"f1": 0.5, "accuracy": 0.5},
        "per_class": {},
        "confusion_matrix": [],
    }
    with patch.object(harness.classifier, "evaluate", return_value=fake_metrics):
        result = harness.run(mode="classifier")
    assert result.passed is False
    assert any("f1_below_threshold" in failure for failure in result.failures)
