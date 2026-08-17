"""Tests for the subject-based automatic evaluation harness in the SDK."""

from __future__ import annotations

from deepiri_helox_sdk.evaluation import AutomaticEvaluationHarness
from deepiri_helox_sdk.evaluation.metrics import (
    classification_metrics,
    score_response,
    token_f1_score,
)
from deepiri_helox_sdk.evaluation.subjects import (
    CallableGenerator,
    OllamaGenerator,
    OpenAIGenerator,
)


def test_full_scoring_surface():
    assert score_response("a", "a", "exact_match") == 1.0
    assert score_response("x 42 y", "41|42|43", "contains_any") == 1.0
    assert score_response("n: 3.14", "3.14", "numeric_match") == 1.0
    assert score_response("a@b.com", r"[\w.]+@[\w.]+\.com", "regex_match") == 1.0
    assert score_response('{"a": 1, "b": 2}', '{"a": 1, "b": 9}', "json_match") == 0.5
    assert token_f1_score("a a b", "a b b") == 2 / 3


def test_classification_metrics_label_names():
    metrics = classification_metrics(
        [0, 1], [0, 1], y_conf=[0.9, 0.8], label_names={0: "debugging", 1: "refactoring"}
    )
    assert metrics["overall"]["accuracy"] == 1.0
    assert set(metrics["per_class"]) == {"debugging", "refactoring"}


def test_evaluate_subject_end_to_end(tmp_path):
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    harness.add_test_suite(
        "gen",
        [
            {"id": "t1", "prompt": "write add", "expected": "def add", "type": "contains"},
            {"id": "t2", "prompt": "write add", "expected": "unrelated", "type": "contains"},
        ],
    )
    subject = CallableGenerator(lambda prompt, max_new_tokens: "def add(a, b)")
    result = harness.evaluate_subject(subject, "gen")
    assert result["total_tests"] == 2
    assert result["passed_tests"] == 1
    assert result["avg_score"] == 0.5
    assert result["passed"] is True


def test_regression_history_is_scoped_to_subject(tmp_path):
    """A weak subject must not be flagged against a *different* subject's history."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])

    strong = CallableGenerator(lambda p, **kw: "good", name="model_a")
    weak = CallableGenerator(lambda p, **kw: "bad", name="model_b")

    for _ in range(2):
        assert harness.evaluate_subject(strong, "gen")["avg_score"] == 1.0

    # model_b has no history of its own, so its low score is a baseline, not a drop.
    first_b = harness.evaluate_subject(weak, "gen")
    assert first_b["avg_score"] == 0.0
    assert "regression" not in first_b

    assert [row["subject"] for row in harness.get_history(suite_name="gen")] == [
        "model_a",
        "model_a",
        "model_b",
    ]
    assert len(harness.get_history(suite_name="gen", subject_name="model_a")) == 2


def test_regression_still_detected_within_one_subject(tmp_path):
    """Scoping by subject must not stop a real drop by the same subject firing."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])

    responses = iter(["good", "bad"])
    subject = CallableGenerator(lambda p, **kw: next(responses), name="model_a")

    harness.evaluate_subject(subject, "gen")
    dropped = harness.evaluate_subject(subject, "gen")

    assert dropped["regression"]["detected"] is True
    assert dropped["regression"]["previous_best"] == 1.0
    assert dropped["passed"] is False


def test_run_evaluation_matrix(tmp_path):
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    harness.add_test_suite("a", [{"prompt": "p", "expected": "good", "type": "contains"}])
    subjects = {"agent1": CallableGenerator(lambda p, **kw: "good answer")}
    matrix = harness.run_evaluation_matrix(subjects)
    assert matrix["matrix"]["a"]["agent1"]["passed_tests"] == 1


def test_export_markdown_report(tmp_path):
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    report = harness.export_markdown_report(
        {
            "gen": {
                "suite_name": "gen",
                "passed": True,
                "pass_rate": 1.0,
                "avg_score": 0.85,
                "total_tests": 1,
                "passed_tests": 1,
                "mode": "generation",
            }
        }
    )
    assert "# Evaluation Report" in report
    assert "| avg_score | 0.850 |" in report


def test_api_generators_require_provider_packages():
    from importlib.util import find_spec

    if find_spec("openai") is None:
        try:
            OpenAIGenerator(name="openai", model="gpt-4o")
        except ImportError as exc:
            assert "openai" in str(exc)
        else:
            raise AssertionError("expected ImportError without openai installed")

    if find_spec("ollama") is None:
        try:
            OllamaGenerator(model="llama3")
        except ImportError as exc:
            assert "ollama" in str(exc)
        else:
            raise AssertionError("expected ImportError without ollama installed")
