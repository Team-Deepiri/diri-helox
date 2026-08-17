"""Tests for the subject-based automatic evaluation harness in the SDK."""

from __future__ import annotations

import json

import pytest

from deepiri_helox_sdk.evaluation import AutomaticEvaluationHarness
from deepiri_helox_sdk.evaluation.automatic_evaluation_harness import (
    is_retryable_error,
    retry_after_seconds,
)
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


def test_config_hash_tracks_score_affecting_settings(tmp_path):
    """The hash must move when the suite or generation settings move, and only then."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    tests = [
        {"id": "t1", "prompt": "p1", "expected": "good", "type": "contains"},
        {"id": "t2", "prompt": "p2", "expected": "good", "type": "contains"},
    ]
    harness.add_test_suite("gen", tests)
    subject = CallableGenerator(lambda p, **kw: "good", name="model_a")

    baseline = harness.evaluate_subject(subject, "gen")["config_hash"]

    # Reordering the same tests is not a change of what is measured.
    harness.add_test_suite("gen", list(reversed(tests)))
    assert harness.evaluate_subject(subject, "gen")["config_hash"] == baseline

    # Changing generation settings is.
    assert harness.evaluate_subject(subject, "gen", max_new_tokens=999)["config_hash"] != baseline

    # So is editing a prompt.
    harness.add_test_suite("gen", [{**tests[0], "prompt": "edited"}, tests[1]])
    assert harness.evaluate_subject(subject, "gen")["config_hash"] != baseline

    assert all(row["config_hash"] for row in harness.get_history(suite_name="gen"))


def test_suite_edit_resets_baseline_instead_of_flagging_regression(tmp_path, caplog):
    """A harder suite is a new baseline, not a model regression."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])
    subject = CallableGenerator(lambda p, **kw: "good", name="model_a")

    for _ in range(2):
        assert harness.evaluate_subject(subject, "gen")["avg_score"] == 1.0

    # Same model, but now asked for something it does not produce.
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "unreachable", "type": "contains"}])
    with caplog.at_level("WARNING"):
        harder = harness.evaluate_subject(subject, "gen")

    assert harder["avg_score"] == 0.0
    assert "regression" not in harder
    assert "baseline reset" in caplog.text

    # The new config builds its own baseline, and drops within it still fire.
    assert len(harness.get_history(suite_name="gen", config_hash=harder["config_hash"])) == 1


def test_score_stderr_shrinks_with_more_tests(tmp_path):
    """stderr is the uncertainty on the mean, so it must fall as the suite grows."""
    subject = CallableGenerator(lambda p, **kw: "good" if p == "hit" else "no", name="model_a")

    def run(repeats):
        harness = AutomaticEvaluationHarness(eval_dir=tmp_path / f"n{repeats}")
        # Half the tests score 1.0 and half 0.0, so the spread is fixed at 0.5.
        harness.add_test_suite(
            "gen",
            [
                {"prompt": "hit" if i % 2 else "miss", "expected": "good", "type": "contains"}
                for i in range(repeats)
            ],
        )
        return harness.evaluate_subject(subject, "gen")

    small, large = run(4), run(64)

    assert small["avg_score"] == large["avg_score"] == 0.5
    # Spread across tests is unchanged; only the confidence in the mean improves.
    assert small["score_std"] == pytest.approx(large["score_std"], abs=0.01)
    assert large["score_stderr"] < small["score_stderr"]
    assert large["score_stderr"] == pytest.approx(0.5 / 64**0.5, rel=0.05)


def test_score_stderr_is_zero_for_a_single_test(tmp_path):
    """A one-test suite has no sample spread, so stderr is undefined and reported as 0."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])
    result = harness.evaluate_subject(CallableGenerator(lambda p, **kw: "good", name="m"), "gen")
    assert result["score_stderr"] == 0.0


def _fixed_suite(harness, size=10):
    """A constant suite, so the config hash stays stable across runs."""
    harness.add_test_suite(
        "gen",
        [{"prompt": "p", "expected": "good", "type": "contains"} for _ in range(size)],
    )


def _subject_scoring(*hits_per_run, size=10):
    """Subject that lands exactly ``hits`` of ``size`` tests on each successive run."""
    responses = iter(["good"] * hits + ["no"] * (size - hits) for hits in hits_per_run)
    current = iter([])

    def generate(prompt, **kwargs):
        nonlocal current
        try:
            return next(current)
        except StopIteration:
            current = iter(next(responses))
            return next(current)

    return CallableGenerator(generate, name="model_a")


def test_noisy_suite_drop_is_treated_as_noise(tmp_path):
    """On a small, high-variance suite a modest drop must not fire on run two."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, min_pass_rate=0.0)
    _fixed_suite(harness)
    subject = _subject_scoring(6, 5)

    assert harness.evaluate_subject(subject, "gen")["avg_score"] == 0.6

    # 0.6 -> 0.5 clears regression_threshold (0.05) but is well inside the
    # combined standard error of two runs that each carry ~0.16 stderr.
    second = harness.evaluate_subject(subject, "gen")
    assert second["avg_score"] == 0.5
    assert second["score_stderr"] > 0.1
    assert "regression" not in second


def test_real_drop_still_fires_on_run_two(tmp_path):
    """The stderr floor must not swallow a drop far larger than the noise."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, min_pass_rate=0.0)
    _fixed_suite(harness)
    subject = _subject_scoring(10, 1)

    harness.evaluate_subject(subject, "gen")
    dropped = harness.evaluate_subject(subject, "gen")

    assert dropped["regression"]["detected"] is True
    assert dropped["regression"]["score_drop"] == pytest.approx(0.9)
    assert dropped["regression"]["score_drop"] > dropped["regression"]["stderr_floor"]


def _seed_history(harness, scores, suite="gen", subject="model_a", config_hash="cfg"):
    """Append prior runs straight to the history file, with no sampling noise."""
    with open(harness.history_file, "a", encoding="utf-8") as handle:
        for score in scores:
            handle.write(
                json.dumps(
                    {
                        "suite_name": suite,
                        "subject": subject,
                        "config_hash": config_hash,
                        "mode": "generation",
                        "avg_score": score,
                        "score_stderr": 0.0,
                    }
                )
                + "\n"
            )


def test_across_run_floor_uses_sample_stddev(tmp_path):
    """Prior runs are a sample, so the spread must be the n-1 estimate."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)

    # Prior runs of 0.9 and 0.5: mean 0.7, population sd 0.200, sample sd 0.283.
    _seed_history(harness, [0.9, 0.5])

    def check(current):
        return harness._check_regression(
            "gen", current, subject_name="model_a", config_hash="cfg", current_stderr=0.0
        )

    # A drop to 0.45 is 0.25 below the mean — outside the population sd but
    # inside the sample sd, so only the n-1 estimate correctly calls it noise.
    assert check(0.45) is None

    # Well past either estimate, so it must still be reported.
    assert check(0.30)["detected"] is True


class _RateLimited(Exception):
    """Stand-in for a provider rate-limit exception."""

    status_code = 429


class _BadPrompt(Exception):
    """Stand-in for a caller error the provider will never accept."""

    status_code = 400


def _no_backoff_harness(tmp_path, **kwargs):
    return AutomaticEvaluationHarness(
        eval_dir=tmp_path, retry_initial_backoff=0.0, retry_max_backoff=0.0, **kwargs
    )


def test_retryable_error_classification():
    assert is_retryable_error(_RateLimited()) is True
    assert is_retryable_error(TimeoutError()) is True
    assert is_retryable_error(ConnectionError()) is True
    # A 400 is our fault; retrying it just burns quota.
    assert is_retryable_error(_BadPrompt()) is False
    assert is_retryable_error(ValueError("bad prompt")) is False


def test_retry_after_header_wins_over_backoff():
    class _WithHeader(Exception):
        status_code = 429
        response = type("R", (), {"headers": {"retry-after": "2.5"}})()

    assert retry_after_seconds(_WithHeader()) == 2.5
    assert retry_after_seconds(_RateLimited()) is None


def test_transient_failures_are_retried(tmp_path):
    """A rate limit must cost a retry, not the whole run."""
    harness = _no_backoff_harness(tmp_path)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])

    attempts = {"n": 0}

    def flaky(prompt, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _RateLimited("slow down")
        return "good"

    result = harness.evaluate_subject(CallableGenerator(flaky, name="model_a"), "gen")

    assert result["avg_score"] == 1.0
    assert result["total_retries"] == 2
    assert result["results"][0]["retries"] == 2


def test_retries_are_capped(tmp_path):
    """Past max_retries the error surfaces instead of looping forever."""
    harness = _no_backoff_harness(tmp_path, max_retries=2)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])

    attempts = {"n": 0}

    def always_limited(prompt, **kwargs):
        attempts["n"] += 1
        raise _RateLimited("slow down")

    with pytest.raises(_RateLimited):
        harness.evaluate_subject(CallableGenerator(always_limited, name="model_a"), "gen")

    assert attempts["n"] == 3  # the original call plus two retries


def test_non_retryable_error_fails_fast(tmp_path):
    """A caller error must not be retried at all."""
    harness = _no_backoff_harness(tmp_path)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])

    attempts = {"n": 0}

    def bad(prompt, **kwargs):
        attempts["n"] += 1
        raise _BadPrompt("malformed")

    with pytest.raises(_BadPrompt):
        harness.evaluate_subject(CallableGenerator(bad, name="model_a"), "gen")

    assert attempts["n"] == 1


def test_retry_backoff_is_excluded_from_latency(tmp_path):
    """Latency must describe the model, not the time we spent waiting on it."""
    harness = AutomaticEvaluationHarness(
        eval_dir=tmp_path, retry_initial_backoff=0.2, retry_max_backoff=0.2
    )
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])

    attempts = {"n": 0}

    def flaky(prompt, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _RateLimited("slow down")
        return "good"

    result = harness.evaluate_subject(CallableGenerator(flaky, name="model_a"), "gen")

    assert result["results"][0]["retries"] == 1
    # The backoff slept at least 100ms; a fast local call must stay well under it.
    assert result["results"][0]["latency_ms"] < 50


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
