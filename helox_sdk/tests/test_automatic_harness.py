"""Tests for the subject-based automatic evaluation harness in the SDK."""

from __future__ import annotations

import json
import time

import pytest

from deepiri_helox_sdk.evaluation import AutomaticEvaluationHarness
from deepiri_helox_sdk.evaluation.automatic_evaluation_harness import (
    is_retryable_error,
    retry_after_seconds,
)
from deepiri_helox_sdk.evaluation.judge import JudgeParseError, LlmJudge
from deepiri_helox_sdk.evaluation.judge_validation import (
    JudgeValidator,
    cohen_kappa,
    validate_judge,
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


def _failing_on(harness, failing_prompts, size=10):
    """Suite of ``size`` tests plus a subject that fails on the named prompts."""
    harness.add_test_suite(
        "gen",
        [
            {"id": str(i), "prompt": str(i), "expected": "good", "type": "contains"}
            for i in range(size)
        ],
    )

    def generate(prompt, **kwargs):
        if prompt in failing_prompts:
            raise _RateLimited("slow down")
        return "good"

    return CallableGenerator(generate, name="model_a")


def test_fail_on_error_true_aborts_on_first_failure(tmp_path):
    """The default must stay strict: one failure ends the run."""
    harness = _no_backoff_harness(tmp_path)
    subject = _failing_on(harness, {"3"})
    with pytest.raises(_RateLimited):
        harness.evaluate_subject(subject, "gen")


def test_fail_on_error_false_completes_the_run(tmp_path):
    """Failed samples are skipped, reported, and kept out of the score."""
    harness = _no_backoff_harness(tmp_path, fail_on_error=False, min_pass_rate=0.0)
    subject = _failing_on(harness, {"3", "7"})

    result = harness.evaluate_subject(subject, "gen")

    assert result["total_tests"] == 10
    assert result["scored_tests"] == 8
    assert result["errored_tests"] == 2
    assert result["error_rate"] == pytest.approx(0.2)
    # The 8 samples that ran all passed; the 2 failures must not drag the
    # score down as if the model had answered badly.
    assert result["avg_score"] == 1.0
    assert result["pass_rate"] == 1.0

    errored = [row for row in result["results"] if row["errored"]]
    assert len(errored) == 2
    assert errored[0]["score"] is None
    assert "_RateLimited" in errored[0]["error"]


def test_fail_on_error_proportion(tmp_path):
    """A proportion aborts only once the share of failures passes the bar."""
    within = _no_backoff_harness(tmp_path / "within", fail_on_error=0.3, min_pass_rate=0.0)
    result = within.evaluate_subject(_failing_on(within, {"1", "2"}), "gen")
    assert result["errored_tests"] == 2  # 0.2 <= 0.3, tolerated

    over = _no_backoff_harness(tmp_path / "over", fail_on_error=0.3, min_pass_rate=0.0)
    with pytest.raises(_RateLimited):
        over.evaluate_subject(_failing_on(over, {"1", "2", "3", "4"}), "gen")


def test_fail_on_error_absolute_count(tmp_path):
    """A value >= 1 is a count of failures, not a proportion."""
    within = _no_backoff_harness(tmp_path / "within", fail_on_error=2, min_pass_rate=0.0)
    result = within.evaluate_subject(_failing_on(within, {"1", "2"}), "gen")
    assert result["errored_tests"] == 2

    over = _no_backoff_harness(tmp_path / "over", fail_on_error=2, min_pass_rate=0.0)
    with pytest.raises(_RateLimited):
        over.evaluate_subject(_failing_on(over, {"1", "2", "3"}), "gen")


def test_error_rate_is_recorded_in_history(tmp_path):
    """An operator reading history must be able to see the run was degraded."""
    harness = _no_backoff_harness(tmp_path, fail_on_error=False, min_pass_rate=0.0)
    harness.evaluate_subject(_failing_on(harness, {"3", "7"}), "gen")
    row = harness.get_history(suite_name="gen")[-1]
    assert row["errored_tests"] == 2
    assert row["error_rate"] == pytest.approx(0.2)


def test_concurrent_run_preserves_suite_order(tmp_path):
    """Results arrive out of order under a pool; rows must still line up."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, min_pass_rate=0.0)
    harness.add_test_suite(
        "gen",
        [
            {"id": str(i), "prompt": str(i), "expected": str(i), "type": "exact_match"}
            for i in range(20)
        ],
    )

    def generate(prompt, **kwargs):
        # Later prompts finish first, so completion order is the reverse of
        # suite order and a naive append would scramble the rows.
        time.sleep((20 - int(prompt)) * 0.002)
        return prompt

    result = harness.evaluate_subject(
        CallableGenerator(generate, name="model_a"), "gen", max_workers=8
    )

    assert result["max_workers"] == 8
    assert result["avg_score"] == 1.0
    assert [row["test_id"] for row in result["results"]] == [str(i) for i in range(20)]
    assert all(row["prompt"] == row["expected"] for row in result["results"])


def test_concurrency_does_not_change_scores(tmp_path):
    """Worker count is an execution detail; the score must be identical."""
    suite = [
        {"id": str(i), "prompt": str(i), "expected": "good" if i % 3 else "no", "type": "contains"}
        for i in range(12)
    ]

    def run(workers):
        harness = AutomaticEvaluationHarness(eval_dir=tmp_path / f"w{workers}", min_pass_rate=0.0)
        harness.add_test_suite("gen", suite)
        return harness.evaluate_subject(
            CallableGenerator(lambda p, **kw: "good", name="model_a"), "gen", max_workers=workers
        )

    sequential, concurrent = run(1), run(6)

    assert sequential["avg_score"] == concurrent["avg_score"]
    assert sequential["pass_rate"] == concurrent["pass_rate"]
    assert [r["score"] for r in sequential["results"]] == [
        r["score"] for r in concurrent["results"]
    ]
    # Scores are unaffected, but the config hash must not encode worker count
    # either, or the two runs would land in separate regression baselines.
    assert sequential["config_hash"] == concurrent["config_hash"]


def test_concurrent_run_is_faster(tmp_path):
    """The point of the pool is wall-clock; prove it actually overlaps."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, min_pass_rate=0.0)
    harness.add_test_suite(
        "gen",
        [{"id": str(i), "prompt": "p", "expected": "good", "type": "contains"} for i in range(16)],
    )
    subject = CallableGenerator(lambda p, **kw: time.sleep(0.02) or "good", name="model_a")

    start = time.perf_counter()
    harness.evaluate_subject(subject, "gen", max_workers=8)
    elapsed = time.perf_counter() - start

    # 16 x 20ms is 320ms sequentially; across 8 workers it should be nearer 40ms.
    assert elapsed < 0.2


def test_concurrent_errors_respect_the_budget(tmp_path):
    """Tolerated failures still land in the right rows under a pool."""
    harness = _no_backoff_harness(tmp_path, fail_on_error=False, min_pass_rate=0.0, max_workers=4)
    subject = _failing_on(harness, {"3", "7"})

    result = harness.evaluate_subject(subject, "gen")

    assert result["errored_tests"] == 2
    assert result["scored_tests"] == 8
    errored_ids = {row["test_id"] for row in result["results"] if row["errored"]}
    assert errored_ids == {"3", "7"}


def test_concurrent_run_aborts_when_budget_spent(tmp_path):
    """Exceeding the budget mid-pool must surface the original exception."""
    harness = _no_backoff_harness(tmp_path, fail_on_error=1, min_pass_rate=0.0, max_workers=4)
    subject = _failing_on(harness, {"1", "2", "3", "4", "5"})

    with pytest.raises(_RateLimited):
        harness.evaluate_subject(subject, "gen")


def _counting_subject(name, response):
    """Subject that records how many times it was actually asked to generate."""
    calls = {"n": 0}

    def generate(prompt, **kwargs):
        calls["n"] += 1
        return response

    return CallableGenerator(generate, name=name), calls


def test_cache_is_off_by_default(tmp_path):
    """Caching must be opt-in, or a rerun silently stops testing the model."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])
    subject, calls = _counting_subject("model_a", "good")

    harness.evaluate_subject(subject, "gen")
    harness.evaluate_subject(subject, "gen")

    assert calls["n"] == 2
    assert not (tmp_path / "cache").exists()


def test_cache_hit_skips_the_model(tmp_path):
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, cache_enabled=True)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])
    subject, calls = _counting_subject("model_a", "good")

    first = harness.evaluate_subject(subject, "gen")
    second = harness.evaluate_subject(subject, "gen")

    assert calls["n"] == 1
    assert first["cache_hits"] == 0
    assert second["cache_hits"] == 1
    assert second["avg_score"] == first["avg_score"] == 1.0
    # A cache hit times the disk, not the model, so it must not enter latency.
    assert second["results"][0]["latency_ms"] is None
    assert "latency" not in second


def test_cache_key_separates_models(tmp_path):
    """Two models must never share an entry, however alike their prompts."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, cache_enabled=True, min_pass_rate=0.0)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])

    good, good_calls = _counting_subject("model_a", "good")
    bad, bad_calls = _counting_subject("model_b", "totally wrong")

    assert harness.evaluate_subject(good, "gen")["avg_score"] == 1.0
    second = harness.evaluate_subject(bad, "gen")

    assert bad_calls["n"] == 1  # not served model_a's answer
    assert second["cache_hits"] == 0
    assert second["avg_score"] == 0.0
    assert good_calls["n"] == 1


def test_cache_key_separates_generation_params(tmp_path):
    """max_new_tokens changes the output, so it must change the key."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, cache_enabled=True)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])
    subject, calls = _counting_subject("model_a", "good")

    harness.evaluate_subject(subject, "gen", max_new_tokens=10)
    harness.evaluate_subject(subject, "gen", max_new_tokens=10)
    assert calls["n"] == 1

    harness.evaluate_subject(subject, "gen", max_new_tokens=500)
    assert calls["n"] == 2


def test_api_subject_cache_key_covers_temperature():
    """A decoding parameter left out of the key would serve the wrong responses."""
    hot = OpenAIGenerator.__new__(OpenAIGenerator)
    hot.name, hot.model, hot.temperature, hot.base_url = "gpt", "gpt-4o", 1.0, None
    cold = OpenAIGenerator.__new__(OpenAIGenerator)
    cold.name, cold.model, cold.temperature, cold.base_url = "gpt", "gpt-4o", 0.0, None

    assert hot.cache_key() != cold.cache_key()
    assert hot.cache_key()["temperature"] == 1.0


def test_expired_cache_entry_is_refetched(tmp_path):
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, cache_enabled=True, cache_ttl=-1.0)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])
    subject, calls = _counting_subject("model_a", "good")

    harness.evaluate_subject(subject, "gen")
    harness.evaluate_subject(subject, "gen")

    assert calls["n"] == 2  # every entry is already stale


def test_clear_cache(tmp_path):
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, cache_enabled=True)
    harness.add_test_suite("gen", [{"prompt": "p", "expected": "good", "type": "contains"}])
    subject, calls = _counting_subject("model_a", "good")

    harness.evaluate_subject(subject, "gen")
    assert harness.clear_cache() == 1

    harness.evaluate_subject(subject, "gen")
    assert calls["n"] == 2


PRICING = {"model_a": {"input_per_1m": 1.0, "output_per_1m": 10.0}}


def test_tokens_are_counted_without_pricing(tmp_path):
    """Token usage is always reported; cost stays zero until rates are given."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    harness.add_test_suite("gen", [{"prompt": "a b c", "expected": "good", "type": "contains"}])
    subject = CallableGenerator(lambda p, **kw: "good enough", name="model_a")

    result = harness.evaluate_subject(subject, "gen")

    assert result["total_prompt_tokens"] == 3
    assert result["total_completion_tokens"] == 2
    assert result["total_cost_usd"] == 0.0


def test_cost_is_priced_per_million_tokens(tmp_path):
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, pricing=PRICING)
    harness.add_test_suite("gen", [{"prompt": "a b c", "expected": "good", "type": "contains"}])
    subject = CallableGenerator(lambda p, **kw: "good enough", name="model_a")

    result = harness.evaluate_subject(subject, "gen")

    expected = (3 / 1_000_000) * 1.0 + (2 / 1_000_000) * 10.0
    assert result["total_cost_usd"] == pytest.approx(expected)
    assert result["results"][0]["cost_usd"] == pytest.approx(expected)
    # Output tokens are the expensive half; a flat per-token rate would miss it.
    assert result["results"][0]["cost_usd"] > (5 / 1_000_000) * 1.0


def test_unpriced_model_costs_nothing(tmp_path):
    """An unknown model reports usage but must not be assigned invented rates."""
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, pricing=PRICING)
    harness.add_test_suite("gen", [{"prompt": "a b c", "expected": "good", "type": "contains"}])
    subject = CallableGenerator(lambda p, **kw: "good enough", name="model_zzz")

    result = harness.evaluate_subject(subject, "gen")

    assert result["total_completion_tokens"] == 2
    assert result["total_cost_usd"] == 0.0


def test_cached_samples_are_not_billed(tmp_path):
    """A cache hit never reached the provider, so it costs nothing."""
    harness = AutomaticEvaluationHarness(
        eval_dir=tmp_path, pricing=PRICING, cache_enabled=True, min_pass_rate=0.0
    )
    harness.add_test_suite("gen", [{"prompt": "a b c", "expected": "good", "type": "contains"}])
    subject = CallableGenerator(lambda p, **kw: "good enough", name="model_a")

    first = harness.evaluate_subject(subject, "gen")
    second = harness.evaluate_subject(subject, "gen")

    assert first["total_cost_usd"] > 0
    assert second["cache_hits"] == 1
    assert second["total_cost_usd"] == 0.0
    assert second["total_prompt_tokens"] == 0
    assert second["total_completion_tokens"] == 0


def test_cost_is_recorded_in_history(tmp_path):
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, pricing=PRICING)
    harness.add_test_suite("gen", [{"prompt": "a b c", "expected": "good", "type": "contains"}])
    harness.evaluate_subject(CallableGenerator(lambda p, **kw: "good", name="model_a"), "gen")

    assert harness.get_history(suite_name="gen")[-1]["total_cost_usd"] > 0


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


# ----------------------------------------------------------------------
# LLM-as-judge scoring
# ----------------------------------------------------------------------


def _judge_returning(*replies):
    """A judge whose model emits the given verdicts, one per call."""
    remaining = list(replies)
    seen = []

    def generate(prompt, **kwargs):
        seen.append(prompt)
        return remaining.pop(0) if remaining else replies[-1]

    return LlmJudge(CallableGenerator(generate, name="judge_model")), seen


def test_judge_normalizes_to_unit_interval():
    judge, _ = _judge_returning('{"score": 5, "reasoning": "perfect"}')
    assert judge.score("q", "a") == 1.0

    judge, _ = _judge_returning('{"score": 1, "reasoning": "wrong"}')
    assert judge.score("q", "a") == 0.0

    judge, _ = _judge_returning('{"score": 4, "reasoning": "close"}')
    assert judge.score("q", "a") == 0.75


def test_judge_prompt_carries_question_and_reference():
    judge, seen = _judge_returning('{"score": 3}')
    judge.score("what is 2+2", "four", reference="4")
    prompt = seen[0]
    assert "what is 2+2" in prompt
    assert "four" in prompt
    assert "[REFERENCE ANSWER]" in prompt and "\n4\n" in prompt


def test_judge_omits_reference_block_when_absent():
    judge, seen = _judge_returning('{"score": 3}')
    judge.score("q", "a")
    assert "[REFERENCE ANSWER]" not in seen[0]


def test_judge_parses_verdicts_wrapped_in_prose():
    judge, _ = _judge_returning('Here is my grade:\n```json\n{"score": 5}\n```\nThanks!')
    assert judge.score("q", "a") == 1.0


def test_judge_falls_back_to_a_bare_score_field():
    judge, _ = _judge_returning("Score: 4 — mostly right")
    assert judge.score("q", "a") == 0.75


def test_judge_clamps_out_of_range_scores():
    judge, _ = _judge_returning('{"score": 9}')
    verdict = judge.judge("q", "a")
    assert verdict["score"] == 1.0
    # The sloppy verdict is preserved so a miscalibrated judge stays visible.
    assert verdict["raw_score"] == 9.0


def test_unparseable_verdict_raises_rather_than_scoring_zero():
    judge, _ = _judge_returning("I would rather not grade this.")
    with pytest.raises(JudgeParseError):
        judge.score("q", "a")

    judge, _ = _judge_returning("")
    with pytest.raises(JudgeParseError):
        judge.score("q", "a")


def test_judge_respects_a_custom_scale():
    judge, _ = _judge_returning('{"score": 5}')
    judge.low, judge.high = 0, 10
    assert judge.score("q", "a") == 0.5

    with pytest.raises(ValueError):
        LlmJudge(CallableGenerator(lambda p, **kw: ""), scale=(5, 5))


def test_harness_scores_llm_judge_tests(tmp_path):
    judge, seen = _judge_returning('{"score": 4, "reasoning": "close"}')
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, judge=judge)
    harness.add_test_suite(
        "gen",
        [
            {
                "id": "j1",
                "prompt": "explain recursion",
                "expected": "a function that calls itself",
                "type": "llm_judge",
            }
        ],
    )
    subject = CallableGenerator(lambda p, **kw: "a function calling itself", name="model_a")
    result = harness.evaluate_subject(subject, "gen")
    assert result["avg_score"] == 0.75
    assert result["results"][0]["test_type"] == "llm_judge"

    # The harness must forward the question and the reference, not just the
    # response — a grader with no question cannot tell relevant from fluent.
    assert "explain recursion" in seen[0]
    assert "a function calling itself" in seen[0]
    assert "a function that calls itself" in seen[0]


def test_judged_and_deterministic_tests_mix_in_one_suite(tmp_path):
    judge, _ = _judge_returning('{"score": 1}')
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, judge=judge)
    harness.add_test_suite(
        "gen",
        [
            {"id": "d1", "prompt": "write add", "expected": "def add", "type": "contains"},
            {"id": "j1", "prompt": "explain it", "expected": "", "type": "llm_judge"},
        ],
    )
    subject = CallableGenerator(lambda p, **kw: "def add(a, b)", name="model_a")
    result = harness.evaluate_subject(subject, "gen")
    scores = {row["test_id"]: row["score"] for row in result["results"]}
    assert scores == {"d1": 1.0, "j1": 0.0}


def test_judge_tests_without_a_judge_fail_validation(tmp_path):
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    harness.add_test_suite(
        "gen", [{"id": "j1", "prompt": "explain", "expected": "", "type": "llm_judge"}]
    )
    report = harness.validate_suite("gen")
    assert report["valid"] is False
    assert any("needs a configured judge" in issue for issue in report["issues"])

    subject = CallableGenerator(lambda p, **kw: "anything", name="model_a")
    with pytest.raises(ValueError, match="requires a judge"):
        harness.evaluate_subject(subject, "gen")


def test_judge_identity_is_recorded_in_run_config(tmp_path):
    judge, _ = _judge_returning('{"score": 3}')
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, judge=judge)
    harness.add_test_suite(
        "gen", [{"id": "j1", "prompt": "explain", "expected": "", "type": "llm_judge"}]
    )
    subject = CallableGenerator(lambda p, **kw: "answer", name="model_a")
    result = harness.evaluate_subject(subject, "gen")
    assert result["config"]["judge"]["judge"] == "judge_model"
    assert result["config"]["judge"]["scale"] == [1, 5]


def test_changing_the_rubric_splits_the_baseline(tmp_path):
    hashes = []
    for rubric in ("score correctness", "score correctness and style"):
        judge, _ = _judge_returning('{"score": 3}')
        judge.rubric = rubric
        harness = AutomaticEvaluationHarness(eval_dir=tmp_path, judge=judge)
        harness.add_test_suite(
            "gen", [{"id": "j1", "prompt": "explain", "expected": "", "type": "llm_judge"}]
        )
        subject = CallableGenerator(lambda p, **kw: "answer", name="model_a")
        hashes.append(harness.evaluate_subject(subject, "gen")["config_hash"])
    assert hashes[0] != hashes[1]


def test_unjudged_suites_carry_no_judge_config(tmp_path):
    judge, _ = _judge_returning('{"score": 3}')
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, judge=judge)
    harness.add_test_suite(
        "gen", [{"id": "d1", "prompt": "write add", "expected": "def add", "type": "contains"}]
    )
    subject = CallableGenerator(lambda p, **kw: "def add(a, b)", name="model_a")
    result = harness.evaluate_subject(subject, "gen")
    assert "judge" not in result["config"]


# ----------------------------------------------------------------------
# Judge bias controls
# ----------------------------------------------------------------------


def _order_sensitive_judge(always_picks="A"):
    """A judge with pure position bias: it always names the same slot."""
    seen = []

    def generate(prompt, **kwargs):
        seen.append(prompt)
        return '{"winner": "%s", "reasoning": "position"}' % always_picks

    return LlmJudge(CallableGenerator(generate, name="judge_model")), seen


def _content_judge(preferred):
    """A judge that picks whichever slot holds ``preferred``, wherever it is."""

    def generate(prompt, **kwargs):
        first = prompt.split("[RESPONSE A]")[1].split("[RESPONSE B]")[0]
        return '{"winner": "A"}' if preferred in first else '{"winner": "B"}'

    return LlmJudge(CallableGenerator(generate, name="judge_model"))


def test_compare_runs_both_orderings():
    judge, seen = _order_sensitive_judge()
    judge.compare("q", "answer one", "answer two")
    assert len(seen) == 2
    # The same two answers, swapped between the slots.
    first_slot = [p.split("[RESPONSE A]")[1].split("[RESPONSE B]")[0].strip() for p in seen]
    assert first_slot == ["answer one", "answer two"]


def test_position_bias_is_caught_and_reported_as_a_tie():
    judge, _ = _order_sensitive_judge(always_picks="A")
    verdict = judge.compare("q", "answer one", "answer two")
    assert verdict["position_bias"] is True
    assert verdict["winner"] == "tie"


def test_consistent_winner_survives_the_order_swap():
    judge = _content_judge("the good answer")
    verdict = judge.compare("q", "the good answer", "the bad answer")
    assert verdict["winner"] == "A"
    assert verdict["position_bias"] is False

    # And the same answer still wins when it is passed in as B.
    flipped = judge.compare("q", "the bad answer", "the good answer")
    assert flipped["winner"] == "B"
    assert flipped["position_bias"] is False


def test_genuine_ties_are_not_reported_as_position_bias():
    judge = LlmJudge(CallableGenerator(lambda p, **kw: '{"winner": "tie"}', name="judge_model"))
    verdict = judge.compare("q", "one", "two")
    assert verdict["winner"] == "tie"
    assert verdict["position_bias"] is False


def test_comparison_verdicts_parse_from_prose_and_fail_loudly():
    judge = LlmJudge(CallableGenerator(lambda p, **kw: "I pick winner: B here", name="j"))
    assert judge.compare("q", "one", "two")["first_pass"]["winner"] == "B"

    judge = LlmJudge(CallableGenerator(lambda p, **kw: "hard to say", name="j"))
    with pytest.raises(JudgeParseError):
        judge.compare("q", "one", "two")


def test_judge_prompts_tell_the_grader_length_is_not_quality():
    judge, seen = _judge_returning('{"score": 3}')
    judge.score("q", "a")
    assert "Length is not quality" in seen[0]

    judge, seen = _order_sensitive_judge()
    judge.compare("q", "one", "two")
    assert "Length is not quality" in seen[0]


def test_self_preference_is_flagged_when_judge_grades_itself(tmp_path):
    judge, _ = _judge_returning('{"score": 5}')
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, judge=judge)
    harness.add_test_suite(
        "gen", [{"id": "j1", "prompt": "explain", "expected": "", "type": "llm_judge"}]
    )

    graded_by_itself = CallableGenerator(lambda p, **kw: "answer", name="judge_model")
    result = harness.evaluate_subject(graded_by_itself, "gen")
    # Flagged, not blocked: same-model grading is a valid smoke test, it just
    # must not pass itself off as independent.
    assert result["judge_self_preference"] is True
    assert result["avg_score"] == 1.0

    other = CallableGenerator(lambda p, **kw: "answer", name="model_a")
    assert harness.evaluate_subject(other, "gen")["judge_self_preference"] is False


def test_self_preference_matches_on_model_not_label():
    judge = LlmJudge(CallableGenerator(lambda p, **kw: "", name="grader"))
    judge.subject.model = "shared-model"

    disguised = CallableGenerator(lambda p, **kw: "", name="totally-different-name")
    disguised.model = "Shared-Model"
    assert judge.self_preference_risk(disguised) is True

    genuine = CallableGenerator(lambda p, **kw: "", name="other")
    genuine.model = "some-other-model"
    assert judge.self_preference_risk(genuine) is False


def test_unjudged_runs_carry_no_self_preference_flag(tmp_path):
    judge, _ = _judge_returning('{"score": 5}')
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path, judge=judge)
    harness.add_test_suite(
        "gen", [{"id": "d1", "prompt": "p", "expected": "good", "type": "contains"}]
    )
    subject = CallableGenerator(lambda p, **kw: "good", name="judge_model")
    assert "judge_self_preference" not in harness.evaluate_subject(subject, "gen")


# ----------------------------------------------------------------------
# Judge validation probes
# ----------------------------------------------------------------------


PROBE_CASES = [
    {"prompt": "how do you add two numbers", "expected": "use the plus operator"},
    {"prompt": "how do you reverse a list", "expected": "call the reverse method"},
    {"prompt": "how do you open a file", "expected": "call the open builtin"},
]


def _reading_judge(correct_score=5.0, length_bias=0.0, wobble=0.0, always_picks=None):
    """
    A judge that actually reads the response.

    Scores ``correct_score`` when the response contains the reference and 1
    when it does not, optionally with a thumb on the scale for length, a
    fixed slot preference, or a per-call score wobble.
    """
    calls = []

    def generate(prompt, **kwargs):
        if "[RESPONSE A]" in prompt:
            if always_picks:
                return '{"winner": "%s"}' % always_picks
            first = prompt.split("[RESPONSE A]")[1].split("[RESPONSE B]")[0]
            reference = prompt.split("[REFERENCE ANSWER]")[1].split("[RESPONSE A]")[0].strip()
            return '{"winner": "A"}' if reference in first else '{"winner": "B"}'

        reference = prompt.split("[REFERENCE ANSWER]")[1].split("[RESPONSE]")[0].strip()
        response = prompt.split("[RESPONSE]")[1].split("Reply with JSON")[0].strip()
        score = correct_score if reference in response else 1.0
        if length_bias:
            score = min(5.0, score + length_bias * len(response) / 100.0)
        # Wobble alternates by call, so it never drains and stays reproducible.
        calls.append(prompt)
        if wobble and len(calls) % 2 == 0:
            score = max(1.0, score - wobble)
        return '{"score": %s}' % score

    return LlmJudge(CallableGenerator(generate, name="judge_model"))


def _flattering_judge():
    """A judge that never reads anything: top marks, and every pairing a tie."""

    def generate(prompt, **kwargs):
        if "[RESPONSE A]" in prompt:
            return '{"winner": "tie"}'
        return '{"score": 5}'

    return LlmJudge(CallableGenerator(generate, name="judge_model"))


def test_a_reading_judge_passes_every_probe():
    report = JudgeValidator(_reading_judge()).validate(PROBE_CASES)
    assert report["passed"] is True
    assert report["failed_probes"] == []
    assert report["probes"]["agreement"]["kappa"] == 1.0
    assert report["probes"]["agreement"]["verdict"] == "strong"


def test_a_judge_that_ignores_the_response_fails_discrimination():
    # Scores everything highly regardless of content — the classic failure.
    report = JudgeValidator(_flattering_judge()).validate(PROBE_CASES)

    assert report["passed"] is False
    assert "discrimination" in report["failed_probes"]
    assert report["probes"]["discrimination"]["mean"] == 0.0

    # And it carries no information, which kappa reports as no agreement at all.
    assert report["probes"]["agreement"]["kappa"] == 0.0
    assert report["probes"]["agreement"]["verdict"] == "unreliable"
    # Raw accuracy would have called this judge 50% correct.
    assert report["probes"]["agreement"]["accuracy"] == 0.5


def test_length_bias_is_caught_by_the_padding_probe():
    # Base score sits below the ceiling, so padding has room to move it.
    biased = _reading_judge(correct_score=3.0, length_bias=1.0)
    report = JudgeValidator(biased).validate(PROBE_CASES)
    assert "length_invariance" in report["failed_probes"]
    assert report["probes"]["length_invariance"]["worst"] > 0.1
    # The padded answer is no more correct, so every case moved the wrong way.
    assert len(report["probes"]["length_invariance"]["failures"]) == len(PROBE_CASES)


def test_position_bias_is_caught_by_the_swap_probe():
    report = JudgeValidator(_reading_judge(always_picks="A")).validate(PROBE_CASES)
    assert "position_invariance" in report["failed_probes"]
    assert report["probes"]["position_invariance"]["bias_rate"] == 1.0


def test_an_unstable_judge_is_caught():
    # Same input, wildly different verdicts across repeats.
    wobble = _reading_judge(wobble=3.0)
    report = JudgeValidator(wobble).validate(PROBE_CASES, include_agreement=False)
    assert "stability" in report["failed_probes"]
    assert report["probes"]["stability"]["worst"] > 0.1


def test_small_score_wobble_is_tolerated():
    jitter = _reading_judge(wobble=0.2)
    report = JudgeValidator(jitter).validate(PROBE_CASES, include_agreement=False)
    assert "stability" not in report["failed_probes"]


def test_cases_are_built_from_an_existing_suite(tmp_path):
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    harness.add_test_suite(
        "gen",
        [
            {"id": "t1", "prompt": "write add", "expected": "def add", "type": "contains"},
            # No reference answer, so there is nothing to perturb.
            {"id": "t2", "prompt": "say hi", "expected": "", "type": "contains"},
            {"id": "t3", "prompt": "write sub", "expected": "def sub", "type": "contains"},
        ],
    )
    cases = JudgeValidator.cases_from_suite(harness.get_suite("gen"))
    assert [case["prompt"] for case in cases] == ["write add", "write sub"]


def test_validate_judge_runs_against_a_shipped_suite(tmp_path):
    harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
    harness.add_test_suite(
        "gen", [{"prompt": c["prompt"], "expected": c["expected"]} for c in PROBE_CASES]
    )
    report = validate_judge(_reading_judge(), harness.get_suite("gen"))
    assert report["passed"] is True
    assert report["cases"] == len(PROBE_CASES)


def test_validation_refuses_to_run_on_a_single_case():
    # One case would be compared against itself and pass vacuously.
    with pytest.raises(ValueError, match="at least 2 cases"):
        JudgeValidator(_reading_judge()).validate(PROBE_CASES[:1])


def test_cohen_kappa_edges():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0
    # A constant rater agrees with nothing beyond chance.
    assert cohen_kappa([1, 0, 1, 0], [1, 1, 1, 1]) == 0.0
    assert cohen_kappa([], []) == 0.0
    assert cohen_kappa([1, 0], [1]) == 0.0


def test_failed_validation_is_logged_loudly(caplog):
    with caplog.at_level("WARNING"):
        JudgeValidator(_flattering_judge()).validate(PROBE_CASES)
    assert "failed validation probes" in caplog.text
    assert "not trustworthy" in caplog.text
