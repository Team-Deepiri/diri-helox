"""
Tests for the automatic evaluation harness (evaluation/automatic_evaluation_harness.py).

Covers scoring primitives, generation and classification evaluation, threshold
gates, persistent history, regression detection (including the distributional
noise floor), summaries, run comparison, and latency benchmarking.
"""

import json

import pytest
import torch

from evaluation.automatic_evaluation_harness import (
    AutomaticEvaluationHarness,
    classification_metrics,
    rouge_l_recall,
    score_response,
    token_f1_score,
    word_overlap_score,
)


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Minimal tokenizer manager stand-in; decodes to a fixed response text."""

    def __init__(self, shared):
        self.shared = shared

    def encode(self, text, add_bos=True, add_eos=False):
        return [0, 0, 0]

    def decode(self, token_ids):
        return self.shared["text"]


class FakeLM(torch.nn.Module):
    """Minimal causal-LM stand-in; emits enough tokens to trigger decoding."""

    def __init__(self, shared):
        super().__init__()
        self.shared = shared
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, input_tensor, max_length):
        return torch.zeros((1, max_length), dtype=torch.long)


@pytest.fixture
def fake_generator():
    return {"text": "def add(a, b): return a + b"}


@pytest.fixture
def harness(tmp_path):
    return AutomaticEvaluationHarness(eval_dir=tmp_path)


def write_history(harness, records):
    with open(harness.history_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------


class TestScoring:
    def test_exact_match(self):
        assert score_response("hello world", "hello world", "exact_match") == 1.0
        assert score_response("hello world", "hello", "exact_match") == 0.0

    def test_contains(self):
        assert score_response("the quick brown fox", "quick brown", "contains") == 1.0
        assert score_response("the quick brown fox", "red herring", "contains") == 0.0

    def test_contains_any(self):
        assert score_response("the answer is 42", "41|42|43", "contains_any") == 1.0
        assert score_response("the answer is 42", "7|8|9", "contains_any") == 0.0

    def test_similarity(self):
        assert score_response("a b c", "a b c d", "similarity") == 0.75
        assert score_response("", "a b c", "similarity") == 0.0

    def test_rouge_l(self):
        assert score_response("a b c", "a b c", "rouge_l") == 1.0
        assert rouge_l_recall("a b x", "a b c") == pytest.approx(2 / 3)

    def test_token_f1(self):
        assert score_response("a a b", "a b b", "token_f1") == pytest.approx(2 / 3)
        assert token_f1_score("x y", "a b c") == 0.0

    def test_numeric_match(self):
        assert score_response("result: 3.14", "3.14", "numeric_match") == 1.0
        assert score_response("result: 3.20", "3.14", "numeric_match") == 0.0
        assert score_response("no number here", "3.14", "numeric_match") == 0.0

    def test_regex_match(self):
        assert score_response("my email is a@b.com", r"[\w.]+@[\w.]+\.com", "regex_match") == 1.0
        assert score_response("no email", r"[\w.]+@[\w.]+\.com", "regex_match") == 0.0

    def test_json_match(self):
        generated = '{"a": 1, "b": 2}'
        assert score_response(generated, '{"a": 1, "b": 2}', "json_match") == 1.0
        assert score_response(generated, '{"a": 1, "b": 99}', "json_match") == 0.5
        assert score_response("not json", '{"a": 1}', "json_match") == 0.0

    def test_word_overlap_bound(self):
        assert 0.0 <= word_overlap_score("a b c d e", "a b c d e") <= 1.0


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_basic(self):
        metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        assert metrics["overall"]["accuracy"] == 0.75
        assert metrics["_num_examples"] == 4
        assert len(metrics["confusion_matrix"]) == 2

    def test_per_class_names(self):
        metrics = classification_metrics([0, 1], [0, 1], label_names={0: "debug", 1: "refactor"})
        assert "debug" in metrics["per_class"]
        assert "refactor" in metrics["per_class"]

    def test_empty(self):
        metrics = classification_metrics([], [])
        assert metrics["overall"] == {}
        assert metrics["confusion_matrix"] == []

    def test_confidence_calibration(self):
        metrics = classification_metrics([0, 0], [0, 1], y_conf=[0.9, 0.8])
        assert metrics["overall"]["avg_confidence_correct"] == pytest.approx(0.9)
        assert metrics["overall"]["avg_confidence_incorrect"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Suite management
# ---------------------------------------------------------------------------


class TestSuiteManagement:
    def test_add_suite_normalizes_defaults(self, harness):
        tests = harness.add_test_suite("gen", [{"prompt": "p", "expected": "e"}])
        assert tests[0]["threshold"] == 0.5
        assert tests[0]["type"] == "similarity"
        assert tests[0]["id"] == ""

    def test_load_suite_from_jsonl(self, harness, tmp_path):
        suite_file = tmp_path / "suite.jsonl"
        suite_file.write_text(
            '{"prompt": "p1", "expected": "e1"}\n'
            '{"prompt": "p2", "expected": "e2", "type": "contains"}\n'
            "not-json\n"
        )
        tests = harness.load_test_suite("code", suite_file)
        assert len(tests) == 2
        assert harness.list_suites() == ["code"]

    def test_load_suites_from_dir(self, harness, tmp_path):
        (tmp_path / "a.jsonl").write_text('{"prompt": "p", "expected": "e"}\n')
        (tmp_path / "b.jsonl").write_text('{"prompt": "p", "expected": "e"}\n')
        loaded = harness.load_suites_from_dir(tmp_path)
        assert set(loaded) == {"a", "b"}

    def test_missing_suite_raises(self, harness):
        with pytest.raises(ValueError):
            harness.evaluate_model(None, None, "missing")

    def test_missing_file_raises(self, harness, tmp_path):
        with pytest.raises(ValueError):
            harness.load_test_suite("x", tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# Generation evaluation
# ---------------------------------------------------------------------------


class TestGenerationEvaluation:
    def _make_harness(self, tmp_path, response_text, tests):
        shared = {"text": response_text}
        harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
        harness.add_test_suite("gen", tests)
        model = FakeLM(shared)
        tokenizer = FakeTokenizer(shared)
        return harness, model, tokenizer

    def test_generation_auto_mode(self, tmp_path, fake_generator):
        tests = [
            {"id": "t1", "prompt": "write add", "expected": "def add", "type": "contains"},
            {"id": "t2", "prompt": "write add", "expected": "unrelated phrase", "type": "contains"},
        ]
        harness, model, tokenizer = self._make_harness(tmp_path, fake_generator["text"], tests)
        result = harness.evaluate_model(model, tokenizer, "gen")
        assert result["mode"] == "generation"
        assert result["total_tests"] == 2
        assert result["passed_tests"] == 1
        assert result["pass_rate"] == 0.5
        assert result["avg_score"] == 0.5
        assert result["latency"]["avg_latency_ms"] >= 0
        assert result["latency"]["avg_tokens_per_sec"] is not None
        assert model.eval_called is True
        assert result["results"][0]["generated"] == fake_generator["text"]

    def test_generation_scoring_per_type(self, tmp_path, fake_generator):
        tests = [
            {"id": "em", "prompt": "p", "expected": "def add(a, b): return a + b", "type": "exact_match"},
            {"id": "rg", "prompt": "p", "expected": r"def add", "type": "regex_match"},
            {"id": "num", "prompt": "p", "expected": "5", "type": "numeric_match"},
        ]
        harness, model, tokenizer = self._make_harness(tmp_path, fake_generator["text"], tests)
        result = harness.evaluate_model(model, tokenizer, "gen")
        by_id = {r["test_id"]: r for r in result["results"]}
        assert by_id["em"]["score"] == 1.0
        assert by_id["rg"]["score"] == 1.0
        assert by_id["num"]["score"] == 0.0

    def test_latency_collection_can_be_disabled(self, tmp_path, fake_generator):
        tests = [{"prompt": "p", "expected": "def add", "type": "contains"}]
        harness, model, tokenizer = self._make_harness(tmp_path, fake_generator["text"], tests)
        result = harness.evaluate_model(model, tokenizer, "gen", collect_latency=False)
        assert "latency" not in result
        assert result["passed_tests"] == 1

    def test_benchmark_latency(self, tmp_path, fake_generator):
        shared = {"text": fake_generator["text"]}
        harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
        model = FakeLM(shared)
        tokenizer = FakeTokenizer(shared)
        stats = harness.benchmark_latency(model, tokenizer, "write add", num_runs=3)
        assert stats["num_runs"] == 3
        assert stats["avg_latency_ms"] >= 0
        assert stats["p95_latency_ms"] >= stats["p50_latency_ms"]


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------


class TestClassificationEvaluation:
    def test_classifier_suite(self, harness):
        harness.add_test_suite(
            "intent",
            [
                {"text": "fix the bug", "label_name": "debugging"},
                {"text": "rename var", "label_name": "refactoring"},
                {"text": "write tests", "label_name": "testing"},
            ],
        )
        predict_fn = lambda texts: [0, 1, 2]  # noqa: E731
        result = harness.evaluate_classifier_suite("intent", predict_fn)
        assert result["mode"] == "classifier"
        assert result["passed_tests"] == 3
        assert result["pass_rate"] == 1.0
        assert result["overall"]["accuracy"] == 1.0
        assert "debugging" in result["per_class"]
        assert len(result["confusion_matrix"]) == 3

    def test_classifier_with_confidence(self, harness):
        harness.add_test_suite(
            "intent",
            [
                {"text": "fix the bug", "label": 0},
                {"text": "write tests", "label": 2},
            ],
        )
        result = harness.evaluate_classifier_suite(
            "intent",
            lambda texts: [(0, 0.95), (1, 0.70)],
        )
        assert result["passed_tests"] == 1
        assert result["overall"]["avg_confidence"] == pytest.approx(0.825)

    def test_classifier_missing_predict_fn_raises(self, harness):
        harness.add_test_suite("intent", [{"text": "fix", "label": 0}])
        with pytest.raises(ValueError):
            harness.evaluate_model(None, None, "intent", mode="auto")

    def test_classifier_prediction_count_mismatch(self, harness):
        harness.add_test_suite("intent", [{"text": "a", "label": 0}, {"text": "b", "label": 1}])
        with pytest.raises(ValueError):
            harness.evaluate_classifier_suite("intent", lambda texts: [0])


# ---------------------------------------------------------------------------
# Threshold gates
# ---------------------------------------------------------------------------


class TestGates:
    def test_pass_rate_gate_fails(self, tmp_path):
        harness = AutomaticEvaluationHarness(eval_dir=tmp_path, min_pass_rate=0.8)
        shared = {"text": "wrong answer"}
        harness.add_test_suite(
            "gen",
            [
                {"prompt": "p", "expected": "right answer", "type": "contains"},
                {"prompt": "p", "expected": "right answer", "type": "contains"},
            ],
        )
        result = harness.evaluate_model(FakeLM(shared), FakeTokenizer(shared), "gen")
        assert result["passed"] is False
        assert any("pass_rate_below_threshold" in f for f in result["failures"])

    def test_avg_score_gate_fails(self, tmp_path):
        harness = AutomaticEvaluationHarness(eval_dir=tmp_path, min_avg_score=0.9)
        shared = {"text": "wrong answer"}
        harness.add_test_suite("gen", [{"prompt": "p", "expected": "right", "type": "contains"}])
        result = harness.evaluate_model(FakeLM(shared), FakeTokenizer(shared), "gen")
        assert result["passed"] is False

    def test_classifier_gates(self, tmp_path):
        harness = AutomaticEvaluationHarness(eval_dir=tmp_path, min_accuracy=1.0, min_f1=1.0)
        harness.add_test_suite("intent", [{"text": "fix", "label": 0}, {"text": "test", "label": 1}])
        result = harness.evaluate_classifier_suite("intent", lambda texts: [0, 0])
        assert result["passed"] is False
        assert any("accuracy_below_threshold" in f for f in result["failures"])


# ---------------------------------------------------------------------------
# Persistence, history, and summaries
# ---------------------------------------------------------------------------


class TestHistory:
    def test_history_is_persisted(self, tmp_path, fake_generator):
        shared = {"text": fake_generator["text"]}
        harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
        harness.add_test_suite("gen", [{"prompt": "p", "expected": "def add", "type": "contains"}])
        harness.evaluate_model(FakeLM(shared), FakeTokenizer(shared), "gen")

        reloaded = AutomaticEvaluationHarness(eval_dir=tmp_path)
        history = reloaded.get_history()
        assert len(history) == 1
        assert history[0]["suite_name"] == "gen"
        assert history[0]["avg_score"] == 1.0
        assert (tmp_path / "history" / "evaluation_history.jsonl").exists()

    def test_reports_written(self, tmp_path, fake_generator):
        shared = {"text": fake_generator["text"]}
        harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
        harness.add_test_suite("gen", [{"prompt": "p", "expected": "def add", "type": "contains"}])
        harness.evaluate_model(FakeLM(shared), FakeTokenizer(shared), "gen")
        report_files = list(tmp_path.glob("eval_gen_*.json"))
        assert len(report_files) == 1

    def test_save_and_load_report(self, harness, tmp_path):
        harness.add_test_suite("gen", [{"prompt": "p", "expected": "e", "type": "contains"}])
        shared = {"text": "e"}
        harness.evaluate_model(FakeLM(shared), FakeTokenizer(shared), "gen")
        report_path = harness.save_report({"suite_name": "gen", "hello": "world"})
        loaded = AutomaticEvaluationHarness.load_report(report_path)
        assert loaded["hello"] == "world"

    def test_summary_aggregates(self, harness):
        write_history(
            harness,
            [
                {"suite_name": "s", "avg_score": 0.80, "pass_rate": 0.9},
                {"suite_name": "s", "avg_score": 0.90, "pass_rate": 1.0},
            ],
        )
        reloaded = AutomaticEvaluationHarness(eval_dir=harness.eval_dir)
        summary = reloaded.get_evaluation_summary()
        assert summary["total_evaluations"] == 2
        assert summary["suites"]["s"]["mean_score"] == pytest.approx(0.85)
        assert summary["suites"]["s"]["max_score"] == 0.90
        assert summary["suites"]["s"]["trend"] == pytest.approx(0.10)

    def test_compare_runs(self, harness):
        write_history(
            harness,
            [
                {"suite_name": "s", "avg_score": 0.70, "pass_rate": 0.8},
                {"suite_name": "s", "avg_score": 0.85, "pass_rate": 0.9},
            ],
        )
        reloaded = AutomaticEvaluationHarness(eval_dir=harness.eval_dir)
        comparison = reloaded.compare_runs("s")
        assert comparison["avg_score_delta"] == pytest.approx(0.15)

    def test_compare_runs_insufficient(self, harness):
        write_history(harness, [{"suite_name": "s", "avg_score": 0.70}])
        reloaded = AutomaticEvaluationHarness(eval_dir=harness.eval_dir)
        assert reloaded.compare_runs("s") is None


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


class TestRegression:
    def test_no_regression_without_history(self, harness):
        assert harness._check_regression("s", 0.9) is None

    def test_regression_below_best(self, harness):
        write_history(harness, [{"suite_name": "s", "avg_score": 0.90}])
        reloaded = AutomaticEvaluationHarness(eval_dir=harness.eval_dir)
        regression = reloaded._check_regression("s", 0.80)
        assert regression is not None
        assert regression["score_drop"] == pytest.approx(0.10)

    def test_no_regression_within_threshold(self, harness):
        write_history(harness, [{"suite_name": "s", "avg_score": 0.90}])
        reloaded = AutomaticEvaluationHarness(eval_dir=harness.eval_dir)
        assert reloaded._check_regression("s", 0.88) is None

    def test_no_regression_within_noise_floor(self, harness):
        # A single lucky 0.95 run sets an unreachable "best" while the
        # historical mean (0.775) is near the current score; the noise floor
        # must suppress the false positive.
        write_history(
            harness,
            [
                {"suite_name": "s", "avg_score": 0.60},
                {"suite_name": "s", "avg_score": 0.95},
            ],
        )
        reloaded = AutomaticEvaluationHarness(eval_dir=harness.eval_dir)
        assert reloaded._check_regression("s", 0.85) is None

    def test_regression_beyond_noise_floor(self, harness):
        # Tight history (mean 0.91, std 0.01) makes a 0.80 run a true outlier.
        write_history(
            harness,
            [
                {"suite_name": "s", "avg_score": 0.90},
                {"suite_name": "s", "avg_score": 0.92},
            ],
        )
        reloaded = AutomaticEvaluationHarness(eval_dir=harness.eval_dir)
        regression = reloaded._check_regression("s", 0.80)
        assert regression is not None
        assert regression["metric"] == "avg_score"

    def test_regression_fails_run(self, tmp_path, fake_generator):
        shared = {"text": fake_generator["text"]}
        harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
        harness.add_test_suite("gen", [{"prompt": "p", "expected": "def add", "type": "contains"}])
        harness.evaluate_model(FakeLM(shared), FakeTokenizer(shared), "gen")

        # Second run with a worse response triggers regression -> run fails.
        shared2 = {"text": "completely wrong"}
        result = harness.evaluate_model(FakeLM(shared2), FakeTokenizer(shared2), "gen")
        assert result["passed"] is False
        assert "regression_detected" in result["failures"]
        assert result["regression"]["detected"] is True


# ---------------------------------------------------------------------------
# Full suite runner
# ---------------------------------------------------------------------------


class TestFullEvaluation:
    def test_run_full_evaluation(self, tmp_path, fake_generator):
        shared = {"text": fake_generator["text"]}
        harness = AutomaticEvaluationHarness(eval_dir=tmp_path)
        harness.add_test_suite("a", [{"prompt": "p", "expected": "def add", "type": "contains"}])
        harness.add_test_suite("b", [{"prompt": "p", "expected": "def add", "type": "contains"}])
        results = harness.run_full_evaluation(FakeLM(shared), FakeTokenizer(shared))
        assert set(results) == {"a", "b"}
        assert results["a"]["passed_tests"] == 1
