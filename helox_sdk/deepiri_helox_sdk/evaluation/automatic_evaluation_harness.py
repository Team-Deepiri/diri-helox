"""
Automatic evaluation harness.

Provides fixed eval prompts, domain-specific test suites, generation and
classification evaluation, latency benchmarking, threshold gates, and
persistent regression tracking for comprehensive model and agent evaluation.

Scoring and classification metrics are provided by :mod:`metrics`; this module
implements the subject-based orchestration (models, agents, classifiers),
threshold gates, and persistence on top of that shared logic.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np

from .metrics import (
    _VALID_TEST_TYPES,
    classification_metrics,
    score_response,
)
from .subjects import (
    CallablePredictor,
    LabelPredictor,
    LegacyModelGenerator,
    ResponseGenerator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring primitives (module-level so they can be reused and unit-tested).
# Invariant: every score is dimensionless and bounded in [0, 1].
# ---------------------------------------------------------------------------


class AutomaticEvaluationHarness:
    """
    Automatic evaluation system for LLMs.

    Features:
    - Fixed evaluation prompts and domain-specific test suites (JSONL)
    - Generation scoring: exact_match, contains, contains_any, similarity,
      rouge_l, token_f1, numeric_match, regex_match, json_match
    - Classification evaluation: accuracy, precision, recall, F1, per-class
      breakdown, confusion matrix, confidence calibration
    - Latency / throughput benchmarking (per-sample and standalone)
    - Pass/fail threshold gates for CI / release promotion
    - Persistent regression tracking with a distributional noise floor
    - Aggregated per-suite summaries, history, and run comparison
    """

    def __init__(
        self,
        eval_dir: Path = Path("evaluation"),
        regression_threshold: float = 0.05,
        min_pass_rate: float = 0.5,
        min_avg_score: float = 0.0,
        min_accuracy: Optional[float] = None,
        min_f1: Optional[float] = None,
        history_file: Optional[Path] = None,
    ):
        """
        Initialize evaluation harness.

        Args:
            eval_dir: Directory for evaluation data and history
            regression_threshold: Threshold for regression detection
            min_pass_rate: Minimum pass rate to pass threshold gates
            min_avg_score: Minimum average score to pass threshold gates
            min_accuracy: Optional minimum accuracy gate for classifier evals
            min_f1: Optional minimum F1 gate for classifier evals
            history_file: Optional override for the persistent history file
        """
        self.eval_dir = Path(eval_dir)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.regression_threshold = regression_threshold
        self.min_pass_rate = min_pass_rate
        self.min_avg_score = min_avg_score
        self.min_accuracy = min_accuracy
        self.min_f1 = min_f1

        self.test_suites: Dict[str, List[Dict[str, Any]]] = {}
        self.history_file = (
            Path(history_file)
            if history_file is not None
            else self.eval_dir / "history" / "evaluation_history.jsonl"
        )
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.evaluation_history: List[Dict[str, Any]] = self._load_history()

    # ------------------------------------------------------------------
    # Test suite management
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_test(raw: Dict[str, Any]) -> Dict[str, Any]:
        test = dict(raw)
        test.setdefault("threshold", 0.5)
        test.setdefault("type", "similarity")
        test.setdefault("id", "")
        test.setdefault("expected", "")
        test.setdefault("prompt", "")
        test.setdefault("text", "")
        return test

    def load_test_suite(self, suite_name: str, test_file: Path) -> List[Dict[str, Any]]:
        """
        Load test suite from a JSONL file (one test per line).

        Args:
            suite_name: Name of test suite
            test_file: Path to test file (JSONL)

        Returns:
            The normalized list of tests
        """
        test_file = Path(test_file)
        if not test_file.exists():
            raise ValueError(f"Test file not found: {test_file}")

        tests: List[Dict[str, Any]] = []
        with open(test_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    logger.warning(f"Skipping invalid JSON on line {line_no} of {test_file}: {exc}")
                    continue
                if not isinstance(raw, dict):
                    logger.warning(f"Skipping non-object on line {line_no} of {test_file}")
                    continue
                tests.append(self._normalize_test(raw))

        if not tests:
            raise ValueError(f"No valid tests found in: {test_file}")

        self.test_suites[suite_name] = tests
        logger.info(f"Loaded {len(tests)} tests for suite: {suite_name}")
        return tests

    def add_test_suite(self, suite_name: str, tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Register a test suite programmatically."""
        normalized = [self._normalize_test(t) for t in tests]
        self.test_suites[suite_name] = normalized
        return normalized

    def load_suites_from_dir(self, suite_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Load every ``*.jsonl`` file in a directory as a named suite."""
        suite_dir = Path(suite_dir)
        loaded: Dict[str, List[Dict[str, Any]]] = {}
        for test_file in sorted(suite_dir.glob("*.jsonl")):
            name = test_file.stem
            self.load_test_suite(name, test_file)
            loaded[name] = self.test_suites[name]
        return loaded

    def list_suites(self) -> List[str]:
        """Return names of all loaded test suites."""
        return sorted(self.test_suites)

    def get_suite(self, suite_name: str) -> List[Dict[str, Any]]:
        """Return the tests of a loaded suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite not found: {suite_name}")
        return self.test_suites[suite_name]

    def validate_suite(self, suite_name: str) -> Dict[str, Any]:
        """
        Validate a suite and report issues.

        Checks that each test carries usable content, a known scoring type,
        and a numeric threshold in [0, 1]. Does not raise; issues are returned.
        """
        tests = self.get_suite(suite_name)
        issues: List[str] = []

        if not tests:
            return {"suite_name": suite_name, "valid": False, "issues": ["empty_suite"]}

        for index, test in enumerate(tests):
            location = f"test[{index}] (id={test.get('id', '')!r})"
            test_type = test.get("type", "similarity")
            if test_type not in _VALID_TEST_TYPES:
                issues.append(f"{location}: unknown test type {test_type!r}")
            try:
                threshold = float(test.get("threshold", 0.5))
            except (TypeError, ValueError):
                issues.append(f"{location}: non-numeric threshold {test.get('threshold')!r}")
            else:
                if not 0.0 <= threshold <= 1.0:
                    issues.append(f"{location}: threshold {threshold} outside [0, 1]")
            has_input = bool(test.get("prompt") or test.get("text"))
            has_expected = bool(test.get("expected"))
            has_label = test.get("label") is not None or bool(test.get("label_name"))
            if not (has_input or has_expected or has_label):
                issues.append(f"{location}: empty test (no prompt/text/expected/label)")

        return {
            "suite_name": suite_name,
            "valid": not issues,
            "total_tests": len(tests),
            "issues": issues,
        }

    # ------------------------------------------------------------------
    # Evaluation entry points
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_mode(tests: List[Dict[str, Any]], mode: str) -> str:
        if mode in {"generation", "classifier"}:
            return mode
        has_labels = any(test.get("label") is not None or test.get("label_name") for test in tests)
        has_generation = any(test.get("prompt") or test.get("expected") for test in tests)
        if has_labels and not has_generation:
            return "classifier"
        if has_generation:
            return "generation"
        return "classifier" if has_labels else "generation"

    def evaluate_model(
        self,
        model,
        tokenizer_manager,
        suite_name: str,
        max_new_tokens: int = 100,
        mode: str = "auto",
        predict_fn: Optional[Callable[[List[str]], List[Any]]] = None,
        collect_latency: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a model (with a tokenizer manager) on a test suite.

        Args:
            model: Model to evaluate (causal LM for generation; unused when a
                predict_fn is supplied for classifier suites)
            tokenizer_manager: Tokenizer manager for generation
            suite_name: Test suite name
            max_new_tokens: Maximum tokens to generate
            mode: "auto", "generation", or "classifier"
            predict_fn: Callable mapping a list of texts to predictions. Each
                prediction is either an int label, a (label, confidence) tuple,
                or a dict with "label"/"confidence" keys.
            collect_latency: Measure per-sample generation latency

        Returns:
            Evaluation results dict
        """
        tests = self.get_suite(suite_name)
        resolved_mode = self._resolve_mode(tests, mode)

        if resolved_mode == "classifier":
            if predict_fn is None:
                raise ValueError(
                    "Suite contains labeled classification tests; pass a predict_fn "
                    "(mapping texts to labels) or use mode='generation'."
                )
            result = self._evaluate_classifier(tests, predict_fn)
        else:
            if not tests:
                return self._empty_generation_result(suite_name, "generation")
            result = self._evaluate_subject_raw(
                LegacyModelGenerator(model, tokenizer_manager),
                tests,
                max_new_tokens=max_new_tokens,
                collect_latency=collect_latency,
            )

        return self._finalize_result(result, suite_name, resolved_mode)

    def evaluate_subject(
        self,
        subject: ResponseGenerator,
        suite_name: str,
        max_new_tokens: int = 100,
        collect_latency: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate any subject (a model or an agent) on a generation suite.

        ``subject`` is a :class:`ResponseGenerator` — e.g.
        :class:`HFModelGenerator` for a HuggingFace causal LM or
        :class:`AgentGenerator` / :class:`CallableGenerator` for an agent.

        Args:
            subject: The model or agent to evaluate
            suite_name: Test suite name
            max_new_tokens: Maximum tokens to generate
            collect_latency: Measure per-sample generation latency

        Returns:
            Evaluation results dict
        """
        tests = self.get_suite(suite_name)
        if not tests:
            return self._empty_generation_result(suite_name, "generation")
        result = self._evaluate_subject_raw(
            subject,
            tests,
            max_new_tokens=max_new_tokens,
            collect_latency=collect_latency,
        )
        return self._finalize_result(result, suite_name, "generation")

    def _evaluate_subject_raw(
        self,
        subject: ResponseGenerator,
        tests: List[Dict[str, Any]],
        max_new_tokens: int = 100,
        collect_latency: bool = True,
    ) -> Dict[str, Any]:
        """Score a subject against an already-loaded suite (no history write)."""
        return self._evaluate_generation(
            self._subject_generate_fn(subject),
            tests,
            max_new_tokens=max_new_tokens,
            collect_latency=collect_latency,
            token_counter=self._subject_token_counter(subject),
        )

    @staticmethod
    def _subject_token_counter(
        subject: ResponseGenerator,
    ) -> Optional[Callable[[str], int]]:
        count = getattr(subject, "count_tokens", None)
        if not callable(count):
            return None
        count = cast(Callable[[str], Optional[int]], count)

        def _count(text: str) -> int:
            counted = count(text)
            if counted is None:
                return len(text.split())
            return counted

        return _count

    def evaluate_predictor(
        self,
        predictor: LabelPredictor,
        suite_name: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a classifier subject (a model or an agent) on a labeled suite.

        Args:
            predictor: A :class:`LabelPredictor` (e.g. CallablePredictor or
                HFClassifierPredictor)
            suite_name: Test suite name

        Returns:
            Evaluation results dict
        """
        tests = self.get_suite(suite_name)
        result = self._evaluate_classifier(tests, predictor.predict)
        return self._finalize_result(result, suite_name, "classifier")

    def evaluate_classifier_suite(
        self,
        suite_name: str,
        predict_fn: Callable[[List[str]], List[Any]],
    ) -> Dict[str, Any]:
        """Evaluate a classification suite using a predict_fn."""
        return self.evaluate_predictor(
            CallablePredictor(predict_fn),
            suite_name,
        )

    def run_full_evaluation(
        self,
        model,
        tokenizer_manager,
        suite_names: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        mode: str = "auto",
        predict_fn: Optional[Callable[[List[str]], List[Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run every loaded suite (or a subset) and return per-suite results."""
        suite_names = suite_names or self.list_suites()
        results: Dict[str, Dict[str, Any]] = {}
        for name in suite_names:
            results[name] = self.evaluate_model(
                model,
                tokenizer_manager,
                name,
                max_new_tokens=max_new_tokens,
                mode=mode,
                predict_fn=predict_fn,
            )
        return results

    def run_full_subject_evaluation(
        self,
        subject: ResponseGenerator,
        suite_names: Optional[List[str]] = None,
        max_new_tokens: int = 100,
    ) -> Dict[str, Dict[str, Any]]:
        """Run every loaded generation suite against a model or agent subject."""
        suite_names = suite_names or self.list_suites()
        results: Dict[str, Dict[str, Any]] = {}
        for name in suite_names:
            results[name] = self.evaluate_subject(subject, name, max_new_tokens=max_new_tokens)
        return results

    def compare_models(
        self,
        model_a,
        tokenizer_a,
        model_b,
        tokenizer_b,
        suite_name: str,
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """Evaluate two models on the same suite and compare per-test scores."""
        result_a = self.evaluate_model(
            model_a, tokenizer_a, suite_name, max_new_tokens=max_new_tokens
        )
        result_b = self.evaluate_model(
            model_b, tokenizer_b, suite_name, max_new_tokens=max_new_tokens
        )
        return self._compare_results(result_a, result_b)

    def compare_subjects(
        self,
        subject_a: ResponseGenerator,
        subject_b: ResponseGenerator,
        suite_name: str,
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """Evaluate two model/agent subjects on the same suite and compare."""
        result_a = self.evaluate_subject(subject_a, suite_name, max_new_tokens=max_new_tokens)
        result_b = self.evaluate_subject(subject_b, suite_name, max_new_tokens=max_new_tokens)
        return self._compare_results(result_a, result_b, names=(subject_a.name, subject_b.name))

    def run_evaluation_matrix(
        self,
        subjects: Dict[str, ResponseGenerator],
        suite_names: Optional[List[str]] = None,
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """
        Evaluate multiple named subjects against multiple suites.

        Returns a matrix of ``avg_score`` / ``pass_rate`` keyed by suite, so a
        comparison can be rendered without re-running each pairing.

        Args:
            subjects: Mapping of display name to subject
            suite_names: Suites to run; defaults to every loaded suite
            max_new_tokens: Maximum tokens to generate per sample
        """
        suite_names = suite_names or self.list_suites()
        matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for suite_name in suite_names:
            matrix[suite_name] = {}
            for name, subject in subjects.items():
                result = self.evaluate_subject(subject, suite_name, max_new_tokens=max_new_tokens)
                matrix[suite_name][name] = {
                    "avg_score": result.get("avg_score", 0.0),
                    "pass_rate": result.get("pass_rate", 0.0),
                    "passed_tests": result.get("passed_tests", 0),
                    "total_tests": result.get("total_tests", 0),
                }
        return {"subjects": list(subjects), "suites": suite_names, "matrix": matrix}

    def export_markdown_report(
        self,
        results: Dict[str, Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Render per-suite results (from ``run_full_evaluation`` or
        ``run_full_subject_evaluation``) as a Markdown table.

        Args:
            results: Mapping of suite name to evaluation result dict
            output_path: Optional destination; when given, the report is written

        Returns:
            The Markdown report string
        """
        lines = [
            "# Evaluation Report",
            "",
            f"- **Generated:** {datetime.now(timezone.utc).isoformat()}",
            f"- **Suites:** {len(results)}",
            "",
        ]
        for suite_name, result in results.items():
            lines.append(f"## {suite_name}")
            lines.append("")
            lines.append("| metric | value |")
            lines.append("| --- | --- |")
            lines.append(f"| total_tests | {result.get('total_tests', 0)} |")
            lines.append(f"| passed_tests | {result.get('passed_tests', 0)} |")
            lines.append(f"| pass_rate | {result.get('pass_rate', 0.0):.3f} |")
            lines.append(f"| avg_score | {result.get('avg_score', 0.0):.3f} |")
            if result.get("mode") == "classifier":
                overall = result.get("overall") or {}
                lines.append(f"| accuracy | {overall.get('accuracy', 0.0):.3f} |")
                lines.append(f"| f1 | {overall.get('f1', 0.0):.3f} |")
            if result.get("avg_latency_ms") is not None:
                lines.append(f"| avg_latency_ms | {result.get('avg_latency_ms', 0.0):.3f} |")
            passed = "passed" if result.get("passed") else "failed"
            lines.append(f"| gates | {passed} |")
            lines.append("")
        report = "\n".join(lines).rstrip()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report + "\n", encoding="utf-8")
        return report

    def _compare_results(
        self,
        result_a: Dict[str, Any],
        result_b: Dict[str, Any],
        names: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, Any]:
        per_test = []
        rows_a = result_a.get("results", [])
        rows_b = result_b.get("results", [])
        for row_a, row_b in zip(rows_a, rows_b):
            per_test.append(
                {
                    "test_id": row_a.get("test_id", ""),
                    "score_a": row_a.get("score"),
                    "score_b": row_b.get("score"),
                    "delta": (row_b.get("score") or 0.0) - (row_a.get("score") or 0.0),
                    "passed_a": row_a.get("passed"),
                    "passed_b": row_b.get("passed"),
                }
            )
        return {
            "suite_name": result_a.get("suite_name", ""),
            "subject_a": names[0] if names else "a",
            "subject_b": names[1] if names else "b",
            "avg_score_a": result_a.get("avg_score", 0.0),
            "avg_score_b": result_b.get("avg_score", 0.0),
            "avg_score_delta": (result_b.get("avg_score", 0.0) - result_a.get("avg_score", 0.0)),
            "pass_rate_a": result_a.get("pass_rate", 0.0),
            "pass_rate_b": result_b.get("pass_rate", 0.0),
            "per_test": per_test,
        }

    # ------------------------------------------------------------------
    # Finalization shared by every evaluation entry point
    # ------------------------------------------------------------------

    def _empty_generation_result(self, suite_name: str, mode: str) -> Dict[str, Any]:
        logger.warning(f"Suite {suite_name!r} is empty; returning empty result")
        return self._finalize_result(
            {
                "total_tests": 0,
                "passed_tests": 0,
                "pass_rate": 0.0,
                "avg_score": 0.0,
                "score_std": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "results": [],
            },
            suite_name,
            mode,
        )

    def _finalize_result(
        self,
        result: Dict[str, Any],
        suite_name: str,
        mode: str,
    ) -> Dict[str, Any]:
        """Attach metadata, apply gates, check regression, and persist."""
        result["suite_name"] = suite_name
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        result["mode"] = mode

        self._apply_gates(result)

        if mode == "classifier":
            overall = result.get("overall") or {}
            f1 = overall.get("f1")
            regression = (
                self._check_regression(suite_name, float(f1), metric_name="f1")
                if f1 is not None
                else None
            )
        else:
            regression = self._check_regression(
                suite_name, result.get("avg_score", 0.0), metric_name="avg_score"
            )

        if regression:
            logger.warning(f"Regression detected in {suite_name}: {regression}")
            result["regression"] = regression
            result["passed"] = False
            result["failures"] = result.get("failures", []) + ["regression_detected"]

        self._save_evaluation(result)

        logger.info(
            f"Evaluation complete: {suite_name} ({mode}) - "
            f"{result.get('passed_tests', 0)}/{result.get('total_tests', 0)} "
            f"passed (avg {result.get('avg_score', 0.0):.2%})"
        )
        return result

    # ------------------------------------------------------------------
    # Generation evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _subject_generate_fn(subject: ResponseGenerator):
        def _fn(prompt: str, max_new_tokens: int) -> str:
            return subject.generate(prompt, max_new_tokens)

        return _fn

    def _evaluate_generation(
        self,
        generate_fn: Callable[[str, int], str],
        tests: List[Dict[str, Any]],
        max_new_tokens: int = 100,
        collect_latency: bool = True,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> Dict[str, Any]:
        """
        Score a generation subject (model or agent) against a suite.

        ``generate_fn(prompt, max_new_tokens) -> str`` abstracts the subject so
        models and agents are evaluated through the same path.
        """
        results: List[Dict[str, Any]] = []
        latencies_ms: List[float] = []
        tokens_per_sec: List[float] = []

        for test in tests:
            prompt = test.get("prompt") or test.get("text") or ""
            expected = test.get("expected", "")
            test_type = test.get("type", "similarity")
            threshold = float(test.get("threshold", 0.5))

            start = time.perf_counter()
            generated_text = generate_fn(prompt, max_new_tokens)
            latency_ms = (time.perf_counter() - start) * 1000.0

            if token_counter is not None:
                new_tokens = token_counter(generated_text)
            else:
                new_tokens = len(generated_text.split())

            if collect_latency:
                latencies_ms.append(latency_ms)
                if latency_ms > 0:
                    tokens_per_sec.append(new_tokens / (latency_ms / 1000.0))

            score = self._score_response(generated_text, expected, test_type)

            results.append(
                {
                    "test_id": test.get("id", ""),
                    "prompt": prompt,
                    "expected": expected,
                    "generated": generated_text,
                    "score": score,
                    "passed": score >= threshold,
                    "test_type": test_type,
                    "latency_ms": latency_ms,
                    "tokens_generated": new_tokens,
                }
            )

        scores = [r["score"] for r in results]
        total = len(results)
        passed = sum(1 for r in results if r["passed"])

        result: Dict[str, Any] = {
            "total_tests": total,
            "passed_tests": passed,
            "pass_rate": passed / total if total else 0.0,
            "avg_score": sum(scores) / total if total else 0.0,
            "score_std": float(np.std(scores)) if scores else 0.0,
            "min_score": float(min(scores)) if scores else 0.0,
            "max_score": float(max(scores)) if scores else 0.0,
            "results": results,
        }

        if collect_latency and latencies_ms:
            lat_arr = np.array(latencies_ms)
            result["latency"] = {
                "avg_latency_ms": float(lat_arr.mean()),
                "p50_latency_ms": float(np.percentile(lat_arr, 50)),
                "p95_latency_ms": float(np.percentile(lat_arr, 95)),
                "max_latency_ms": float(lat_arr.max()),
                "min_latency_ms": float(lat_arr.min()),
                "avg_tokens_per_sec": (float(np.mean(tokens_per_sec)) if tokens_per_sec else None),
            }
            result["avg_latency_ms"] = float(lat_arr.mean())

        return result

    def benchmark_subject(
        self,
        subject: ResponseGenerator,
        prompt: str,
        max_new_tokens: int = 50,
        num_runs: int = 5,
    ) -> Dict[str, Any]:
        """Benchmark generation latency and throughput for a model/agent subject."""
        if num_runs < 1:
            raise ValueError("num_runs must be >= 1")

        latencies: List[float] = []
        tokens_per_sec: List[float] = []
        counter = self._subject_token_counter(subject)
        for _ in range(num_runs):
            start = time.perf_counter()
            text = subject.generate(prompt, max_new_tokens)
            latency_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(latency_ms)
            if latency_ms > 0:
                tokens = counter(text) if counter is not None else len(text.split())
                tokens_per_sec.append(tokens / (latency_ms / 1000.0))

        lat_arr = np.array(latencies)
        return {
            "subject": subject.name,
            "prompt": prompt,
            "num_runs": num_runs,
            "avg_latency_ms": float(lat_arr.mean()),
            "p50_latency_ms": float(np.percentile(lat_arr, 50)),
            "p95_latency_ms": float(np.percentile(lat_arr, 95)),
            "max_latency_ms": float(lat_arr.max()),
            "min_latency_ms": float(lat_arr.min()),
            "avg_tokens_per_sec": float(np.mean(tokens_per_sec)) if tokens_per_sec else None,
        }

    def benchmark_latency(
        self,
        model,
        tokenizer_manager,
        prompt: str,
        max_new_tokens: int = 50,
        num_runs: int = 5,
    ) -> Dict[str, Any]:
        """Benchmark generation latency and throughput for a model."""
        return self.benchmark_subject(
            LegacyModelGenerator(model, tokenizer_manager),
            prompt,
            max_new_tokens=max_new_tokens,
            num_runs=num_runs,
        )

    # ------------------------------------------------------------------
    # Classification evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_predictions(
        predictions: List[Any],
    ) -> Tuple[List[int], List[Optional[float]]]:
        labels: List[int] = []
        confidences: List[Optional[float]] = []
        for pred in predictions:
            if isinstance(pred, (tuple, list)) and len(pred) == 2:
                labels.append(int(pred[0]))
                confidences.append(float(pred[1]))
            elif isinstance(pred, dict):
                labels.append(int(pred["label"]))
                conf = pred.get("confidence")
                confidences.append(float(conf) if conf is not None else None)
            else:
                labels.append(int(pred))
                confidences.append(None)
        return labels, confidences

    def _evaluate_classifier(
        self,
        tests: List[Dict[str, Any]],
        predict_fn: Callable[[List[str]], List[Any]],
    ) -> Dict[str, Any]:
        labeled: List[Dict[str, Any]] = []
        name_to_id: Dict[str, int] = {}
        id_to_name: Dict[int, str] = {}

        for test in tests:
            text = test.get("text") or test.get("prompt") or ""
            label = test.get("label")
            label_name = test.get("label_name")
            if label is None and label_name is not None:
                if label_name not in name_to_id:
                    name_to_id[label_name] = len(name_to_id)
                label = name_to_id[label_name]
            if label is None:
                continue
            label_id = int(label)
            if label_id not in id_to_name:
                id_to_name[label_id] = str(label_name or f"category_{label_id}")
            entry = dict(test)
            entry["text"] = text
            entry["label"] = label_id
            labeled.append(entry)

        if not labeled:
            raise ValueError("No labeled classification tests found in suite")

        texts = [t["text"] for t in labeled]
        y_true = [t["label"] for t in labeled]
        predictions = predict_fn(texts)
        y_pred, y_conf = self._normalize_predictions(predictions)
        if len(y_pred) != len(y_true):
            raise ValueError(
                f"predict_fn returned {len(y_pred)} predictions for {len(y_true)} texts"
            )

        metrics = classification_metrics(y_true, y_pred, y_conf, label_names=id_to_name)
        total = len(y_true)
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)

        per_test = []
        for entry, pred_id, conf in zip(labeled, y_pred, y_conf):
            true_label = entry["label"]
            per_test.append(
                {
                    "test_id": entry.get("id", ""),
                    "text": entry["text"],
                    "expected_label": true_label,
                    "expected_label_name": id_to_name.get(true_label, str(true_label)),
                    "predicted_label": pred_id,
                    "predicted_label_name": id_to_name.get(pred_id, str(pred_id)),
                    "confidence": conf,
                    "passed": pred_id == true_label,
                }
            )

        result: Dict[str, Any] = {
            "total_tests": total,
            "passed_tests": correct,
            "pass_rate": correct / total,
            "avg_score": metrics["overall"].get("accuracy", 0.0),
            "overall": metrics["overall"],
            "per_class": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
            "results": per_test,
        }
        return result

    # ------------------------------------------------------------------
    # Scoring / gating / persistence
    # ------------------------------------------------------------------

    def _score_response(self, generated: str, expected: str, test_type: str) -> float:
        """Score a generated response (delegates to module-level scorer)."""
        return score_response(generated, expected, test_type)

    def _apply_gates(self, result: Dict[str, Any]) -> None:
        """Apply configured pass/fail threshold gates to a result."""
        failures: List[str] = []
        pass_rate = result.get("pass_rate", 0.0)
        avg_score = result.get("avg_score", 0.0)

        if pass_rate < self.min_pass_rate:
            failures.append(f"pass_rate_below_threshold:{pass_rate:.4f}")
        if avg_score < self.min_avg_score:
            failures.append(f"avg_score_below_threshold:{avg_score:.4f}")

        if result.get("mode") == "classifier":
            overall = result.get("overall") or {}
            accuracy = overall.get("accuracy")
            f1 = overall.get("f1")
            if (
                self.min_accuracy is not None
                and accuracy is not None
                and accuracy < self.min_accuracy
            ):
                failures.append(f"accuracy_below_threshold:{accuracy:.4f}")
            if self.min_f1 is not None and f1 is not None and f1 < self.min_f1:
                failures.append(f"f1_below_threshold:{f1:.4f}")

        result["passed"] = len(failures) == 0
        result["failures"] = failures

    @staticmethod
    def _history_record(result: Dict[str, Any]) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "suite_name": result.get("suite_name", ""),
            "mode": result.get("mode", ""),
            "timestamp": result.get("timestamp", ""),
            "total_tests": result.get("total_tests", 0),
            "passed_tests": result.get("passed_tests", 0),
            "pass_rate": result.get("pass_rate", 0.0),
            "avg_score": result.get("avg_score", 0.0),
        }
        overall = result.get("overall") or {}
        if overall:
            record["accuracy"] = overall.get("accuracy")
            record["f1"] = overall.get("f1")
        if result.get("avg_latency_ms") is not None:
            record["avg_latency_ms"] = result.get("avg_latency_ms")
        return record

    def _save_evaluation(self, result: Dict[str, Any]) -> Path:
        """Persist the evaluation result as a JSON report and history record."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        result_file = self.eval_dir / f"eval_{result['suite_name']}_{timestamp}.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        record = self._history_record(result)
        self.evaluation_history.append(record)
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return result_file

    def _load_history(self, suite_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load persisted evaluation records, optionally filtered by suite."""
        if not self.history_file.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if suite_name is not None and row.get("suite_name") != suite_name:
                    continue
                records.append(row)
        return records

    def get_history(self, suite_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return persisted evaluation history, optionally filtered by suite."""
        return self._load_history(suite_name=suite_name)

    # ------------------------------------------------------------------
    # Regression tracking
    # ------------------------------------------------------------------

    def _check_regression(
        self,
        suite_name: str,
        current_score: float,
        metric_name: str = "avg_score",
    ) -> Optional[Dict[str, Any]]:
        """
        Detect regression against the best and mean of prior runs.

        Invariant: scores are dimensionless and bounded in [0, 1], so the drop
        is measured in the same units as ``regression_threshold``.

        Dropping below the historical best by more than ``regression_threshold``
        is a *necessary* condition. Once enough prior runs exist (n >= 2), the
        drop must also exceed the historical noise floor (max of the configured
        threshold and one sample standard deviation) so that a single lucky best
        run cannot set an unreachable bar.
        """
        prior_scores = [
            float(row[metric_name])
            for row in self._load_history(suite_name=suite_name)
            if metric_name in row and row[metric_name] is not None
        ]
        if not prior_scores:
            return None

        best_previous = max(prior_scores)
        score_drop = best_previous - current_score
        if score_drop <= self.regression_threshold:
            return None

        if len(prior_scores) >= 2:
            mean = sum(prior_scores) / len(prior_scores)
            variance = sum((s - mean) ** 2 for s in prior_scores) / len(prior_scores)
            std = variance**0.5
            noise_floor = max(self.regression_threshold, std)
            if mean - current_score <= noise_floor:
                return None

        return {
            "detected": True,
            "metric": metric_name,
            "current_score": current_score,
            "previous_best": best_previous,
            "score_drop": score_drop,
        }

    # ------------------------------------------------------------------
    # Summaries and reporting
    # ------------------------------------------------------------------

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Return aggregate statistics across all recorded evaluations."""
        if not self.evaluation_history:
            return {"total_evaluations": 0, "suites": {}}

        suite_summaries: Dict[str, Dict[str, Any]] = {}
        for record in self.evaluation_history:
            suite_name = str(record.get("suite_name", "unknown"))
            bucket = suite_summaries.setdefault(
                suite_name, {"count": 0, "scores": [], "pass_rates": []}
            )
            bucket["count"] += 1
            if record.get("avg_score") is not None:
                bucket["scores"].append(float(record["avg_score"]))
            if record.get("pass_rate") is not None:
                bucket["pass_rates"].append(float(record["pass_rate"]))

        for suite_name, bucket in suite_summaries.items():
            scores = bucket.pop("scores")
            pass_rates = bucket.pop("pass_rates")
            if scores:
                bucket["mean_score"] = sum(scores) / len(scores)
                bucket["max_score"] = max(scores)
                bucket["min_score"] = min(scores)
                bucket["std_score"] = float(np.std(scores)) if len(scores) > 1 else 0.0
                bucket["latest_score"] = scores[-1]
                bucket["trend"] = scores[-1] - scores[-2] if len(scores) > 1 else None
            if pass_rates:
                bucket["mean_pass_rate"] = sum(pass_rates) / len(pass_rates)

        return {
            "total_evaluations": len(self.evaluation_history),
            "suites": suite_summaries,
        }

    def compare_runs(self, suite_name: str, n: int = 2) -> Optional[Dict[str, Any]]:
        """Compare the last ``n`` recorded runs of a suite."""
        history = self._load_history(suite_name=suite_name)
        if len(history) < 2:
            return None
        recent = history[-n:]
        first, last = recent[0], recent[-1]

        def _summary(record: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "timestamp": record.get("timestamp", ""),
                "avg_score": record.get("avg_score"),
                "pass_rate": record.get("pass_rate"),
                "accuracy": record.get("accuracy"),
                "f1": record.get("f1"),
            }

        return {
            "suite_name": suite_name,
            "runs_compared": len(recent),
            "first": _summary(first),
            "last": _summary(last),
            "avg_score_delta": (last.get("avg_score") or 0.0) - (first.get("avg_score") or 0.0),
        }

    def save_report(self, result: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """Persist a full evaluation result as a JSON report."""
        output_path = (
            Path(output_path)
            if output_path is not None
            else (self.eval_dir / f"report_{result.get('suite_name', 'eval')}.json")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Report saved: {output_path}")
        return output_path

    @staticmethod
    def load_report(report_path: Path) -> Dict[str, Any]:
        """Load a JSON evaluation report."""
        with open(report_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {"payload": payload}

    def write_aggregate_report(
        self,
        results: Dict[str, Dict[str, Any]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Combine per-suite results into a single aggregate JSON report.

        Args:
            results: Mapping of suite name to evaluation result dict
                (e.g. from ``run_full_evaluation``)
            output_path: Destination path; defaults to
                ``eval_dir/aggregate_report.json``

        Returns:
            The report path
        """
        summary = {
            "total_suites": len(results),
            "passed_suites": sum(1 for r in results.values() if r.get("passed")),
            "overall_pass_rate": float(
                np.mean([r.get("pass_rate", 0.0) for r in results.values()]) if results else 0.0
            ),
            "overall_avg_score": float(
                np.mean([r.get("avg_score", 0.0) for r in results.values()]) if results else 0.0
            ),
        }
        aggregate = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "suites": results,
        }
        output_path = (
            Path(output_path)
            if output_path is not None
            else (self.eval_dir / "aggregate_report.json")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)
        logger.info(f"Aggregate report saved: {output_path}")
        return output_path
