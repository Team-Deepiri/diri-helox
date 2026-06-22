"""Post-training evaluation orchestrator."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmark import InferenceBenchmark
from .classifier import ClassifierEvaluator
from .generation import GenerationEvaluator
from .parity import InferenceParityTester
from .regression import RegressionTracker
from .report import save_eval_report
from .samples import EvaluationSample, load_jsonl_suite
from .schemas import EvalRunConfig, EvalRunResult, EvalThresholds


class PostTrainingEvalHarness:
    """
    End-to-end post-training evaluation harness.

    Supports:
    - Classifier metrics (accuracy, F1, per-class, confusion matrix)
    - Generation suites (exact match, contains, similarity, rouge_l)
    - Optional parity and latency benchmarks
    - Regression tracking against prior runs
    - Threshold gates for CI / release promotion
    """

    def __init__(
        self,
        config: EvalRunConfig,
        suites: Optional[Dict[str, List[EvaluationSample]]] = None,
    ) -> None:
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.suites = suites or {}
        self.regression = RegressionTracker(
            history_dir=self.config.output_dir / "history",
            regression_threshold=self.config.regression_threshold,
        )
        self.classifier = ClassifierEvaluator(
            model_path=self.config.model_path,
            num_labels=self.config.num_labels,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
        )
        self.generation = GenerationEvaluator(max_new_tokens=self.config.max_new_tokens)
        self.parity = InferenceParityTester()
        self.benchmark = InferenceBenchmark()

    def load_suite(self, suite_name: str, suite_path: Path) -> None:
        """Load or replace a named evaluation suite from JSONL."""
        self.suites[suite_name] = load_jsonl_suite(suite_path)

    def run(
        self,
        suite_name: Optional[str] = None,
        mode: str = "auto",
        parity_input_ids=None,
        parity_model=None,
    ) -> EvalRunResult:
        """
        Execute a full evaluation run.

        mode:
            auto — classifier if labels present, else generation
            classifier — force classification metrics
            generation — force generation scoring
        """
        suite_name = suite_name or self.config.suite_name
        samples = self.suites.get(suite_name, [])
        run_id = uuid.uuid4().hex[:12]
        result = EvalRunResult(
            run_id=run_id,
            model_path=str(self.config.model_path),
            suite_name=suite_name,
            metadata=dict(self.config.metadata),
        )

        resolved_mode = self._resolve_mode(samples, mode)
        if resolved_mode == "classifier":
            result.classification = self.classifier.evaluate(samples)
            self._apply_classifier_gates(result)
            self._track_classifier_regression(result)
        else:
            result.generation = self._run_generation(samples)
            self._apply_generation_gates(result)
            self._track_generation_regression(result)

        if self.config.run_parity and parity_model is not None and parity_input_ids is not None:
            result.parity = self.parity.run_full_parity_suite(parity_model, parity_input_ids)
            if not result.parity.get("all_tests_passed", False):
                result.passed = False
                result.failures.append("parity_suite_failed")

        if self.config.run_benchmark and samples:
            result.benchmark = self._run_benchmark(samples)

        report_path = self.config.output_dir / f"eval_{suite_name}_{run_id}.json"
        save_eval_report(result, report_path)
        self.regression.record(self._history_record(result))
        return result

    def get_history_summary(self) -> Dict[str, Any]:
        return self.regression.summary()

    def _resolve_mode(self, samples: List[EvaluationSample], mode: str) -> str:
        if mode in {"classifier", "generation"}:
            return mode
        has_labels = any(
            sample.label is not None or sample.label_name for sample in samples
        )
        has_generation = any(sample.prompt or sample.expected for sample in samples)
        if has_labels and not has_generation:
            return "classifier"
        if has_generation:
            return "generation"
        return "classifier" if has_labels else "generation"

    def _run_generation(self, samples: List[EvaluationSample]) -> Dict[str, Any]:
        if not samples:
            return {
                "total_tests": 0,
                "passed_tests": 0,
                "pass_rate": 0.0,
                "avg_score": 0.0,
                "results": [],
            }

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(self.config.model_path))
        model = AutoModelForCausalLM.from_pretrained(str(self.config.model_path))
        return self.generation.evaluate_hf_causal_lm(model, tokenizer, samples)

    def _run_benchmark(self, samples: List[EvaluationSample]) -> Dict[str, Any]:
        texts = [sample.text for sample in samples if sample.text][: self.config.benchmark_samples]
        if not texts:
            return {}

        def _predict_batch(batch: List[str]) -> None:
            self.classifier.predict(batch)

        return self.benchmark.benchmark_texts(_predict_batch, texts, batch_size=1)

    def _apply_classifier_gates(self, result: EvalRunResult) -> None:
        thresholds = self.config.thresholds
        overall = (result.classification or {}).get("overall", {})
        accuracy = overall.get("accuracy")
        f1 = overall.get("f1")
        if accuracy is not None and accuracy < thresholds.min_accuracy:
            result.passed = False
            result.failures.append(f"accuracy_below_threshold:{accuracy:.4f}")
        if f1 is not None and f1 < thresholds.min_f1:
            result.passed = False
            result.failures.append(f"f1_below_threshold:{f1:.4f}")

    def _apply_generation_gates(self, result: EvalRunResult) -> None:
        thresholds = self.config.thresholds
        generation = result.generation or {}
        pass_rate = generation.get("pass_rate", 0.0)
        avg_score = generation.get("avg_score", 0.0)
        if pass_rate < thresholds.min_pass_rate:
            result.passed = False
            result.failures.append(f"pass_rate_below_threshold:{pass_rate:.4f}")
        if avg_score < thresholds.min_avg_score:
            result.passed = False
            result.failures.append(f"avg_score_below_threshold:{avg_score:.4f}")

    def _track_classifier_regression(self, result: EvalRunResult) -> None:
        overall = (result.classification or {}).get("overall", {})
        f1 = overall.get("f1")
        if f1 is None:
            return
        regression = self.regression.check_score_regression(
            result.suite_name,
            metric_name="f1",
            current_score=float(f1),
        )
        if regression:
            result.regression = regression
            if regression["score_drop"] > self.config.thresholds.max_regression_drop:
                result.passed = False
                result.failures.append("classification_regression_detected")

    def _track_generation_regression(self, result: EvalRunResult) -> None:
        generation = result.generation or {}
        avg_score = generation.get("avg_score")
        if avg_score is None:
            return
        regression = self.regression.check_score_regression(
            result.suite_name,
            metric_name="avg_score",
            current_score=float(avg_score),
        )
        if regression:
            result.regression = regression
            if regression["score_drop"] > self.config.thresholds.max_regression_drop:
                result.passed = False
                result.failures.append("generation_regression_detected")

    @staticmethod
    def _history_record(result: EvalRunResult) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "run_id": result.run_id,
            "suite_name": result.suite_name,
            "model_path": result.model_path,
            "timestamp": result.timestamp,
            "passed": result.passed,
        }
        if result.classification:
            record["f1"] = result.classification.get("overall", {}).get("f1")
            record["accuracy"] = result.classification.get("overall", {}).get("accuracy")
        if result.generation:
            record["avg_score"] = result.generation.get("avg_score")
            record["pass_rate"] = result.generation.get("pass_rate")
        return record
