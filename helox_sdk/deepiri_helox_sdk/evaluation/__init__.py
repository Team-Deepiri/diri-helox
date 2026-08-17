"""Post-training evaluation harness for Helox-trained models."""

from .automatic_evaluation_harness import AutomaticEvaluationHarness
from .benchmark import InferenceBenchmark
from .categories import CATEGORIES, LABEL_TO_ID
from .classifier import ClassifierEvaluator
from .comparison import ModelComparisonReport, compare_model_directories
from .generation import GenerationEvaluator
from .harness import PostTrainingEvalHarness
from .judge import DEFAULT_RUBRIC, JUDGE_TEST_TYPE, JudgeParseError, LlmJudge
from .metrics import (
    classification_metrics,
    rouge_l_recall,
    score_response,
    score_text_response,
    token_f1_score,
    word_overlap_score,
)
from .parity import InferenceParityTester
from .regression import RegressionTracker
from .report import load_eval_report, save_eval_report
from .samples import EvaluationSample, load_jsonl_suite
from .schemas import EvalRunConfig, EvalThresholds, EvalRunResult
from .subjects import (
    AgentGenerator,
    AnthropicGenerator,
    ApiGenerator,
    CallableGenerator,
    CallablePredictor,
    HFClassifierPredictor,
    HFModelGenerator,
    LabelPredictor,
    LegacyModelGenerator,
    OllamaGenerator,
    OpenAIGenerator,
    ResponseGenerator,
)

__all__ = [
    "AgentGenerator",
    "AnthropicGenerator",
    "ApiGenerator",
    "AutomaticEvaluationHarness",
    "CATEGORIES",
    "CallableGenerator",
    "CallablePredictor",
    "ClassifierEvaluator",
    "DEFAULT_RUBRIC",
    "EvalRunConfig",
    "EvalRunResult",
    "EvalThresholds",
    "EvaluationSample",
    "GenerationEvaluator",
    "HFClassifierPredictor",
    "HFModelGenerator",
    "InferenceBenchmark",
    "InferenceParityTester",
    "JUDGE_TEST_TYPE",
    "JudgeParseError",
    "LABEL_TO_ID",
    "LabelPredictor",
    "LegacyModelGenerator",
    "LlmJudge",
    "ModelComparisonReport",
    "OllamaGenerator",
    "OpenAIGenerator",
    "PostTrainingEvalHarness",
    "RegressionTracker",
    "ResponseGenerator",
    "classification_metrics",
    "compare_model_directories",
    "load_eval_report",
    "load_jsonl_suite",
    "rouge_l_recall",
    "save_eval_report",
    "score_response",
    "score_text_response",
    "token_f1_score",
    "word_overlap_score",
]
