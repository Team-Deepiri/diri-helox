"""
In-repo evaluation entrypoint.

Portable installs should use ``deepiri_helox_sdk.evaluation`` from the ``helox_sdk``
subdirectory package.
"""

try:
    from deepiri_helox_sdk.evaluation import (  # noqa: F401
        ClassifierEvaluator,
        GenerationEvaluator,
        InferenceBenchmark,
        InferenceParityTester,
        PostTrainingEvalHarness,
        RegressionTracker,
        compare_model_directories,
        load_eval_report,
        load_jsonl_suite,
        save_eval_report,
    )
except ImportError:
    from .automatic_evaluation_harness import AutomaticEvaluationHarness  # noqa: F401
    from .inference_parity_tester import InferenceParityTester  # noqa: F401
    from .model_evaluator import ModelEvaluator  # noqa: F401

from .automatic_evaluation_harness import (  # noqa: F401
    AutomaticEvaluationHarness,
    classification_metrics,
    rouge_l_recall,
    score_response,
    token_f1_score,
    word_overlap_score,
)
from .subjects import (  # noqa: F401
    AgentGenerator,
    CallableGenerator,
    CallablePredictor,
    HFClassifierPredictor,
    HFModelGenerator,
    LabelPredictor,
    LegacyModelGenerator,
    ResponseGenerator,
)
