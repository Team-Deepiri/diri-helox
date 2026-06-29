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
