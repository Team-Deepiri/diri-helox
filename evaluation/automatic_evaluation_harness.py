"""
Re-export of the automatic evaluation harness from the Helox SDK.

The canonical implementation lives in ``deepiri_helox_sdk.evaluation`` so the
same harness (and its scoring logic in ``metrics``) is shared between in-repo
code and cross-repo consumers such as diri-persola through the installable SDK.
"""

from deepiri_helox_sdk.evaluation.automatic_evaluation_harness import (  # noqa: F401
    AutomaticEvaluationHarness,
)
from deepiri_helox_sdk.evaluation.metrics import (  # noqa: F401
    classification_metrics,
    rouge_l_recall,
    score_response,
    token_f1_score,
    word_overlap_score,
)

__all__ = [
    "AutomaticEvaluationHarness",
    "classification_metrics",
    "rouge_l_recall",
    "score_response",
    "token_f1_score",
    "word_overlap_score",
]
