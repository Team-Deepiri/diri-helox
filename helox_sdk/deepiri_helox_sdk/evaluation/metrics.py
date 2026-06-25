"""Scoring and classification metrics for post-training evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .categories import CATEGORIES


def score_text_response(
    generated: str,
    expected: str,
    test_type: str = "similarity",
) -> float:
    """Score a generated response against an expected reference."""
    generated = generated.strip()
    expected = expected.strip()
    if test_type == "exact_match":
        return 1.0 if generated == expected else 0.0
    if test_type == "contains":
        return 1.0 if expected.lower() in generated.lower() else 0.0
    if test_type == "similarity":
        return word_overlap_score(generated, expected)
    if test_type == "rouge_l":
        return rouge_l_recall(generated, expected)
    return word_overlap_score(generated, expected)


def word_overlap_score(generated: str, expected: str) -> float:
    """Jaccard-like word overlap against expected tokens."""
    gen_words = set(generated.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    return len(gen_words & exp_words) / len(exp_words)


def rouge_l_recall(generated: str, expected: str) -> float:
    """Longest-common-subsequence recall (lightweight ROUGE-L proxy)."""
    gen_tokens = generated.lower().split()
    exp_tokens = expected.lower().split()
    if not exp_tokens:
        return 0.0
    lcs_len = _lcs_length(gen_tokens, exp_tokens)
    return lcs_len / len(exp_tokens)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    rows = len(a) + 1
    cols = len(b) + 1
    table = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        for j in range(1, cols):
            if a[i - 1] == b[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[rows - 1][cols - 1]


def classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_conf: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute sklearn-style classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    if not y_true:
        return {"overall": {}, "per_class": {}, "confusion_matrix": []}

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    prec_cls, rec_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    conf_arr = np.array(y_conf or [0.0] * len(y_true), dtype=float)
    correct_mask = np.array(y_true) == np.array(y_pred)
    avg_conf_correct = float(np.mean(conf_arr[correct_mask])) if correct_mask.any() else 0.0
    avg_conf_incorrect = float(np.mean(conf_arr[~correct_mask])) if (~correct_mask).any() else 0.0

    per_class = {
        CATEGORIES.get(i, f"category_{i}"): {
            "precision": float(prec_cls[i]),
            "recall": float(rec_cls[i]),
            "f1": float(f1_cls[i]),
            "support": int(sup_cls[i]),
        }
        for i in range(len(prec_cls))
    }

    return {
        "_num_examples": len(y_true),
        "overall": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "avg_confidence": float(np.mean(conf_arr)) if len(conf_arr) else 0.0,
            "avg_confidence_correct": avg_conf_correct,
            "avg_confidence_incorrect": avg_conf_incorrect,
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def aggregate_generation_scores(scores: List[float], passed_flags: List[bool]) -> Dict[str, Any]:
    """Aggregate per-sample generation scores."""
    total = len(scores)
    passed = sum(1 for flag in passed_flags if flag)
    avg_score = sum(scores) / total if total else 0.0
    return {
        "total_tests": total,
        "passed_tests": passed,
        "pass_rate": passed / total if total else 0.0,
        "avg_score": avg_score,
    }


def pick_number(*values: Any) -> Optional[float]:
    """Return the first value parseable as float."""
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def efficiency_score(
    quality: Optional[float],
    throughput_per_sec: Optional[float],
    avg_latency_ms: Optional[float],
) -> Optional[float]:
    """Quality-adjusted efficiency metric used in model comparison."""
    if quality is not None and throughput_per_sec is not None and throughput_per_sec > 0:
        return quality * throughput_per_sec
    if quality is not None and avg_latency_ms is not None and avg_latency_ms > 0:
        return quality / avg_latency_ms
    return None
