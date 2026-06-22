from __future__ import annotations

from deepiri_helox_sdk.evaluation.metrics import (
    efficiency_score,
    pick_number,
    rouge_l_recall,
    score_text_response,
    word_overlap_score,
)


def test_exact_match():
    assert score_text_response("ok", "ok", "exact_match") == 1.0
    assert score_text_response("no", "ok", "exact_match") == 0.0


def test_contains():
    assert score_text_response("status is ok today", "ok", "contains") == 1.0


def test_word_overlap():
    assert word_overlap_score("a b c", "b c d") == 2 / 3


def test_rouge_l_recall():
    score = rouge_l_recall("the cat sat", "the cat")
    assert score == 1.0


def test_pick_number():
    assert pick_number(None, "0.5", 2) == 0.5


def test_efficiency_score_throughput():
    assert efficiency_score(0.9, 100.0, None) == 90.0


def test_efficiency_score_latency():
    assert efficiency_score(0.8, None, 4.0) == 0.2
