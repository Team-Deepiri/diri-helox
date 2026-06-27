from __future__ import annotations

from pathlib import Path

import pytest

from deepiri_helox_sdk.evaluation.samples import EvaluationSample, load_jsonl_suite


FIXTURES = Path(__file__).parent / "fixtures"


def test_from_dict_classifier_fields():
    sample = EvaluationSample.from_dict(
        {"text": "hello", "label": 2, "label_name": "writing_code"}
    )
    assert sample.text == "hello"
    assert sample.label == 2
    assert sample.label_name == "writing_code"


def test_from_dict_generation_fields():
    sample = EvaluationSample.from_dict(
        {
            "prompt": "Say hi",
            "expected": "hello",
            "type": "contains",
            "threshold": 0.8,
            "id": "t1",
        }
    )
    assert sample.prompt == "Say hi"
    assert sample.expected == "hello"
    assert sample.test_type == "contains"
    assert sample.threshold == 0.8
    assert sample.test_id == "t1"


def test_load_jsonl_suite():
    samples = load_jsonl_suite(FIXTURES / "intent_holdout.jsonl")
    assert len(samples) == 4
    assert samples[0].label_name == "debugging"


def test_load_jsonl_invalid_line(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_jsonl_suite(bad)
