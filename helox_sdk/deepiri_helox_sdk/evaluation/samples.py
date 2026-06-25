"""Evaluation sample types and JSONL loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass(slots=True)
class EvaluationSample:
    """Universal sample for classifier or generation evaluation."""

    text: str = ""
    label: Optional[int] = None
    label_name: Optional[str] = None
    prompt: Optional[str] = None
    expected: Optional[str] = None
    test_type: str = "similarity"
    threshold: float = 0.5
    test_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvaluationSample":
        text = str(raw.get("text") or raw.get("input") or "")
        prompt = raw.get("prompt")
        if prompt is not None:
            prompt = str(prompt)
        expected = raw.get("expected") or raw.get("output")
        if expected is not None:
            expected = str(expected)
        label = raw.get("label")
        if label is not None:
            label = int(label)
        label_name = raw.get("label_name") or raw.get("category")
        if label_name is not None:
            label_name = str(label_name)
        return cls(
            text=text,
            label=label,
            label_name=label_name,
            prompt=str(prompt) if prompt is not None else None,
            expected=expected,
            test_type=str(raw.get("type", "similarity")),
            threshold=float(raw.get("threshold", 0.5)),
            test_id=str(raw.get("id", "")),
            metadata=dict(raw.get("metadata") or {}),
        )


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file."""
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_no} of {path}")
            yield payload


def load_jsonl_suite(path: Path) -> List[EvaluationSample]:
    """Load an evaluation suite from JSONL."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation suite not found: {path}")
    return [EvaluationSample.from_dict(row) for row in iter_jsonl(path)]
