"""Evaluation report persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from .schemas import EvalRunResult


def save_eval_report(
    report: Union[Dict[str, Any], EvalRunResult],
    output_path: Path,
) -> Path:
    """Write an evaluation report as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict() if isinstance(report, EvalRunResult) else report
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def load_eval_report(path: Path) -> Dict[str, Any]:
    """Load an evaluation report from disk."""
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload
