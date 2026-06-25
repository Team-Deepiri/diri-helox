"""Regression tracking across evaluation runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class RegressionTracker:
    """Persist evaluation history and detect metric regressions."""

    def __init__(
        self,
        history_dir: Path,
        regression_threshold: float = 0.05,
    ) -> None:
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.regression_threshold = regression_threshold
        self.history_file = self.history_dir / "evaluation_history.jsonl"

    def record(self, payload: Dict[str, Any]) -> None:
        """Append one evaluation record to the history log."""
        with open(self.history_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def load_history(self, suite_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load prior evaluation records, optionally filtered by suite."""
        if not self.history_file.exists():
            return []
        records: List[Dict[str, Any]] = []
        with open(self.history_file, encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if suite_name and row.get("suite_name") != suite_name:
                    continue
                records.append(row)
        return records

    def check_score_regression(
        self,
        suite_name: str,
        metric_name: str,
        current_score: float,
    ) -> Optional[Dict[str, Any]]:
        """Compare current score against the best prior score for a suite."""
        prior = self.load_history(suite_name=suite_name)
        prior_scores = [
            float(row[metric_name])
            for row in prior
            if metric_name in row and row[metric_name] is not None
        ]
        if not prior_scores:
            return None

        best_previous = max(prior_scores)
        score_drop = best_previous - current_score
        if score_drop <= self.regression_threshold:
            return None

        return {
            "detected": True,
            "metric": metric_name,
            "current_score": current_score,
            "previous_best": best_previous,
            "score_drop": score_drop,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def summary(self) -> Dict[str, Any]:
        """Summarize stored evaluation history by suite."""
        records = self.load_history()
        if not records:
            return {"total_evaluations": 0, "suites": {}}

        suites: Dict[str, Dict[str, Any]] = {}
        for row in records:
            suite = str(row.get("suite_name", "unknown"))
            bucket = suites.setdefault(suite, {"count": 0, "avg_scores": []})
            bucket["count"] += 1
            if row.get("avg_score") is not None:
                bucket["avg_scores"].append(float(row["avg_score"]))

        for suite, bucket in suites.items():
            scores = bucket.pop("avg_scores")
            if scores:
                bucket["mean_score"] = sum(scores) / len(scores)
                bucket["max_score"] = max(scores)
                bucket["min_score"] = min(scores)

        return {"total_evaluations": len(records), "suites": suites}
