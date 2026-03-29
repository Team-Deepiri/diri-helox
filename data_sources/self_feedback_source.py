"""
SelfFeedbackDataSource: uses a trained model's own high-confidence predictions
as training data. This enables the "pointed at its own output" requirement.

It reads an inference log (JSONL with text + predicted_label + confidence)
and returns only samples above the confidence threshold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from .base import DataSample, DataSource, DataSourceConfig


class SelfFeedbackDataSource(DataSource):
    """
    Loads high-confidence model predictions as training samples.

    Config params:
        inference_log_path   (str): path to JSONL inference log
        confidence_threshold (float): minimum confidence to accept (default: 0.90)
        text_field           (str): field for input text (default: "text")
        label_field          (str): field for predicted label id (default: "predicted_label")
        label_name_field     (str): field for label name (default: "predicted_label_name")
        confidence_field     (str): field for confidence score (default: "confidence")
        max_samples          (int | None): cap on samples loaded
    """

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        self._log_path = Path(config.params.get("inference_log_path", ""))
        self._threshold = config.params.get("confidence_threshold", 0.90)
        self._text_field = config.params.get("text_field", "text")
        self._label_field = config.params.get("label_field", "predicted_label")
        self._label_name_field = config.params.get("label_name_field", "predicted_label_name")
        self._confidence_field = config.params.get("confidence_field", "confidence")
        self._max_samples = config.params.get("max_samples", None)

    def _iter_log(self) -> Iterator[DataSample]:
        if not self._log_path or not self._log_path.exists():
            return
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                confidence = item.get(self._confidence_field, 0.0)
                if confidence < self._threshold:
                    continue
                text = item.get(self._text_field, "")
                if not text:
                    continue
                label = item.get(self._label_field)
                label_name = item.get(self._label_name_field)
                yield DataSample(
                    text=text,
                    label=label if isinstance(label, int) else None,
                    label_name=label_name,
                    metadata={
                        "confidence": confidence,
                        "self_feedback": True,
                    },
                    source=f"self_feedback:{self.name}",
                )

    def load(self) -> List[DataSample]:
        samples = []
        for sample in self._iter_log():
            samples.append(sample)
            if self._max_samples and len(samples) >= self._max_samples:
                break
        return samples

    def stream(self) -> Iterator[DataSample]:
        count = 0
        for sample in self._iter_log():
            yield sample
            count += 1
            if self._max_samples and count >= self._max_samples:
                return

    def get_info(self) -> Dict[str, Any]:
        return {
            "source_type": "self_feedback",
            "name": self.name,
            "inference_log_path": str(self._log_path),
            "log_exists": self._log_path.exists() if self._log_path else False,
            "confidence_threshold": self._threshold,
        }
