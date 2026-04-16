"""
SelfFeedbackDataSource: uses a trained model's own high-confidence predictions
as training data. This enables the "pointed at its own output" requirement.

It reads an inference log (JSONL with text + predicted_label + confidence)
and returns only samples above the confidence threshold.

If inference_log_path is not supplied, the source can optionally look up the
latest Production model path from MLflow via deepiri-training-orchestrator's
ModelRegistry and use its co-located inference log.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .base import DataSample, DataSource, DataSourceConfig

logger = logging.getLogger(__name__)


class SelfFeedbackDataSource(DataSource):
    """
    Loads high-confidence model predictions as training samples.

    Config params:
        inference_log_path   (str): path to JSONL inference log. If omitted, the source
                                    queries MLflow ModelRegistry for the latest Production
                                    model and looks for an inference_log.jsonl beside it.
        model_name           (str): MLflow registered model name (default: "intent-classifier")
        mlflow_tracking_uri  (str): MLflow tracking URI (default: "http://localhost:5000")
        confidence_threshold (float): minimum confidence to accept (default: 0.90)
        text_field           (str): field for input text (default: "text")
        label_field          (str): field for predicted label id (default: "predicted_label")
        label_name_field     (str): field for label name (default: "predicted_label_name")
        confidence_field     (str): field for confidence score (default: "confidence")
        max_samples          (int | None): cap on samples loaded
    """

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        self._log_path: Optional[Path] = (
            Path(config.params["inference_log_path"])
            if config.params.get("inference_log_path")
            else None
        )
        self._model_name = config.params.get("model_name", "intent-classifier")
        self._mlflow_uri = config.params.get("mlflow_tracking_uri", "http://localhost:5000")
        self._threshold = config.params.get("confidence_threshold", 0.90)
        self._text_field = config.params.get("text_field", "text")
        self._label_field = config.params.get("label_field", "predicted_label")
        self._label_name_field = config.params.get("label_name_field", "predicted_label_name")
        self._confidence_field = config.params.get("confidence_field", "confidence")
        self._max_samples = config.params.get("max_samples", None)

    def _resolve_log_path(self) -> Optional[Path]:
        """
        Return the inference log path, falling back to MLflow ModelRegistry
        if no explicit path was configured.
        """
        if self._log_path:
            return self._log_path

        # Dynamic discovery: ask ModelRegistry for the latest Production model
        try:
            from deepiri_training_orchestrator import ModelRegistry

            registry = ModelRegistry(tracking_uri=self._mlflow_uri)
            model_dir = registry.get_latest_model(self._model_name, stage="Production")
            if model_dir:
                candidate = Path(model_dir) / "inference_log.jsonl"
                if candidate.exists():
                    return candidate
        except Exception as exc:
            # Best-effort lookup: if registry access fails, fall back to configured path.
            logger.debug(
                "SelfFeedbackDataSource: failed to resolve log path from model registry",
                exc_info=exc,
            )

        return None

    def _iter_log(self) -> Iterator[DataSample]:
        log_path = self._resolve_log_path()
        if not log_path or not log_path.exists():
            return
        with open(log_path, "r", encoding="utf-8") as f:
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
        resolved = self._resolve_log_path()
        return {
            "source_type": "self_feedback",
            "name": self.name,
            "inference_log_path": str(resolved) if resolved else None,
            "log_exists": resolved.exists() if resolved else False,
            "confidence_threshold": self._threshold,
            "model_name": self._model_name,
        }
