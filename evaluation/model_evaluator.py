"""
ModelEvaluator: reusable evaluation class for trained classification models.
Extracts and wraps the logic from scripts/evaluation/evaluate_trained_model.py.
Uses the full 31-category mapping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_HELOX_ROOT = Path(__file__).parent.parent
if str(_HELOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_HELOX_ROOT))

try:
    import torch
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _IMPORTS_OK = True
except ImportError as exc:
    _IMPORTS_OK = False
    _IMPORT_ERR = exc

from data_sources.base import DataSample  # noqa: E402

# Full 31-category mapping (must match LABEL_MAPPING in generate_synthetic_data.py)
CATEGORIES: Dict[int, str] = {
    0: "debugging",
    1: "refactoring",
    2: "writing_code",
    3: "programming",
    4: "running_code",
    5: "inspecting",
    6: "writing",
    7: "learning_research",
    8: "learning_study",
    9: "learning_training",
    10: "learning_practice",
    11: "creative",
    12: "administrative",
    13: "team_organization",
    14: "team_collaboration",
    15: "team_planning",
    16: "research",
    17: "planning",
    18: "communication",
    19: "big_data_analytics",
    20: "data_processing",
    21: "design",
    22: "qa",
    23: "testing",
    24: "validation",
    25: "reporting",
    26: "documentation",
    27: "system_admin",
    28: "ux_ui",
    29: "security",
    30: "data_privacy",
}
LABEL_TO_ID: Dict[str, int] = {name: idx for idx, name in CATEGORIES.items()}


class ModelEvaluator:
    """
    Runs evaluation on a trained classification model.

    Usage:
        evaluator = ModelEvaluator(model_path="models/intent_classifier")
        metrics = evaluator.evaluate(test_samples)
        evaluator.save_report(metrics, "models/intent_classifier/evaluation_report.json")
    """

    def __init__(
        self,
        model_path: str,
        num_labels: int = 31,
        batch_size: int = 32,
        max_length: int = 128,
    ) -> None:
        if not _IMPORTS_OK:
            raise ImportError(
                f"ModelEvaluator requires torch, transformers, sklearn: {_IMPORT_ERR}"
            )
        self.model_path = Path(model_path)
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, test_samples: List[DataSample]) -> Dict[str, Any]:
        """Run full evaluation. Returns metrics dict."""
        if not test_samples:
            return {"overall": {}, "per_class": {}, "confusion_matrix": []}

        labeled_samples: List[DataSample] = []
        skipped_unlabeled = 0
        for sample in test_samples:
            if sample.label is not None:
                labeled_samples.append(sample)
                continue
            if sample.label_name and sample.label_name in LABEL_TO_ID:
                labeled_samples.append(
                    DataSample(
                        text=sample.text,
                        label=LABEL_TO_ID[sample.label_name],
                        label_name=sample.label_name,
                        metadata=sample.metadata,
                        source=sample.source,
                    )
                )
                continue
            skipped_unlabeled += 1

        if not labeled_samples:
            print("Warning: all evaluation samples are unlabeled; returning empty metrics")
            return {"overall": {}, "per_class": {}, "confusion_matrix": []}

        model, tokenizer, device = self._load_model()
        texts = [s.text for s in labeled_samples]
        true_labels = [int(s.label) for s in labeled_samples if s.label is not None]

        predictions, confidences = self._predict_batch(model, tokenizer, device, texts)
        metrics = self._calculate_metrics(true_labels, predictions, confidences)
        if skipped_unlabeled:
            metrics["_skipped_unlabeled"] = skipped_unlabeled
        return metrics

    def save_report(self, metrics: Dict[str, Any], output_path: str) -> None:
        """Persist evaluation metrics as a JSON report."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model_path": str(self.model_path),
            "num_test_examples": metrics.get("_num_examples", 0),
            "metrics": metrics,
        }
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Evaluation report saved: {out}")

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Return predictions + confidence for a list of texts."""
        model, tokenizer, device = self._load_model()
        predictions, confidences = self._predict_batch(model, tokenizer, device, texts)
        return [
            {
                "text": text,
                "predicted_label": int(pred),
                "predicted_label_name": CATEGORIES.get(int(pred), f"category_{pred}"),
                "confidence": float(conf),
            }
            for text, pred, conf in zip(texts, predictions, confidences)
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            print(f"Loading model from {self.model_path}...")
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self._model.eval()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
        return self._model, self._tokenizer, self._device

    def _predict_batch(self, model, tokenizer, device, texts: List[str]):
        predictions = []
        confidences = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1).cpu().numpy()
                confs = torch.max(probs, dim=-1)[0].cpu().numpy()
            predictions.extend(preds.tolist())
            confidences.extend(confs.tolist())
        return predictions, confidences

    def _calculate_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        y_conf: List[float],
    ) -> Dict[str, Any]:
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        prec_cls, rec_cls, f1_cls, sup_cls = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)

        conf_arr = np.array(y_conf)
        correct_mask = np.array(y_true) == np.array(y_pred)
        avg_conf_correct = float(np.mean(conf_arr[correct_mask])) if correct_mask.any() else 0.0
        avg_conf_incorrect = (
            float(np.mean(conf_arr[~correct_mask])) if (~correct_mask).any() else 0.0
        )

        num_cls = len(prec_cls)
        per_class = {
            CATEGORIES.get(i, f"category_{i}"): {
                "precision": float(prec_cls[i]),
                "recall": float(rec_cls[i]),
                "f1": float(f1_cls[i]),
                "support": int(sup_cls[i]),
            }
            for i in range(num_cls)
        }

        return {
            "_num_examples": len(y_true),
            "overall": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "avg_confidence": float(np.mean(conf_arr)),
                "avg_confidence_correct": avg_conf_correct,
                "avg_confidence_incorrect": avg_conf_incorrect,
            },
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }
