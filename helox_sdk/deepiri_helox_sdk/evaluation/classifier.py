"""Classification model evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .categories import LABEL_TO_ID
from .metrics import classification_metrics
from .samples import EvaluationSample


class ClassifierEvaluator:
    """Evaluate a HuggingFace sequence-classification checkpoint."""

    def __init__(
        self,
        model_path: str | Path,
        num_labels: int = 31,
        batch_size: int = 32,
        max_length: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.max_length = max_length
        self.device_override = device
        self._model = None
        self._tokenizer = None
        self._device = None

    def evaluate(self, samples: List[EvaluationSample]) -> Dict[str, Any]:
        """Run full classification evaluation."""
        labeled, skipped = self._resolve_labels(samples)
        if not labeled:
            return {
                "overall": {},
                "per_class": {},
                "confusion_matrix": [],
                "_skipped_unlabeled": skipped,
            }

        model, tokenizer, device = self._load_model()
        texts = [sample.text for sample in labeled]
        true_labels = [int(sample.label) for sample in labeled if sample.label is not None]
        predictions, confidences = self._predict_batch(model, tokenizer, device, texts)
        metrics = classification_metrics(true_labels, predictions, confidences)
        if skipped:
            metrics["_skipped_unlabeled"] = skipped
        return metrics

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Return label predictions and confidences for raw texts."""
        from .categories import CATEGORIES

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

    def _resolve_labels(
        self, samples: List[EvaluationSample]
    ) -> Tuple[List[EvaluationSample], int]:
        labeled: List[EvaluationSample] = []
        skipped = 0
        for sample in samples:
            if sample.label is not None:
                labeled.append(sample)
                continue
            if sample.label_name and sample.label_name in LABEL_TO_ID:
                labeled.append(
                    EvaluationSample(
                        text=sample.text,
                        label=LABEL_TO_ID[sample.label_name],
                        label_name=sample.label_name,
                        metadata=sample.metadata,
                    )
                )
                continue
            skipped += 1
        return labeled, skipped

    def _load_model(self):
        if self._model is not None:
            return self._model, self._tokenizer, self._device

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self._model.eval()
        if self.device_override:
            self._device = torch.device(self.device_override)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        return self._model, self._tokenizer, self._device

    def _predict_batch(self, model, tokenizer, device, texts: List[str]):
        import torch

        predictions: List[int] = []
        confidences: List[float] = []
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
