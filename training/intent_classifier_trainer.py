"""
IntentClassifierTrainer: reusable class for fine-tuning BERT/DeBERTa models
for 31-category task classification.

Extracts and wraps the logic from scripts/training/train_intent_classifier.py.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_HELOX_ROOT = Path(__file__).parent.parent
if str(_HELOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_HELOX_ROOT))

# Must happen before importing torch/transformers when falling back to CPU
_force_cpu = os.environ.get("FORCE_CPU", "").lower() == "true"


def _disable_deepspeed_features() -> None:
    """Prevent DeepSpeed from being imported (avoids CUDA compilation errors on CPU)."""
    os.environ["ACCELERATE_USE_CPU"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["DS_SKIP_CUDA_CHECK"] = "1"
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    os.environ.setdefault("CUDA_HOME", "")

    original_find_spec = importlib.util.find_spec

    def patched_find_spec(name, package=None):
        if name == "deepspeed" or (isinstance(name, str) and name.startswith("deepspeed.")):
            return None
        return original_find_spec(name, package)

    if importlib.util.find_spec is not patched_find_spec:
        importlib.util.find_spec = patched_find_spec


import torch  # noqa: E402 (must be after env setup above)

if torch.cuda.is_available() and not _force_cpu:
    try:
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 7:
            _force_cpu = True
            _disable_deepspeed_features()
    except Exception:
        _force_cpu = True
        _disable_deepspeed_features()
else:
    _disable_deepspeed_features()

from datasets import Dataset  # noqa: E402
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from data_sources.base import DataSample  # noqa: E402

# 31-category mapping shared across train / evaluate
CATEGORY_MAP: Dict[int, str] = {
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


def _prepare_device(force_cpu: bool = False) -> Dict[str, Any]:
    info: Dict[str, Any] = {"device": "cpu", "use_gpu": False, "gpu_name": None, "reason": ""}
    if force_cpu or not torch.cuda.is_available():
        info["reason"] = "forced_cpu" if force_cpu else "cuda_unavailable"
        return info
    try:
        gpu_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 7:
            info["reason"] = "capability_too_low"
            return info
        test = torch.tensor([1.0]).cuda()
        _ = test * 2
        del test
        torch.cuda.empty_cache()
        info.update(
            {"device": "cuda", "use_gpu": True, "gpu_name": gpu_name, "reason": "gpu_compatible"}
        )
    except Exception as e:
        info["reason"] = f"gpu_error:{e}"
    if not info["use_gpu"]:
        _disable_deepspeed_features()
    return info


class _DeviceAwareTrainer(Trainer):
    """HuggingFace Trainer that respects detected device config."""

    def __init__(
        self, *args, device_info: Optional[Dict] = None, force_cpu: bool = False, **kwargs
    ):
        self.device_info = device_info or _prepare_device(force_cpu=force_cpu)
        training_args = kwargs.get("args")
        if training_args is not None:
            training_args.no_cuda = not self.device_info["use_gpu"]
            if hasattr(training_args, "use_cpu"):
                training_args.use_cpu = not self.device_info["use_gpu"]
        super().__init__(*args, **kwargs)


def _compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


class IntentClassifierTrainer:
    """
    Reusable fine-tuning wrapper for BERT/DeBERTa intent classification.

    Usage:
        trainer = IntentClassifierTrainer(model_name="bert-base-uncased")
        metrics = trainer.train(train_samples, val_samples)
        trainer.save()   # saves to output_dir
        print(trainer.get_model_path())
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 31,
        output_dir: str = "models/intent_classifier",
        num_epochs: int = 2,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        max_length: int = 128,
        **kwargs,  # absorbs extra config keys like "trainer_type"
    ) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length

        self._tokenizer = None
        self._model = None
        self._trainer: Optional[_DeviceAwareTrainer] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(
        self,
        train_samples: List[DataSample],
        val_samples: List[DataSample],
    ) -> Dict[str, Any]:
        """
        Fine-tune the model on train_samples, evaluate on val_samples.
        Returns eval metrics dict.
        """
        print("=" * 60)
        print(f"IntentClassifierTrainer — {self.model_name}")
        print(f"  Labels : {self.num_labels}")
        print(f"  Epochs : {self.num_epochs} | Batch: {self.batch_size} | LR: {self.learning_rate}")
        print("=" * 60)

        tokenizer = self._load_tokenizer()
        model = self._load_model()

        train_hf = self._to_hf_dataset(train_samples, tokenizer)
        val_hf = self._to_hf_dataset(val_samples, tokenizer)

        device_info = _prepare_device(force_cpu=_force_cpu)
        use_gpu = device_info["use_gpu"]
        print(f"  Device : {'CUDA – ' + str(device_info.get('gpu_name')) if use_gpu else 'CPU'}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        eval_steps = max(1, min(100, len(train_samples) // max(self.batch_size, 1)))

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            push_to_hub=False,
            report_to="none",
            no_cuda=not use_gpu,
        )
        if hasattr(training_args, "use_cpu"):
            training_args.use_cpu = not use_gpu

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self._trainer = _DeviceAwareTrainer(
            device_info=device_info,
            model=model,
            args=training_args,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
        )

        print("\nStarting training...")
        self._trainer.train()
        eval_results = self._trainer.evaluate()
        print(f"\nValidation accuracy : {eval_results.get('eval_accuracy', 0):.4f}")
        print(f"Validation F1       : {eval_results.get('eval_f1', 0):.4f}")
        return eval_results

    def save(self, path: Optional[str] = None) -> None:
        """Save model, tokenizer, category_map.json, training_info.json."""
        if self._trainer is None:
            raise RuntimeError("Call train() before save()")
        out = Path(path) if path else self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        self._trainer.save_model(str(out))
        self._tokenizer.save_pretrained(str(out))

        id2label = {i: CATEGORY_MAP.get(i, f"category_{i}") for i in range(self.num_labels)}
        label2id = {v: k for k, v in id2label.items()}
        with open(out / "category_map.json", "w") as f:
            json.dump({"id2label": id2label, "label2id": label2id}, f, indent=2)

        eval_res = self._trainer.evaluate() if self._trainer else {}
        with open(out / "training_info.json", "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "num_labels": self.num_labels,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "eval_accuracy": float(eval_res.get("eval_accuracy", 0)),
                    "eval_f1": float(eval_res.get("eval_f1", 0)),
                },
                f,
                indent=2,
            )

        print(f"Model saved to: {out}")

    def get_model_path(self) -> str:
        return str(self.output_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_tokenizer(self):
        if self._tokenizer is None:
            print(f"Loading tokenizer: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def _load_model(self):
        if self._model is None:
            print(f"Loading model: {self.model_name}")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )
        return self._model

    def _to_hf_dataset(self, samples: List[DataSample], tokenizer) -> Dataset:
        """Convert DataSample list to a tokenized HuggingFace Dataset."""
        texts = [s.text for s in samples]
        labels = [s.label if s.label is not None else 0 for s in samples]

        raw = Dataset.from_dict({"text": texts, "label": labels})

        def tokenize_fn(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])
        return tokenized
