"""
IntentClassifierTrainer: reusable class for fine-tuning BERT/DeBERTa models
for 31-category task classification.

Extracts and wraps the logic from scripts/training/train_intent_classifier.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from core.gpu_utils import detect_device
from data_sources.base import DataSample

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


def _disable_deepspeed_features() -> None:
    """
    Prevent DeepSpeed from being imported.

    DeepSpeed requires CUDA compilation headers that are not present on CPU-only
    or low-capability GPU machines.  Setting these env vars tells Accelerate and
    DeepSpeed to skip CUDA compilation checks entirely.
    """
    import importlib.util

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


def _check_gpu_capability() -> bool:
    """
    Check whether the available GPU meets the minimum compute capability.

    BERT training requires CUDA compute capability ≥ 7.0 (Volta+) to run
    efficiently with mixed precision.  Older GPUs (e.g. Kepler/Maxwell) fall
    back to CPU to avoid silent numerical errors.

    Returns True if a usable GPU is available, False otherwise.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 7:
            return False
        # Sanity-check: run a small tensor op to confirm the GPU works
        test = torch.tensor([1.0]).cuda()
        _ = test * 2
        del test
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False


def _compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


class _OrchestratorCallbackAdapter(TrainerCallback):
    """
    Bridges HuggingFace Trainer events to deepiri-training-orchestrator callbacks.

    HF fires on_log every ``logging_steps`` and on_evaluate after each eval pass.
    We translate those events into on_step_end / on_eval_end so that
    CheckpointCallback, LoggingCallback, and EarlyStoppingCallback actually fire.
    When EarlyStoppingCallback sets ctx.extra["stop_training"], we propagate that
    back to HF via TrainerControl.should_training_stop.
    """

    def __init__(self, orchestrator_callbacks: list, pipeline: Any = None) -> None:
        from deepiri_training_orchestrator import TrainingContext

        self._cbs = orchestrator_callbacks
        self._pipeline = pipeline
        self._ctx = TrainingContext()

    def on_train_begin(
        self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self._ctx.max_steps = state.max_steps
        for cb in self._cbs:
            cb.on_train_begin(self._pipeline, self._ctx)

    def on_log(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return
        self._ctx.step = state.global_step
        self._ctx.epoch = int(state.epoch or 0)
        metrics = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
        for cb in self._cbs:
            cb.on_step_end(self._pipeline, self._ctx, metrics)

    def on_evaluate(
        self,
        args: Any,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TrainerControl:
        if not metrics:
            return control
        self._ctx.step = state.global_step
        eval_metrics = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        for cb in self._cbs:
            cb.on_eval_end(self._pipeline, self._ctx, eval_metrics)
        if self._ctx.extra.get("stop_training"):
            control.should_training_stop = True
        return control

    def on_train_end(
        self, args: Any, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self._ctx.step = state.global_step
        for cb in self._cbs:
            cb.on_train_end(self._pipeline, self._ctx)


class _DeviceAwareTrainer(Trainer):
    """HuggingFace Trainer that applies the correct device configuration."""

    def __init__(self, *args, use_gpu: bool = False, **kwargs):
        training_args = kwargs.get("args")
        if training_args is not None:
            training_args.use_cpu = not use_gpu
        super().__init__(*args, **kwargs)


class IntentClassifierTrainer:
    """
    Reusable fine-tuning wrapper for BERT/DeBERTa intent classification.

    Device selection is determined at train() time using core.gpu_utils.detect_device(),
    which delegates to deepiri-gpu-utils for consistent behaviour across all services.
    Pass force_device="cpu" to override (e.g. for CI/CD or low-memory environments).

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
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_length: int = 128,
        force_device: Optional[str] = None,
        orchestrator_callbacks: Optional[List[Any]] = None,
        **kwargs,  # absorbs extra config keys like "trainer_type"
    ) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.force_device = force_device
        self.orchestrator_callbacks = orchestrator_callbacks or []

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

        # Determine device at train time (not import time)
        device = detect_device(force=self.force_device)
        use_gpu = device.type in ("cuda", "mps")

        # Check GPU compute capability for CUDA; disable DeepSpeed if using CPU
        if device.type == "cuda" and not _check_gpu_capability():
            print("  Warning: GPU compute capability < 7.0 — falling back to CPU")
            device = detect_device(force="cpu")
            use_gpu = False

        if not use_gpu:
            _disable_deepspeed_features()

        print(f"  Device : {device}")

        tokenizer = self._load_tokenizer()
        model = self._load_model()

        train_hf = self._to_hf_dataset(train_samples, tokenizer)
        val_hf = self._to_hf_dataset(val_samples, tokenizer)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        eval_steps = max(1, min(100, len(train_samples) // max(self.batch_size, 1)))

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=0.01,
            # Warmup over 6% of total steps — standard for BERT fine-tuning per
            # the original paper (Devlin et al. 2019). Prevents large gradient
            # updates from destabilising the pre-trained weights at step 0.
            warmup_ratio=0.06,
            lr_scheduler_type="linear",
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
            use_cpu=not use_gpu,
        )

        hf_callbacks: List[TrainerCallback] = []
        if self.orchestrator_callbacks:
            hf_callbacks.append(
                _OrchestratorCallbackAdapter(self.orchestrator_callbacks, pipeline=self)
            )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self._trainer = _DeviceAwareTrainer(
            use_gpu=use_gpu,
            model=model,
            args=training_args,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
            callbacks=hf_callbacks or None,
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
        label2id = {name: idx for idx, name in CATEGORY_MAP.items()}
        texts: List[str] = []
        labels: List[int] = []
        dropped = 0
        for sample in samples:
            label: Optional[int] = sample.label
            if label is None and sample.label_name:
                label = label2id.get(sample.label_name)
            if label is None or label < 0 or label >= self.num_labels:
                dropped += 1
                continue
            texts.append(sample.text)
            labels.append(int(label))

        if dropped:
            print(
                f"  Warning: dropped {dropped} unlabeled/out-of-range samples "
                "during dataset conversion"
            )
        if not texts:
            raise ValueError("No labeled samples available for intent classifier training")

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
