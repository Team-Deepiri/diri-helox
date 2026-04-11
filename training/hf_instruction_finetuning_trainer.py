"""
HFInstructionFinetuningTrainer: HuggingFace-based causal LM fine-tuning with
response-only loss masking — for Persola personality model training.

Persola data flow:
  PersonaEngine builds a system prompt from personality knobs (creativity, humor,
  formality, empathy, …) → formats samples as:
      instruction: <system_prompt + user_message>
      response:    <persona-consistent reply>

  This trainer computes loss ONLY on response tokens by setting instruction
  token labels to -100 (HuggingFace ignore index), so the model learns to
  generate personality-consistent replies without memorising the persona prompt.

DataSample format (two supported layouts):
  1. Metadata fields  — metadata["instruction"] + metadata["response"]
  2. Full-text        — text is used as the full sequence (no masking applied)

Usage in DynamicTrainingPipeline config:
  {
    "trainer_type": "instruction_finetuning",
    "model_name": "meta-llama/Llama-3.2-1B",   # or any HF causal LM
    "output_dir": "models/persola_finetuned",
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-5,
    "max_length": 512
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from core.gpu_utils import detect_device
from data_sources.base import DataSample

# Label value that HuggingFace Trainer ignores when computing loss
_IGNORE_INDEX = -100

# Delimiter tokens wrapping the response portion (Llama/Mistral chat style)
_INST_END = "[/INST]"


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # Flatten and filter ignored positions
    mask = labels.reshape(-1) != _IGNORE_INDEX
    preds_flat = predictions.reshape(-1)[mask]
    labels_flat = labels.reshape(-1)[mask]
    accuracy = accuracy_score(labels_flat, preds_flat)
    return {"token_accuracy": accuracy}


class _OrchestratorCallbackAdapter(TrainerCallback):
    """Bridges HF Trainer events to deepiri-training-orchestrator callbacks."""

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
            k.removeprefix("eval_") if k.startswith("eval_") else k: float(v)
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


class HFInstructionFinetuningTrainer:
    """
    HuggingFace-based causal LM trainer with response-only loss masking.

    Designed for Persola personality fine-tuning: the model learns to generate
    persona-consistent responses without computing loss on the instruction/system
    prompt tokens.

    Compatible with any HuggingFace AutoModelForCausalLM checkpoint (Llama,
    Mistral, Phi, etc.).

    Args:
        model_name:             HuggingFace model ID or local path.
        output_dir:             Where to save the fine-tuned model.
        num_epochs:             Training epochs (default: 3).
        batch_size:             Per-device batch size (default: 4).
        learning_rate:          AdamW learning rate (default: 2e-5).
        max_length:             Max tokenised sequence length (default: 512).
        force_device:           Override device — "cpu", "cuda", or "mps".
        orchestrator_callbacks: deepiri-training-orchestrator callbacks list.
    """

    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        output_dir: str = "models/persola_finetuned",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        max_length: int = 512,
        force_device: Optional[str] = None,
        orchestrator_callbacks: Optional[List[Any]] = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.force_device = force_device
        self.orchestrator_callbacks = orchestrator_callbacks or []

        self._tokenizer = None
        self._model = None
        self._trainer: Optional[Trainer] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(
        self,
        train_samples: List[DataSample],
        val_samples: List[DataSample],
    ) -> Dict[str, Any]:
        """Fine-tune on train_samples, evaluate on val_samples."""
        print("=" * 60)
        print(f"HFInstructionFinetuningTrainer — {self.model_name}")
        print(f"  Epochs: {self.num_epochs} | Batch: {self.batch_size} | LR: {self.learning_rate}")
        print("=" * 60)

        device = detect_device(force=self.force_device)
        use_gpu = device.type in ("cuda", "mps")
        print(f"  Device: {device}")

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
            # Warmup over 3% of total steps with cosine decay — standard for
            # causal LM instruction fine-tuning. Warmup prevents large updates
            # from destabilising pre-trained weights at the start of training.
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,
            report_to="none",
            use_cpu=not use_gpu,
        )

        hf_callbacks = []
        if self.orchestrator_callbacks:
            hf_callbacks.append(
                _OrchestratorCallbackAdapter(self.orchestrator_callbacks, pipeline=self)
            )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=_IGNORE_INDEX, padding=True
        )

        self._trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            data_collator=data_collator,
            compute_metrics=_compute_metrics,
            callbacks=hf_callbacks or None,
        )

        print("\nStarting instruction fine-tuning...")
        self._trainer.train()
        eval_results = self._trainer.evaluate()
        print(f"\nValidation token accuracy: {eval_results.get('eval_token_accuracy', 0):.4f}")
        return eval_results

    def save(self, path: Optional[str] = None) -> None:
        """Save model, tokenizer, and training_info.json."""
        if self._trainer is None:
            raise RuntimeError("Call train() before save()")
        out = Path(path) if path else self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        self._trainer.save_model(str(out))
        self._tokenizer.save_pretrained(str(out))

        eval_res = self._trainer.evaluate() if self._trainer else {}
        with open(out / "training_info.json", "w") as f:
            json.dump(
                {
                    "trainer_type": "instruction_finetuning",
                    "model_name": self.model_name,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "token_accuracy": float(eval_res.get("eval_token_accuracy", 0)),
                    "use_case": "persola_personality_finetuning",
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
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def _load_model(self):
        if self._model is None:
            print(f"Loading model: {self.model_name}")
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        return self._model

    def _build_sequence(self, sample: DataSample) -> tuple[str, str]:
        """
        Extract (instruction, response) from a DataSample.

        Priority:
          1. metadata["instruction"] + metadata["response"]
          2. text split on [/INST] delimiter (Llama/Mistral chat format)
          3. text as full response (no instruction masking)
        """
        instruction = sample.metadata.get("instruction", "")
        response = sample.metadata.get("response", "")

        if not instruction and not response:
            if _INST_END in sample.text:
                parts = sample.text.split(_INST_END, 1)
                instruction = parts[0] + _INST_END
                response = parts[1].strip()
            else:
                # No instruction context — treat entire text as response
                instruction = ""
                response = sample.text

        return instruction, response

    def _tokenize_with_mask(self, sample: DataSample, tokenizer) -> Dict[str, Any]:
        """
        Tokenise (instruction + response) and mask instruction labels to _IGNORE_INDEX.

        Returns input_ids, attention_mask, and labels with instruction positions set
        to -100 so cross-entropy loss is only computed on response tokens.

        Instruction length is measured by tokenising the instruction *within* the
        full sequence (not separately) to avoid BOS-token boundary misalignment.
        Tokenisers like Llama/Mistral add a BOS token only at the very start of the
        sequence; tokenising the instruction alone would yield a different token ID
        list (with a leading BOS) than what appears in the full tokenised sequence,
        causing the mask to cover the wrong tokens.
        """
        instruction, response = self._build_sequence(sample)
        full_text = instruction + response

        full_enc = tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        # Build labels: mask instruction prefix
        labels = list(input_ids)
        if instruction:
            # Tokenise just the response to find where it starts in the full sequence.
            # Using add_special_tokens=False ensures no BOS is prepended, so the
            # response tokens match exactly what appears at the end of the full encoding.
            resp_enc = tokenizer(
                response,
                truncation=True,
                max_length=self.max_length,
                return_tensors=None,
                add_special_tokens=False,
            )
            resp_len = len(resp_enc["input_ids"])
            # Mask everything before the response tokens
            inst_len = len(input_ids) - resp_len
            for i in range(max(0, inst_len)):
                labels[i] = _IGNORE_INDEX

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _to_hf_dataset(self, samples: List[DataSample], tokenizer) -> Dataset:
        """Convert DataSample list to a tokenised HuggingFace Dataset."""
        rows = [self._tokenize_with_mask(s, tokenizer) for s in samples]
        return Dataset.from_list(rows)
