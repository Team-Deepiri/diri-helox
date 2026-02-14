"""
Complete Training Pipeline
Production-ready training with LoRA, QLoRA, distributed training, and experiment tracking
"""
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from mlops.infrastructure import LayeredModelAdapter, LayerConfig, LayerType
from datasets import load_dataset, Dataset
from accelerate import Accelerator
try:
    from deepspeed import init_distributed
except ModuleNotFoundError:
    init_distributed = None
import mlflow
import wandb
import json
from pathlib import Path
import os
from typing import Dict, Optional
import argparse
from mlops.infrastructure.lora_training import LoRATrainer, QLoRATrainingPipeline
from mlops.infrastructure.experiment_tracker import ExperimentTracker
import logging

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger("helox.pipeline")


class FullTrainingPipeline:
    """Complete training pipeline with all features."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.accelerator = Accelerator()
        self.tracker = None
        self.model = None
        self.tokenizer = None
        
    def setup_experiment_tracking(self):
        """Setup MLflow and W&B tracking."""
        self.tracker = ExperimentTracker(
            experiment_name=self.config.get("experiment_name", "deepiri_training"),
            tracking_uri=self.config.get("mlflow_uri", "file:./mlruns"),

            use_wandb=self.config.get("use_wandb", False),
            wandb_project=self.config.get("wandb_project", "deepiri")
        )
        self.tracker.start_run()
        #self.tracker.log_git_info()
        
    def load_and_prepare_data(self):
        """Load and prepare training dataset."""
        dataset_path = self.config["train_dataset_path"]
        logger.info(f"Loading dataset: {dataset_path}")
        
        if dataset_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=dataset_path, split='train')
        else:
            dataset = load_dataset(dataset_path, split='train')
        
        def tokenize_function(examples):
            texts = examples.get('text', examples.get('input', []))
            return self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.config.get("max_length", 512),
                return_tensors="pt"
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        if "validation_dataset_path" in self.config:
            val_dataset = load_dataset('json', data_files=self.config["validation_dataset_path"], split='train')
            val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
        else:
            val_tokenized = tokenized.train_test_split(test_size=0.1)['test']
        
        return tokenized, val_tokenized
    
    def setup_model(self):
        """Setup model with quantization and LoRA."""
        model_name = self.config.get("base_model", "mistralai/Mistral-7B-v0.1")
        use_qlora = self.config.get("use_qlora", True)
        
        logger.info(f"Loading model: model={model_name} qlora={use_qlora}")
        
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
            
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Layered adapter attach (Phase 1) ---
        mgr = LayeredModelAdapter(self.model)

        layer_type_str = self.config.get("layer_type", "task")   # "domain" | "task" | "user" | "tenant"
        layer_name = self.config.get("layer_name", "default")    # e.g., "b2b", "classification", "user_123"

        layer_cfg = LayerConfig(
            layer_type=LayerType(layer_type_str),
            name=layer_name,
            rank=self.config.get("lora_rank", 16),
            alpha=self.config.get("lora_alpha", 32),
            dropout=self.config.get("lora_dropout", 0.05),
            bias=self.config.get("lora_bias", "none"),
            target_modules=self.config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
        )

        adapter_name = mgr.attach_new_lora_layer(layer_cfg)
        mgr.freeze_all_adapters_except(adapter_name)

        self.model = mgr.model
        self._layer_mgr = mgr  # keep for saving later

        try:
            self.model.print_trainable_parameters()
        except Exception:
            pass
        
        if self.tracker:
            self.tracker.log_params({
        "model": model_name,
        "use_qlora": use_qlora,
        "layer_type": layer_cfg.layer_type.value,
        "layer_name": layer_cfg.name,
        "adapter_name": adapter_name,
        "lora_rank": layer_cfg.rank,
        "lora_alpha": layer_cfg.alpha,
        "lora_dropout": layer_cfg.dropout,
        "lora_bias": layer_cfg.bias,
        "target_modules": layer_cfg.target_modules,
    })
    
    def train(self, train_dataset, val_dataset):
        """Train the model."""
        output_dir = self.config.get("output_dir", "./models/trained")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 4),
            per_device_eval_batch_size=self.config.get("eval_batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 16),
            learning_rate=self.config.get("learning_rate", 2e-4),
            fp16=True,
            logging_steps=self.config.get("logging_steps", 10),
            save_steps=self.config.get("save_steps", 500),
            eval_strategy="steps",
            eval_steps=self.config.get("eval_steps", 500),
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to=["mlflow", "wandb"] if self.config.get("use_wandb") else ["mlflow"],
            deepspeed=self.config.get("deepspeed_config") if self.config.get("use_deepspeed") else None,
            warmup_steps=self.config.get("warmup_steps", 100),
            weight_decay=self.config.get("weight_decay", 0.01),
            lr_scheduler_type="cosine"
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        logger.info("Starting training")
        trainer.train()
        
                # Save only the trained adapter if we used the layered manager
        if hasattr(self, "_layer_mgr") and self._layer_mgr is not None:
            self._layer_mgr.save_active_adapter(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        
        if self.tracker:
            # Log the saved adapter/tokenizer folder instead of pickling the in-memory model
            self.tracker.log_model(output_dir, "lora_adapter")
            eval_results = trainer.evaluate()
            self.tracker.log_metrics(eval_results)



        logger.info("Training complete: output_dir=%s", output_dir)
        return output_dir
    
    def run(self):
        """Run complete training pipeline."""
        self.setup_experiment_tracking()
        self.setup_model()
        train_dataset, val_dataset = self.load_and_prepare_data()
        
        if self.tracker:
            self.tracker.log_dataset(self.config["train_dataset_path"])
        
        output_dir = self.train(train_dataset, val_dataset)
        
        if self.tracker:
            self.tracker.end_run()
        
        return output_dir


def main():
    parser = argparse.ArgumentParser(description="Full Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Config JSON file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    pipeline = FullTrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()


