"""
Versioned Training Pipeline
Training pipeline that uses dataset versioning for reproducible training
"""
import torch
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
    TaskType
)
from datasets import load_dataset, Dataset
import json
from pathlib import Path
import os
from typing import Dict, Optional
import argparse

from ...utils.dataset_versioning import DatasetVersionManager, DatasetType
from ...utils.dataset_validation import validate_dataset_quality
from ...mlops.infrastructure.experiment_tracker import ExperimentTracker
from deepiri_modelkit.logging import get_logger

logger = get_logger("helox.versioned_pipeline")


class VersionedTrainingPipeline:
    """
    Training pipeline that uses dataset versioning for reproducible results.

    This ensures that:
    - Training runs are reproducible with exact dataset versions
    - Dataset lineage is tracked and auditable
    - Model performance can be correlated with specific data versions
    - Rollbacks to previous dataset versions are possible
    """

    def __init__(self, config: Dict):
        self.config = config
        self.version_manager = DatasetVersionManager(
            db_url=config.get("version_db_url", "sqlite:///dataset_versions.db"),
            storage_backend=config.get("storage_backend", "local")
        )
        self.tracker = None
        self.model = None
        self.tokenizer = None
        self.dataset_version = None

    def setup_experiment_tracking(self):
        """Setup experiment tracking with dataset version info."""
        from ...mlops.infrastructure.experiment_tracker import ExperimentTracker

        self.tracker = ExperimentTracker(
            experiment_name=self.config.get("experiment_name", "versioned_training"),
            tracking_uri=self.config.get("mlflow_uri", "http://localhost:5000"),
            use_wandb=self.config.get("use_wandb", False),
            wandb_project=self.config.get("wandb_project", "deepiri")
        )
        self.tracker.start_run()
        self.tracker.log_git_info()

        # Log dataset version info
        if self.dataset_version:
            self.tracker.log_params({
                "dataset_name": self.dataset_version.dataset_name,
                "dataset_version": self.dataset_version.version,
                "dataset_checksum": self.dataset_version.data_checksum,
                "total_samples": self.dataset_version.total_samples,
                "dataset_type": self.dataset_version.dataset_type.value
            })

    def load_versioned_dataset(self):
        """
        Load dataset using versioning system.

        Supports:
        - Specific version: "dataset_name@version"
        - Latest version: "dataset_name@latest"
        - Dataset path with auto-versioning: "/path/to/data"
        """
        dataset_spec = self.config["dataset_spec"]
        dataset_type = DatasetType(self.config.get("dataset_type", "lease_abstraction"))

        logger.info("Loading versioned dataset", spec=dataset_spec)

        if "@" in dataset_spec:
            # Versioned dataset specification: "dataset_name@version"
            dataset_name, version_spec = dataset_spec.split("@", 1)

            if version_spec == "latest":
                self.dataset_version = self.version_manager.get_latest_version(
                    dataset_name, dataset_type
                )
                if not self.dataset_version:
                    raise ValueError(f"No versions found for dataset {dataset_name}")
            else:
                self.dataset_version = self.version_manager.get_version(
                    dataset_name, version_spec, dataset_type
                )
                if not self.dataset_version:
                    raise ValueError(f"Version {version_spec} not found for dataset {dataset_name}")

            # Use versioned dataset path
            dataset_path = self.dataset_version.storage_path
            logger.info("Using versioned dataset",
                       name=self.dataset_version.dataset_name,
                       version=self.dataset_version.version,
                       path=dataset_path)

        else:
            # Regular path - optionally create version for tracking
            dataset_path = dataset_spec
            if self.config.get("auto_version", False):
                logger.info("Auto-versioning dataset", path=dataset_path)
                self.dataset_version = self.version_manager.create_version(
                    dataset_name=self.config.get("auto_dataset_name", "auto_versioned"),
                    dataset_type=dataset_type,
                    data_path=Path(dataset_path),
                    change_summary=f"Auto-versioned for training run",
                    tags=["auto_versioned", "training"]
                )
                dataset_path = self.dataset_version.storage_path

        return dataset_path

    def load_and_prepare_data(self):
        """Load and prepare training dataset using versioned data."""
        dataset_path = self.load_versioned_dataset()

        logger.info("Loading dataset from path", path=dataset_path)

        # Load dataset
        if str(dataset_path).endswith('.jsonl'):
            dataset = load_dataset('json', data_files=str(dataset_path), split='train')
        else:
            # Handle directory with multiple files
            jsonl_files = list(Path(dataset_path).glob("*.jsonl"))
            if jsonl_files:
                dataset = load_dataset('json', data_files=[str(f) for f in jsonl_files], split='train')
            else:
                raise ValueError(f"No .jsonl files found in {dataset_path}")

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

        # Create validation split
        if "validation_dataset_spec" in self.config:
            val_dataset_path = self.config["validation_dataset_spec"]
            if val_dataset_path.endswith('.jsonl'):
                val_dataset = load_dataset('json', data_files=val_dataset_path, split='train')
            else:
                val_jsonl_files = list(Path(val_dataset_path).glob("*.jsonl"))
                val_dataset = load_dataset('json', data_files=[str(f) for f in val_jsonl_files], split='train')

            val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)
        else:
            val_tokenized = tokenized.train_test_split(test_size=0.1)['test']

        return tokenized, val_tokenized

    def setup_model(self):
        """Setup model with quantization and LoRA."""
        model_name = self.config.get("base_model", "mistralai/Mistral-7B-v0.1")
        use_qlora = self.config.get("use_qlora", True)

        logger.info("Loading model", model=model_name, qlora=use_qlora)

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

        lora_config = LoraConfig(
            r=self.config.get("lora_rank", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            target_modules=self.config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        if self.tracker:
            self.tracker.log_params({
                "model": model_name,
                "lora_rank": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "use_qlora": use_qlora,
                "dataset_version": self.dataset_version.version if self.dataset_version else None
            })

    def train(self, train_dataset, val_dataset):
        """Train the model with versioned dataset tracking."""
        output_dir = self.config.get("output_dir", "./models/versioned_training")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Include dataset version in output directory
        if self.dataset_version:
            version_suffix = f"_v{self.dataset_version.version.replace('.', '_')}"
            output_dir = f"{output_dir}{version_suffix}"

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
            evaluation_strategy="steps",
            eval_steps=self.config.get("eval_steps", 500),
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to=["mlflow", "wandb"] if self.config.get("use_wandb") else ["mlflow"],
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

        logger.info("Starting versioned training",
                   dataset=self.dataset_version.dataset_name if self.dataset_version else "unknown",
                   version=self.dataset_version.version if self.dataset_version else "unknown")

        trainer.train()

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save training metadata with dataset version info
        training_metadata = {
            "dataset_name": self.dataset_version.dataset_name if self.dataset_version else None,
            "dataset_version": self.dataset_version.version if self.dataset_version else None,
            "dataset_checksum": self.dataset_version.data_checksum if self.dataset_version else None,
            "total_samples": self.dataset_version.total_samples if self.dataset_version else None,
            "training_config": self.config,
            "output_dir": output_dir
        }

        metadata_path = Path(output_dir) / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2, default=str)

        if self.tracker:
            self.tracker.log_model(self.model, "model")
            eval_results = trainer.evaluate()
            self.tracker.log_metrics(eval_results)

            # Log dataset version as artifact
            if self.dataset_version:
                self.tracker.log_artifact(str(metadata_path))

        logger.info("Versioned training complete",
                   output_dir=output_dir,
                   dataset_version=self.dataset_version.version if self.dataset_version else "unknown")

        return output_dir, training_metadata

    def run(self):
        """Run complete versioned training pipeline."""
        self.setup_experiment_tracking()
        self.setup_model()
        train_dataset, val_dataset = self.load_and_prepare_data()
        output_dir, metadata = self.train(train_dataset, val_dataset)

        if self.tracker:
            self.tracker.end_run()

        return output_dir, metadata


def main():
    parser = argparse.ArgumentParser(description="Versioned Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Config JSON file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    pipeline = VersionedTrainingPipeline(config)
    output_dir, metadata = pipeline.run()

    print(f"Training complete! Output: {output_dir}")
    print(f"Dataset version: {metadata.get('dataset_version', 'N/A')}")


if __name__ == "__main__":
    main()
