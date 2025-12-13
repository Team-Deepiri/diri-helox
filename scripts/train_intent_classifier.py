#!/usr/bin/env python3
"""
Train Tier 1: Intent Classifier (50 predefined abilities)
Fine-tunes DeBERTa for maximum reliability
"""
import os
import sys
import json
import types
import importlib.util
from pathlib import Path

# IMPORTANT: Set environment variables BEFORE importing torch/transformers
# This prevents DeepSpeed from trying to compile CUDA ops when using CPU
# Check if we should use CPU mode early
_force_cpu = os.environ.get('FORCE_CPU', '').lower() == 'true'

def _disable_deepspeed_features():
    """
    Disable CUDA/DeepSpeed features to avoid MissingCUDAException.
    Uses a more robust approach: prevents deepspeed from being imported
    by making importlib think it doesn't exist, rather than creating a fake module.
    """
    os.environ['ACCELERATE_USE_CPU'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['DS_SKIP_CUDA_CHECK'] = '1'
    os.environ['ACCELERATE_USE_DEEPSPEED'] = 'false'
    os.environ.setdefault('CUDA_HOME', '')
    
    # Monkey-patch importlib.util.find_spec to return None for deepspeed
    # This makes transformers think deepspeed is not available
    # This is cleaner than creating a fake module
    original_find_spec = importlib.util.find_spec
    
    def patched_find_spec(name, package=None):
        if name == 'deepspeed' or (isinstance(name, str) and name.startswith('deepspeed.')):
            return None  # Tell transformers deepspeed doesn't exist
        return original_find_spec(name, package)
    
    # Only patch if not already patched
    if importlib.util.find_spec is not patched_find_spec:
        importlib.util.find_spec = patched_find_spec

import torch

# If CUDA is available, check compatibility for environment setup
if torch.cuda.is_available() and not _force_cpu:
    try:
        cuda_capability = torch.cuda.get_device_capability(0)
        if cuda_capability[0] < 7:
            _force_cpu = True
            _disable_deepspeed_features()
    except Exception:
        _force_cpu = True
        _disable_deepspeed_features()
else:
    _disable_deepspeed_features()

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def disable_deepspeed_env():
    """Public helper to disable DeepSpeed/CUDA features."""
    _disable_deepspeed_features()

def detect_device(force_cpu: bool = False) -> dict:
    """
    Detect and return detailed device information.
    Returns dict with keys:
      - device (str): "cuda" or "cpu"
      - use_gpu (bool)
      - gpu_name (str | None)
      - capability (tuple | None)
      - reason (str)
    """
    info = {
        "device": "cpu",
        "use_gpu": False,
        "gpu_name": None,
        "capability": None,
        "reason": ""
    }
    
    if force_cpu or not torch.cuda.is_available():
        reason = "forced_cpu" if force_cpu else "cuda_unavailable"
        if not torch.cuda.is_available():
            print("ℹ️  CUDA not available, using CPU")
        info["reason"] = reason
        return info
    
    try:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_capability = torch.cuda.get_device_capability(0)
        info["gpu_name"] = gpu_name
        info["capability"] = cuda_capability
        
        print(f"ℹ️  Detected GPU: {gpu_name}")
        print(f"   CUDA Capability: {cuda_capability[0]}.{cuda_capability[1]}")
        
        if cuda_capability[0] < 7:
            print(f"⚠️  GPU has CUDA capability {cuda_capability[0]}.{cuda_capability[1]}")
            print("   PyTorch requires 7.0+, falling back to CPU")
            info["reason"] = "capability_too_low"
            return info
        
        # Try a simple tensor operation to verify GPU works
        test_tensor = torch.tensor([1.0]).cuda()
        _ = test_tensor * 2
        del test_tensor
        torch.cuda.empty_cache()
        
        print("✓ GPU is compatible, using CUDA")
        info["device"] = "cuda"
        info["use_gpu"] = True
        info["reason"] = "gpu_compatible"
        return info
    except Exception as e:
        print(f"⚠️  GPU detected but not compatible: {e}")
        print("   Falling back to CPU")
        info["reason"] = f"gpu_error:{e}"
        return info

def prepare_device_environment(force_cpu: bool = False) -> dict:
    """
    Prepare environment variables and return device information.
    Ensures DeepSpeed stays disabled when falling back to CPU.
    """
    info = detect_device(force_cpu=force_cpu)
    if not info["use_gpu"]:
        disable_deepspeed_env()
    return info


class DeviceAwareTrainer(Trainer):
    """Trainer subclass that enforces detected device configuration."""
    
    def __init__(self, *args, device_info=None, force_cpu=False, **kwargs):
        self.device_info = device_info or prepare_device_environment(force_cpu=force_cpu)
        training_args = kwargs.get("args")
        if training_args is not None:
            training_args.no_cuda = not self.device_info["use_gpu"]
            if hasattr(training_args, "use_cpu"):
                training_args.use_cpu = not self.device_info["use_gpu"]
        super().__init__(*args, **kwargs)

def compute_metrics(eval_pred):
    """Compute accuracy and F1 score"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_intent_classifier(
    model_name: str = "bert-base-uncased",
    train_file: str = "app/train/data/classification_train.jsonl",
    val_file: str = "app/train/data/classification_val.jsonl",
    output_dir: str = "app/train/models/intent_classifier",
    num_abilities: int = 31,  # 31 task categories
    num_epochs: int = 2,  # Reduced from 3
    batch_size: int = 16,
    learning_rate: float = 1e-5  # Reduced from 2e-5
):
    """Train intent classifier"""
    
    print("=" * 60)
    print("Training Tier 1: Task Category Classifier")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Categories: {num_abilities}")
    print(f"  Coding: debugging, refactoring, writing_code, programming, running_code, inspecting")
    print(f"  Core: writing, learning_research, learning_study, learning_training, learning_practice")
    print(f"  Team: creative, administrative, team_organization, team_collaboration, team_planning")
    print(f"  Work: research, planning, communication, big_data_analytics, data_processing, design")
    print(f"  Quality: qa, testing, validation, reporting, documentation, system_admin")
    print(f"  Specialized: ux_ui, security, data_privacy")
    print(f"Epochs: {num_epochs}")
    print()
    
    # Check if data exists
    train_path = Path(train_file)
    val_path = Path(val_file)
    
    if not train_path.exists():
        print(f"❌ Training file not found: {train_file}")
        print("   Run: python app/train/scripts/prepare_training_data.py")
        print("   Or generate synthetic data first")
        return
    
    if not val_path.exists():
        print(f"⚠ Validation file not found: {val_file}")
        print("   Using train file for validation (not ideal)")
        val_path = train_path
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset('json', data_files={
            'train': str(train_path),
            'validation': str(val_path)
        })
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("   Make sure your JSONL file has 'text' and 'label' fields")
        return
    
    print(f"✓ Train: {len(dataset['train'])} examples")
    print(f"✓ Val: {len(dataset['validation'])} examples")
    
    if len(dataset['train']) < 100:
        print("⚠ Warning: Very small dataset. Consider generating more data.")
    
    # Load model and tokenizer
    print(f"\nLoading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_abilities
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Make sure transformers is installed: pip install transformers")
        return
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=128
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset['train'].column_names if col != 'label']
    )
    
    # Detect device (GPU or CPU) via trainer detection layer
    print("\n🔍 Detecting device (trainer-managed)...")
    device_info = prepare_device_environment(force_cpu=_force_cpu)
    device = device_info["device"]
    use_gpu = device_info["use_gpu"]
    if use_gpu:
        print(f"   ✓ Using GPU: {device_info.get('gpu_name')}")
    else:
        print(f"   ℹ️  Using CPU (reason: {device_info.get('reason', 'cpu_fallback')})")
    print()
    
    # Training arguments
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=str(output_path / "logs"),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=min(100, len(dataset['train']) // batch_size),
        save_strategy="steps",
        save_steps=min(100, len(dataset['train']) // batch_size),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        report_to="none",  # Disable wandb/tensorboard by default
        no_cuda=not use_gpu,
    )
    if hasattr(training_args, "use_cpu"):
        training_args.use_cpu = not use_gpu
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer with detection layer
    trainer = DeviceAwareTrainer(
        device_info=device_info,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n🚀 Starting training...")
    print(f"   Device: {device.upper()} ({'GPU' if use_gpu else 'CPU'})")
    if use_gpu:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print()
    
    try:
        trainer.train()
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate
    print("\n📊 Evaluating...")
    eval_results = trainer.evaluate()
    print(f"\nResults:")
    print(f"  Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  Validation F1: {eval_results['eval_f1']:.4f}")
    print(f"  Validation Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Validation Recall: {eval_results['eval_recall']:.4f}")
    
    # Save
    print(f"\n💾 Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save category mapping (31 categories)
    category_map = {
        0: "debugging", 1: "refactoring", 2: "writing_code", 3: "programming", 4: "running_code", 5: "inspecting",
        6: "writing", 7: "learning_research", 8: "learning_study", 9: "learning_training", 10: "learning_practice",
        11: "creative", 12: "administrative", 13: "team_organization", 14: "team_collaboration", 15: "team_planning",
        16: "research", 17: "planning", 18: "communication", 19: "big_data_analytics", 20: "data_processing",
        21: "design", 22: "qa", 23: "testing", 24: "validation", 25: "reporting",
        26: "documentation", 27: "system_admin", 28: "ux_ui", 29: "security", 30: "data_privacy"
    }
    id_to_label = {i: category_map.get(i, f"category_{i}") for i in range(num_abilities)}
    label_to_id = {v: k for k, v in id_to_label.items()}
    
    with open(f"{output_dir}/category_map.json", 'w') as f:
        json.dump({
            "id2label": id_to_label,
            "label2id": label_to_id
        }, f, indent=2)
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "num_abilities": num_abilities,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_examples": len(dataset['train']),
        "val_examples": len(dataset['validation']),
        "eval_accuracy": float(eval_results['eval_accuracy']),
        "eval_f1": float(eval_results['eval_f1'])
    }
    with open(f"{output_dir}/training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Use this path in CommandRouter: {output_dir}")
    print(f"\n   To use in production:")
    print(f"   from app.services.command_router import get_command_router")
    print(f"   router = get_command_router(model_path='{output_dir}')")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Tier 1 Intent Classifier")
    parser.add_argument("--model", default="bert-base-uncased",
                       help="Base model to fine-tune")
    parser.add_argument("--train-file", default="app/train/data/classification_train.jsonl",
                       help="Training data file")
    parser.add_argument("--val-file", default="app/train/data/classification_val.jsonl",
                       help="Validation data file")
    parser.add_argument("--output-dir", default="app/train/models/intent_classifier",
                       help="Output directory for model")
    parser.add_argument("--num-abilities", type=int, default=31,
                       help="Number of task categories (default: 31)")
    parser.add_argument("--epochs", type=int, default=2,
                       help="Number of training epochs (default: 2)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate (default: 1e-5)")
    
    args = parser.parse_args()
    
    train_intent_classifier(
        model_name=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        num_abilities=args.num_abilities,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
