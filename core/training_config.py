"""
Configuration management for LLM training.

Provides structured configuration classes for model, training, and data settings.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Architecture
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    vocab_size: int = 50000
    max_position_embeddings: int = 8192
    
    # Advanced components
    use_rms_norm: bool = True
    use_swiglu: bool = True
    use_rope_embeddings: bool = True
    
    # Dropout
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    # Activation
    activation_function: str = "swiglu"  # swiglu, gelu, relu
    
    # Initialization
    initializer_range: float = 0.02
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Path) -> "ModelConfig":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Paths
    raw_data_dir: Path = Path("data/datasets/raw")
    processed_data_dir: Path = Path("data/datasets/processed")
    tokenized_data_dir: Path = Path("data/datasets/tokenized")
    
    # Tokenizer
    tokenizer_model_path: Optional[Path] = None
    tokenizer_vocab_size: int = 50000
    tokenizer_model_type: str = "bpe"  # bpe, unigram, word, char
    
    # Data processing
    min_text_length: int = 50
    max_text_length: int = 8192
    remove_duplicates: bool = True
    max_urls_per_document: int = 5
    
    # Dataset splits
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    
    # Data formats
    input_format: str = "text"  # text, jsonl
    output_format: str = "arrow"  # arrow, bin, jsonl
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.raw_data_dir, str):
            self.raw_data_dir = Path(self.raw_data_dir)
        if isinstance(self.processed_data_dir, str):
            self.processed_data_dir = Path(self.processed_data_dir)
        if isinstance(self.tokenized_data_dir, str):
            self.tokenized_data_dir = Path(self.tokenized_data_dir)
        if isinstance(self.tokenizer_model_path, str):
            self.tokenizer_model_path = Path(self.tokenizer_model_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert Path objects to strings
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Path) -> "DataConfig":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Device
    device: Optional[str] = None  # None = auto-detect
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Training hyperparameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    total_steps: int = 300000
    
    # Optimizer
    optimizer_type: str = "adamw"  # adamw, adam, sgd
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Scheduler
    scheduler_type: str = "cosine"  # cosine, linear, constant
    min_learning_rate: float = 1e-6
    
    # Checkpointing
    save_steps: int = 10000
    save_total_limit: int = 3
    checkpoint_dir: Path = Path("models/checkpoints")
    resume_from_checkpoint: Optional[Path] = None
    
    # Evaluation
    eval_steps: int = 1000
    eval_batch_size: int = 4
    eval_metrics: list = field(default_factory=lambda: ["perplexity", "loss"])
    
    # Logging
    logging_steps: int = 100
    log_dir: Path = Path("logs")
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    # Data
    max_sequence_length: int = 8192
    data_loader_num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.log_dir, str):
            self.log_dir = Path(self.log_dir)
        if isinstance(self.resume_from_checkpoint, str):
            self.resume_from_checkpoint = Path(self.resume_from_checkpoint)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert Path objects to strings
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_file(cls, path: Path) -> "TrainingConfig":
        """Load from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save(self, path: Path):
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_effective_batch_size(self) -> int:
        """Get effective batch size (batch_size * gradient_accumulation_steps)."""
        return self.batch_size * self.gradient_accumulation_steps

