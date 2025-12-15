#!/usr/bin/env python3
"""
Main script for training LLM from scratch.

This script orchestrates the complete training pipeline:
1. Data collection and cleaning
2. Tokenizer training
3. Dataset preparation
4. Model pretraining
5. Instruction fine-tuning (optional)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.device_manager import DeviceManager
from core.training_config import TrainingConfig, ModelConfig, DataConfig
from data_processing.data_collector import DataCollector
from data_processing.text_cleaner import TextCleaner
from data_processing.dataset_builder import DatasetBuilder
from tokenization.tokenizer_trainer import TokenizerTrainer
from tokenization.tokenizer_manager import TokenizerManager
from models.transformer_lm import create_model_from_config
from training.pretraining_trainer import PretrainingTrainer
from training.instruction_finetuning_trainer import InstructionFinetuningTrainer
from torch.utils.data import DataLoader
from datasets import load_from_disk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_directories(config: DataConfig):
    """Create necessary directories."""
    config.raw_data_dir.mkdir(parents=True, exist_ok=True)
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    config.tokenized_data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Directories created")


def collect_and_clean_data(config: DataConfig) -> Path:
    """Collect and clean raw data."""
    logger.info("Starting data collection and cleaning...")
    
    cleaner = TextCleaner(
        min_length=config.min_text_length,
        max_urls=config.max_urls_per_document,
    )
    
    collector = DataCollector(
        output_dir=config.processed_data_dir,
        cleaner=cleaner,
    )
    
    # Collect from raw data directory
    if config.raw_data_dir.exists():
        output_file = collector.collect_from_directory(
            source_dir=config.raw_data_dir,
            recursive=True,
        )
        logger.info(f"Data collection complete: {output_file}")
        return output_file
    else:
        logger.warning(f"Raw data directory not found: {config.raw_data_dir}")
        logger.info("Skipping data collection. Please add data to raw_data_dir.")
        return None


def train_tokenizer(
    data_file: Path,
    config: DataConfig,
) -> Path:
    """Train SentencePiece tokenizer."""
    logger.info("Training tokenizer...")
    
    trainer = TokenizerTrainer(
        vocab_size=config.tokenizer_vocab_size,
        model_type=config.tokenizer_model_type,
    )
    
    model_path, vocab_path = trainer.train_from_jsonl(
        jsonl_file=data_file,
        text_field="text",
        output_prefix="deepiri_tokenizer",
        output_dir=config.processed_data_dir,
    )
    
    logger.info(f"Tokenizer training complete: {model_path}")
    return model_path


def prepare_dataset(
    data_file: Path,
    tokenizer_path: Path,
    config: DataConfig,
) -> Path:
    """Prepare tokenized dataset."""
    logger.info("Preparing dataset...")
    
    builder = DatasetBuilder(
        tokenizer_model_path=tokenizer_path,
        max_length=config.max_text_length,
    )
    
    dataset = builder.build_from_jsonl(
        jsonl_file=data_file,
        text_field="text",
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
    )
    
    # Save dataset
    output_dir = config.tokenized_data_dir / "pretraining_dataset"
    builder.save_dataset(dataset, output_dir, format=config.output_format)
    
    logger.info(f"Dataset prepared: {output_dir}")
    return output_dir


def create_data_loader(
    dataset_path: Path,
    config: TrainingConfig,
    split: str = "train",
    shuffle: bool = True,
) -> DataLoader:
    """Create data loader from dataset."""
    dataset = load_from_disk(str(dataset_path))
    split_dataset = dataset[split]
    
    def collate_fn(batch):
        """Collate function for batching."""
        input_ids = [item["input_ids"] for item in batch]
        
        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)
        max_len = min(max_len, config.max_sequence_length)
        
        padded_ids = []
        attention_masks = []
        
        for ids in input_ids:
            # Truncate if necessary
            if len(ids) > max_len:
                ids = ids[:max_len]
            
            # Pad
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }
    
    return DataLoader(
        split_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=config.data_loader_num_workers,
        pin_memory=config.pin_memory,
    )


def main():
    parser = argparse.ArgumentParser(description="Train LLM from scratch")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Configuration directory",
    )
    parser.add_argument(
        "--skip-data-collection",
        action="store_true",
        help="Skip data collection step",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip tokenizer training (use existing)",
    )
    parser.add_argument(
        "--skip-pretraining",
        action="store_true",
        help="Skip pretraining (only do fine-tuning)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device (cpu, cuda, mps)",
    )
    
    args = parser.parse_args()
    
    # Load configurations
    config_dir = args.config_dir
    model_config = ModelConfig.from_file(config_dir / "model_config.json")
    data_config = DataConfig.from_file(config_dir / "data_config.json")
    training_config = TrainingConfig.from_file(config_dir / "training_config.json")
    
    # Override device if specified
    if args.device:
        training_config.device = args.device
    
    # Setup device
    device_manager = DeviceManager(force_device=training_config.device)
    logger.info(f"Using device: {device_manager.get_device_info()}")
    
    # Setup directories
    setup_directories(data_config)
    
    # Step 1: Data collection and cleaning
    if not args.skip_data_collection:
        data_file = collect_and_clean_data(data_config)
        if data_file is None:
            logger.error("No data file available. Cannot continue.")
            return
    else:
        # Find existing data file
        data_files = list(data_config.processed_data_dir.glob("*.jsonl"))
        if data_files:
            data_file = data_files[0]
            logger.info(f"Using existing data file: {data_file}")
        else:
            logger.error("No data file found. Cannot continue.")
            return
    
    # Step 2: Train tokenizer
    if not args.skip_tokenizer:
        tokenizer_path = train_tokenizer(data_file, data_config)
        data_config.tokenizer_model_path = tokenizer_path
    else:
        # Use existing tokenizer
        tokenizer_path = data_config.tokenizer_model_path
        if not tokenizer_path or not Path(tokenizer_path).exists():
            logger.error("Tokenizer not found. Cannot continue.")
            return
        logger.info(f"Using existing tokenizer: {tokenizer_path}")
    
    # Step 3: Prepare dataset
    dataset_path = prepare_dataset(data_file, tokenizer_path, data_config)
    
    # Step 4: Create data loaders
    train_loader = create_data_loader(
        dataset_path,
        training_config,
        split="train",
        shuffle=True,
    )
    val_loader = create_data_loader(
        dataset_path,
        training_config,
        split="validation",
        shuffle=False,
    )
    
    # Step 5: Create model
    logger.info("Creating model...")
    model = create_model_from_config(model_config)
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    # Step 6: Pretraining
    if not args.skip_pretraining:
        logger.info("Starting pretraining...")
        trainer = PretrainingTrainer(
            model=model,
            config=training_config,
            model_config=model_config,
            device_manager=device_manager,
        )
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            resume_from_checkpoint=training_config.resume_from_checkpoint,
        )
        logger.info("Pretraining complete!")
    
    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main()

