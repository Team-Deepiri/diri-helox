#!/usr/bin/env python3
"""
Script for instruction fine-tuning.

Fine-tunes a pretrained model on instruction-following data.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.device_manager import DeviceManager
from core.training_config import TrainingConfig, ModelConfig
from models.transformer_lm import TransformerLanguageModel, create_model_from_config
from training.instruction_finetuning_trainer import InstructionFinetuningTrainer
from torch.utils.data import DataLoader
import torch
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_instruction_data_loader(
    jsonl_file: Path,
    tokenizer_manager,
    config: TrainingConfig,
    shuffle: bool = True,
) -> DataLoader:
    """Create data loader for instruction fine-tuning."""
    from datasets import Dataset
    
    # Load instruction data
    examples = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                examples.append({
                    "instruction": data.get("instruction", ""),
                    "response": data.get("response", ""),
                })
            except json.JSONDecodeError:
                continue
    
    # Format: "Instruction: {instruction}\n\nResponse: {response}"
    formatted_texts = []
    instruction_masks = []
    
    for example in examples:
        instruction = example["instruction"]
        response = example["response"]
        
        # Create formatted text
        formatted = f"Instruction: {instruction}\n\nResponse: {response}"
        
        # Tokenize
        tokens = tokenizer_manager.encode(formatted, add_bos=True, add_eos=True)
        
        # Create mask (0 for instruction, 1 for response)
        instruction_text = f"Instruction: {instruction}\n\nResponse: "
        instruction_tokens = tokenizer_manager.encode(instruction_text, add_bos=True, add_eos=False)
        
        mask = [0] * len(instruction_tokens) + [1] * (len(tokens) - len(instruction_tokens))
        
        # Truncate if necessary
        max_len = config.max_sequence_length
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
            mask = mask[:max_len]
        
        formatted_texts.append(tokens)
        instruction_masks.append(mask)
    
    # Create dataset
    dataset = Dataset.from_dict({
        "input_ids": formatted_texts,
        "instruction_mask": instruction_masks,
    })
    
    def collate_fn(batch):
        """Collate function."""
        input_ids = [item["input_ids"] for item in batch]
        instruction_masks = [item["instruction_mask"] for item in batch]
        
        # Pad
        max_len = max(len(ids) for ids in input_ids)
        max_len = min(max_len, config.max_sequence_length)
        
        padded_ids = []
        padded_masks = []
        attention_masks = []
        
        for ids, mask in zip(input_ids, instruction_masks):
            if len(ids) > max_len:
                ids = ids[:max_len]
                mask = mask[:max_len]
            
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)
            padded_masks.append(mask + [0] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "instruction_mask": torch.tensor(padded_masks, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=config.data_loader_num_workers,
        pin_memory=config.pin_memory,
    )


def main():
    parser = argparse.ArgumentParser(description="Instruction fine-tuning")
    parser.add_argument(
        "--pretrained-model",
        type=Path,
        required=True,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--instruction-data",
        type=Path,
        required=True,
        help="Path to instruction data JSONL file",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        required=True,
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Configuration directory",
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
    training_config = TrainingConfig.from_file(config_dir / "training_config.json")
    
    if args.device:
        training_config.device = args.device
    
    # Setup device
    device_manager = DeviceManager(force_device=training_config.device)
    logger.info(f"Using device: {device_manager.get_device_info()}")
    
    # Load tokenizer
    from tokenization.tokenizer_manager import TokenizerManager
    tokenizer_manager = TokenizerManager(args.tokenizer_model)
    
    # Create model
    logger.info("Creating model...")
    model = create_model_from_config(model_config)
    
    # Load pretrained weights
    logger.info(f"Loading pretrained model from {args.pretrained_model}")
    checkpoint_path = args.pretrained_model / "model.pt"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device_manager.get_device()))
        logger.info("Pretrained weights loaded")
    else:
        logger.warning("Pretrained model not found. Starting from scratch.")
    
    # Create data loader
    logger.info("Preparing instruction data...")
    train_loader = create_instruction_data_loader(
        args.instruction_data,
        tokenizer_manager,
        training_config,
        shuffle=True,
    )
    
    # Create trainer
    logger.info("Starting instruction fine-tuning...")
    trainer = InstructionFinetuningTrainer(
        model=model,
        config=training_config,
        model_config=model_config,
        device_manager=device_manager,
    )
    
    # Train
    trainer.train(train_loader=train_loader)
    
    logger.info("Instruction fine-tuning complete!")


if __name__ == "__main__":
    main()

