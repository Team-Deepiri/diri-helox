"""
Dataset building utilities for creating training datasets.

Converts cleaned text into tokenized, ready-to-train datasets.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Iterator
import numpy as np
from datasets import Dataset, DatasetDict
import sentencepiece as spm

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Builds training datasets from cleaned text data.
    
    Handles tokenization, splitting, and format conversion.
    """
    
    def __init__(
        self,
        tokenizer_model_path: Optional[Path] = None,
        max_length: int = 8192,
    ):
        """
        Initialize dataset builder.
        
        Args:
            tokenizer_model_path: Path to SentencePiece tokenizer model
            max_length: Maximum sequence length
        """
        self.tokenizer_model_path = tokenizer_model_path
        self.max_length = max_length
        self.tokenizer = None
        
        if tokenizer_model_path and Path(tokenizer_model_path).exists():
            self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Load SentencePiece tokenizer."""
        if self.tokenizer_model_path:
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(str(self.tokenizer_model_path))
            logger.info(f"Loaded tokenizer from {self.tokenizer_model_path}")
    
    def build_from_jsonl(
        self,
        jsonl_file: Path,
        text_field: str = "text",
        train_split: float = 0.9,
        val_split: float = 0.05,
        test_split: float = 0.05,
    ) -> DatasetDict:
        """
        Build dataset from JSONL file.
        
        Args:
            jsonl_file: Input JSONL file
            text_field: Field containing text
            train_split: Training split ratio
            val_split: Validation split ratio
            test_split: Test split ratio
            
        Returns:
            DatasetDict with train/val/test splits
        """
        if not jsonl_file.exists():
            raise ValueError(f"JSONL file does not exist: {jsonl_file}")
        
        # Load texts
        texts = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get(text_field, "")
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(texts)} texts from {jsonl_file}")
        
        # Tokenize if tokenizer available
        if self.tokenizer:
            tokenized = self._tokenize_texts(texts)
        else:
            # Use raw texts (tokenization will happen during training)
            tokenized = [{"text": text} for text in texts]
        
        # Create dataset
        dataset = Dataset.from_list(tokenized)
        
        # Split dataset
        splits = dataset.train_test_split(
            test_size=1 - train_split,
            seed=42,
        )
        
        val_test_size = test_split / (val_split + test_split)
        val_test = splits["test"].train_test_split(
            test_size=val_test_size,
            seed=42,
        )
        
        dataset_dict = DatasetDict({
            "train": splits["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        })
        
        logger.info(f"Created dataset with splits: {dict(dataset_dict.num_rows)}")
        
        return dataset_dict
    
    def _tokenize_texts(self, texts: List[str]) -> List[dict]:
        """Tokenize texts using SentencePiece."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")
        
        tokenized = []
        for text in texts:
            try:
                # Encode to token IDs
                token_ids = self.tokenizer.encode(text, out_type=int)
                
                # Truncate if necessary
                if len(token_ids) > self.max_length:
                    token_ids = token_ids[:self.max_length]
                
                tokenized.append({
                    "input_ids": token_ids,
                    "text": text,  # Keep original for reference
                })
            except Exception as e:
                logger.warning(f"Error tokenizing text: {e}")
                continue
        
        return tokenized
    
    def save_dataset(
        self,
        dataset: DatasetDict,
        output_dir: Path,
        format: str = "arrow",
    ):
        """
        Save dataset to disk.
        
        Args:
            dataset: DatasetDict to save
            output_dir: Output directory
            format: Format to save (arrow, jsonl)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if format == "arrow":
            dataset.save_to_disk(str(output_dir))
            logger.info(f"Saved dataset to {output_dir} (Arrow format)")
        elif format == "jsonl":
            for split_name, split_dataset in dataset.items():
                output_file = output_dir / f"{split_name}.jsonl"
                with open(output_file, "w", encoding="utf-8") as f:
                    for item in split_dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                logger.info(f"Saved {split_name} split to {output_file}")
        else:
            raise ValueError(f"Unsupported format: {format}")

