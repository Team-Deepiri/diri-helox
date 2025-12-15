"""
SentencePiece tokenizer training.

Trains custom tokenizers optimized for your data.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, List
import sentencepiece as spm

logger = logging.getLogger(__name__)


class TokenizerTrainer:
    """
    Train SentencePiece tokenizers for LLM training.
    
    Creates custom tokenizers optimized for your specific data.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
    ):
        """
        Initialize tokenizer trainer.
        
        Args:
            vocab_size: Vocabulary size
            model_type: Model type (bpe, unigram, word, char)
            character_coverage: Character coverage ratio
        """
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
    
    def train_from_file(
        self,
        input_file: Path,
        output_prefix: str,
        output_dir: Optional[Path] = None,
    ) -> tuple[Path, Path]:
        """
        Train tokenizer from text file.
        
        Args:
            input_file: Input text file (one sentence per line)
            output_prefix: Output model prefix
            output_dir: Output directory (default: same as input)
            
        Returns:
            Tuple of (model_path, vocab_path)
        """
        input_file = Path(input_file)
        if not input_file.exists():
            raise ValueError(f"Input file does not exist: {input_file}")
        
        if output_dir is None:
            output_dir = input_file.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f"{output_prefix}.model"
        vocab_path = output_dir / f"{output_prefix}.vocab"
        
        logger.info(f"Training tokenizer from {input_file}")
        logger.info(
            f"Vocab size: {self.vocab_size}, Model type: {self.model_type}"
        )
        
        # Train using SentencePiece
        spm.SentencePieceTrainer.train(
            input=str(input_file),
            model_prefix=str(output_dir / output_prefix),
            vocab_size=self.vocab_size,
            model_type=self.model_type,
            character_coverage=self.character_coverage,
            input_sentence_size=1000000,  # Limit for memory
            shuffle_input_sentence=True,
            seed_sentencepiece_size=1000000,
            shrinking_factor=0.75,
            num_threads=4,
            max_sentence_length=4192,
        )
        
        logger.info(f"Tokenizer training complete")
        logger.info(f"Model: {model_path}")
        logger.info(f"Vocab: {vocab_path}")
        
        return model_path, vocab_path
    
    def train_from_texts(
        self,
        texts: List[str],
        output_prefix: str,
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """
        Train tokenizer from list of texts.
        
        Args:
            texts: List of text strings
            output_prefix: Output model prefix
            output_dir: Output directory
            
        Returns:
            Tuple of (model_path, vocab_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write texts to temporary file
        temp_file = output_dir / f"{output_prefix}_temp.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text.strip() + "\n")
        
        try:
            return self.train_from_file(temp_file, output_prefix, output_dir)
        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
    
    def train_from_jsonl(
        self,
        jsonl_file: Path,
        text_field: str = "text",
        output_prefix: str = "tokenizer",
        output_dir: Optional[Path] = None,
    ) -> tuple[Path, Path]:
        """
        Train tokenizer from JSONL file.
        
        Args:
            jsonl_file: Input JSONL file
            text_field: Field containing text
            output_prefix: Output model prefix
            output_dir: Output directory
            
        Returns:
            Tuple of (model_path, vocab_path)
        """
        import json
        
        jsonl_file = Path(jsonl_file)
        if not jsonl_file.exists():
            raise ValueError(f"JSONL file does not exist: {jsonl_file}")
        
        if output_dir is None:
            output_dir = jsonl_file.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract texts
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
        
        logger.info(f"Extracted {len(texts)} texts from {jsonl_file}")
        
        return self.train_from_texts(texts, output_prefix, output_dir)

