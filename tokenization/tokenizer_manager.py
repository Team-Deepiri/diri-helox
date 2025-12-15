"""
Tokenizer management and utilities.

Handles loading, saving, and using trained tokenizers.
"""

import logging
from pathlib import Path
from typing import List, Optional
import sentencepiece as spm

logger = logging.getLogger(__name__)


class TokenizerManager:
    """
    Manages SentencePiece tokenizer instances.
    
    Provides convenient interface for encoding/decoding text.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize tokenizer manager.
        
        Args:
            model_path: Path to tokenizer model file
        """
        self.model_path = model_path
        self.tokenizer = None
        
        if model_path:
            self.load(model_path)
    
    def load(self, model_path: Path):
        """
        Load tokenizer from file.
        
        Args:
            model_path: Path to .model file
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Tokenizer model not found: {model_path}")
        
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(str(model_path))
        self.model_path = model_path
        
        logger.info(f"Loaded tokenizer from {model_path}")
        logger.info(f"Vocabulary size: {self.tokenizer.get_piece_size()}")
    
    def encode(
        self,
        text: str,
        out_type: type = int,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> List:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            out_type: Output type (int or str)
            add_bos: Add beginning-of-sequence token
            add_eos: Add end-of-sequence token
            
        Returns:
            List of token IDs or token strings
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")
        
        return self.tokenizer.encode(
            text,
            out_type=out_type,
            add_bos=add_bos,
            add_eos=add_eos,
        )
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")
        
        return self.tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")
        
        return self.tokenizer.get_piece_size()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subword tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of token strings
        """
        return self.encode(text, out_type=str)
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Detokenize tokens back to text.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Detokenized text
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not loaded")
        
        return self.tokenizer.decode_pieces(tokens)

