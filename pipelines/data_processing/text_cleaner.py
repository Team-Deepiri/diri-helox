"""
Text cleaning and filtering utilities.

Provides robust text cleaning functions for preparing training data.
"""

import re
import logging
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Text cleaning and filtering for training data.
    
    Removes boilerplate, duplicates, and low-quality content.
    """
    
    def __init__(
        self,
        min_length: int = 50,
        max_urls: int = 5,
        remove_boilerplate: bool = True,
    ):
        """
        Initialize text cleaner.
        
        Args:
            min_length: Minimum text length in characters
            max_urls: Maximum URLs per document
            remove_boilerplate: Whether to remove common boilerplate
        """
        self.min_length = min_length
        self.max_urls = max_urls
        self.remove_boilerplate = remove_boilerplate
        
        # Common boilerplate patterns
        self.boilerplate_patterns = [
            r"cookie policy",
            r"privacy policy",
            r"terms of service",
            r"all rights reserved",
            r"© \d{4}",
            r"page \d+ of \d+",
            r"click here",
            r"read more",
        ]
        
        # URL pattern
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
    
    def clean(self, text: str) -> Optional[str]:
        """
        Clean a single text document.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text or None if filtered out
        """
        if not text or not isinstance(text, str):
            return None
        
        # Strip whitespace
        text = text.strip()
        
        # Check minimum length
        if len(text) < self.min_length:
            return None
        
        # Remove excessive URLs
        urls = self.url_pattern.findall(text)
        if len(urls) > self.max_urls:
            return None
        
        # Remove boilerplate
        if self.remove_boilerplate:
            text = self._remove_boilerplate(text)
            if not text or len(text.strip()) < self.min_length:
                return None
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        
        # Final length check
        if len(text) < self.min_length:
            return None
        
        return text
    
    def _remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate patterns."""
        text_lower = text.lower()
        
        for pattern in self.boilerplate_patterns:
            text = re.sub(pattern, "", text_lower, flags=re.IGNORECASE)
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Clean a batch of texts.
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of cleaned texts (filtered)
        """
        cleaned = []
        for text in texts:
            result = self.clean(text)
            if result:
                cleaned.append(result)
        
        return cleaned
    
    def remove_duplicates(self, texts: List[str]) -> List[str]:
        """
        Remove duplicate texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List with duplicates removed
        """
        seen = set()
        unique = []
        
        for text in texts:
            # Normalize for comparison
            normalized = text.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(text)
        
        return unique


def clean_text_document(
    text: str,
    min_length: int = 50,
    max_urls: int = 5,
) -> Optional[str]:
    """
    Convenience function to clean a single text document.
    
    Args:
        text: Raw text
        min_length: Minimum length
        max_urls: Maximum URLs
        
    Returns:
        Cleaned text or None
    """
    cleaner = TextCleaner(min_length=min_length, max_urls=max_urls)
    return cleaner.clean(text)

