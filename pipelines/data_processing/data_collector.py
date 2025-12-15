"""
Data collection utilities for gathering training data from various sources.

Supports multiple data sources and formats.
"""

import json
import logging
from pathlib import Path
from typing import List, Iterator, Optional
from .text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects and organizes training data from various sources.
    
    Supports:
    - Plain text files
    - JSONL files
    - Directory crawling
    - Common Crawl format
    """
    
    def __init__(
        self,
        output_dir: Path,
        cleaner: Optional[TextCleaner] = None,
    ):
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to save collected data
            cleaner: Optional text cleaner instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cleaner = cleaner or TextCleaner()
        
        # Statistics
        self.stats = {
            "total_documents": 0,
            "cleaned_documents": 0,
            "filtered_documents": 0,
        }
    
    def collect_from_directory(
        self,
        source_dir: Path,
        file_extensions: List[str] = [".txt", ".md", ".py", ".js", ".ts"],
        recursive: bool = True,
    ) -> Path:
        """
        Collect text from files in a directory.
        
        Args:
            source_dir: Source directory
            file_extensions: File extensions to include
            recursive: Whether to search recursively
            
        Returns:
            Path to output file
        """
        source_dir = Path(source_dir)
        if not source_dir.exists():
            raise ValueError(f"Source directory does not exist: {source_dir}")
        
        output_file = self.output_dir / "collected_texts.jsonl"
        
        collected_texts = []
        
        pattern = "**/*" if recursive else "*"
        for ext in file_extensions:
            for file_path in source_dir.glob(f"{pattern}{ext}"):
                try:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    cleaned = self.cleaner.clean(text)
                    if cleaned:
                        collected_texts.append({
                            "text": cleaned,
                            "source": str(file_path),
                            "type": "file",
                        })
                        self.stats["cleaned_documents"] += 1
                    else:
                        self.stats["filtered_documents"] += 1
                    self.stats["total_documents"] += 1
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
        
        # Remove duplicates
        unique_texts = self.cleaner.remove_duplicates(
            [item["text"] for item in collected_texts]
        )
        collected_texts = [
            item for item in collected_texts if item["text"] in unique_texts
        ]
        
        # Save to JSONL
        self._save_jsonl(collected_texts, output_file)
        
        logger.info(
            f"Collected {len(collected_texts)} documents from {source_dir}"
        )
        logger.info(f"Statistics: {self.stats}")
        
        return output_file
    
    def collect_from_jsonl(
        self,
        source_file: Path,
        text_field: str = "text",
    ) -> Path:
        """
        Collect and clean data from JSONL file.
        
        Args:
            source_file: Source JSONL file
            text_field: Field name containing text
            
        Returns:
            Path to output file
        """
        source_file = Path(source_file)
        if not source_file.exists():
            raise ValueError(f"Source file does not exist: {source_file}")
        
        output_file = self.output_dir / f"cleaned_{source_file.name}"
        
        collected_texts = []
        
        with open(source_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    text = data.get(text_field, "")
                    if not text:
                        continue
                    
                    cleaned = self.cleaner.clean(text)
                    if cleaned:
                        collected_texts.append({
                            "text": cleaned,
                            "source": str(source_file),
                            "line": line_num,
                            "type": "jsonl",
                        })
                        self.stats["cleaned_documents"] += 1
                    else:
                        self.stats["filtered_documents"] += 1
                    self.stats["total_documents"] += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
        
        # Save to JSONL
        self._save_jsonl(collected_texts, output_file)
        
        logger.info(
            f"Collected {len(collected_texts)} documents from {source_file}"
        )
        
        return output_file
    
    def _save_jsonl(self, data: List[dict], output_file: Path):
        """Save data to JSONL file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    def get_statistics(self) -> dict:
        """Get collection statistics."""
        return self.stats.copy()

