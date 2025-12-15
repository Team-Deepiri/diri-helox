"""
Data leakage and memorization detection system.

Detects N-gram overlap, train/eval contamination, and memorization
patterns for legal and IP safety.
"""

import logging
import hashlib
from typing import Dict, Any, List, Set, Tuple
from collections import Counter, defaultdict
import json

logger = logging.getLogger(__name__)


class DataLeakageDetector:
    """
    Detects data leakage and memorization in training data.
    
    Detects:
    - N-gram overlap between train/eval
    - Exact duplicate detection
    - Near-duplicate detection
    - Memorization patterns
    """
    
    def __init__(
        self,
        ngram_size: int = 5,
        overlap_threshold: float = 0.8,
    ):
        """
        Initialize leakage detector.
        
        Args:
            ngram_size: Size of N-grams for overlap detection
            overlap_threshold: Threshold for overlap detection
        """
        self.ngram_size = ngram_size
        self.overlap_threshold = overlap_threshold
    
    def extract_ngrams(self, text: str) -> Set[str]:
        """Extract N-grams from text."""
        words = text.lower().split()
        ngrams = set()
        
        for i in range(len(words) - self.ngram_size + 1):
            ngram = " ".join(words[i:i + self.ngram_size])
            ngrams.add(ngram)
        
        return ngrams
    
    def detect_train_eval_contamination(
        self,
        train_texts: List[str],
        eval_texts: List[str],
    ) -> Dict[str, Any]:
        """
        Detect contamination between train and eval sets.
        
        Args:
            train_texts: Training texts
            eval_texts: Evaluation texts
            
        Returns:
            Contamination report
        """
        logger.info("Checking for train/eval contamination...")
        
        # Extract N-grams from train set
        train_ngrams = set()
        for text in train_texts:
            train_ngrams.update(self.extract_ngrams(text))
        
        # Check eval texts for overlap
        contamination_count = 0
        contaminated_texts = []
        
        for i, eval_text in enumerate(eval_texts):
            eval_ngrams = self.extract_ngrams(eval_text)
            
            if not eval_ngrams:
                continue
            
            overlap = len(train_ngrams & eval_ngrams) / len(eval_ngrams)
            
            if overlap > self.overlap_threshold:
                contamination_count += 1
                contaminated_texts.append({
                    "index": i,
                    "overlap": overlap,
                    "text_preview": eval_text[:100],
                })
        
        contamination_rate = contamination_count / len(eval_texts) if eval_texts else 0.0
        
        report = {
            "contamination_detected": contamination_count > 0,
            "contamination_count": contamination_count,
            "contamination_rate": contamination_rate,
            "total_eval_texts": len(eval_texts),
            "contaminated_texts": contaminated_texts[:10],  # First 10
        }
        
        if contamination_count > 0:
            logger.warning(
                f"Train/eval contamination detected: {contamination_count} texts "
                f"({contamination_rate:.2%})"
            )
        else:
            logger.info("No train/eval contamination detected")
        
        return report
    
    def detect_exact_duplicates(
        self,
        texts: List[str],
    ) -> Dict[str, Any]:
        """
        Detect exact duplicate texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Duplicate report
        """
        logger.info("Checking for exact duplicates...")
        
        # Hash texts
        text_hashes = {}
        duplicates = defaultdict(list)
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            if text_hash in text_hashes:
                duplicates[text_hash].append(i)
            else:
                text_hashes[text_hash] = i
                duplicates[text_hash] = [i]
        
        # Find actual duplicates (more than one occurrence)
        duplicate_groups = {
            hash_val: indices
            for hash_val, indices in duplicates.items()
            if len(indices) > 1
        }
        
        total_duplicates = sum(len(indices) - 1 for indices in duplicate_groups.values())
        
        report = {
            "duplicates_detected": len(duplicate_groups) > 0,
            "duplicate_groups": len(duplicate_groups),
            "total_duplicate_instances": total_duplicates,
            "duplicate_rate": total_duplicates / len(texts) if texts else 0.0,
        }
        
        if duplicate_groups:
            logger.warning(
                f"Exact duplicates detected: {len(duplicate_groups)} groups, "
                f"{total_duplicates} duplicate instances"
            )
        else:
            logger.info("No exact duplicates detected")
        
        return report
    
    def detect_near_duplicates(
        self,
        texts: List[str],
        similarity_threshold: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Detect near-duplicate texts using N-gram similarity.
        
        Args:
            texts: List of texts
            similarity_threshold: Similarity threshold
            
        Returns:
            Near-duplicate report
        """
        logger.info("Checking for near-duplicates...")
        
        # Extract N-grams for all texts
        text_ngrams = [self.extract_ngrams(text) for text in texts]
        
        near_duplicates = []
        
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                ngrams_i = text_ngrams[i]
                ngrams_j = text_ngrams[j]
                
                if not ngrams_i or not ngrams_j:
                    continue
                
                # Jaccard similarity
                intersection = len(ngrams_i & ngrams_j)
                union = len(ngrams_i | ngrams_j)
                similarity = intersection / union if union > 0 else 0.0
                
                if similarity >= similarity_threshold:
                    near_duplicates.append({
                        "text1_index": i,
                        "text2_index": j,
                        "similarity": similarity,
                    })
        
        report = {
            "near_duplicates_detected": len(near_duplicates) > 0,
            "near_duplicate_pairs": len(near_duplicates),
            "near_duplicate_rate": len(near_duplicates) / (len(texts) * (len(texts) - 1) / 2) if len(texts) > 1 else 0.0,
        }
        
        if near_duplicates:
            logger.warning(
                f"Near-duplicates detected: {len(near_duplicates)} pairs"
            )
        else:
            logger.info("No near-duplicates detected")
        
        return report
    
    def detect_memorization_patterns(
        self,
        texts: List[str],
        min_frequency: int = 10,
    ) -> Dict[str, Any]:
        """
        Detect potential memorization patterns (high-frequency N-grams).
        
        Args:
            texts: List of texts
            min_frequency: Minimum frequency for pattern detection
            
        Returns:
            Memorization pattern report
        """
        logger.info("Checking for memorization patterns...")
        
        # Count N-gram frequencies
        ngram_counter = Counter()
        
        for text in texts:
            ngrams = self.extract_ngrams(text)
            ngram_counter.update(ngrams)
        
        # Find high-frequency patterns
        high_freq_patterns = {
            ngram: count
            for ngram, count in ngram_counter.items()
            if count >= min_frequency
        }
        
        report = {
            "patterns_detected": len(high_freq_patterns) > 0,
            "high_frequency_patterns": len(high_freq_patterns),
            "top_patterns": dict(
                sorted(high_freq_patterns.items(), key=lambda x: x[1], reverse=True)[:20]
            ),
        }
        
        if high_freq_patterns:
            logger.info(
                f"High-frequency patterns detected: {len(high_freq_patterns)} patterns"
            )
        
        return report

