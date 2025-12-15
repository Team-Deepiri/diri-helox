"""
Semantic deduplication engine.

Uses embedding-based similarity filtering and near-duplicate suppression
to prevent duplicate data from poisoning LLMs.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Set
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class SemanticDeduplicationEngine:
    """
    Semantic deduplication using embeddings.
    
    Features:
    - Embedding-based similarity
    - Near-duplicate suppression
    - Configurable similarity threshold
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        embedding_model=None,  # Will be injected
        cache_dir: Path = Path("data/deduplication_cache"),
    ):
        """
        Initialize semantic deduplication engine.
        
        Args:
            similarity_threshold: Similarity threshold for duplicates
            embedding_model: Embedding model for semantic similarity
            cache_dir: Cache directory for embeddings
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_cache: Dict[str, np.ndarray] = {}
    
    def compute_embedding(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Compute embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Compute embedding
        if self.embedding_model:
            embedding = self.embedding_model.encode(text)
        else:
            # Fallback: simple hash-based embedding
            embedding = np.array([hash(text) % 1000] * 384, dtype=np.float32)
        
        # Cache
        self.embedding_cache[text_hash] = embedding
        
        return embedding
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_duplicates(
        self,
        texts: List[str],
    ) -> Dict[str, List[int]]:
        """
        Find duplicate texts using semantic similarity.
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary of text indices to duplicate indices
        """
        logger.info(f"Finding semantic duplicates in {len(texts)} texts...")
        
        # Compute embeddings
        embeddings = [self.compute_embedding(text) for text in texts]
        
        # Find duplicates
        duplicates: Dict[int, List[int]] = {}
        processed = set()
        
        for i in range(len(texts)):
            if i in processed:
                continue
            
            duplicates[i] = []
            
            for j in range(i + 1, len(texts)):
                if j in processed:
                    continue
                
                similarity = self.compute_similarity(embeddings[i], embeddings[j])
                
                if similarity >= self.similarity_threshold:
                    duplicates[i].append(j)
                    processed.add(j)
        
        # Filter out non-duplicates
        duplicates = {k: v for k, v in duplicates.items() if v}
        
        if duplicates:
            total_duplicates = sum(len(v) for v in duplicates.values())
            logger.info(f"Found {len(duplicates)} duplicate groups, {total_duplicates} duplicates")
        else:
            logger.info("No semantic duplicates found")
        
        return duplicates
    
    def filter_duplicates(
        self,
        texts: List[str],
    ) -> List[str]:
        """
        Filter out duplicate texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Filtered list without duplicates
        """
        duplicates = self.find_duplicates(texts)
        
        # Get indices to keep (first occurrence of each group)
        keep_indices = set(duplicates.keys())
        for dup_group in duplicates.values():
            keep_indices.update(dup_group)
        
        # Filter
        filtered = [texts[i] for i in range(len(texts)) if i not in keep_indices or i in duplicates]
        
        removed_count = len(texts) - len(filtered)
        logger.info(f"Filtered {removed_count} duplicate texts")
        
        return filtered

