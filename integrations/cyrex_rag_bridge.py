"""
Cyrex RAG Bridge for Helox Training.

Provides seamless integration between Helox training and Cyrex RAG pipeline.
Handles connection, error handling, and fallback behavior.
"""

import logging
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CyrexRAGBridge:
    """
    Bridge to Cyrex RAG pipeline for training integration.
    
    Handles:
    - Connection to Cyrex RAG service
    - Fallback when RAG unavailable
    - Error handling and retries
    """
    
    def __init__(
        self,
        cyrex_rag_pipeline=None,
        fallback_enabled: bool = True,
    ):
        """
        Initialize Cyrex RAG bridge.
        
        Args:
            cyrex_rag_pipeline: Cyrex RAG pipeline instance
            fallback_enabled: Enable fallback when RAG unavailable
        """
        self.rag_pipeline = cyrex_rag_pipeline
        self.fallback_enabled = fallback_enabled
        self.available = self.rag_pipeline is not None
        
        if self.available:
            logger.info("Cyrex RAG bridge initialized - RAG features enabled")
        else:
            logger.warning("Cyrex RAG bridge initialized without pipeline - RAG features disabled")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        task_type_filter: Optional[str] = None,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from Cyrex RAG.
        
        Args:
            query: Query string
            top_k: Number of results
            task_type_filter: Optional task type filter
            rerank: Whether to rerank results
            
        Returns:
            List of retrieved chunks
        """
        if not self.available:
            if self.fallback_enabled:
                logger.debug(f"RAG unavailable - returning empty results for query: {query[:50]}")
                return []
            else:
                raise RuntimeError("RAG pipeline not available")
        
        try:
            results = self.rag_pipeline.retrieve(
                query=query,
                top_k=top_k,
                task_type_filter=task_type_filter,
                rerank=rerank,
            )
            
            # Normalize results format
            normalized_results = []
            for r in results:
                if isinstance(r, dict):
                    # Extract text content
                    text = (
                        r.get("challenge_text") or
                        r.get("content") or
                        r.get("text") or
                        str(r)
                    )
                    normalized_results.append({
                        "content": text,
                        "score": r.get("score", 0.0),
                        "metadata": r.get("metadata", {}),
                    })
                else:
                    normalized_results.append({
                        "content": str(r),
                        "score": 0.0,
                        "metadata": {},
                    })
            
            return normalized_results
        
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            if self.fallback_enabled:
                return []
            else:
                raise
    
    def is_available(self) -> bool:
        """Check if RAG is available."""
        return self.available


def create_cyrex_rag_bridge(
    cyrex_path: Optional[Path] = None,
    auto_discover: bool = True,
) -> CyrexRAGBridge:
    """
    Create Cyrex RAG bridge with auto-discovery.
    
    Args:
        cyrex_path: Path to Cyrex directory
        auto_discover: Auto-discover Cyrex if path not provided
        
    Returns:
        CyrexRAGBridge instance
    """
    rag_pipeline = None
    
    # Try to import and initialize Cyrex RAG
    try:
        if cyrex_path:
            import sys
            sys.path.insert(0, str(cyrex_path))
        
        # Try importing Cyrex RAG
        try:
            from app.integrations.rag_pipeline import initialize_rag_system
            rag_pipeline = initialize_rag_system()
            logger.info("Successfully initialized Cyrex RAG pipeline")
        except ImportError:
            if auto_discover:
                # Try to find Cyrex
                current_path = Path(__file__).parent.parent.parent
                cyrex_candidates = [
                    current_path / "diri-cyrex",
                    current_path.parent / "diri-cyrex",
                    Path("/app/diri-cyrex"),  # Docker path
                ]
                
                for candidate in cyrex_candidates:
                    if candidate.exists():
                        import sys
                        sys.path.insert(0, str(candidate))
                        try:
                            from app.integrations.rag_pipeline import initialize_rag_system
                            rag_pipeline = initialize_rag_system()
                            logger.info(f"Auto-discovered and initialized Cyrex RAG from {candidate}")
                            break
                        except Exception:
                            continue
            else:
                logger.debug("Cyrex RAG not found - RAG features will be disabled")
        
    except Exception as e:
        logger.warning(f"Failed to initialize Cyrex RAG: {e}")
    
    return CyrexRAGBridge(cyrex_rag_pipeline=rag_pipeline)

