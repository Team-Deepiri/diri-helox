"""
RAG-aware training integration.

Enables retriever-aware training with context window packing,
retrieval-conditioned prompts, and chunk boundary awareness
for RAG-native model training.
"""

import logging
from typing import Dict, Any, List, Optional
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class RAGAwareTrainingIntegrator:
    """
    Integrates RAG capabilities into training.
    
    Features:
    - Context window packing with retrieved context
    - Retrieval-conditioned prompts
    - Chunk boundary awareness
    - Integration with Cyrex RAG system
    """
    
    def __init__(
        self,
        rag_pipeline=None,  # Will be injected from Cyrex
        max_context_length: int = 8192,
        retrieval_ratio: float = 0.3,  # 30% of context for retrieval
    ):
        """
        Initialize RAG-aware training integrator.
        
        Args:
            rag_pipeline: RAG pipeline instance (from Cyrex)
            max_context_length: Maximum context length
            retrieval_ratio: Ratio of context for retrieved content
        """
        self.rag_pipeline = rag_pipeline
        self.max_context_length = max_context_length
        self.retrieval_ratio = retrieval_ratio
    
    def pack_context_with_retrieval(
        self,
        query: str,
        base_text: str,
        tokenizer_manager,
        num_retrievals: int = 3,
    ) -> Dict[str, Any]:
        """
        Pack context with retrieved content.
        
        Args:
            query: Query for retrieval
            base_text: Base text to include
            tokenizer_manager: Tokenizer manager
            num_retrievals: Number of retrieved chunks
            
        Returns:
            Packed context dictionary
        """
        # Retrieve relevant chunks from Cyrex RAG pipeline
        retrieved_chunks = []
        if self.rag_pipeline:
            try:
                # Handle both direct RAG pipeline and bridge
                if hasattr(self.rag_pipeline, 'retrieve'):
                    # Direct RAG pipeline
                    results = self.rag_pipeline.retrieve(
                        query=query,
                        top_k=num_retrievals,
                        task_type_filter=None,
                        rerank=True,
                    )
                elif hasattr(self.rag_pipeline, 'is_available') and self.rag_pipeline.is_available():
                    # RAG bridge
                    results = self.rag_pipeline.retrieve(
                        query=query,
                        top_k=num_retrievals,
                    )
                else:
                    results = []
                
                # Extract text from results
                for r in results:
                    if isinstance(r, dict):
                        chunk_text = (
                            r.get("challenge_text") or
                            r.get("content") or
                            r.get("text") or
                            ""
                        )
                    else:
                        chunk_text = str(r)
                    
                    if chunk_text:
                        retrieved_chunks.append(chunk_text)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                # Fallback: use empty retrieval
                retrieved_chunks = []
        
        # Calculate token budgets
        max_tokens = self.max_context_length
        retrieval_tokens = int(max_tokens * self.retrieval_ratio)
        base_tokens = max_tokens - retrieval_tokens
        
        # Tokenize base text
        base_token_ids = tokenizer_manager.encode(base_text, add_bos=True, add_eos=False)
        if len(base_token_ids) > base_tokens:
            base_token_ids = base_token_ids[:base_tokens]
        
        # Tokenize and pack retrieved chunks
        retrieval_token_ids = []
        for chunk in retrieved_chunks:
            chunk_ids = tokenizer_manager.encode(chunk, add_bos=False, add_eos=False)
            if len(retrieval_token_ids) + len(chunk_ids) <= retrieval_tokens:
                retrieval_token_ids.extend(chunk_ids)
            else:
                break
        
        # Combine: [retrieval] [base]
        combined_ids = retrieval_token_ids + base_token_ids
        
        # Truncate if necessary
        if len(combined_ids) > max_tokens:
            combined_ids = combined_ids[:max_tokens]
        
        # Create mask: 0 for retrieval, 1 for base
        retrieval_mask = [0] * len(retrieval_token_ids) + [1] * len(base_token_ids)
        retrieval_mask = retrieval_mask[:len(combined_ids)]
        
        return {
            "input_ids": combined_ids,
            "retrieval_mask": retrieval_mask,
            "retrieved_chunks": retrieved_chunks,
            "base_text": base_text,
        }
    
    def create_retrieval_conditioned_prompt(
        self,
        instruction: str,
        retrieved_context: List[str],
        tokenizer_manager,
    ) -> Dict[str, Any]:
        """
        Create retrieval-conditioned prompt.
        
        Args:
            instruction: Instruction text
            retrieved_context: Retrieved context chunks
            tokenizer_manager: Tokenizer manager
            
        Returns:
            Formatted prompt with retrieval context
        """
        # Format: "Context: [retrieved]\n\nInstruction: [instruction]\n\nResponse:"
        context_text = "\n\n".join(retrieved_context)
        formatted = f"Context:\n{context_text}\n\nInstruction: {instruction}\n\nResponse:"
        
        # Tokenize
        token_ids = tokenizer_manager.encode(formatted, add_bos=True, add_eos=True)
        
        # Create mask: 0 for context+instruction, 1 for response (to be generated)
        context_instruction_text = formatted.split("Response:")[0] + "Response:"
        context_instruction_ids = tokenizer_manager.encode(
            context_instruction_text,
            add_bos=True,
            add_eos=False,
        )
        
        mask = [0] * len(context_instruction_ids) + [1] * (len(token_ids) - len(context_instruction_ids))
        mask = mask[:len(token_ids)]
        
        return {
            "input_ids": token_ids,
            "instruction_mask": mask,
            "formatted_text": formatted,
        }
    
    def mark_chunk_boundaries(
        self,
        text: str,
        chunk_size: int = 512,
        tokenizer_manager=None,
    ) -> List[int]:
        """
        Mark chunk boundaries in text.
        
        Args:
            text: Input text
            chunk_size: Chunk size in tokens
            tokenizer_manager: Tokenizer manager
            
        Returns:
            List of boundary positions
        """
        if not tokenizer_manager:
            return []
        
        # Tokenize
        token_ids = tokenizer_manager.encode(text, add_bos=True, add_eos=True)
        
        # Mark boundaries
        boundaries = []
        for i in range(0, len(token_ids), chunk_size):
            boundaries.append(i)
        
        return boundaries

