"""
Training-aware scheduling and curriculum learning.

Provides adaptive batch sizing, dynamic sequence length curriculum,
and difficulty-based sample progression for improved convergence.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
import torch

logger = logging.getLogger(__name__)


class CurriculumLearningScheduler:
    """
    Implements curriculum learning for training.
    
    Features:
    - Difficulty scoring per sample
    - Easy → hard progression
    - Topic balancing
    - Dynamic sequence length
    """
    
    def __init__(
        self,
        initial_seq_len: int = 512,
        max_seq_len: int = 8192,
        growth_rate: float = 1.1,
        difficulty_window: int = 1000,
    ):
        """
        Initialize curriculum scheduler.
        
        Args:
            initial_seq_len: Initial sequence length
            max_seq_len: Maximum sequence length
            growth_rate: Rate of sequence length growth
            difficulty_window: Window for difficulty calculation
        """
        self.initial_seq_len = initial_seq_len
        self.max_seq_len = max_seq_len
        self.growth_rate = growth_rate
        self.difficulty_window = difficulty_window
        
        self.current_seq_len = initial_seq_len
        self.step = 0
        self.difficulty_scores: deque = deque(maxlen=difficulty_window)
    
    def get_current_sequence_length(self, step: int) -> int:
        """
        Get current sequence length based on curriculum.
        
        Args:
            step: Current training step
            
        Returns:
            Current sequence length
        """
        self.step = step
        
        # Gradually increase sequence length
        growth_steps = 1000
        if step % growth_steps == 0 and step > 0:
            new_len = int(self.current_seq_len * self.growth_rate)
            self.current_seq_len = min(new_len, self.max_seq_len)
            logger.info(f"Step {step}: Increased sequence length to {self.current_seq_len}")
        
        return self.current_seq_len
    
    def score_difficulty(
        self,
        text: str,
        loss: Optional[float] = None,
    ) -> float:
        """
        Score difficulty of a sample.
        
        Args:
            text: Sample text
            loss: Optional loss value for this sample
            
        Returns:
            Difficulty score (0-1)
        """
        # Base difficulty on text characteristics
        word_count = len(text.split())
        char_count = len(text)
        
        # Normalize
        word_score = min(word_count / 1000, 1.0)
        char_score = min(char_count / 5000, 1.0)
        
        # Combine with loss if available
        if loss is not None:
            loss_score = min(loss / 10.0, 1.0)
            difficulty = (word_score + char_score + loss_score) / 3.0
        else:
            difficulty = (word_score + char_score) / 2.0
        
        self.difficulty_scores.append(difficulty)
        
        return difficulty
    
    def should_include_sample(
        self,
        difficulty: float,
        step: int,
    ) -> bool:
        """
        Determine if sample should be included based on curriculum.
        
        Args:
            difficulty: Sample difficulty
            step: Current step
            
        Returns:
            True if sample should be included
        """
        # Gradually increase difficulty threshold
        initial_threshold = 0.3
        final_threshold = 1.0
        total_steps = 100000
        
        threshold = initial_threshold + (final_threshold - initial_threshold) * min(
            step / total_steps,
            1.0,
        )
        
        return difficulty <= threshold
    
    def get_adaptive_batch_size(
        self,
        base_batch_size: int,
        step: int,
        memory_usage: Optional[float] = None,
    ) -> int:
        """
        Get adaptive batch size based on training progress.
        
        Args:
            base_batch_size: Base batch size
            step: Current step
            memory_usage: Optional current memory usage (0-1)
            
        Returns:
            Adaptive batch size
        """
        # Start with smaller batches, increase over time
        if step < 1000:
            return base_batch_size // 2
        elif step < 10000:
            return base_batch_size
        else:
            # Can increase if memory allows
            if memory_usage and memory_usage < 0.8:
                return int(base_batch_size * 1.5)
            return base_batch_size


class AdaptiveBatchScheduler:
    """
    Adaptive batch size scheduling.
    
    Adjusts batch size based on:
    - Training progress
    - Memory usage
    - Gradient norms
    - Loss stability
    """
    
    def __init__(
        self,
        initial_batch_size: int = 2,
        max_batch_size: int = 32,
        min_batch_size: int = 1,
    ):
        """
        Initialize adaptive batch scheduler.
        
        Args:
            initial_batch_size: Initial batch size
            max_batch_size: Maximum batch size
            min_batch_size: Minimum batch size
        """
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        
        self.current_batch_size = initial_batch_size
        self.gradient_history: deque = deque(maxlen=100)
    
    def update_batch_size(
        self,
        gradient_norm: float,
        memory_usage: float,
        step: int,
    ) -> int:
        """
        Update batch size based on metrics.
        
        Args:
            gradient_norm: Current gradient norm
            memory_usage: Current memory usage (0-1)
            step: Current step
            
        Returns:
            New batch size
        """
        self.gradient_history.append(gradient_norm)
        
        # Increase if gradients are stable and memory allows
        if len(self.gradient_history) > 10:
            recent_stability = np.std(list(self.gradient_history)[-10:])
            
            if recent_stability < 0.1 and memory_usage < 0.8:
                new_size = min(self.current_batch_size * 2, self.max_batch_size)
                if new_size != self.current_batch_size:
                    logger.info(f"Step {step}: Increased batch size to {new_size}")
                    self.current_batch_size = new_size
            elif memory_usage > 0.9:
                new_size = max(self.current_batch_size // 2, self.min_batch_size)
                if new_size != self.current_batch_size:
                    logger.warning(f"Step {step}: Decreased batch size to {new_size} (memory)")
                    self.current_batch_size = new_size
        
        return self.current_batch_size

