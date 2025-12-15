"""
Token distribution monitoring system.

Detects token frequency drift, rare-token collapse alerts,
and distribution anomalies for maintaining model knowledge.
"""

import logging
from typing import Dict, Any, List
from collections import Counter, defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class TokenDistributionMonitor:
    """
    Monitors token distribution during training.
    
    Detects:
    - Token frequency drift
    - Rare-token collapse
    - Distribution anomalies
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        rare_token_threshold: float = 0.001,
        drift_threshold: float = 0.1,
    ):
        """
        Initialize token distribution monitor.
        
        Args:
            vocab_size: Vocabulary size
            rare_token_threshold: Threshold for rare tokens
            drift_threshold: Threshold for frequency drift
        """
        self.vocab_size = vocab_size
        self.rare_token_threshold = rare_token_threshold
        self.drift_threshold = drift_threshold
        
        self.token_frequencies: Dict[int, List[float]] = defaultdict(list)
        self.baseline_frequencies: Dict[int, float] = {}
    
    def update_token_frequencies(
        self,
        token_ids: List[int],
        step: int,
    ):
        """
        Update token frequency tracking.
        
        Args:
            token_ids: List of token IDs
            step: Current step
        """
        # Count tokens
        token_counts = Counter(token_ids)
        total_tokens = len(token_ids)
        
        # Update frequencies
        for token_id, count in token_counts.items():
            frequency = count / total_tokens
            self.token_frequencies[token_id].append(frequency)
    
    def establish_baseline(
        self,
        token_ids: List[int],
    ):
        """
        Establish baseline token frequencies.
        
        Args:
            token_ids: Baseline token IDs
        """
        token_counts = Counter(token_ids)
        total_tokens = len(token_ids)
        
        for token_id, count in token_counts.items():
            self.baseline_frequencies[token_id] = count / total_tokens
        
        logger.info(f"Established baseline for {len(self.baseline_frequencies)} tokens")
    
    def detect_rare_token_collapse(
        self,
        current_step: int,
    ) -> Dict[str, Any]:
        """
        Detect rare token collapse.
        
        Args:
            current_step: Current training step
            
        Returns:
            Collapse detection results
        """
        collapsed_tokens = []
        
        for token_id, frequencies in self.token_frequencies.items():
            if len(frequencies) < 10:
                continue
            
            # Check if frequency dropped below threshold
            recent_freq = np.mean(frequencies[-100:])
            baseline_freq = self.baseline_frequencies.get(token_id, 0.0)
            
            if baseline_freq > self.rare_token_threshold:
                if recent_freq < baseline_freq * 0.1:  # 90% drop
                    collapsed_tokens.append({
                        "token_id": token_id,
                        "baseline_freq": baseline_freq,
                        "current_freq": recent_freq,
                        "drop_ratio": recent_freq / baseline_freq if baseline_freq > 0 else 0.0,
                    })
        
        if collapsed_tokens:
            logger.warning(
                f"Rare token collapse detected: {len(collapsed_tokens)} tokens"
            )
        
        return {
            "collapse_detected": len(collapsed_tokens) > 0,
            "collapsed_tokens": collapsed_tokens[:20],  # Top 20
            "total_collapsed": len(collapsed_tokens),
        }
    
    def detect_frequency_drift(
        self,
        current_step: int,
    ) -> Dict[str, Any]:
        """
        Detect token frequency drift.
        
        Args:
            current_step: Current training step
            
        Returns:
            Drift detection results
        """
        drifted_tokens = []
        
        for token_id, frequencies in self.token_frequencies.items():
            if len(frequencies) < 10:
                continue
            
            recent_freq = np.mean(frequencies[-100:])
            baseline_freq = self.baseline_frequencies.get(token_id, 0.0)
            
            if baseline_freq > 0:
                drift_ratio = abs(recent_freq - baseline_freq) / baseline_freq
                
                if drift_ratio > self.drift_threshold:
                    drifted_tokens.append({
                        "token_id": token_id,
                        "baseline_freq": baseline_freq,
                        "current_freq": recent_freq,
                        "drift_ratio": drift_ratio,
                    })
        
        if drifted_tokens:
            logger.warning(f"Token frequency drift detected: {len(drifted_tokens)} tokens")
        
        return {
            "drift_detected": len(drifted_tokens) > 0,
            "drifted_tokens": drifted_tokens[:20],
            "total_drifted": len(drifted_tokens),
        }

