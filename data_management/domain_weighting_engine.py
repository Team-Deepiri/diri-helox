"""
Domain weighting engine.

Provides per-domain sampling ratios, runtime domain rebalancing,
and domain-aware training for steering model personality.
"""

import logging
import random
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class DomainWeightingEngine:
    """
    Manages domain weighting for training data.
    
    Features:
    - Per-domain sampling ratios
    - Runtime domain rebalancing
    - Domain-aware sampling
    """
    
    def __init__(
        self,
        domain_weights: Optional[Dict[str, float]] = None,
        auto_rebalance: bool = True,
    ):
        """
        Initialize domain weighting engine.
        
        Args:
            domain_weights: Initial domain weights
            auto_rebalance: Enable automatic rebalancing
        """
        self.domain_weights = domain_weights or {}
        self.auto_rebalance = auto_rebalance
        
        self.domain_counts: Dict[str, int] = defaultdict(int)
        self.domain_losses: Dict[str, List[float]] = defaultdict(list)
    
    def assign_domain(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Assign domain to text sample.
        
        Args:
            text: Text sample
            metadata: Optional metadata with domain info
            
        Returns:
            Domain name
        """
        if metadata and "domain" in metadata:
            return metadata["domain"]
        
        # Simple heuristic-based assignment
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ["code", "function", "class", "import"]):
            return "code"
        elif any(keyword in text_lower for keyword in ["question", "answer", "explain"]):
            return "qa"
        elif any(keyword in text_lower for keyword in ["article", "news", "blog"]):
            return "articles"
        else:
            return "general"
    
    def get_sampling_weight(
        self,
        domain: str,
    ) -> float:
        """
        Get sampling weight for domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Sampling weight
        """
        return self.domain_weights.get(domain, 1.0)
    
    def should_sample(
        self,
        domain: str,
    ) -> bool:
        """
        Determine if sample should be included based on domain weight.
        
        Args:
            domain: Domain name
            
        Returns:
            True if should sample
        """
        weight = self.get_sampling_weight(domain)
        return random.random() < weight
    
    def update_domain_metrics(
        self,
        domain: str,
        loss: float,
    ):
        """
        Update domain metrics.
        
        Args:
            domain: Domain name
            loss: Loss value
        """
        self.domain_counts[domain] += 1
        self.domain_losses[domain].append(loss)
    
    def rebalance_weights(
        self,
        target_ratios: Optional[Dict[str, float]] = None,
    ):
        """
        Rebalance domain weights.
        
        Args:
            target_ratios: Target domain ratios
        """
        if not self.auto_rebalance:
            return
        
        if target_ratios:
            # Set to target ratios
            total_weight = sum(target_ratios.values())
            self.domain_weights = {
                domain: weight / total_weight
                for domain, weight in target_ratios.items()
            }
        else:
            # Auto-balance based on loss
            domain_avg_losses = {
                domain: np.mean(losses) if losses else 1.0
                for domain, losses in self.domain_losses.items()
            }
            
            if domain_avg_losses:
                # Weight inversely proportional to loss (higher loss = more samples)
                total_loss = sum(domain_avg_losses.values())
                self.domain_weights = {
                    domain: loss / total_loss
                    for domain, loss in domain_avg_losses.items()
                }
        
        logger.info(f"Rebalanced domain weights: {self.domain_weights}")

