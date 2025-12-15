"""
Gradient monitoring and clipping system.

Provides per-step gradient norm logging, adaptive clipping thresholds,
and gradient explosion detection.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class GradientMonitoringSystem:
    """
    Monitors and manages gradients during training.
    
    Tracks:
    - Per-layer gradient norms
    - Global gradient norm
    - Gradient distribution
    - Clipping statistics
    """
    
    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
        adaptive_clipping: bool = True,
        history_size: int = 1000,
    ):
        """
        Initialize gradient monitoring system.
        
        Args:
            max_norm: Maximum gradient norm for clipping
            norm_type: Norm type (2.0 for L2, float('inf') for L-inf)
            adaptive_clipping: Enable adaptive clipping threshold
            history_size: Size of gradient norm history
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.adaptive_clipping = adaptive_clipping
        self.history_size = history_size
        
        self.gradient_norm_history = deque(maxlen=history_size)
        self.per_layer_norms: Dict[str, List[float]] = {}
        self.clip_count = 0
        self.total_steps = 0
    
    def compute_gradient_norms(
        self,
        model: nn.Module,
    ) -> Dict[str, float]:
        """
        Compute gradient norms for all parameters.
        
        Args:
            model: Model with gradients
            
        Returns:
            Dictionary of layer names to gradient norms
        """
        norms = {}
        total_norm = 0.0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(self.norm_type)
                norms[name] = param_norm.item()
                total_norm += param_norm.item() ** self.norm_type
        
        total_norm = total_norm ** (1.0 / self.norm_type)
        norms["total"] = total_norm
        
        return norms
    
    def clip_gradients(
        self,
        model: nn.Module,
        step: int,
    ) -> Dict[str, Any]:
        """
        Clip gradients and return statistics.
        
        Args:
            model: Model with gradients
            step: Current training step
            
        Returns:
            Statistics dictionary
        """
        # Compute norms before clipping
        norms_before = self.compute_gradient_norms(model)
        total_norm_before = norms_before["total"]
        
        # Adaptive clipping threshold
        if self.adaptive_clipping:
            if len(self.gradient_norm_history) > 10:
                recent_median = np.median(list(self.gradient_norm_history)[-100:])
                adaptive_max_norm = max(self.max_norm, recent_median * 1.5)
            else:
                adaptive_max_norm = self.max_norm
        else:
            adaptive_max_norm = self.max_norm
        
        # Clip gradients
        if total_norm_before > adaptive_max_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                adaptive_max_norm,
                norm_type=self.norm_type,
            )
            self.clip_count += 1
            was_clipped = True
        else:
            was_clipped = False
        
        # Compute norms after clipping
        norms_after = self.compute_gradient_norms(model)
        total_norm_after = norms_after["total"]
        
        # Update history
        self.gradient_norm_history.append(total_norm_after)
        
        # Track per-layer norms
        for name, norm in norms_before.items():
            if name != "total":
                if name not in self.per_layer_norms:
                    self.per_layer_norms[name] = []
                self.per_layer_norms[name].append(norm)
        
        self.total_steps += 1
        
        # Log if clipped
        if was_clipped:
            logger.warning(
                f"Step {step}: Gradients clipped "
                f"(norm: {total_norm_before:.4f} -> {total_norm_after:.4f}, "
                f"threshold: {adaptive_max_norm:.4f})"
            )
        
        return {
            "gradient_norm": total_norm_after,
            "gradient_norm_before": total_norm_before,
            "was_clipped": was_clipped,
            "clip_threshold": adaptive_max_norm,
            "per_layer_norms": {
                k: v for k, v in norms_before.items() if k != "total"
            },
        }
    
    def detect_exploding_gradients(
        self,
        threshold_multiplier: float = 10.0,
    ) -> bool:
        """
        Detect if gradients are exploding.
        
        Args:
            threshold_multiplier: Multiplier for median to detect explosion
            
        Returns:
            True if explosion detected
        """
        if len(self.gradient_norm_history) < 10:
            return False
        
        recent_norms = list(self.gradient_norm_history)[-100:]
        median_norm = np.median(recent_norms)
        current_norm = recent_norms[-1]
        
        if current_norm > median_norm * threshold_multiplier:
            logger.error(
                f"Gradient explosion detected: "
                f"current={current_norm:.4f}, median={median_norm:.4f}"
            )
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gradient statistics."""
        if not self.gradient_norm_history:
            return {}
        
        recent_norms = list(self.gradient_norm_history)[-100:]
        
        return {
            "current_norm": recent_norms[-1] if recent_norms else 0.0,
            "mean_norm": np.mean(recent_norms) if recent_norms else 0.0,
            "median_norm": np.median(recent_norms) if recent_norms else 0.0,
            "max_norm": np.max(recent_norms) if recent_norms else 0.0,
            "min_norm": np.min(recent_norms) if recent_norms else 0.0,
            "clip_count": self.clip_count,
            "clip_rate": self.clip_count / max(self.total_steps, 1),
            "total_steps": self.total_steps,
        }
    
    def get_top_layers_by_gradient_norm(self, top_k: int = 10) -> List[tuple]:
        """
        Get top layers by average gradient norm.
        
        Args:
            top_k: Number of top layers to return
            
        Returns:
            List of (layer_name, avg_norm) tuples
        """
        layer_avg_norms = [
            (name, np.mean(norms))
            for name, norms in self.per_layer_norms.items()
        ]
        
        layer_avg_norms.sort(key=lambda x: x[1], reverse=True)
        
        return layer_avg_norms[:top_k]

