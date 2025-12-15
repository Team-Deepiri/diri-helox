"""
Multi-objective training support.

Enables weighted losses, auxiliary objectives (format compliance, etc.),
and complex training objectives beyond simple next-token prediction.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)


class MultiObjectiveLoss:
    """
    Multi-objective loss function.
    
    Combines multiple loss objectives with weights.
    """
    
    def __init__(
        self,
        objectives: Dict[str, Callable],
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-objective loss.
        
        Args:
            objectives: Dictionary of objective names to loss functions
            weights: Optional weights for each objective
        """
        self.objectives = objectives
        self.weights = weights or {name: 1.0 for name in objectives.keys()}
    
    def __call__(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss.
        
        Args:
            predictions: Dictionary of prediction tensors
            targets: Dictionary of target tensors
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0
        
        for name, loss_fn in self.objectives.items():
            if name in predictions and name in targets:
                loss = loss_fn(predictions[name], targets[name])
                weighted_loss = loss * self.weights.get(name, 1.0)
                
                losses[name] = loss
                losses[f"{name}_weighted"] = weighted_loss
                total_loss += weighted_loss
        
        losses["total"] = total_loss
        
        return losses


class FormatComplianceLoss(nn.Module):
    """
    Loss for format compliance (e.g., JSON structure, code syntax).
    
    Encourages model to follow specific output formats.
    """
    
    def __init__(self, format_type: str = "json"):
        """
        Initialize format compliance loss.
        
        Args:
            format_type: Type of format (json, code, etc.)
        """
        super().__init__()
        self.format_type = format_type
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute format compliance loss.
        
        Args:
            predictions: Predicted tokens
            targets: Target tokens
            
        Returns:
            Format compliance loss
        """
        # For now, use a simple structural loss
        # In practice, this would parse and validate format
        
        if self.format_type == "json":
            # Encourage balanced braces/brackets
            # This is a simplified version
            return self._json_structure_loss(predictions, targets)
        else:
            # Default: no additional loss
            return torch.tensor(0.0, device=predictions.device)
    
    def _json_structure_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute JSON structure loss."""
        # Simplified: encourage matching structure
        # In practice, would parse and validate JSON
        
        # For now, return small regularization term
        return torch.tensor(0.01, device=predictions.device)


class MultiObjectiveTrainer:
    """
    Trainer that supports multiple training objectives.
    
    Enables:
    - Weighted loss combinations
    - Auxiliary objectives
    - Format compliance
    - Task-specific objectives
    """
    
    def __init__(
        self,
        model: nn.Module,
        objectives: Dict[str, Callable],
        objective_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize multi-objective trainer.
        
        Args:
            model: Model to train
            objectives: Dictionary of objective functions
            objective_weights: Weights for each objective
        """
        self.model = model
        self.loss_fn = MultiObjectiveLoss(objectives, objective_weights)
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss.
        
        Args:
            outputs: Model outputs
            targets: Training targets
            
        Returns:
            Dictionary of losses
        """
        return self.loss_fn(outputs, targets)
    
    def update_objective_weights(
        self,
        weights: Dict[str, float],
    ):
        """Update objective weights dynamically."""
        self.loss_fn.weights.update(weights)
        logger.info(f"Updated objective weights: {weights}")

