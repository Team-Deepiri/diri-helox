"""
Numerical stability and loss scaling manager.

Provides dynamic loss scaling, NaN/Inf detection, and gradient overflow recovery
for stable mixed precision training.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class DynamicLossScaler:
    """
    Dynamic loss scaler for mixed precision training.
    
    Automatically adjusts loss scale based on gradient overflow detection.
    """
    
    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        scale_factor: float = 2.0,
        scale_window: int = 2000,
        min_scale: float = 1.0,
        max_scale: float = 2.0 ** 24,
    ):
        """
        Initialize dynamic loss scaler.
        
        Args:
            init_scale: Initial loss scale
            scale_factor: Factor to multiply/divide scale by
            scale_window: Steps before increasing scale
            min_scale: Minimum loss scale
            max_scale: Maximum loss scale
        """
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        self.steps_since_scale = 0
        self.overflow_count = 0
        self.total_steps = 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        return loss * self.scale
    
    def unscale_gradients(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        """
        Unscale gradients and check for overflow.
        
        Args:
            optimizer: Optimizer with scaled gradients
            
        Returns:
            True if overflow detected
        """
        # Check for inf/NaN in gradients
        has_overflow = False
        
        for param in optimizer.param_groups[0]["params"]:
            if param.grad is not None:
                # Check for inf
                if torch.isinf(param.grad).any():
                    has_overflow = True
                    break
                # Check for NaN
                if torch.isnan(param.grad).any():
                    has_overflow = True
                    break
        
        if has_overflow:
            self.overflow_count += 1
            self._decrease_scale()
            logger.warning(
                f"Gradient overflow detected. Decreasing scale to {self.scale:.2e}"
            )
        else:
            self.steps_since_scale += 1
            if self.steps_since_scale >= self.scale_window:
                self._increase_scale()
        
        self.total_steps += 1
        
        return has_overflow
    
    def _increase_scale(self):
        """Increase loss scale."""
        new_scale = self.scale * self.scale_factor
        self.scale = min(new_scale, self.max_scale)
        self.steps_since_scale = 0
        
        if self.scale < self.max_scale:
            logger.info(f"Increasing loss scale to {self.scale:.2e}")
    
    def _decrease_scale(self):
        """Decrease loss scale."""
        new_scale = self.scale / self.scale_factor
        self.scale = max(new_scale, self.min_scale)
        self.steps_since_scale = 0
    
    def get_state(self) -> Dict[str, Any]:
        """Get scaler state."""
        return {
            "scale": self.scale,
            "steps_since_scale": self.steps_since_scale,
            "overflow_count": self.overflow_count,
            "total_steps": self.total_steps,
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load scaler state."""
        self.scale = state.get("scale", self.scale)
        self.steps_since_scale = state.get("steps_since_scale", 0)
        self.overflow_count = state.get("overflow_count", 0)
        self.total_steps = state.get("total_steps", 0)


class NumericalStabilityMonitor:
    """
    Monitors numerical stability during training.
    
    Detects:
    - NaN/Inf in activations
    - Gradient overflow
    - Loss explosion
    """
    
    def __init__(self, alert_threshold: float = 1e6):
        """
        Initialize stability monitor.
        
        Args:
            alert_threshold: Loss threshold for alerting
        """
        self.alert_threshold = alert_threshold
        self.nan_count = 0
        self.inf_count = 0
        self.overflow_count = 0
    
    def check_activations(
        self,
        outputs: Dict[str, torch.Tensor],
        step: int,
    ) -> bool:
        """
        Check activations for NaN/Inf.
        
        Args:
            outputs: Model outputs dictionary
            step: Current training step
            
        Returns:
            True if issues detected
        """
        issues_detected = False
        
        for key, tensor in outputs.items():
            if torch.is_tensor(tensor):
                if torch.isnan(tensor).any():
                    self.nan_count += 1
                    logger.error(f"NaN detected in {key} at step {step}")
                    issues_detected = True
                
                if torch.isinf(tensor).any():
                    self.inf_count += 1
                    logger.error(f"Inf detected in {key} at step {step}")
                    issues_detected = True
        
        return issues_detected
    
    def check_loss(self, loss: torch.Tensor, step: int) -> bool:
        """
        Check loss for anomalies.
        
        Args:
            loss: Loss tensor
            step: Current training step
            
        Returns:
            True if loss is anomalous
        """
        loss_value = loss.item()
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"Loss is NaN/Inf at step {step}")
            return True
        
        if loss_value > self.alert_threshold:
            logger.warning(
                f"Loss explosion detected: {loss_value:.2e} at step {step}"
            )
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, int]:
        """Get stability statistics."""
        return {
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "overflow_count": self.overflow_count,
        }

