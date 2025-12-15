"""
Training health monitoring system.

Detects early divergence, stagnation, dead heads/neurons, and other
training anomalies to save weeks of wasted compute.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from collections import deque
import torch

logger = logging.getLogger(__name__)


class TrainingHealthMonitor:
    """
    Monitors training health and detects anomalies.
    
    Detects:
    - Early divergence
    - Stagnation
    - Dead heads/neurons
    - Loss plateaus
    - Learning rate issues
    """
    
    def __init__(
        self,
        divergence_threshold: float = 10.0,
        stagnation_window: int = 1000,
        stagnation_threshold: float = 0.01,
        dead_neuron_threshold: float = 1e-6,
    ):
        """
        Initialize health monitor.
        
        Args:
            divergence_threshold: Loss increase multiplier for divergence
            stagnation_window: Window for stagnation detection
            stagnation_threshold: Minimum loss change for stagnation
            dead_neuron_threshold: Activation threshold for dead neurons
        """
        self.divergence_threshold = divergence_threshold
        self.stagnation_window = stagnation_window
        self.stagnation_threshold = stagnation_threshold
        self.dead_neuron_threshold = dead_neuron_threshold
        
        self.loss_history = deque(maxlen=10000)
        self.alert_callbacks: List[Callable] = []
        
        self.divergence_detected = False
        self.stagnation_detected = False
    
    def check_loss(self, loss: float, step: int) -> Dict[str, Any]:
        """
        Check loss for anomalies.
        
        Args:
            loss: Current loss
            step: Current step
            
        Returns:
            Dictionary with health status
        """
        self.loss_history.append(loss)
        
        health_status = {
            "healthy": True,
            "warnings": [],
            "alerts": [],
        }
        
        # Check divergence
        if self._check_divergence():
            health_status["healthy"] = False
            health_status["alerts"].append("divergence_detected")
            self.divergence_detected = True
            logger.error(f"Step {step}: Training divergence detected!")
        
        # Check stagnation
        if self._check_stagnation():
            health_status["warnings"].append("stagnation_detected")
            self.stagnation_detected = True
            logger.warning(f"Step {step}: Training stagnation detected")
        
        # Check loss explosion
        if self._check_loss_explosion(loss):
            health_status["alerts"].append("loss_explosion")
            logger.error(f"Step {step}: Loss explosion detected!")
        
        return health_status
    
    def _check_divergence(self) -> bool:
        """Check for training divergence."""
        if len(self.loss_history) < 100:
            return False
        
        recent_losses = list(self.loss_history)[-100:]
        baseline_loss = np.median(list(self.loss_history)[:100])
        current_loss = recent_losses[-1]
        
        if current_loss > baseline_loss * self.divergence_threshold:
            return True
        
        return False
    
    def _check_stagnation(self) -> bool:
        """Check for training stagnation."""
        if len(self.loss_history) < self.stagnation_window:
            return False
        
        recent_losses = list(self.loss_history)[-self.stagnation_window:]
        loss_change = abs(recent_losses[-1] - recent_losses[0])
        
        if loss_change < self.stagnation_threshold:
            return True
        
        return False
    
    def _check_loss_explosion(self, current_loss: float) -> bool:
        """Check for loss explosion."""
        if len(self.loss_history) < 10:
            return False
        
        recent_median = np.median(list(self.loss_history)[-10:])
        
        if current_loss > recent_median * 5.0:
            return True
        
        return False
    
    def detect_dead_neurons(
        self,
        activations: Dict[str, torch.Tensor],
    ) -> Dict[str, List[int]]:
        """
        Detect dead neurons in activations.
        
        Args:
            activations: Dictionary of layer names to activation tensors
            
        Returns:
            Dictionary of layer names to dead neuron indices
        """
        dead_neurons = {}
        
        for layer_name, activation in activations.items():
            if not torch.is_tensor(activation):
                continue
            
            # Compute mean activation per neuron
            if activation.dim() > 2:
                # Flatten spatial dimensions
                activation = activation.view(activation.size(0), -1)
            
            mean_activations = activation.mean(dim=0)
            
            # Find dead neurons (mean activation below threshold)
            dead_mask = mean_activations < self.dead_neuron_threshold
            dead_indices = torch.where(dead_mask)[0].tolist()
            
            if dead_indices:
                dead_neurons[layer_name] = dead_indices
                logger.warning(
                    f"Dead neurons detected in {layer_name}: {len(dead_indices)} neurons"
                )
        
        return dead_neurons
    
    def detect_dead_attention_heads(
        self,
        attention_weights: torch.Tensor,
        threshold: float = 0.01,
    ) -> List[int]:
        """
        Detect dead attention heads.
        
        Args:
            attention_weights: Attention weights [batch, heads, seq_len, seq_len]
            threshold: Entropy threshold for dead heads
            
        Returns:
            List of dead head indices
        """
        # Compute entropy per head
        probs = torch.softmax(attention_weights, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        # Average over sequence
        mean_entropy = entropy.mean(dim=(0, 2, 3))
        
        # Find dead heads
        dead_mask = mean_entropy < threshold
        dead_heads = torch.where(dead_mask)[0].tolist()
        
        if dead_heads:
            logger.warning(f"Dead attention heads detected: {dead_heads}")
        
        return dead_heads
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get training health summary."""
        if not self.loss_history:
            return {"status": "unknown", "loss_count": 0}
        
        recent_losses = list(self.loss_history)[-100:]
        
        return {
            "status": "healthy" if not (self.divergence_detected or self.stagnation_detected) else "unhealthy",
            "loss_count": len(self.loss_history),
            "current_loss": recent_losses[-1] if recent_losses else 0.0,
            "mean_loss": np.mean(recent_losses) if recent_losses else 0.0,
            "divergence_detected": self.divergence_detected,
            "stagnation_detected": self.stagnation_detected,
        }
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, message)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

