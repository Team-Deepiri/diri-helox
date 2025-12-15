"""
Precision-aware layer control system.

Enables per-layer precision, sensitive layers kept in FP32,
and attention vs MLP precision split for optimal mixed precision.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class PrecisionAwareLayerControl:
    """
    Controls precision per layer for mixed precision training.
    
    Features:
    - Per-layer precision control
    - Sensitive layers in FP32
    - Attention vs MLP precision split
    """
    
    def __init__(
        self,
        default_precision: torch.dtype = torch.float16,
        fp32_layers: Optional[List[str]] = None,
        attention_precision: Optional[torch.dtype] = None,
        mlp_precision: Optional[torch.dtype] = None,
    ):
        """
        Initialize precision controller.
        
        Args:
            default_precision: Default precision for layers
            fp32_layers: List of layer names to keep in FP32
            attention_precision: Precision for attention layers
            mlp_precision: Precision for MLP layers
        """
        self.default_precision = default_precision
        self.fp32_layers = fp32_layers or []
        self.attention_precision = attention_precision or default_precision
        self.mlp_precision = mlp_precision or default_precision
    
    def apply_precision_control(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Apply precision control to model.
        
        Args:
            model: Model to apply precision to
            
        Returns:
            Model with precision control
        """
        for name, module in model.named_modules():
            # Check if layer should be FP32
            if any(fp32_name in name for fp32_name in self.fp32_layers):
                module = module.float()
                logger.debug(f"Layer {name} set to FP32")
            
            # Apply attention/MLP precision
            elif "attention" in name.lower() or "attn" in name.lower():
                if self.attention_precision != self.default_precision:
                    module = self._convert_module_precision(module, self.attention_precision)
                    logger.debug(f"Layer {name} (attention) set to {self.attention_precision}")
            
            elif "mlp" in name.lower() or "feed_forward" in name.lower():
                if self.mlp_precision != self.default_precision:
                    module = self._convert_module_precision(module, self.mlp_precision)
                    logger.debug(f"Layer {name} (MLP) set to {self.mlp_precision}")
        
        return model
    
    def _convert_module_precision(
        self,
        module: nn.Module,
        dtype: torch.dtype,
    ) -> nn.Module:
        """Convert module to specified precision."""
        # Convert parameters
        for param in module.parameters():
            if param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                param.data = param.data.to(dtype)
        
        # Convert buffers
        for buffer in module.buffers():
            if buffer.dtype in (torch.float32, torch.float16, torch.bfloat16):
                buffer.data = buffer.data.to(dtype)
        
        return module
    
    def identify_sensitive_layers(
        self,
        model: nn.Module,
        gradient_norms: Dict[str, float],
    ) -> List[str]:
        """
        Identify layers sensitive to precision (high gradient variance).
        
        Args:
            model: Model
            gradient_norms: Dictionary of layer names to gradient norms
            
        Returns:
            List of sensitive layer names
        """
        if not gradient_norms:
            return []
        
        # Find layers with high gradient variance
        norms = list(gradient_norms.values())
        mean_norm = sum(norms) / len(norms) if norms else 0.0
        std_norm = (
            sum((n - mean_norm) ** 2 for n in norms) / len(norms)
        ) ** 0.5 if norms else 0.0
        
        threshold = mean_norm + 2 * std_norm
        
        sensitive_layers = [
            name for name, norm in gradient_norms.items()
            if norm > threshold
        ]
        
        if sensitive_layers:
            logger.info(f"Identified {len(sensitive_layers)} sensitive layers for FP32")
        
        return sensitive_layers

