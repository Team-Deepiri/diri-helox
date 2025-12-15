"""
Device management utilities with automatic CPU/GPU detection.

This module provides intelligent device selection and management for training,
automatically detecting and utilizing available hardware resources.
"""

import logging
import os
from typing import Optional, Tuple
import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Manages device selection and configuration for training.
    
    Automatically detects available hardware and selects optimal device.
    Supports both CPU and GPU (CUDA/MPS) training with fallback logic.
    """
    
    def __init__(self, force_device: Optional[str] = None):
        """
        Initialize device manager.
        
        Args:
            force_device: Optional device override ('cpu', 'cuda', 'mps')
        """
        self.force_device = force_device
        self.device = self._detect_optimal_device()
        self.device_info = self._get_device_info()
        
        logger.info(f"DeviceManager initialized: {self.device_info}")
    
    def _detect_optimal_device(self) -> torch.device:
        """
        Automatically detect and select the optimal device.
        
        Priority:
        1. Forced device (if specified)
        2. CUDA (if available)
        3. MPS (Apple Silicon, if available)
        4. CPU (fallback)
        
        Returns:
            torch.device: Selected device
        """
        if self.force_device:
            device_str = self.force_device.lower()
            if device_str == "cpu":
                return torch.device("cpu")
            elif device_str == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif device_str == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                logger.warning(
                    f"Forced device '{device_str}' not available, falling back to auto-detection"
                )
        
        # Auto-detection
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return device
        
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple Silicon (MPS) available")
            return device
        
        logger.info("Using CPU (no GPU detected)")
        return torch.device("cpu")
    
    def _get_device_info(self) -> dict:
        """Get detailed information about the selected device."""
        info = {
            "device": str(self.device),
            "device_type": self.device.type,
            "is_cuda": self.device.type == "cuda",
            "is_mps": self.device.type == "mps",
            "is_cpu": self.device.type == "cpu",
        }
        
        if self.device.type == "cuda":
            info.update({
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
                "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9,
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            })
        elif self.device.type == "mps":
            info["backend"] = "Metal Performance Shaders"
        
        return info
    
    def get_device(self) -> torch.device:
        """Get the selected device."""
        return self.device
    
    def get_device_info(self) -> dict:
        """Get device information dictionary."""
        return self.device_info.copy()
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self.device.type in ("cuda", "mps")
    
    def get_batch_size_recommendation(
        self,
        model_size_mb: float,
        sequence_length: int,
        base_batch_size: int = 1,
    ) -> int:
        """
        Recommend batch size based on available device memory.
        
        Args:
            model_size_mb: Model size in megabytes
            sequence_length: Sequence length for training
            base_batch_size: Base batch size to start from
            
        Returns:
            Recommended batch size
        """
        if self.device.type == "cpu":
            # Conservative batch size for CPU
            return max(1, base_batch_size // 4)
        
        if self.device.type == "cuda":
            total_memory_gb = self.device_info.get("memory_total_gb", 0)
            if total_memory_gb >= 40:  # A100 or similar
                return base_batch_size * 4
            elif total_memory_gb >= 24:  # RTX 4090
                return base_batch_size * 2
            else:
                return base_batch_size
        
        # MPS or other
        return base_batch_size
    
    def clear_cache(self):
        """Clear device memory cache if applicable."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
        elif self.device.type == "mps":
            torch.mps.empty_cache()
            logger.debug("MPS cache cleared")


def get_optimal_device(force_device: Optional[str] = None) -> torch.device:
    """
    Convenience function to get optimal device.
    
    Args:
        force_device: Optional device override
        
    Returns:
        torch.device: Optimal device
    """
    manager = DeviceManager(force_device=force_device)
    return manager.get_device()


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        dict: Device information
    """
    manager = DeviceManager()
    return manager.get_device_info()

