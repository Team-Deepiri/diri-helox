"""Core utilities for LLM training infrastructure.

Symbols are **explicitly** re-exported here so static analysis (e.g. CodeQL) and ``__all__``
agree on what the package defines.
"""

from __future__ import annotations

from .device_manager import DeviceManager, get_optimal_device
from .gpu_utils import detect_device, get_gpu_info, is_gpu_available
from .mudspeed_gpu import resolve_mudspeed_torch_device
from .training_config import DataConfig, ModelConfig, TrainingConfig

__all__ = [
    "DeviceManager",
    "detect_device",
    "get_gpu_info",
    "is_gpu_available",
    "get_optimal_device",
    "resolve_mudspeed_torch_device",
    "TrainingConfig",
    "ModelConfig",
    "DataConfig",
]
