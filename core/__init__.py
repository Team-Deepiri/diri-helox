"""Core utilities for LLM training infrastructure.

Public symbols are loaded **lazily** (``__getattr__``) so ``import core`` does not eagerly
import every submodule—reduces import-time side effects and keeps optional stacks isolated
until used (review feedback on package ``__init__`` import style).
"""

from __future__ import annotations

import importlib
from typing import Any

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

_EXPORTS: dict[str, tuple[str, str]] = {
    "DeviceManager": ("device_manager", "DeviceManager"),
    "get_optimal_device": ("device_manager", "get_optimal_device"),
    "detect_device": ("gpu_utils", "detect_device"),
    "get_gpu_info": ("gpu_utils", "get_gpu_info"),
    "is_gpu_available": ("gpu_utils", "is_gpu_available"),
    "resolve_mudspeed_torch_device": ("mudspeed_gpu", "resolve_mudspeed_torch_device"),
    "TrainingConfig": ("training_config", "TrainingConfig"),
    "ModelConfig": ("training_config", "ModelConfig"),
    "DataConfig": ("training_config", "DataConfig"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = _EXPORTS[name]
    mod = importlib.import_module(f"{__name__}.{mod_name}")
    val = getattr(mod, attr_name)
    globals()[name] = val
    return val


def __dir__() -> list[str]:  # noqa: D401
    return sorted(set(globals()) | set(__all__))
