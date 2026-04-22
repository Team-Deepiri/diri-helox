"""
Device management — thin wrapper around core.gpu_utils.

All GPU detection logic lives in gpu_utils.py and is imported here so that
existing code using DeviceManager continues to work without changes.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from .gpu_utils import (
    clear_device_cache,
    detect_device,
    get_gpu_info,
    is_gpu_available,
    recommend_batch_size,
)

logger = logging.getLogger(__name__)

# Backward-compatibility shim:
# legacy tests and integrations patch `core.device_manager.resolve_torch_device`.
try:
    from deepiri_gpu_utils.torch_device import resolve_torch_device
except ImportError:  # pragma: no cover - optional dependency
    resolve_torch_device = None  # type: ignore[misc,assignment]


class DeviceManager:
    """
    Manages device selection and configuration for training.

    Delegates all detection and inspection logic to core.gpu_utils.
    Supports CPU, CUDA, and MPS (Apple Silicon).
    """

    def __init__(self, force_device: Optional[str] = None) -> None:
        self.force_device = force_device
        if force_device is None and resolve_torch_device is not None:
            decision = resolve_torch_device("auto")
            self.device = torch.device(decision.device)
        else:
            self.device = detect_device(force=force_device)
        self.device_info = get_gpu_info(self.device)
        logger.info("DeviceManager initialized: %s", self.device_info)

    def get_device(self) -> torch.device:
        return self.device

    def get_device_info(self) -> dict:
        return self.device_info.copy()

    def is_gpu_available(self) -> bool:
        return is_gpu_available()

    def get_batch_size_recommendation(
        self,
        model_size_mb: float,
        sequence_length: int,
        base_batch_size: int = 1,
    ) -> int:
        return recommend_batch_size(
            self.device,
            model_size_mb=model_size_mb,
            sequence_length=sequence_length,
            base_batch_size=base_batch_size,
        )

    def clear_cache(self) -> None:
        clear_device_cache(self.device)


# ---------------------------------------------------------------------------
# Module-level convenience functions (kept for backwards compatibility)
# ---------------------------------------------------------------------------


def get_optimal_device(force_device: Optional[str] = None) -> torch.device:
    return detect_device(force=force_device)


def get_device_info() -> dict:
    return get_gpu_info()
