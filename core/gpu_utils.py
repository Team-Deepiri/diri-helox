"""
GPU detection and memory utilities for diri-helox.

Device detection delegates to deepiri-modelkit (the shared platform library),
which is the single source of truth for GPU/CUDA/MPS detection across all
Deepiri services (cyrex, helox, etc.).

Helox-specific helpers (batch size recommendation, cache clearing, detailed
GPU info) are defined here because they are training-specific and do not
belong in the shared contract library.

Usage:
    from core.gpu_utils import detect_device, get_gpu_info, is_gpu_available
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core device detection — delegates to deepiri-modelkit
# ---------------------------------------------------------------------------

try:
    from deepiri_modelkit.utils.device import get_torch_device as _modelkit_get_torch_device
    from deepiri_modelkit.utils.device import get_device as _modelkit_get_device

    def detect_device(force: Optional[str] = None) -> torch.device:
        """
        Detect the optimal compute device.

        Delegates to deepiri-modelkit for consistent device detection across
        all Deepiri services.  Priority: CUDA → MPS (Apple Silicon) → CPU.

        Args:
            force: Optional override string — "cpu", "cuda", or "mps".
                   When set, attempts to use that device only.

        Returns:
            torch.device
        """
        if force:
            key = force.lower()
            if key == "cpu":
                return torch.device("cpu")
            if key == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if key == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning(
                "Forced device '%s' not available — falling back to auto-detection", force
            )
        return _modelkit_get_torch_device()  # type: ignore[no-any-return]

    def is_gpu_available() -> bool:
        """Return True if any GPU (CUDA or MPS) is available."""
        return _modelkit_get_device() != "cpu"  # type: ignore[no-any-return]

except ImportError:
    logger.warning(
        "deepiri-modelkit not installed — falling back to local GPU detection. "
        "Install with: pip install git+git@github.com:Team-Deepiri/deepiri-modelkit.git"
    )

    def detect_device(force: Optional[str] = None) -> torch.device:  # type: ignore[misc]
        if force:
            key = force.lower()
            if key == "cpu":
                return torch.device("cpu")
            if key == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if key == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning(
                "Forced device '%s' not available — falling back to auto-detection", force
            )
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def is_gpu_available() -> bool:  # type: ignore[misc]
        return torch.cuda.is_available() or torch.backends.mps.is_available()


# ---------------------------------------------------------------------------
# Helox-specific training utilities (not in modelkit)
# ---------------------------------------------------------------------------


def get_gpu_info(device: Optional[torch.device] = None) -> dict:
    """
    Return training-relevant hardware info for the given device.

    Args:
        device: torch.device to describe. If None, auto-detects.

    Returns:
        dict with keys: device, device_type, is_cuda, is_mps, is_cpu,
        and CUDA-specific memory/capability keys when applicable.
    """
    if device is None:
        device = detect_device()

    info: dict = {
        "device": str(device),
        "device_type": device.type,
        "is_cuda": device.type == "cuda",
        "is_mps": device.type == "mps",
        "is_cpu": device.type == "cpu",
    }

    if device.type == "cuda":
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 3),
                "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 3),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3),
            }
        )
    elif device.type == "mps":
        info["backend"] = "Metal Performance Shaders"

    return info


def clear_device_cache(device: torch.device) -> None:
    """Free GPU memory cache for the given device (call after each training epoch)."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        logger.debug("CUDA cache cleared")
    elif device.type == "mps":
        torch.mps.empty_cache()
        logger.debug("MPS cache cleared")


def recommend_batch_size(
    device: torch.device,
    model_size_mb: float,
    sequence_length: int,
    base_batch_size: int = 1,
) -> int:
    """
    Suggest a batch size based on available device memory.

    Heuristic for intent classifier training (~110 MB BERT-base):
    - GPU ≥ 40 GB  → 4× base
    - GPU ≥ 24 GB  → 2× base
    - GPU < 24 GB  → base
    - CPU/MPS      → ½ base (conservative)

    Args:
        device: The compute device.
        model_size_mb: Approximate model size in megabytes.
        sequence_length: Max sequence length for training.
        base_batch_size: Starting batch size.

    Returns:
        Recommended batch size (always ≥ 1).
    """
    if device.type == "cpu":
        return max(1, base_batch_size // 2)

    if device.type == "cuda":
        info = get_gpu_info(device)
        total_gb = info.get("memory_total_gb", 0)
        if total_gb >= 40:
            return base_batch_size * 4
        if total_gb >= 24:
            return base_batch_size * 2
        return base_batch_size

    # MPS or other
    return max(1, base_batch_size // 2)
