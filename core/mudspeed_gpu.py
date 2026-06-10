"""Shared GPU device resolution entry point for Mudspeed (in-repo).

For **pip-installed** consumers, use the ``deepiri-helox-sdk`` package under
``helox_sdk/`` in this repository (same policy via ``core.gpu_utils`` →
``deepiri-gpu-utils``).

When diri-helox is on ``PYTHONPATH``, Mudspeed may import this module so policy
stays aligned with ``core.gpu_utils``.
"""

from __future__ import annotations

from typing import Optional

import torch

from .gpu_utils import detect_device


def resolve_mudspeed_torch_device(force: Optional[str] = None) -> torch.device:
    """Return the compute device for Mudspeed using Helox’s shared detection."""
    return detect_device(force=force)


__all__ = ["resolve_mudspeed_torch_device"]
