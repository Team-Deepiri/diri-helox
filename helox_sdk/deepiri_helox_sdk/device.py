"""Torch device resolution aligned with diri-helox ``core.gpu_utils`` policy.

Delegates to ``deepiri-gpu-utils`` (same stack as ``core.gpu_utils``). Kept in this
package so services can depend on a **pip-installable** artifact from the Helox
repo without putting the whole monorepo on ``PYTHONPATH``.
"""

from __future__ import annotations

from typing import Optional

import torch


def resolve_mudspeed_torch_device(force: Optional[str] = None) -> torch.device:
    """Return the compute device using the shared Deepiri torch device policy."""
    from deepiri_gpu_utils.torch_device import resolve_torch_device

    policy = force.lower() if force else "auto"
    decision = resolve_torch_device(policy)
    return torch.device(decision.device)
