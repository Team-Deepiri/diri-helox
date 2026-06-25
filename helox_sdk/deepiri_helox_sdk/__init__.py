"""Installable surface from diri-helox for cross-repo consumers (e.g. Mudspeed)."""

from .device import resolve_mudspeed_torch_device

__all__ = ["resolve_mudspeed_torch_device"]

try:
    from . import evaluation as evaluation

    __all__.append("evaluation")
except ImportError:  # pragma: no cover - optional heavy deps at import time
    evaluation = None  # type: ignore[assignment,misc]
