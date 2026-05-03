"""Installable surface from diri-helox for cross-repo consumers (e.g. Mudspeed)."""

from .device import resolve_mudspeed_torch_device

__all__ = ["resolve_mudspeed_torch_device"]
