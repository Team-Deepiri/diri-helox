"""
Centralized logging adapter for diri-helox.

Wires the deepiri-logger submodule (deps/deepiri-logger/python) as the primary
logger so every module emits structured JSON following the Deepiri golden schema
with PII masking.  Falls back gracefully to stdlib logging if the submodule is
not present (e.g. CI environments that skip submodule init).

Usage in any module::

    from helox_logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add deepiri-logger Python package to sys.path
# ---------------------------------------------------------------------------
_PKG_PATH = Path(__file__).parent / "deps" / "deepiri-logger" / "python"
if _PKG_PATH.exists() and str(_PKG_PATH) not in sys.path:
    sys.path.insert(0, str(_PKG_PATH))

# ---------------------------------------------------------------------------
# Initialise and export
# ---------------------------------------------------------------------------
try:
    from deepiri_logger import get_logger, init as _init  # type: ignore[import]

    _init(service_name="diri-helox")
except Exception:  # pragma: no cover

    def get_logger(name: str | None = None):  # type: ignore[misc]
        """Fallback stdlib logger when deepiri-logger is unavailable."""
        return logging.getLogger(name or __name__)


__all__ = ["get_logger"]
