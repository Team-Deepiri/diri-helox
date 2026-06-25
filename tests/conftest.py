"""Shared pytest configuration for diri-helox."""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root must be importable as `core`, `data_sources`, `pipelines`, etc.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
