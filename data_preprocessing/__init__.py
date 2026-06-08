"""Backward-compatible top-level alias for preprocessing modules.

Some training pipelines import ``data_preprocessing.*`` directly. In this repo,
the implementation lives under ``pipelines.data_preprocessing``.
"""

from pipelines.data_preprocessing.base import *  # noqa: F403
from pipelines.data_preprocessing.base import __all__ as _base_all

__all__ = list(_base_all)
