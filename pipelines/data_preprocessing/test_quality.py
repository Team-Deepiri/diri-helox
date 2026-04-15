"""Shim: canonical quality tests live in ``deepiri-dataset-processor/tests/test_quality.py``.

This file loads that module so ``pytest pipelines/data_preprocessing`` exercises the same
cases without duplicating test logic. The real implementations are tested via
``deepiri_dataset_processor`` (see submodule ``tests/``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SHIM_DIR = Path(__file__).resolve().parent
_DSP_ROOT = _SHIM_DIR.parents[2] / "deepiri-dataset-processor"
_CANONICAL = _DSP_ROOT / "tests" / "test_quality.py"
_MODULE_NAME = "deepiri_dataset_processor.tests._test_quality_via_shim"


def _load_canonical() -> None:
    if not _CANONICAL.is_file():
        raise FileNotFoundError(
            f"Expected canonical tests at {_CANONICAL}. "
            "Initialize the git submodule: git submodule update --init deepiri-dataset-processor"
        )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _CANONICAL)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load quality tests from {_CANONICAL}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    skip = frozenset(
        {"__name__", "__file__", "__cached__", "__loader__", "__package__", "__spec__"}
    )
    g = globals()
    for key, val in vars(mod).items():
        if key in skip:
            continue
        g[key] = val


_load_canonical()
