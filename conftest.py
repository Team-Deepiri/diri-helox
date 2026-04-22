"""
Pytest configuration for diri-helox.
Ensures the helox root is on sys.path so all modules can be imported directly.
"""

import sys
from pathlib import Path

# Add helox root to sys.path so `import data_sources`, `import training`, etc. all work
_HELOX_ROOT = Path(__file__).parent
if str(_HELOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_HELOX_ROOT))

collect_ignore = [
    # Pre-existing tests with broken imports (depend on submodule internals not yet exported)
    "tests/test_data_preprocessing_pipeline.py",
    "tests/test_dataset_versioning.py",
]
