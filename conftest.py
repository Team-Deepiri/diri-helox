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

# Prevent pytest from collecting the root __init__.py which has relative imports
# that can't be resolved when pytest treats the root as a standalone module.
collect_ignore = ["__init__.py"]
