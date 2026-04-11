"""
Utility modules for Helox training pipelines.

Submodules such as ``utils.dataset_versioning`` import without ``deepiri_modelkit``.
When that package is installed, semantic helpers are re-exported here.
"""

__all__: list[str] = []

try:
    from deepiri_modelkit.ml.confidence import ConfidenceLevel
    from deepiri_modelkit.ml.semantic import get_semantic_analyzer
except ImportError:
    pass
else:
    __all__ = ["get_semantic_analyzer", "ConfidenceLevel"]
