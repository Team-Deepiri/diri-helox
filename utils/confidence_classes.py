try:
    from deepiri_modelkit.ml.confidence import ConfidenceLevel
except ImportError:
    ConfidenceLevel = None


__all__ = [
    "ConfidenceLevel",
]
