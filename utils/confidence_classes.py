# Migrated to deepiri-modelkit. Re-exporting for backwards compatibility.
try:
    from deepiri_modelkit.ml import confidence as _confidence
except ImportError:
    _confidence = None

ConfidenceLevel = getattr(_confidence, "ConfidenceLevel", None)
ConfidenceSource = getattr(_confidence, "ConfidenceSource", None)
ConfidenceAttributes = getattr(_confidence, "ConfidenceAttributes", None)
ConfidenceCalculator = getattr(_confidence, "ConfidenceCalculator", None)
get_confidence_calculator = getattr(_confidence, "get_confidence_calculator", None)

__all__ = [
    "ConfidenceLevel",
    "ConfidenceSource",
    "ConfidenceAttributes",
    "ConfidenceCalculator",
    "get_confidence_calculator",
]
