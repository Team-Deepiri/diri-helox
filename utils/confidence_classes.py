# Migrated to deepiri-modelkit. Re-exporting for backwards compatibility.
try:
    from deepiri_modelkit.ml import confidence as _confidence
except ImportError:
    _confidence = None

if _confidence is not None and hasattr(_confidence, "ConfidenceLevel"):
    ConfidenceLevel = _confidence.ConfidenceLevel
else:
    ConfidenceLevel = None

_OPTIONAL_EXPORTS = (
    "ConfidenceSource",
    "ConfidenceAttributes",
    "ConfidenceCalculator",
    "get_confidence_calculator",
)

for _name in _OPTIONAL_EXPORTS:
    if _confidence is not None and hasattr(_confidence, _name):
        globals()[_name] = getattr(_confidence, _name)

__all__ = ["ConfidenceLevel"] + [name for name in _OPTIONAL_EXPORTS if name in globals()]
