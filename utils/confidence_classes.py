# Migrated to deepiri-modelkit. Re-exporting for backwards compatibility.
from deepiri_modelkit.ml import confidence as _confidence

ConfidenceLevel = _confidence.ConfidenceLevel

_OPTIONAL_EXPORTS = (
    "ConfidenceSource",
    "ConfidenceAttributes",
    "ConfidenceCalculator",
    "get_confidence_calculator",
)

for _name in _OPTIONAL_EXPORTS:
    if hasattr(_confidence, _name):
        globals()[_name] = getattr(_confidence, _name)

__all__ = ["ConfidenceLevel"] + [name for name in _OPTIONAL_EXPORTS if name in globals()]
