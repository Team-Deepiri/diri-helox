try:
    from deepiri_modelkit.ml.confidence import (
        ConfidenceLevel,
        ConfidenceSource,
        ConfidenceAttributes,
        ConfidenceCalculator,
        get_confidence_calculator,
    )
except ImportError:
    ConfidenceAttributes = None
    ConfidenceCalculator = None
    ConfidenceLevel = None
    ConfidenceSource = None
    ConfidenceAttributes = None
    ConfidenceCalculator = None
    get_confidence_calculator = None


__all__ = [
    "ConfidenceLevel",
    "ConfidenceSource",
    "ConfidenceAttributes",
    "ConfidenceCalculator",
    "get_confidence_calculator",
]
