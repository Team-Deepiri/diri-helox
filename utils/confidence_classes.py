"""
Redesigned Confidence Classes System
Provides structured confidence assessment with multiple attributes
"""
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


class ConfidenceLevel(str, Enum):
    """Confidence level categories"""
    VERY_HIGH = "very_high"  # 0.9-1.0
    HIGH = "high"  # 0.75-0.9
    MEDIUM = "medium"  # 0.5-0.75
    LOW = "low"  # 0.25-0.5
    VERY_LOW = "very_low"  # 0.0-0.25


class ConfidenceSource(str, Enum):
    """Sources of confidence information"""
    MODEL_PREDICTION = "model_prediction"
    TRAINING_DATA_COVERAGE = "training_data_coverage"
    FEATURE_QUALITY = "feature_quality"
    CONTEXT_MATCH = "context_match"
    HISTORICAL_ACCURACY = "historical_accuracy"
    ENSEMBLE_AGREEMENT = "ensemble_agreement"


@dataclass
class ConfidenceAttributes:
    """
    Comprehensive confidence attributes for model predictions
    
    Attributes:
        base_score: Raw model confidence score (0.0-1.0)
        level: Categorical confidence level
        sources: Dictionary of confidence sources and their contributions
        uncertainty: Measure of prediction uncertainty
        calibration: How well-calibrated the prediction is
        reliability: Overall reliability score
        explanation: Human-readable explanation
    """
    base_score: float
    level: ConfidenceLevel
    sources: Dict[str, float]
    uncertainty: float
    calibration: float
    reliability: float
    explanation: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "base_score": self.base_score,
            "level": self.level.value,
            "sources": self.sources,
            "uncertainty": self.uncertainty,
            "calibration": self.calibration,
            "reliability": self.reliability,
            "explanation": self.explanation
        }


class ConfidenceCalculator:
    """
    Calculate comprehensive confidence scores with multiple attributes
    """
    
    def __init__(self):
        self.confidence_thresholds = {
            ConfidenceLevel.VERY_HIGH: 0.9,
            ConfidenceLevel.HIGH: 0.75,
            ConfidenceLevel.MEDIUM: 0.5,
            ConfidenceLevel.LOW: 0.25,
            ConfidenceLevel.VERY_LOW: 0.0
        }
    
    def calculate_confidence(
        self,
        model_probabilities: np.ndarray,
        top_k_probs: Optional[List[float]] = None,
        training_coverage: Optional[float] = None,
        feature_quality: Optional[float] = None,
        context_match: Optional[float] = None,
        historical_accuracy: Optional[Dict[int, float]] = None
    ) -> ConfidenceAttributes:
        """
        Calculate comprehensive confidence attributes
        
        Args:
            model_probabilities: Model output probabilities for all classes
            top_k_probs: Top-k probabilities (for ensemble agreement)
            training_coverage: How well training data covers this example (0-1)
            feature_quality: Quality of input features (0-1)
            context_match: How well context matches expected patterns (0-1)
            historical_accuracy: Historical accuracy per class {class_id: accuracy}
        
        Returns:
            ConfidenceAttributes object
        """
        # Base score: maximum probability
        base_score = float(np.max(model_probabilities))
        
        # Uncertainty: entropy-based measure
        entropy = -np.sum(model_probabilities * np.log(model_probabilities + 1e-10))
        max_entropy = np.log(len(model_probabilities))
        uncertainty = float(entropy / max_entropy)  # Normalized to [0, 1]
        
        # Calibration: difference between top-2 probabilities (margin)
        sorted_probs = np.sort(model_probabilities)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
        calibration = float(margin)  # Higher margin = better calibration
        
        # Source contributions
        sources = {}
        
        # Model prediction contribution
        sources[ConfidenceSource.MODEL_PREDICTION.value] = base_score
        
        # Training data coverage
        if training_coverage is not None:
            sources[ConfidenceSource.TRAINING_DATA_COVERAGE.value] = training_coverage
        else:
            sources[ConfidenceSource.TRAINING_DATA_COVERAGE.value] = 0.7  # Default moderate
        
        # Feature quality
        if feature_quality is not None:
            sources[ConfidenceSource.FEATURE_QUALITY.value] = feature_quality
        else:
            sources[ConfidenceSource.FEATURE_QUALITY.value] = 0.8  # Default good
        
        # Context match
        if context_match is not None:
            sources[ConfidenceSource.CONTEXT_MATCH.value] = context_match
        else:
            sources[ConfidenceSource.CONTEXT_MATCH.value] = 0.7  # Default moderate
        
        # Historical accuracy
        if historical_accuracy:
            predicted_class = int(np.argmax(model_probabilities))
            hist_acc = historical_accuracy.get(predicted_class, 0.7)
            sources[ConfidenceSource.HISTORICAL_ACCURACY.value] = hist_acc
        else:
            sources[ConfidenceSource.HISTORICAL_ACCURACY.value] = 0.7  # Default moderate
        
        # Ensemble agreement (if top_k_probs provided)
        if top_k_probs:
            agreement = float(np.std(top_k_probs))  # Lower std = higher agreement
            sources[ConfidenceSource.ENSEMBLE_AGREEMENT.value] = 1.0 - min(agreement, 1.0)
        else:
            sources[ConfidenceSource.ENSEMBLE_AGREEMENT.value] = 0.7  # Default moderate
        
        # Weighted reliability score
        weights = {
            ConfidenceSource.MODEL_PREDICTION.value: 0.4,
            ConfidenceSource.TRAINING_DATA_COVERAGE.value: 0.15,
            ConfidenceSource.FEATURE_QUALITY.value: 0.15,
            ConfidenceSource.CONTEXT_MATCH.value: 0.1,
            ConfidenceSource.HISTORICAL_ACCURACY.value: 0.1,
            ConfidenceSource.ENSEMBLE_AGREEMENT.value: 0.1
        }
        
        reliability = sum(
            sources[source] * weights.get(source, 0.0)
            for source in sources
        )
        
        # Adjust reliability based on uncertainty and calibration
        reliability = reliability * (1.0 - uncertainty * 0.3) * (0.7 + calibration * 0.3)
        reliability = max(0.0, min(1.0, reliability))
        
        # Determine confidence level
        level = self._get_confidence_level(reliability)
        
        # Generate explanation
        explanation = self._generate_explanation(
            reliability, level, sources, uncertainty, calibration
        )
        
        return ConfidenceAttributes(
            base_score=base_score,
            level=level,
            sources=sources,
            uncertainty=uncertainty,
            calibration=calibration,
            reliability=reliability,
            explanation=explanation
        )
    
    def _get_confidence_level(self, reliability: float) -> ConfidenceLevel:
        """Get confidence level from reliability score"""
        if reliability >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif reliability >= 0.75:
            return ConfidenceLevel.HIGH
        elif reliability >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif reliability >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_explanation(
        self,
        reliability: float,
        level: ConfidenceLevel,
        sources: Dict[str, float],
        uncertainty: float,
        calibration: float
    ) -> str:
        """Generate human-readable explanation"""
        parts = []
        
        # Main confidence statement
        parts.append(f"Confidence: {level.value.replace('_', ' ').title()} ({reliability:.2%})")
        
        # Key factors
        key_factors = []
        if sources.get(ConfidenceSource.MODEL_PREDICTION.value, 0) > 0.8:
            key_factors.append("strong model prediction")
        if sources.get(ConfidenceSource.TRAINING_DATA_COVERAGE.value, 0) > 0.8:
            key_factors.append("good training coverage")
        if uncertainty < 0.3:
            key_factors.append("low uncertainty")
        if calibration > 0.5:
            key_factors.append("clear class separation")
        
        if key_factors:
            parts.append(f"Key factors: {', '.join(key_factors)}")
        
        # Concerns
        concerns = []
        if uncertainty > 0.6:
            concerns.append("high uncertainty")
        if calibration < 0.2:
            concerns.append("unclear class separation")
        if sources.get(ConfidenceSource.TRAINING_DATA_COVERAGE.value, 0) < 0.5:
            concerns.append("limited training coverage")
        
        if concerns:
            parts.append(f"Concerns: {', '.join(concerns)}")
        
        return ". ".join(parts) + "."
    
    def should_accept_prediction(
        self,
        confidence: ConfidenceAttributes,
        min_reliability: float = 0.7,
        min_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    ) -> Tuple[bool, str]:
        """
        Determine if prediction should be accepted based on confidence
        
        Returns:
            (should_accept, reason)
        """
        level_order = {
            ConfidenceLevel.VERY_LOW: 0,
            ConfidenceLevel.LOW: 1,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.HIGH: 3,
            ConfidenceLevel.VERY_HIGH: 4
        }
        
        if confidence.reliability < min_reliability:
            return False, f"Reliability {confidence.reliability:.2%} below threshold {min_reliability:.2%}"
        
        if level_order[confidence.level] < level_order[min_level]:
            return False, f"Confidence level {confidence.level.value} below required {min_level.value}"
        
        return True, "Confidence meets requirements"


# Singleton instance
_confidence_calculator = None


def get_confidence_calculator() -> ConfidenceCalculator:
    """Get singleton ConfidenceCalculator instance"""
    global _confidence_calculator
    if _confidence_calculator is None:
        _confidence_calculator = ConfidenceCalculator()
    return _confidence_calculator

