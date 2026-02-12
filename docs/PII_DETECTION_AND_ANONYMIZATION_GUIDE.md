# PII Detection & Anonymization Pipeline for Helox

## Table of Contents

1. [Introduction](#introduction)
2. [Trivia Warmups - Understanding PII](#trivia-warmups---understanding-pii)
3. [Current Architecture](#current-architecture)
4. [Implementation Strategy](#implementation-strategy)
5. [Beginner Implementation](#beginner-implementation)
6. [Advanced Implementation](#advanced-implementation)
7. [Integration with Helox Pipeline](#integration-with-helox-pipeline)
8. [Testing and Validation](#testing-and-validation)
9. [Production Deployment](#production-deployment)
10. [Monitoring and Compliance](#monitoring-and-compliance)

---

## Introduction

This guide provides a comprehensive walkthrough for implementing a production-grade PII (Personally Identifiable Information) detection and anonymization pipeline in Helox. The pipeline ensures that sensitive data is identified and anonymized before it enters the training data, protecting user privacy and ensuring regulatory compliance.

### What You'll Learn

- Fundamentals of PII and why it matters in ML training
- Current state of PII handling in the Deepiri platform
- Step-by-step implementation from basic to advanced
- Integration with Helox's data preprocessing pipeline
- Production deployment and monitoring strategies

### Prerequisites

- Basic understanding of Python
- Familiarity with data pipelines
- Knowledge of the Deepiri platform architecture (Cyrex and Helox)
- Understanding of privacy regulations (GDPR, CCPA, HIPAA)

---

## Trivia Warmups - Understanding PII

### What is PII?

Personally Identifiable Information (PII) is any data that can be used to identify, contact, or locate a specific individual. In the context of machine learning, PII in training data poses several risks:

1. **Privacy Violations**: Training models on PII can lead to memorization and potential data leakage
2. **Regulatory Compliance**: GDPR, CCPA, HIPAA require protection of personal data
3. **Security Risks**: Exposed PII can be used for identity theft or fraud
4. **Legal Liability**: Mishandling PII can result in significant fines and legal action

### Types of PII

PII can be categorized into several types:

**Direct Identifiers** (uniquely identify an individual):
- Social Security Numbers (SSN)
- Credit card numbers
- Passport numbers
- Driver's license numbers
- Email addresses (in some contexts)
- Phone numbers
- IP addresses

**Indirect Identifiers** (can identify when combined):
- Date of birth
- ZIP code
- Gender
- Occupation
- Medical record numbers
- Biometric data

**Quasi-Identifiers** (identify when combined with other data):
- Age + ZIP code + gender
- Job title + company + location

### Why PII Detection is Hard

1. **Format Variations**: Phone numbers can be written as (555) 123-4567, 555-123-4567, or 5551234567
2. **Context Dependency**: "John" might be a name or just a common word
3. **False Positives**: Patterns that look like PII but aren't (e.g., "123-45-6789" in a math problem)
4. **False Negatives**: PII written in non-standard formats
5. **Multilingual Support**: Different languages have different PII formats

### Current State in Deepiri Platform

The Deepiri platform currently has basic PII detection in two places:

1. **Cyrex Guardrails** (`diri-cyrex/app/core/guardrails.py`): Detects PII using regex patterns for safety checks
2. **Data Transformer** (`diri-cyrex/app/core/realtime_data_pipeline.py`): Basic redaction using simple regex patterns

**Current Limitations**:
- Only handles SSN, credit cards, and emails
- Uses simple regex (no context awareness)
- No confidence scoring
- No anonymization strategies (only redaction)
- No audit trail
- Limited to English patterns

---

## Current Architecture

### Data Flow Overview

```
User Input / Agent Interactions
    ↓
Cyrex RealtimeDataPipeline
    ├── Validation Layer
    ├── Transformation Layer (basic PII redaction)
    │   └── DataTransformer._redact_pii()
    │
    ├── Route 1: Helox Training
    │   ├── Redis Stream: pipeline.helox-training.raw
    │   └── Redis Stream: pipeline.helox-training.structured
    │
    └── Route 2: Cyrex Runtime
        └── Memory Manager / Synapse

Helox RealtimeIngestion
    ├── Consumes from Redis Streams
    ├── Validates records
    └── Writes to JSONL files
        ├── data/datasets/pipeline/raw/
        └── data/datasets/pipeline/structured/

Helox Preprocessing Pipeline
    ├── DataLoadingStage
    ├── DataCleaningStage
    ├── DataValidationStage
    ├── DataRoutingStage
    ├── LabelValidationStage
    └── DataTransformationStage
```

### Current PII Redaction Implementation

The current implementation in `DataTransformer` uses simple regex patterns:

```python
_PII_PATTERNS = [
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN_REDACTED]'),
    (re.compile(r'\b\d{16}\b'), '[CARD_REDACTED]'),
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL_REDACTED]'),
]
```

**Issues with Current Approach**:
1. Only handles three PII types
2. No validation (e.g., credit card Luhn algorithm)
3. No context awareness
4. Fixed replacement strings (not configurable)
5. No detection confidence scores
6. No audit logging

---

## Implementation Strategy

### Design Principles

1. **Defense in Depth**: Multiple layers of PII detection
2. **Configurable**: Support different anonymization strategies
3. **Auditable**: Log all PII detection and anonymization events
4. **Performant**: Minimal impact on pipeline throughput
5. **Extensible**: Easy to add new PII types and detection methods

### Architecture Layers

```
Layer 1: Detection
    ├── Rule-based (regex patterns)
    ├── ML-based (Presidio Analyzer)
    └── Custom patterns

Layer 2: Validation
    ├── Format validation (Luhn, date ranges)
    ├── Context analysis
    └── Confidence scoring

Layer 3: Anonymization
    ├── Redaction (replace with placeholder)
    ├── Masking (partial replacement)
    ├── Hashing (one-way)
    ├── Encryption (reversible)
    └── Pseudonymization (consistent replacement)

Layer 4: Audit & Compliance
    ├── Detection logs
    ├── Anonymization logs
    ├── Compliance reports
    └── Metrics tracking
```

### Integration Points

1. **Cyrex Pipeline**: Enhanced PII detection before routing to Helox
2. **Helox Ingestion**: Additional validation and detection
3. **Helox Preprocessing**: Dedicated PII anonymization stage
4. **Training Pipeline**: Final check before model training

---

## Beginner Implementation

### Step 1: Install Dependencies

First, install the required packages. Presidio is already listed in `pyproject.toml` but commented out in `requirements.txt`.

```bash
# In diri-helox directory
poetry add presidio-analyzer presidio-anonymizer
# Or with pip
pip install presidio-analyzer presidio-anonymizer
```

**Note**: Presidio requires spaCy models. Install the English model:

```bash
python -m spacy download en_core_web_lg
```

### Step 2: Create Basic PII Detector

Create a new file: `diri-helox/data_safety/pii_detector.py`

```python
"""
Basic PII Detection Module

Provides rule-based and ML-based PII detection capabilities.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of PII that can be detected"""
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    IP_ADDRESS = "IP_ADDRESS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    PERSON_NAME = "PERSON_NAME"
    ADDRESS = "ADDRESS"
    DRIVER_LICENSE = "DRIVER_LICENSE"
    PASSPORT = "PASSPORT"


@dataclass
class PIIDetection:
    """Represents a detected PII instance"""
    pii_type: PIIType
    value: str
    start_pos: int
    end_pos: int
    confidence: float
    context: Optional[str] = None


class BasicPIIDetector:
    """
    Basic PII detector using regex patterns.
    
    This is a simple implementation for learning purposes.
    Production systems should use Presidio Analyzer for better accuracy.
    """
    
    def __init__(self):
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> List[Tuple[re.Pattern, PIIType, float]]:
        """Load regex patterns for PII detection"""
        patterns = [
            # SSN: 123-45-6789 or 123 45 6789
            (re.compile(r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b'), PIIType.SSN, 0.8),
            
            # Credit card: 16 digits, optionally with dashes/spaces
            (re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'), PIIType.CREDIT_CARD, 0.7),
            
            # Email
            (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), PIIType.EMAIL, 0.9),
            
            # Phone: (555) 123-4567, 555-123-4567, 5551234567
            (re.compile(r'\b\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'), PIIType.PHONE, 0.7),
            
            # IP address
            (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), PIIType.IP_ADDRESS, 0.6),
        ]
        return patterns
    
    def detect(self, text: str) -> List[PIIDetection]:
        """
        Detect PII in text using regex patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected PII instances
        """
        detections = []
        
        for pattern, pii_type, confidence in self.patterns:
            for match in pattern.finditer(text):
                detections.append(PIIDetection(
                    pii_type=pii_type,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence,
                    context=self._extract_context(text, match.start(), match.end())
                ))
        
        # Remove overlapping detections (keep higher confidence)
        detections = self._remove_overlaps(detections)
        
        return detections
    
    def _extract_context(self, text: str, start: int, end: int, context_size: int = 20) -> str:
        """Extract context around detected PII"""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end]
    
    def _remove_overlaps(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """Remove overlapping detections, keeping higher confidence ones"""
        if not detections:
            return []
        
        # Sort by confidence (descending)
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        non_overlapping = []
        
        for detection in sorted_detections:
            overlaps = False
            for existing in non_overlapping:
                if self._overlaps(detection, existing):
                    overlaps = True
                    break
            if not overlaps:
                non_overlapping.append(detection)
        
        return non_overlapping
    
    def _overlaps(self, d1: PIIDetection, d2: PIIDetection) -> bool:
        """Check if two detections overlap"""
        return not (d1.end_pos <= d2.start_pos or d2.end_pos <= d1.start_pos)
```

### Step 3: Create Basic Anonymizer

Create: `diri-helox/data_safety/pii_anonymizer.py`

```python
"""
Basic PII Anonymization Module

Provides simple anonymization strategies for detected PII.
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from .pii_detector import PIIDetection, PIIType

logger = logging.getLogger(__name__)


class AnonymizationStrategy(str, Enum):
    """Strategies for anonymizing PII"""
    REDACT = "redact"  # Replace with placeholder
    MASK = "mask"  # Partial replacement (e.g., 123-45-****)
    HASH = "hash"  # One-way hash
    PSEUDONYMIZE = "pseudonymize"  # Consistent replacement


class BasicPIIAnonymizer:
    """
    Basic PII anonymizer with multiple strategies.
    """
    
    def __init__(self, default_strategy: AnonymizationStrategy = AnonymizationStrategy.REDACT):
        self.default_strategy = default_strategy
        self.placeholders = {
            PIIType.SSN: "[SSN_REDACTED]",
            PIIType.CREDIT_CARD: "[CARD_REDACTED]",
            PIIType.EMAIL: "[EMAIL_REDACTED]",
            PIIType.PHONE: "[PHONE_REDACTED]",
            PIIType.IP_ADDRESS: "[IP_REDACTED]",
            PIIType.DATE_OF_BIRTH: "[DOB_REDACTED]",
            PIIType.PERSON_NAME: "[NAME_REDACTED]",
            PIIType.ADDRESS: "[ADDRESS_REDACTED]",
        }
    
    def anonymize(
        self,
        text: str,
        detections: List[PIIDetection],
        strategy: Optional[AnonymizationStrategy] = None
    ) -> str:
        """
        Anonymize detected PII in text.
        
        Args:
            text: Original text
            detections: List of PII detections
            strategy: Anonymization strategy (uses default if not specified)
            
        Returns:
            Anonymized text
        """
        if not detections:
            return text
        
        strategy = strategy or self.default_strategy
        
        # Sort detections by position (reverse order to maintain indices)
        sorted_detections = sorted(detections, key=lambda x: x.start_pos, reverse=True)
        
        anonymized_text = text
        
        for detection in sorted_detections:
            replacement = self._get_replacement(detection, strategy)
            anonymized_text = (
                anonymized_text[:detection.start_pos] +
                replacement +
                anonymized_text[detection.end_pos:]
            )
        
        return anonymized_text
    
    def _get_replacement(
        self,
        detection: PIIDetection,
        strategy: AnonymizationStrategy
    ) -> str:
        """Get replacement text based on strategy"""
        if strategy == AnonymizationStrategy.REDACT:
            return self.placeholders.get(detection.pii_type, "[PII_REDACTED]")
        
        elif strategy == AnonymizationStrategy.MASK:
            return self._mask_value(detection)
        
        elif strategy == AnonymizationStrategy.HASH:
            return self._hash_value(detection)
        
        elif strategy == AnonymizationStrategy.PSEUDONYMIZE:
            return self._pseudonymize_value(detection)
        
        else:
            return self.placeholders.get(detection.pii_type, "[PII_REDACTED]")
    
    def _mask_value(self, detection: PIIDetection) -> str:
        """Mask PII value (show partial, hide rest)"""
        value = detection.value
        
        if detection.pii_type == PIIType.SSN:
            # Show last 4: ***-**-1234
            if len(value) >= 4:
                return f"***-**-{value[-4:]}"
            return "***-**-****"
        
        elif detection.pii_type == PIIType.CREDIT_CARD:
            # Show last 4: ****-****-****-1234
            if len(value.replace("-", "").replace(" ", "")) >= 4:
                digits = value.replace("-", "").replace(" ", "")
                return f"****-****-****-{digits[-4:]}"
            return "****-****-****-****"
        
        elif detection.pii_type == PIIType.EMAIL:
            # Show domain: ***@example.com
            if "@" in value:
                local, domain = value.split("@", 1)
                return f"***@{domain}"
            return "***@***"
        
        else:
            # Default: show first and last character
            if len(value) > 2:
                return f"{value[0]}***{value[-1]}"
            return "***"
    
    def _hash_value(self, detection: PIIDetection) -> str:
        """Hash PII value (one-way)"""
        hash_obj = hashlib.sha256(detection.value.encode())
        hash_hex = hash_obj.hexdigest()[:16]  # First 16 chars
        return f"[HASHED_{detection.pii_type.value}_{hash_hex}]"
    
    def _pseudonymize_value(self, detection: PIIType) -> str:
        """Pseudonymize PII (consistent replacement)"""
        # Simple pseudonymization using hash
        hash_obj = hashlib.sha256(detection.value.encode())
        hash_hex = hash_obj.hexdigest()[:8]  # First 8 chars
        
        # Generate consistent replacement based on type
        if detection.pii_type == PIIType.EMAIL:
            return f"user_{hash_hex}@example.com"
        elif detection.pii_type == PIIType.PHONE:
            return f"555-{hash_hex[:3]}-{hash_hex[3:7]}"
        else:
            return f"[PSEUDO_{detection.pii_type.value}_{hash_hex}]"
```

### Step 4: Simple Usage Example

Create a test script: `diri-helox/examples/test_basic_pii.py`

```python
"""
Simple test of basic PII detection and anonymization
"""

from data_safety.pii_detector import BasicPIIDetector
from data_safety.pii_anonymizer import BasicPIIAnonymizer, AnonymizationStrategy

# Sample text with PII
text = """
Hi, my name is John Doe and my email is john.doe@example.com.
You can reach me at (555) 123-4567.
My SSN is 123-45-6789 and my credit card is 4532-1234-5678-9010.
"""

# Detect PII
detector = BasicPIIDetector()
detections = detector.detect(text)

print("Detected PII:")
for det in detections:
    print(f"  - {det.pii_type.value}: {det.value} (confidence: {det.confidence:.2f})")

# Anonymize with different strategies
anonymizer = BasicPIIAnonymizer()

strategies = [
    AnonymizationStrategy.REDACT,
    AnonymizationStrategy.MASK,
    AnonymizationStrategy.HASH,
]

for strategy in strategies:
    anonymized = anonymizer.anonymize(text, detections, strategy)
    print(f"\n{strategy.value.upper()} strategy:")
    print(anonymized)
```

---

## Advanced Implementation

### Step 1: Presidio-Based Detection

Presidio provides ML-based PII detection with higher accuracy. Create: `diri-helox/data_safety/presidio_detector.py`

```python
"""
Presidio-based PII Detection

Uses Microsoft Presidio for advanced ML-based PII detection.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

from .pii_detector import PIIDetection, PIIType

logger = logging.getLogger(__name__)


class PresidioPIIDetector:
    """
    Advanced PII detector using Presidio Analyzer.
    
    Presidio provides:
    - ML-based entity recognition
    - Context-aware detection
    - Multiple language support
    - Custom entity recognition
    """
    
    # Map Presidio entity types to our PIIType enum
    PRESIDIO_TO_PII = {
        "US_SSN": PIIType.SSN,
        "CREDIT_CARD": PIIType.CREDIT_CARD,
        "EMAIL_ADDRESS": PIIType.EMAIL,
        "PHONE_NUMBER": PIIType.PHONE,
        "IP_ADDRESS": PIIType.IP_ADDRESS,
        "DATE_TIME": PIIType.DATE_OF_BIRTH,
        "PERSON": PIIType.PERSON_NAME,
        "LOCATION": PIIType.ADDRESS,
        "US_DRIVER_LICENSE": PIIType.DRIVER_LICENSE,
        "US_PASSPORT": PIIType.PASSPORT,
    }
    
    def __init__(self, language: str = "en"):
        """
        Initialize Presidio analyzer.
        
        Args:
            language: Language code (default: "en")
        """
        if not PRESIDIO_AVAILABLE:
            raise ImportError(
                "Presidio not available. Install with: pip install presidio-analyzer"
            )
        
        try:
            # Initialize NLP engine
            provider = NlpEngineProvider(ner_model_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": language, "model_name": "en_core_web_lg"}]
            })
            nlp_engine = provider.create_engine()
            
            # Initialize analyzer
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            self.language = language
            
            logger.info("Presidio analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Presidio: {e}")
            raise
    
    def detect(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        score_threshold: float = 0.5
    ) -> List[PIIDetection]:
        """
        Detect PII using Presidio.
        
        Args:
            text: Input text
            entities: List of entity types to detect (None = all)
            score_threshold: Minimum confidence score
            
        Returns:
            List of detected PII instances
        """
        try:
            # Run Presidio analysis
            results = self.analyzer.analyze(
                text=text,
                language=self.language,
                entities=entities,
                score_threshold=score_threshold
            )
            
            detections = []
            for result in results:
                pii_type = self.PRESIDIO_TO_PII.get(
                    result.entity_type,
                    None
                )
                
                if pii_type:
                    detections.append(PIIDetection(
                        pii_type=pii_type,
                        value=text[result.start:result.end],
                        start_pos=result.start,
                        end_pos=result.end,
                        confidence=result.score,
                        context=self._extract_context(text, result.start, result.end)
                    ))
            
            return detections
        
        except Exception as e:
            logger.error(f"Presidio detection failed: {e}")
            return []
    
    def _extract_context(self, text: str, start: int, end: int, context_size: int = 30) -> str:
        """Extract context around detected PII"""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end]
```

### Step 2: Advanced Anonymization with Presidio

Create: `diri-helox/data_safety/presidio_anonymizer.py`

```python
"""
Presidio-based PII Anonymization

Uses Microsoft Presidio Anonymizer for advanced anonymization.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_ANONYMIZER_AVAILABLE = True
except ImportError:
    PRESIDIO_ANONYMIZER_AVAILABLE = False

from .pii_detector import PIIDetection, PIIType
from .pii_anonymizer import AnonymizationStrategy

logger = logging.getLogger(__name__)


class PresidioPIIAnonymizer:
    """
    Advanced PII anonymizer using Presidio Anonymizer.
    
    Supports:
    - Multiple anonymization operators
    - Custom operators
    - Consistent anonymization (pseudonymization)
    """
    
    def __init__(self):
        if not PRESIDIO_ANONYMIZER_AVAILABLE:
            raise ImportError(
                "Presidio Anonymizer not available. "
                "Install with: pip install presidio-anonymizer"
            )
        
        self.anonymizer = AnonymizerEngine()
        self._setup_operators()
    
    def _setup_operators(self):
        """Setup default anonymization operators"""
        self.operators = {
            "default": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
            "mask": OperatorConfig("mask", {"chars_to_mask": 4, "masking_char": "*", "from_end": True}),
            "hash": OperatorConfig("hash"),
            "redact": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
        }
    
    def anonymize(
        self,
        text: str,
        detections: List[PIIDetection],
        strategy: Optional[AnonymizationStrategy] = None
    ) -> str:
        """
        Anonymize text using Presidio.
        
        Args:
            text: Original text
            detections: List of PII detections
            strategy: Anonymization strategy
            
        Returns:
            Anonymized text
        """
        if not detections:
            return text
        
        # Convert detections to Presidio format
        analyzer_results = []
        for det in detections:
            analyzer_results.append({
                "entity_type": self._pii_to_presidio_type(det.pii_type),
                "start": det.start_pos,
                "end": det.end_pos,
                "score": det.confidence
            })
        
        # Get operator config based on strategy
        operator_config = self._get_operator_config(strategy)
        
        # Create operator configs for each entity type
        operators = {}
        for det in detections:
            entity_type = self._pii_to_presidio_type(det.pii_type)
            operators[entity_type] = operator_config
        
        # Anonymize
        try:
            result = self.anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators
            )
            return result.text
        except Exception as e:
            logger.error(f"Presidio anonymization failed: {e}")
            return text
    
    def _pii_to_presidio_type(self, pii_type: PIIType) -> str:
        """Convert PIIType to Presidio entity type"""
        mapping = {
            PIIType.SSN: "US_SSN",
            PIIType.CREDIT_CARD: "CREDIT_CARD",
            PIIType.EMAIL: "EMAIL_ADDRESS",
            PIIType.PHONE: "PHONE_NUMBER",
            PIIType.IP_ADDRESS: "IP_ADDRESS",
            PIIType.DATE_OF_BIRTH: "DATE_TIME",
            PIIType.PERSON_NAME: "PERSON",
            PIIType.ADDRESS: "LOCATION",
            PIIType.DRIVER_LICENSE: "US_DRIVER_LICENSE",
            PIIType.PASSPORT: "US_PASSPORT",
        }
        return mapping.get(pii_type, "DEFAULT")
    
    def _get_operator_config(self, strategy: Optional[AnonymizationStrategy]):
        """Get Presidio operator config for strategy"""
        if strategy == AnonymizationStrategy.MASK:
            return self.operators["mask"]
        elif strategy == AnonymizationStrategy.HASH:
            return self.operators["hash"]
        else:
            return self.operators["redact"]
```

### Step 3: Unified PII Pipeline

Create a unified interface: `diri-helox/data_safety/pii_pipeline.py`

```python
"""
Unified PII Detection and Anonymization Pipeline

Combines detection and anonymization into a single pipeline.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from .pii_detector import BasicPIIDetector, PIIDetection
from .pii_anonymizer import BasicPIIAnonymizer, AnonymizationStrategy

try:
    from .presidio_detector import PresidioPIIDetector
    from .presidio_anonymizer import PresidioPIIAnonymizer
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PIIProcessingResult:
    """Result of PII processing"""
    original_text: str
    anonymized_text: str
    detections: List[PIIDetection]
    detection_count: int
    anonymization_strategy: str
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PIIPipeline:
    """
    Unified pipeline for PII detection and anonymization.
    
    Supports:
    - Multiple detection backends (basic regex, Presidio)
    - Multiple anonymization strategies
    - Audit logging
    - Performance metrics
    """
    
    def __init__(
        self,
        use_presidio: bool = True,
        default_strategy: AnonymizationStrategy = AnonymizationStrategy.REDACT,
        enable_audit: bool = True
    ):
        """
        Initialize PII pipeline.
        
        Args:
            use_presidio: Use Presidio for detection (falls back to basic if unavailable)
            default_strategy: Default anonymization strategy
            enable_audit: Enable audit logging
        """
        self.use_presidio = use_presidio and PRESIDIO_AVAILABLE
        self.default_strategy = default_strategy
        self.enable_audit = enable_audit
        
        # Initialize detectors
        if self.use_presidio:
            try:
                self.detector = PresidioPIIDetector()
                logger.info("Using Presidio for PII detection")
            except Exception as e:
                logger.warning(f"Presidio initialization failed: {e}, falling back to basic detector")
                self.detector = BasicPIIDetector()
                self.use_presidio = False
        else:
            self.detector = BasicPIIDetector()
            logger.info("Using basic regex-based PII detection")
        
        # Initialize anonymizers
        if self.use_presidio:
            try:
                self.anonymizer = PresidioPIIAnonymizer()
                logger.info("Using Presidio for PII anonymization")
            except Exception as e:
                logger.warning(f"Presidio anonymizer initialization failed: {e}, falling back to basic anonymizer")
                self.anonymizer = BasicPIIAnonymizer(default_strategy)
        else:
            self.anonymizer = BasicPIIAnonymizer(default_strategy)
        
        # Audit log
        self.audit_log: List[Dict[str, Any]] = []
    
    def process(
        self,
        text: str,
        strategy: Optional[AnonymizationStrategy] = None,
        min_confidence: float = 0.5
    ) -> PIIProcessingResult:
        """
        Process text: detect and anonymize PII.
        
        Args:
            text: Input text
            strategy: Anonymization strategy (uses default if not specified)
            min_confidence: Minimum confidence for detection
            
        Returns:
            PIIProcessingResult with anonymized text and metadata
        """
        import time
        start_time = time.time()
        
        strategy = strategy or self.default_strategy
        
        # Detect PII
        if self.use_presidio:
            detections = self.detector.detect(text, score_threshold=min_confidence)
        else:
            detections = self.detector.detect(text)
            # Filter by confidence
            detections = [d for d in detections if d.confidence >= min_confidence]
        
        # Anonymize
        if detections:
            anonymized_text = self.anonymizer.anonymize(text, detections, strategy)
        else:
            anonymized_text = text
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Create result
        result = PIIProcessingResult(
            original_text=text,
            anonymized_text=anonymized_text,
            detections=detections,
            detection_count=len(detections),
            anonymization_strategy=strategy.value,
            processing_time_ms=processing_time,
            metadata={
                "detector_type": "presidio" if self.use_presidio else "basic",
                "anonymizer_type": "presidio" if self.use_presidio else "basic",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Audit logging
        if self.enable_audit:
            self._log_audit(result)
        
        return result
    
    def _log_audit(self, result: PIIProcessingResult):
        """Log PII processing for audit trail"""
        audit_entry = {
            "timestamp": result.metadata["timestamp"],
            "detection_count": result.detection_count,
            "strategy": result.anonymization_strategy,
            "processing_time_ms": result.processing_time_ms,
            "detections": [
                {
                    "type": det.pii_type.value,
                    "confidence": det.confidence,
                    "value_preview": det.value[:10] + "..." if len(det.value) > 10 else det.value
                }
                for det in result.detections
            ]
        }
        self.audit_log.append(audit_entry)
        
        # Log to logger
        logger.info(
            f"PII processed: {result.detection_count} detections, "
            f"strategy={result.anonymization_strategy}, "
            f"time={result.processing_time_ms:.2f}ms"
        )
    
    def get_audit_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        if limit:
            return self.audit_log[-limit:]
        return self.audit_log
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        if not self.audit_log:
            return {
                "total_processed": 0,
                "total_detections": 0,
                "avg_processing_time_ms": 0.0,
                "detection_types": {}
            }
        
        total_processed = len(self.audit_log)
        total_detections = sum(entry["detection_count"] for entry in self.audit_log)
        avg_time = sum(entry["processing_time_ms"] for entry in self.audit_log) / total_processed
        
        # Count detection types
        detection_types = {}
        for entry in self.audit_log:
            for det in entry.get("detections", []):
                det_type = det["type"]
                detection_types[det_type] = detection_types.get(det_type, 0) + 1
        
        return {
            "total_processed": total_processed,
            "total_detections": total_detections,
            "avg_processing_time_ms": avg_time,
            "detection_types": detection_types
        }
```

---

## Integration with Helox Pipeline

### Step 1: Create Preprocessing Stage

Create: `diri-helox/pipelines/data_preprocessing/stages.py` (add to existing file)

Add a new stage class:

```python
class PIIAnonymizationStage(PreprocessingStage):
    """
    Stage for PII detection and anonymization.
    
    This stage detects and anonymizes PII in training data before it
    enters the model training pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="pii_anonymization", config=config)
        
        # Configuration
        self.use_presidio = config.get("use_presidio", True) if config else True
        self.strategy = config.get("strategy", "redact") if config else "redact"
        self.min_confidence = config.get("min_confidence", 0.5) if config else 0.5
        self.enable_audit = config.get("enable_audit", True) if config else True
        
        # Initialize PII pipeline
        from ...data_safety.pii_pipeline import PIIPipeline, AnonymizationStrategy
        
        strategy_enum = AnonymizationStrategy(self.strategy)
        self.pii_pipeline = PIIPipeline(
            use_presidio=self.use_presidio,
            default_strategy=strategy_enum,
            enable_audit=self.enable_audit
        )
    
    def get_dependencies(self) -> List[str]:
        """PII anonymization should run after data cleaning"""
        return ["data_cleaning"]
    
    def process(self, data: Any) -> StageResult:
        """
        Detect and anonymize PII in data.
        
        Args:
            data: Data from previous stage
            
        Returns:
            StageResult with anonymized data
        """
        try:
            # Extract data and metadata
            actual_data, original_metadata = self._extract_data_and_metadata(data)
            
            # Process items
            anonymized_data = self._process_items(
                actual_data,
                self._anonymize_single_item
            )
            
            # Update metadata with PII statistics
            pii_stats = self.pii_pipeline.get_statistics()
            metadata_updates = {
                "pii_anonymized": True,
                "pii_detection_count": pii_stats.get("total_detections", 0),
                "pii_anonymization_strategy": self.strategy
            }
            
            return self._create_result(
                processed_data=anonymized_data,
                original_metadata=original_metadata,
                metadata_updates=metadata_updates
            )
        
        except Exception as e:
            return self._create_result(
                processed_data=None,
                original_metadata={},
                metadata_updates={},
                success=False,
                error=f"PII anonymization failed: {str(e)}"
            )
    
    def _anonymize_single_item(self, item: Dict) -> Dict:
        """Anonymize PII in a single data item"""
        if not isinstance(item, dict):
            return item
        
        anonymized_item = item.copy()
        
        # Anonymize text fields
        text_fields = ["text", "input", "output", "instruction", "context"]
        for field in text_fields:
            if field in anonymized_item and isinstance(anonymized_item[field], str):
                result = self.pii_pipeline.process(
                    anonymized_item[field],
                    min_confidence=self.min_confidence
                )
                anonymized_item[field] = result.anonymized_text
                
                # Store PII metadata
                if result.detection_count > 0:
                    if "_pii_metadata" not in anonymized_item:
                        anonymized_item["_pii_metadata"] = {}
                    anonymized_item["_pii_metadata"][field] = {
                        "detection_count": result.detection_count,
                        "detections": [
                            {
                                "type": det.pii_type.value,
                                "confidence": det.confidence
                            }
                            for det in result.detections
                        ]
                    }
        
        return anonymized_item
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate that data can be anonymized.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        actual_data = self._extract_data(data)
        
        try:
            self._check_data_not_none(actual_data, context="validate")
        except ValueError as e:
            errors.append(str(e))
            return self._create_validation_result(errors, warnings)
        
        # Validate PII pipeline is initialized
        if not hasattr(self, "pii_pipeline") or self.pii_pipeline is None:
            errors.append("PII pipeline not initialized")
            return self._create_validation_result(errors, warnings)
        
        # Check if data has text fields to anonymize
        item_errors, item_warnings = self._validate_items(
            actual_data,
            self._validate_anonymization_diagnostics,
            empty_error="Data is empty - nothing to anonymize"
        )
        errors.extend(item_errors)
        warnings.extend(item_warnings)
        
        return self._create_validation_result(errors, warnings)
    
    def _validate_anonymization_diagnostics(self, item: Dict) -> tuple[List[str], List[str]]:
        """Validate that an item can be anonymized"""
        errors = []
        warnings = []
        
        if not isinstance(item, dict):
            errors.append(f"Item must be a dictionary, got {type(item).__name__}")
            return errors, warnings
        
        # Check for text fields
        text_fields = ["text", "input", "output", "instruction", "context"]
        has_text_field = any(
            field in item and isinstance(item[field], str) and item[field].strip()
            for field in text_fields
        )
        
        if not has_text_field:
            warnings.append("No text fields found - nothing to anonymize")
        
        return errors, warnings
```

### Step 2: Integrate into Preprocessing Orchestrator

Update: `diri-helox/pipelines/data_preprocessing/orchestrator.py`

Add PII anonymization stage to the default pipeline:

```python
# In the pipeline configuration
stages = [
    DataLoadingStage(config={"source": "..."}),
    DataCleaningStage(),
    PIIAnonymizationStage(config={
        "use_presidio": True,
        "strategy": "redact",
        "min_confidence": 0.5,
        "enable_audit": True
    }),
    DataValidationStage(),
    # ... rest of stages
]
```

### Step 3: Integrate into Cyrex Pipeline

Update: `diri-cyrex/app/core/realtime_data_pipeline.py`

Enhance the `DataTransformer._redact_pii` method to use the advanced pipeline:

```python
# At the top of the file
try:
    import sys
    from pathlib import Path
    # Add helox to path (adjust based on your structure)
    helox_path = Path(__file__).parent.parent.parent.parent / "diri-helox"
    if helox_path.exists():
        sys.path.insert(0, str(helox_path))
        from data_safety.pii_pipeline import PIIPipeline, AnonymizationStrategy
        PII_PIPELINE_AVAILABLE = True
    else:
        PII_PIPELINE_AVAILABLE = False
except ImportError:
    PII_PIPELINE_AVAILABLE = False

# In DataTransformer class
@classmethod
def _redact_pii(cls, record: PipelineRecord) -> PipelineRecord:
    """Redact PII from text fields before sending to training"""
    if PII_PIPELINE_AVAILABLE:
        # Use advanced PII pipeline
        pipeline = PIIPipeline(
            use_presidio=True,
            default_strategy=AnonymizationStrategy.REDACT,
            enable_audit=True
        )
        
        for attr in ("input_text", "output_text", "instruction", "context"):
            text = getattr(record, attr, "")
            if text:
                result = pipeline.process(text)
                setattr(record, attr, result.anonymized_text)
    else:
        # Fallback to basic regex
        for attr in ("input_text", "output_text", "instruction", "context"):
            text = getattr(record, attr, "")
            if text:
                for pattern, replacement in cls._PII_PATTERNS:
                    text = pattern.sub(replacement, text)
                setattr(record, attr, text)
    
    return record
```

### Step 4: Add to Helox Ingestion

Update: `diri-helox/integrations/realtime_ingestion.py`

Add PII validation during ingestion:

```python
# At the top
from data_safety.pii_pipeline import PIIPipeline, AnonymizationStrategy

# In HeloxRealtimeIngestion.__init__
self.pii_pipeline = PIIPipeline(
    use_presidio=True,
    default_strategy=AnonymizationStrategy.REDACT,
    enable_audit=True
)

# In _parse_record or before buffering
def _check_and_anonymize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
    """Check for PII and anonymize if needed"""
    # Check text fields
    text_fields = ["text", "input", "output", "instruction"]
    for field in text_fields:
        if field in record and isinstance(record[field], str):
            result = self.pii_pipeline.process(record[field])
            if result.detection_count > 0:
                record[field] = result.anonymized_text
                # Log detection
                logger.warning(
                    f"PII detected in {field}: {result.detection_count} instances"
                )
    
    return record
```

---

## Testing and Validation

### Unit Tests

Create: `diri-helox/tests/test_pii_detection.py`

```python
"""
Unit tests for PII detection and anonymization
"""

import pytest
from data_safety.pii_detector import BasicPIIDetector, PIIType
from data_safety.pii_anonymizer import BasicPIIAnonymizer, AnonymizationStrategy
from data_safety.pii_pipeline import PIIPipeline


class TestBasicPIIDetector:
    """Test basic PII detector"""
    
    def test_detect_ssn(self):
        detector = BasicPIIDetector()
        text = "My SSN is 123-45-6789"
        detections = detector.detect(text)
        
        assert len(detections) == 1
        assert detections[0].pii_type == PIIType.SSN
        assert detections[0].value == "123-45-6789"
    
    def test_detect_email(self):
        detector = BasicPIIDetector()
        text = "Contact me at john@example.com"
        detections = detector.detect(text)
        
        assert len(detections) == 1
        assert detections[0].pii_type == PIIType.EMAIL
        assert detections[0].value == "john@example.com"
    
    def test_detect_multiple(self):
        detector = BasicPIIDetector()
        text = "Email: john@example.com, Phone: (555) 123-4567"
        detections = detector.detect(text)
        
        assert len(detections) >= 2
        types = [d.pii_type for d in detections]
        assert PIIType.EMAIL in types
        assert PIIType.PHONE in types


class TestBasicPIIAnonymizer:
    """Test basic PII anonymizer"""
    
    def test_redact_strategy(self):
        detector = BasicPIIDetector()
        anonymizer = BasicPIIAnonymizer()
        
        text = "My SSN is 123-45-6789"
        detections = detector.detect(text)
        anonymized = anonymizer.anonymize(text, detections, AnonymizationStrategy.REDACT)
        
        assert "[SSN_REDACTED]" in anonymized
        assert "123-45-6789" not in anonymized
    
    def test_mask_strategy(self):
        detector = BasicPIIDetector()
        anonymizer = BasicPIIAnonymizer()
        
        text = "My SSN is 123-45-6789"
        detections = detector.detect(text)
        anonymized = anonymizer.anonymize(text, detections, AnonymizationStrategy.MASK)
        
        assert "6789" in anonymized  # Last 4 digits should be visible
        assert "123-45" not in anonymized or "***" in anonymized


class TestPIIPipeline:
    """Test unified PII pipeline"""
    
    def test_pipeline_detection_and_anonymization(self):
        pipeline = PIIPipeline(use_presidio=False)  # Use basic for testing
        
        text = "Contact john@example.com or call (555) 123-4567"
        result = pipeline.process(text)
        
        assert result.detection_count > 0
        assert result.anonymized_text != text
        assert "@example.com" not in result.anonymized_text or "[EMAIL_REDACTED]" in result.anonymized_text
    
    def test_pipeline_statistics(self):
        pipeline = PIIPipeline(use_presidio=False)
        
        texts = [
            "Email: test@example.com",
            "Phone: (555) 123-4567",
            "SSN: 123-45-6789"
        ]
        
        for text in texts:
            pipeline.process(text)
        
        stats = pipeline.get_statistics()
        assert stats["total_processed"] == 3
        assert stats["total_detections"] > 0
```

### Integration Tests

Create: `diri-helox/tests/test_pii_integration.py`

```python
"""
Integration tests for PII pipeline with Helox preprocessing
"""

import pytest
from pipelines.data_preprocessing.stages import PIIAnonymizationStage
from pipelines.data_preprocessing.orchestrator import PreprocessingOrchestrator


class TestPIIIntegration:
    """Test PII integration with preprocessing pipeline"""
    
    def test_pii_stage_in_pipeline(self):
        """Test PII anonymization stage in full pipeline"""
        config = {
            "use_presidio": False,  # Use basic for testing
            "strategy": "redact",
            "min_confidence": 0.5
        }
        
        stage = PIIAnonymizationStage(config=config)
        
        data = {
            "text": "Contact john@example.com for details",
            "label": "contact"
        }
        
        result = stage.process(data)
        
        assert result.success
        assert "[EMAIL_REDACTED]" in result.processed_data.data["text"]
    
    def test_pii_with_preprocessing_orchestrator(self):
        """Test PII stage in full preprocessing orchestrator"""
        # Create orchestrator with PII stage
        orchestrator = PreprocessingOrchestrator()
        
        # Add stages including PII anonymization
        # ... configure stages ...
        
        # Process data
        # ... test processing ...
        pass
```

---

## Production Deployment

### Configuration

Create: `diri-helox/configs/pii_config.json`

```json
{
  "detection": {
    "use_presidio": true,
    "min_confidence": 0.6,
    "entities": null
  },
  "anonymization": {
    "strategy": "redact",
    "enable_audit": true,
    "audit_retention_days": 90
  },
  "performance": {
    "batch_size": 100,
    "max_processing_time_ms": 1000
  },
  "compliance": {
    "gdpr_enabled": true,
    "hipaa_enabled": false,
    "ccpa_enabled": true
  }
}
```

### Monitoring

Add metrics collection:

```python
# In pii_pipeline.py
class PIIPipeline:
    def __init__(self, ...):
        # ... existing code ...
        self.metrics = {
            "total_processed": 0,
            "total_detections": 0,
            "detection_types": {},
            "processing_times": [],
            "errors": 0
        }
    
    def process(self, ...):
        try:
            # ... processing ...
            self.metrics["total_processed"] += 1
            self.metrics["total_detections"] += result.detection_count
            self.metrics["processing_times"].append(result.processing_time_ms)
            
            for det in result.detections:
                det_type = det.pii_type.value
                self.metrics["detection_types"][det_type] = \
                    self.metrics["detection_types"].get(det_type, 0) + 1
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"PII processing error: {e}")
            raise
```

### Error Handling

Implement robust error handling:

```python
class PIIPipeline:
    def process(self, text: str, ...) -> PIIProcessingResult:
        try:
            # ... detection ...
        except Exception as e:
            logger.error(f"PII detection failed: {e}")
            # Fallback: return original text with warning
            return PIIProcessingResult(
                original_text=text,
                anonymized_text=text,  # Return original on error
                detections=[],
                detection_count=0,
                anonymization_strategy="none",
                processing_time_ms=0.0,
                metadata={"error": str(e), "fallback": True}
            )
```

---

## Monitoring and Compliance

### Audit Logging

Implement comprehensive audit logging:

```python
class PIIAuditLogger:
    """Audit logger for PII processing"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self._setup_logging()
    
    def log_processing(self, result: PIIProcessingResult, user_id: Optional[str] = None):
        """Log PII processing event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "detection_count": result.detection_count,
            "strategy": result.anonymization_strategy,
            "processing_time_ms": result.processing_time_ms,
            "detections": [
                {
                    "type": det.pii_type.value,
                    "confidence": det.confidence,
                    "position": (det.start_pos, det.end_pos)
                }
                for det in result.detections
            ]
        }
        
        # Write to audit log file
        with open(self.log_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
```

### Compliance Reports

Generate compliance reports:

```python
class PIIComplianceReporter:
    """Generate compliance reports for PII processing"""
    
    def generate_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for date range"""
        # Load audit logs
        # Calculate metrics
        # Generate report
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_records_processed": 0,
                "total_pii_detected": 0,
                "pii_types_detected": {},
                "anonymization_strategies_used": {},
                "compliance_status": "compliant"
            },
            "details": {
                # Detailed breakdown
            }
        }
```

### Performance Optimization

Optimize for production:

1. **Batch Processing**: Process multiple texts in batch
2. **Caching**: Cache detection results for identical texts
3. **Async Processing**: Use async/await for non-blocking operations
4. **Resource Pooling**: Reuse Presidio analyzers across requests

```python
class OptimizedPIIPipeline(PIIPipeline):
    """Optimized PII pipeline for production"""
    
    def __init__(self, ...):
        super().__init__(...)
        self._cache = {}  # Simple cache for identical texts
    
    def process(self, text: str, ...) -> PIIProcessingResult:
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Process
        result = super().process(text, ...)
        
        # Cache result
        self._cache[cache_key] = result
        
        return result
```

---

## Conclusion

This guide has walked you through implementing a comprehensive PII detection and anonymization pipeline for Helox, from basic regex-based detection to advanced ML-based solutions using Presidio.

### Key Takeaways

1. **Start Simple**: Begin with basic regex patterns, then upgrade to ML-based detection
2. **Layer Defense**: Use multiple detection methods for better coverage
3. **Audit Everything**: Maintain comprehensive audit logs for compliance
4. **Test Thoroughly**: Unit tests and integration tests are essential
5. **Monitor Performance**: Track metrics and optimize for production workloads

### Next Steps

1. Implement the basic detector and anonymizer
2. Test with real data samples
3. Integrate into preprocessing pipeline
4. Add Presidio for advanced detection
5. Implement audit logging and compliance reporting
6. Deploy to production with monitoring

### Additional Resources

- [Presidio Documentation](https://microsoft.github.io/presidio/)
- [GDPR Compliance Guide](https://gdpr.eu/)
- [NIST Privacy Framework](https://www.nist.gov/privacy-framework)

---

**End of Guide**

