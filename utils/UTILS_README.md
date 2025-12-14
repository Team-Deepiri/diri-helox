**Purpose**

This document explains two utility modules used by the training and data-augmentation pipeline:

- `semantic_analyzer.py` — interfaces with an Ollama LLM to generate semantic augmentations (verbs, prefixes, suffixes, paraphrases) and perform light semantic structure analysis.
- `confidence_classes.py` — provides a structured, multi-source confidence-scoring system to evaluate model predictions and decide whether to accept them.

**Files Covered**

- `app/train/utils/semantic_analyzer.py`
- `app/train/utils/confidence_classes.py`

**Quick Summary**

- `semantic_analyzer.py`: Useful for data augmentation and generating natural variations of task text (paraphrases, prefix/suffix alternatives, semantically-related verbs). It calls Ollama (via `ollama` Python package or HTTP using `httpx`/`requests`) and caches category-level outputs.
- `confidence_classes.py`: Computes a composite `reliability` score from multiple sources (model prediction, training coverage, feature quality, context match, historical accuracy, ensemble agreement), maps it to a categorical `ConfidenceLevel`, and emits a human-readable explanation.

**Usage: `semantic_analyzer.py`**

- Purpose: generate training-time paraphrases and contextual variants, and extract simple semantic structure.
- How to get an instance:

```python
from app.train.utils.semantic_analyzer import get_semantic_analyzer

# Use environment variable OLLAMA_BASE_URL or pass explicit url
analyzer = get_semantic_analyzer()  # returns SemanticAnalyzer or None if Ollama not reachable
```

- Example calls:

```python
verbs = analyzer.extract_semantic_verbs(text="Create a report", category="productivity")
prefixes = analyzer.generate_semantic_prefixes(text="", category="productivity")
suffixes = analyzer.generate_semantic_suffixes(text="", category="productivity")
paraphrases = analyzer.generate_paraphrases("Summarize this document.", category="summary", num_paraphrases=3)
structure = analyzer.analyze_semantic_structure("Please summarize the quarterly report by Monday")
```

- Notes / environment:
  - The module will try the `ollama` Python package first, then fall back to HTTP via `httpx` or `requests` depending on availability.
  - Default Ollama base URL: `http://localhost:11434`. The factory `get_semantic_analyzer()` reads `OLLAMA_BASE_URL` and `OLLAMA_MODEL` environment variables.
  - The code expects Ollama to return JSON-like arrays/objects; it extracts JSON with regex when needed. If Ollama is unreachable or responses fail, functions return safe defaults (empty lists or simple heuristic analyses).
  - Calls are synchronous and may block — consider running them in a background worker for bulk augmentation.

**Usage: `confidence_classes.py`**

- Purpose: attach structured confidence metadata to predictions and decide whether to accept or reject a prediction.
- Get the calculator (singleton):

```python
from app.train.utils.confidence_classes import get_confidence_calculator
calc = get_confidence_calculator()
```

- Example usage:

```python
import numpy as np
from app.train.utils.confidence_classes import get_confidence_calculator

calc = get_confidence_calculator()
probs = np.array([0.05, 0.9, 0.05])  # model output probabilities
attrs = calc.calculate_confidence(
    model_probabilities=probs,
    training_coverage=0.8,
    feature_quality=0.9,
    context_match=0.7,
    historical_accuracy={1: 0.85}
)

print(attrs.to_dict())
accept, reason = calc.should_accept_prediction(attrs, min_reliability=0.7)
print(accept, reason)
```

- Key points about the computation:
  - `base_score` is the max probability from the model output.
  - `uncertainty` is entropy normalized by max entropy (range ~0–1).
  - `calibration` is a simple margin between top-1 and top-2 probabilities.
  - Individual source contributions have defaults when optional inputs are missing (e.g., training coverage defaults to 0.7).
  - `reliability` is a weighted sum of sources, adjusted by uncertainty and calibration, clamped to [0,1].
  - `should_accept_prediction()` compares `reliability` and ordinal `ConfidenceLevel` against thresholds.

**Recommendations & Caveats**

- Validate inputs (e.g., `model_probabilities` should be a proper probability distribution and non-empty) before calling `calculate_confidence`.
- The choice of default values and weights is opinionated and hard-coded. If you need different behavior, consider wrapping `ConfidenceCalculator` or adding parameters to override weights and defaults.
- `semantic_analyzer.py` relies on an LLM service (Ollama). For reproducible offline unit tests, mock `_call_ollama` or `get_semantic_analyzer()` to return a stubbed analyzer.
- Both modules are designed for convenience and readability; for production-critical paths, add more robust validation and monitoring (timeouts, retries, rate limiting, structured logging).

**Quick developer notes**

- Files live under: `app/train/utils/semantic_analyzer.py` and `app/train/utils/confidence_classes.py`.
- Environment variables:
  - `OLLAMA_BASE_URL` — base URL for local Ollama instance (default `http://localhost:11434`).
  - `OLLAMA_MODEL` — model identifier used by the analyzer (module reads it in factory).