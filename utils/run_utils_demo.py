"""Runnable demo that exercises `semantic_analyzer.py` and `confidence_classes.py`.

This script:
- Attempts to instantiate a `SemanticAnalyzer` via `get_semantic_analyzer()`.
- Falls back to a small `MockSemanticAnalyzer` when Ollama is not reachable.
- Uses `get_confidence_calculator()` to compute confidence for a sample model output.

Run from diri-cyrex/app/train/utils:

    python -m run_utils_demo

The script is intentionally simple so it can be used as a smoke test or as
an example for integrating these utilities into pipelines.
"""
from __future__ import annotations

import argparse
import json
from typing import List

from confidence_classes import get_confidence_calculator
from semantic_analyzer import get_semantic_analyzer
import numpy as np


class MockSemanticAnalyzer:
    """Minimal mock replacement for SemanticAnalyzer used when Ollama is unavailable."""

    def extract_semantic_verbs(self, text: str, category: str) -> List[str]:
        return ["create", "generate", "produce"]

    def generate_semantic_prefixes(self, text: str, category: str) -> List[str]:
        return ["Please", "Could you", "I need to"]

    def generate_semantic_suffixes(self, text: str, category: str) -> List[str]:
        return ["today", "this week", "when possible"]

    def generate_paraphrases(self, text: str, category: str, num_paraphrases: int = 3) -> List[str]:
        return [f"Paraphrase {i+1}: {text}" for i in range(num_paraphrases)]

    def analyze_semantic_structure(self, text: str) -> dict:
        words = text.split()
        return {
            "action_verb": words[0] if words else "unknown",
            "object": " ".join(words[1:]) if len(words) > 1 else "",
            "modifiers": [],
            "temporal": None,
            "urgency": None,
        }


def run_demo(force_mock: bool = False) -> None:
    # Semantic analyzer
    analyzer = None if force_mock else get_semantic_analyzer()
    if analyzer is None:
        print("[demo] Ollama not available or forced mock — using MockSemanticAnalyzer")
        analyzer = MockSemanticAnalyzer()
    else:
        print("[demo] Using real SemanticAnalyzer")

    sample_text = "Summarize the quarterly results by Friday"
    category = "summary"

    print("\n--- Semantic Analyzer Outputs ---")
    verbs = analyzer.extract_semantic_verbs(sample_text, category)
    prefixes = analyzer.generate_semantic_prefixes(sample_text, category)
    suffixes = analyzer.generate_semantic_suffixes(sample_text, category)
    paraphrases = analyzer.generate_paraphrases(sample_text, category, num_paraphrases=3)
    structure = analyzer.analyze_semantic_structure(sample_text)

    print("Verbs:", verbs)
    print("Prefixes:", prefixes)
    print("Suffixes:", suffixes)
    print("Paraphrases:")
    for p in paraphrases:
        print(" -", p)
    print("Structure:", json.dumps(structure, indent=2))

    # Confidence calculator
    print("\n--- Confidence Calculator Demo ---")
    calc = get_confidence_calculator()

    # Example model probabilities (3-class)
    probs = np.array([0.05, 0.9, 0.05])
    attrs = calc.calculate_confidence(
        model_probabilities=probs,
        training_coverage=0.8,
        feature_quality=0.9,
        context_match=0.7,
        historical_accuracy={1: 0.85}
    )

    print("Confidence attributes:")
    print(json.dumps(attrs.to_dict(), indent=2))

    accept, reason = calc.should_accept_prediction(attrs, min_reliability=0.7)
    print(f"Should accept prediction? {accept} — {reason}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo runner for train/utils modules")
    p.add_argument("--mock", action="store_true", help="Force using the mock analyzer")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_demo(force_mock=args.mock)
