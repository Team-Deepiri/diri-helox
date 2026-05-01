"""
Unit tests for IntentClassifierTrainer utility logic.
These tests avoid model downloads and focus on deterministic helper behavior.
"""

import sys
from pathlib import Path

import pytest

_HELOX_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_HELOX_ROOT))

pytest.importorskip("datasets")
pytest.importorskip("transformers")
pytest.importorskip("torch")

from datasets import Dataset
from training.intent_classifier_trainer import IntentClassifierTrainer


class TestClassWeightComputation:
    def test_balanced_weights_favor_minority_classes(self):
        labels = [0] * 90 + [1] * 10
        weights = IntentClassifierTrainer._compute_balanced_class_weights(
            labels=labels,
            num_labels=3,
            min_weight=0.1,
            max_weight=10.0,
        )
        # Minority class should get larger weight than majority class.
        assert weights[1] > weights[0]
        # Missing class should remain neutral (1.0).
        assert weights[2] == 1.0

    def test_balanced_weights_clip_range(self):
        labels = [0] * 99 + [1]
        weights = IntentClassifierTrainer._compute_balanced_class_weights(
            labels=labels,
            num_labels=2,
            min_weight=0.5,
            max_weight=2.0,
        )
        assert min(weights) >= 0.5
        assert max(weights) <= 2.0

    def test_balanced_weights_invalid_label_range_raises(self):
        with pytest.raises(ValueError, match="out-of-range"):
            IntentClassifierTrainer._compute_balanced_class_weights(
                labels=[0, 1, 4],  # 4 is out of range for num_labels=3
                num_labels=3,
            )

    def test_resolve_class_weights_none_returns_none(self):
        trainer = IntentClassifierTrainer(class_weighting="none")
        train_hf = Dataset.from_dict({"label": [0, 0, 1], "input_ids": [[1], [2], [3]]})
        assert trainer._resolve_class_weights(train_hf) is None

    def test_resolve_class_weights_invalid_mode_raises(self):
        trainer = IntentClassifierTrainer(class_weighting="unknown_mode")
        train_hf = Dataset.from_dict({"label": [0, 1], "input_ids": [[1], [2]]})
        with pytest.raises(ValueError, match="Unsupported class_weighting mode"):
            trainer._resolve_class_weights(train_hf)
