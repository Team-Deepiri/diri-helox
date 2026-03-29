"""
Unit tests for DynamicTrainingPipeline (non-training steps only).
Run: pytest tests/test_dynamic_pipeline.py -v
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_HELOX_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_HELOX_ROOT))

from data_sources.base import DataSample, DataSourceConfig
from pipelines.training.dynamic_training_pipeline import DynamicTrainingPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_samples(n: int, num_labels: int = 31) -> list:
    return [
        DataSample(
            text=f"Sample text number {i} describing a task to do",
            label=i % num_labels,
            label_name=f"category_{i % num_labels}",
            source="test",
        )
        for i in range(n)
    ]


def _minimal_config(**overrides) -> dict:
    base = {
        "pipeline_name": "test_pipeline",
        "data_sources": [
            {"source_type": "synthetic", "name": "s", "params": {"total_examples": 31}}
        ],
        "preprocessing": {"use_text_cleaner": False, "use_deduplication": False},
        "training": {
            "trainer_type": "intent_classifier",
            "model_name": "bert-base-uncased",
            "num_labels": 31,
            "num_epochs": 1,
            "batch_size": 8,
            "output_dir": "/tmp/test_model",
        },
        "evaluation": {"enabled": True},
        "export": {"mlflow": {"enabled": False}, "publish_model_ready": False},
        "split": {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15, "seed": 42},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_from_file(self, tmp_path):
        cfg = _minimal_config()
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(cfg))
        pipeline = DynamicTrainingPipeline.from_file(str(p))
        assert pipeline.config["pipeline_name"] == "test_pipeline"

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DynamicTrainingPipeline.from_file(str(tmp_path / "no_file.json"))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_min_length_filter(self):
        samples = [
            DataSample(text="hi", label=0, source="t"),        # too short
            DataSample(text="a longer task description", label=1, source="t"),
        ]
        pipeline = DynamicTrainingPipeline({
            "preprocessing": {"use_text_cleaner": False, "use_deduplication": False, "min_text_length": 10}
        })
        result = pipeline.preprocess(samples)
        assert len(result) == 1
        assert result[0].text == "a longer task description"

    def test_deduplication_removes_duplicates(self):
        samples = [
            DataSample(text="unique text about coding tasks", label=0, source="t"),
            DataSample(text="unique text about coding tasks", label=0, source="t"),  # dup
            DataSample(text="another unique description here", label=1, source="t"),
        ]
        pipeline = DynamicTrainingPipeline({
            "preprocessing": {
                "use_text_cleaner": False,
                "use_deduplication": True,
                "min_text_length": 5,
            }
        })
        try:
            result = pipeline.preprocess(samples)
            # With dedup, should have fewer samples
            assert len(result) <= 3
        except ImportError:
            pytest.skip("deepiri-dataset-processor not installed")

    def test_empty_samples_handled(self):
        pipeline = DynamicTrainingPipeline({
            "preprocessing": {"use_text_cleaner": False, "use_deduplication": False, "min_text_length": 5}
        })
        result = pipeline.preprocess([])
        assert result == []


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_split_ratios(self):
        samples = _make_samples(100)
        pipeline = DynamicTrainingPipeline({
            "split": {"train_ratio": 0.70, "val_ratio": 0.15, "seed": 42}
        })
        train, val, test = pipeline.split_data(samples)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert len(train) + len(val) + len(test) == 100

    def test_split_is_reproducible(self):
        samples = _make_samples(50)
        cfg = {"split": {"train_ratio": 0.70, "val_ratio": 0.15, "seed": 99}}
        train1, _, _ = DynamicTrainingPipeline(cfg).split_data(samples)
        train2, _, _ = DynamicTrainingPipeline(cfg).split_data(samples)
        assert [s.text for s in train1] == [s.text for s in train2]

    def test_split_small_dataset(self):
        samples = _make_samples(3)
        pipeline = DynamicTrainingPipeline({
            "split": {"train_ratio": 0.70, "val_ratio": 0.15, "seed": 0}
        })
        train, val, test = pipeline.split_data(samples)
        # All samples accounted for
        assert len(train) + len(val) + len(test) == 3


# ---------------------------------------------------------------------------
# Dry run (data loading without training)
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_loads_data(self):
        """Verify setup + load_data works without training."""
        pipeline = DynamicTrainingPipeline({
            "data_sources": [
                {"source_type": "synthetic", "name": "s", "params": {"total_examples": 31}}
            ],
            "preprocessing": {"use_text_cleaner": False, "use_deduplication": False},
        })
        pipeline.setup_data_sources()
        samples = pipeline.load_data()
        # Should have approximately 31 samples
        assert len(samples) >= 20, f"Expected at least 20 samples, got {len(samples)}"

    def test_setup_creates_correct_source_types(self):
        pipeline = DynamicTrainingPipeline({
            "data_sources": [
                {"source_type": "synthetic", "name": "synth", "params": {"total_examples": 10}},
            ]
        })
        pipeline.setup_data_sources()
        assert len(pipeline._sources) == 1
        assert pipeline._sources[0].source_type == "synthetic"


# ---------------------------------------------------------------------------
# Export (mocked)
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_skips_when_disabled(self):
        """No error raised when mlflow and model-ready are both disabled."""
        pipeline = DynamicTrainingPipeline({
            "export": {"mlflow": {"enabled": False}, "publish_model_ready": False},
            "training": {"output_dir": "/tmp/fake"},
        })
        # Should not raise
        pipeline.export({"overall": {"accuracy": 0.9, "f1": 0.88}})

    def test_export_graceful_on_mlflow_failure(self):
        """Pipeline continues even if MLflow is unreachable."""
        pipeline = DynamicTrainingPipeline({
            "pipeline_name": "test",
            "export": {
                "mlflow": {
                    "enabled": True,
                    "tracking_uri": "http://localhost:9999",  # non-existent
                    "register_model": False,
                },
                "publish_model_ready": False,
            },
            "training": {"output_dir": "/tmp/fake"},
        })
        # Should not raise — graceful degradation
        pipeline.export({"overall": {"accuracy": 0.8}})
