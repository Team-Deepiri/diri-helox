"""
Unit tests for DynamicTrainingPipeline (non-training steps only).
Run: pytest tests/test_dynamic_pipeline.py -v
"""

import json
import sys
from pathlib import Path
import pytest

_HELOX_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_HELOX_ROOT))

from data_sources.base import DataSample
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

    def test_from_file_expands_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_MODEL_DIR", "/tmp/env_model_dir")
        cfg = _minimal_config()
        cfg["training"]["output_dir"] = "${TEST_MODEL_DIR}"
        p = tmp_path / "cfg_env.json"
        p.write_text(json.dumps(cfg))

        pipeline = DynamicTrainingPipeline.from_file(str(p))
        assert pipeline.config["training"]["output_dir"] == "/tmp/env_model_dir"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def test_min_length_filter(self):
        samples = [
            DataSample(text="hi", label=0, source="t"),  # too short
            DataSample(text="a longer task description", label=1, source="t"),
        ]
        pipeline = DynamicTrainingPipeline(
            {
                "preprocessing": {
                    "use_text_cleaner": False,
                    "use_deduplication": False,
                    "min_text_length": 10,
                }
            }
        )
        result = pipeline.preprocess(samples)
        assert len(result) == 1
        assert result[0].text == "a longer task description"

    def test_deduplication_removes_duplicates(self):
        samples = [
            DataSample(text="unique text about coding tasks", label=0, source="t"),
            DataSample(text="unique text about coding tasks", label=0, source="t"),  # dup
            DataSample(text="another unique description here", label=1, source="t"),
        ]
        pipeline = DynamicTrainingPipeline(
            {
                "preprocessing": {
                    "use_text_cleaner": False,
                    "use_deduplication": True,
                    "min_text_length": 5,
                }
            }
        )
        result = pipeline.preprocess(samples)
        assert len(result) == 2

    def test_empty_samples_handled(self):
        pipeline = DynamicTrainingPipeline(
            {
                "preprocessing": {
                    "use_text_cleaner": False,
                    "use_deduplication": False,
                    "min_text_length": 5,
                }
            }
        )
        result = pipeline.preprocess([])
        assert result == []

    def test_create_dataset_version_snapshot(self, tmp_path):
        samples = [
            DataSample(text="alpha task text", label=0, label_name="debugging", source="test"),
            DataSample(text="beta task text", label=1, label_name="testing", source="test"),
        ]
        snapshot = tmp_path / "processed" / "snapshot.jsonl"
        metadata_dir = tmp_path / "metadata"
        pipeline = DynamicTrainingPipeline(
            {
                "pipeline_name": "versioning_test",
                "preprocessing": {
                    "use_text_cleaner": False,
                    "use_deduplication": False,
                    "create_dataset_version": True,
                    "dataset_snapshot_path": str(snapshot),
                    "dataset_metadata_dir": str(metadata_dir),
                },
            }
        )
        result = pipeline.preprocess(samples)
        assert len(result) == 2
        assert snapshot.exists()
        assert (metadata_dir / "versioning_test_version.json").exists()

    def test_semantic_dedup_skips_without_model_by_default(self):
        samples = [
            DataSample(text="first unique sentence", label=0, source="test"),
            DataSample(text="second unique sentence", label=1, source="test"),
        ]
        pipeline = DynamicTrainingPipeline(
            {
                "preprocessing": {
                    "use_text_cleaner": False,
                    "use_deduplication": False,
                    "use_semantic_deduplication": True,
                    "semantic_deduplication": {
                        "similarity_threshold": 0.95,
                        # No embedding_model_name on purpose.
                    },
                }
            }
        )
        result = pipeline.preprocess(samples)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


class TestSplitData:
    def test_split_ratios(self):
        samples = _make_samples(100)
        pipeline = DynamicTrainingPipeline(
            {"split": {"train_ratio": 0.70, "val_ratio": 0.15, "seed": 42}}
        )
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
        pipeline = DynamicTrainingPipeline(
            {"split": {"train_ratio": 0.70, "val_ratio": 0.15, "seed": 0}}
        )
        train, val, test = pipeline.split_data(samples)
        # All samples accounted for
        assert len(train) + len(val) + len(test) == 3

    def test_split_invalid_ratios_raise(self):
        samples = _make_samples(10)
        pipeline = DynamicTrainingPipeline(
            {"split": {"train_ratio": 0.9, "val_ratio": 0.2, "seed": 42}}
        )
        with pytest.raises(ValueError, match="Invalid split ratios"):
            pipeline.split_data(samples)

    def test_stratified_split_preserves_class_presence(self):
        samples = []
        for i in range(30):
            samples.append(
                DataSample(
                    text=f"class_a_{i}",
                    label=0,
                    label_name="debugging",
                    source="test",
                )
            )
            samples.append(
                DataSample(
                    text=f"class_b_{i}",
                    label=1,
                    label_name="testing",
                    source="test",
                )
            )

        pipeline = DynamicTrainingPipeline(
            {
                "split": {
                    "train_ratio": 0.70,
                    "val_ratio": 0.15,
                    "seed": 42,
                    "strategy": "stratified",
                }
            }
        )
        train, val, test = pipeline.split_data(samples)

        train_labels = {s.label for s in train}
        val_labels = {s.label for s in val}
        test_labels = {s.label for s in test}
        assert train_labels == {0, 1}
        assert val_labels == {0, 1}
        assert test_labels == {0, 1}

    def test_stratified_auto_falls_back_on_sparse_labels(self):
        samples = [
            DataSample(text=f"s{i}", label=i, label_name=f"c{i}", source="test") for i in range(6)
        ]
        pipeline = DynamicTrainingPipeline(
            {
                "split": {
                    "train_ratio": 0.70,
                    "val_ratio": 0.15,
                    "seed": 42,
                    "strategy": "auto",
                }
            }
        )
        train, val, test = pipeline.split_data(samples)
        assert len(train) + len(val) + len(test) == 6

    def test_stratified_auto_falls_back_when_eval_pool_smaller_than_classes(self):
        samples = []
        for cls in range(20):
            samples.append(
                DataSample(text=f"class_{cls}_a", label=cls, label_name=f"c{cls}", source="test")
            )
            samples.append(
                DataSample(text=f"class_{cls}_b", label=cls, label_name=f"c{cls}", source="test")
            )

        pipeline = DynamicTrainingPipeline(
            {
                "split": {
                    "train_ratio": 0.80,
                    "val_ratio": 0.10,
                    "seed": 42,
                    "strategy": "auto",
                }
            }
        )
        train, val, test = pipeline.split_data(samples)
        assert len(train) + len(val) + len(test) == 40

    def test_split_with_leakage_check_enabled(self):
        samples = _make_samples(30)
        pipeline = DynamicTrainingPipeline(
            {
                "split": {
                    "train_ratio": 0.70,
                    "val_ratio": 0.15,
                    "seed": 42,
                    "run_leakage_check": True,
                    "leakage_ngram_size": 3,
                    "leakage_overlap_threshold": 0.7,
                }
            }
        )
        train, val, test = pipeline.split_data(samples)
        assert len(train) + len(val) + len(test) == 30


# ---------------------------------------------------------------------------
# Dry run (data loading without training)
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_loads_data(self):
        """Verify setup + load_data works without training."""
        pipeline = DynamicTrainingPipeline(
            {
                "data_sources": [
                    {"source_type": "synthetic", "name": "s", "params": {"total_examples": 31}}
                ],
                "preprocessing": {"use_text_cleaner": False, "use_deduplication": False},
            }
        )
        pipeline.setup_data_sources()
        samples = pipeline.load_data()
        # Should have approximately 31 samples
        assert len(samples) >= 20, f"Expected at least 20 samples, got {len(samples)}"

    def test_setup_creates_correct_source_types(self):
        pipeline = DynamicTrainingPipeline(
            {
                "data_sources": [
                    {"source_type": "synthetic", "name": "synth", "params": {"total_examples": 10}},
                ]
            }
        )
        pipeline.setup_data_sources()
        assert len(pipeline._sources) == 1
        assert pipeline._sources[0].source_type == "synthetic"

    def test_setup_wraps_weighted_top_level_sources(self, tmp_path):
        p1 = tmp_path / "source_a.jsonl"
        p2 = tmp_path / "source_b.jsonl"
        p1.write_text('{"text":"alpha task","label":1}\n')
        p2.write_text('{"text":"beta task","label":2}\n')

        pipeline = DynamicTrainingPipeline(
            {
                "data_sources": [
                    {
                        "source_type": "static",
                        "name": "a",
                        "weight": 0.8,
                        "params": {"file_paths": [str(p1)]},
                    },
                    {
                        "source_type": "static",
                        "name": "b",
                        "weight": 0.2,
                        "params": {"file_paths": [str(p2)]},
                    },
                ],
            }
        )
        pipeline.setup_data_sources()
        assert len(pipeline._sources) == 1
        assert pipeline._sources[0].source_type == "composite"


# ---------------------------------------------------------------------------
# Export (mocked)
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_skips_when_disabled(self):
        """No error raised when mlflow and model-ready are both disabled."""
        pipeline = DynamicTrainingPipeline(
            {
                "export": {"mlflow": {"enabled": False}, "publish_model_ready": False},
                "training": {"output_dir": "/tmp/fake"},
            }
        )
        # Should not raise
        pipeline.export({"overall": {"accuracy": 0.9, "f1": 0.88}})

    def test_export_graceful_on_mlflow_failure(self):
        """Pipeline continues even if MLflow is unreachable."""
        pipeline = DynamicTrainingPipeline(
            {
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
            }
        )
        # Should not raise — graceful degradation
        pipeline.export({"overall": {"accuracy": 0.8}})
