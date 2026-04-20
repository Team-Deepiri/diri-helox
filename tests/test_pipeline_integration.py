"""
Integration tests for the full dynamic training pipeline.

These tests are marked with @pytest.mark.slow as they actually fine-tune a model.
Run all tests:   pytest tests/test_pipeline_integration.py -v
Run slow tests:  pytest tests/test_pipeline_integration.py -v -m slow

Requirements: torch, transformers, datasets, scikit-learn, numpy
"""

import json
import sys
from pathlib import Path

import pytest

_HELOX_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_HELOX_ROOT))

# Skip the entire module if ML deps are unavailable
try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import datasets  # noqa: F401

    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _ML_AVAILABLE,
    reason="torch / transformers / datasets not installed",
)


from pipelines.training.dynamic_training_pipeline import DynamicTrainingPipeline

# ---------------------------------------------------------------------------
# Full end-to-end pipeline (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
class TestEndToEndPipeline:
    def test_synthetic_100_full_pipeline(self, tmp_path):
        """
        Train an intent classifier on 100 synthetic samples for 1 epoch.
        Verifies model files exist and metrics are returned.
        """
        model_dir = tmp_path / "models" / "intent_classifier"
        config = {
            "pipeline_name": "integration_test",
            "data_sources": [
                {
                    "source_type": "synthetic",
                    "name": "synth_100",
                    "params": {"total_examples": 100, "use_ollama": False},
                }
            ],
            "preprocessing": {
                "use_text_cleaner": False,
                "use_deduplication": False,
                "min_text_length": 5,
            },
            "training": {
                "trainer_type": "intent_classifier",
                "model_name": "bert-base-uncased",
                "num_labels": 31,
                "num_epochs": 1,
                "batch_size": 8,
                "learning_rate": 1e-5,
                "max_length": 64,
                "output_dir": str(model_dir),
            },
            "evaluation": {
                "enabled": True,
                "report_path": str(model_dir / "evaluation_report.json"),
            },
            "export": {
                "mlflow": {"enabled": False},
                "publish_model_ready": False,
            },
            "split": {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15, "seed": 42},
        }

        pipeline = DynamicTrainingPipeline(config)
        metrics = pipeline.run()

        # Model files created
        assert model_dir.exists(), "Model directory was not created"
        assert (model_dir / "config.json").exists(), "config.json missing"
        assert (model_dir / "category_map.json").exists(), "category_map.json missing"
        assert (model_dir / "training_info.json").exists(), "training_info.json missing"

        # Evaluation report created
        report_path = model_dir / "evaluation_report.json"
        assert report_path.exists(), "evaluation_report.json missing"
        with open(report_path) as f:
            report = json.load(f)
        assert "metrics" in report

        # Metrics structure is sane
        assert "overall" in metrics
        overall = metrics["overall"]
        assert "accuracy" in overall
        assert 0.0 <= overall["accuracy"] <= 1.0
        assert "f1" in overall

        print(f"\n  Accuracy: {overall['accuracy']:.4f}")
        print(f"  F1:       {overall['f1']:.4f}")

    def test_dry_run_does_not_train(self, tmp_path):
        """Dry run should return data without creating model files."""
        model_dir = tmp_path / "models" / "dry_run"
        config = {
            "pipeline_name": "dry_run_test",
            "data_sources": [
                {"source_type": "synthetic", "name": "s", "params": {"total_examples": 31}}
            ],
            "preprocessing": {"use_text_cleaner": False, "use_deduplication": False},
            "training": {"trainer_type": "intent_classifier", "output_dir": str(model_dir)},
            "export": {"mlflow": {"enabled": False}, "publish_model_ready": False},
        }
        pipeline = DynamicTrainingPipeline(config)
        pipeline.setup_data_sources()
        samples = pipeline.load_data()
        samples = pipeline.preprocess(samples)
        train, val, test = pipeline.split_data(samples)

        # No model created
        assert not model_dir.exists()
        # Data was loaded
        assert len(train) + len(val) + len(test) == len(samples)
        assert len(samples) >= 20
