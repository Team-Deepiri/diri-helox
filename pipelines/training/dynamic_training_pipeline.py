"""
DynamicTrainingPipeline: config-driven training orchestrator that can be
pointed at any data source — static JSONL, Redis streams, synthetic generation,
self-feedback, or a composite of any of the above.

Pipeline flow:
  1. Load data from all configured sources
  2. Preprocess (TextCleaner + ExactDeduplicator via deepiri-dataset-processor)
  3. Split into train / val / test
  4. Train (IntentClassifierTrainer)
  5. Evaluate (ModelEvaluator)
  6. Export to MLflow + publish model-ready event (graceful if unavailable)
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HELOX_ROOT = Path(__file__).parent.parent.parent
if str(_HELOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_HELOX_ROOT))

from data_sources.base import DataSample
from data_sources.factory import create_data_sources_from_config


class DynamicTrainingPipeline:
    """
    Config-driven end-to-end training pipeline.

    Args:
        config (dict): pipeline configuration (can be loaded from JSON via from_file())

    Example config (minimal):
        {
          "pipeline_name": "my_pipeline",
          "data_sources": [
            {"source_type": "synthetic", "name": "s100", "params": {"total_examples": 100}}
          ],
          "training": {"trainer_type": "intent_classifier", "model_name": "bert-base-uncased"},
          "evaluation": {"enabled": true},
          "export": {"mlflow": {"enabled": false}, "publish_model_ready": false}
        }
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._sources = []
        self._trainer = None
        self._evaluator = None

    @classmethod
    def from_file(cls, path: str) -> "DynamicTrainingPipeline":
        with open(path) as f:
            return cls(json.load(f))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_data_sources(self) -> None:
        """Instantiate all configured data sources."""
        self._sources = create_data_sources_from_config(
            self.config.get("data_sources", [])
        )

    def load_data(self) -> List[DataSample]:
        """Collect samples from all sources."""
        all_samples: List[DataSample] = []
        for source in self._sources:
            print(f"  Loading source '{source.name}' ({source.source_type})...")
            samples = source.load()
            print(f"    -> {len(samples)} samples")
            all_samples.extend(samples)
        print(f"  Total loaded: {len(all_samples)} samples")
        return all_samples

    def preprocess(self, samples: List[DataSample]) -> List[DataSample]:
        """
        Apply text cleaning and deduplication via deepiri-dataset-processor.
        Falls back gracefully if the package is unavailable.
        """
        cfg = self.config.get("preprocessing", {})
        min_len = cfg.get("min_text_length", 5)

        # Filter empty / too-short samples
        samples = [s for s in samples if s.text and len(s.text.strip()) >= min_len]

        if cfg.get("use_text_cleaner", True):
            try:
                from deepiri_dataset_processor.cleaning.text_cleaner import TextCleaner
                cleaner = TextCleaner(min_length=min_len, remove_boilerplate=False)
                cleaned = []
                for s in samples:
                    text = cleaner.clean(s.text) or s.text.strip()
                    cleaned.append(DataSample(
                        text=text,
                        label=s.label,
                        label_name=s.label_name,
                        metadata=s.metadata,
                        source=s.source,
                    ))
                samples = cleaned
                print(f"  Text cleaning applied: {len(samples)} samples remain")
            except ImportError:
                print("  Warning: deepiri-dataset-processor not available, skipping text cleaning")

        if cfg.get("use_deduplication", True):
            try:
                from deepiri_dataset_processor.deduplication.exact_dedup import ExactDeduplicator
                dedup = ExactDeduplicator()
                texts = [s.text for s in samples]
                unique_texts = set(dedup.filter_duplicates(texts))
                before = len(samples)
                samples = [s for s in samples if s.text in unique_texts]
                print(f"  Deduplication: {before} -> {len(samples)} samples")
            except ImportError:
                print("  Warning: deepiri-dataset-processor not available, skipping deduplication")

        return samples

    def split_data(
        self, samples: List[DataSample]
    ) -> Tuple[List[DataSample], List[DataSample], List[DataSample]]:
        """Split into train / val / test according to config ratios."""
        split_cfg = self.config.get("split", {})
        train_ratio = split_cfg.get("train_ratio", 0.70)
        val_ratio = split_cfg.get("val_ratio", 0.15)
        seed = split_cfg.get("seed", 42)

        rng = random.Random(seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = shuffled[:n_train]
        val = shuffled[n_train: n_train + n_val]
        test = shuffled[n_train + n_val:]

        print(f"  Split: {len(train)} train / {len(val)} val / {len(test)} test")
        return train, val, test

    def train(
        self, train_samples: List[DataSample], val_samples: List[DataSample]
    ) -> Dict[str, Any]:
        """Train using the configured trainer type."""
        training_cfg = dict(self.config.get("training", {}))
        trainer_type = training_cfg.pop("trainer_type", "intent_classifier")

        if trainer_type == "intent_classifier":
            from training.intent_classifier_trainer import IntentClassifierTrainer
            self._trainer = IntentClassifierTrainer(**training_cfg)
        else:
            raise ValueError(f"Unknown trainer_type: '{trainer_type}'")

        metrics = self._trainer.train(train_samples, val_samples)
        self._trainer.save()
        return metrics

    def evaluate(self, test_samples: List[DataSample]) -> Dict[str, Any]:
        """Evaluate the trained model on test samples."""
        from evaluation.model_evaluator import ModelEvaluator

        model_path = self._trainer.get_model_path() if self._trainer else \
            self.config.get("training", {}).get("output_dir", "models/intent_classifier")

        self._evaluator = ModelEvaluator(model_path=model_path)
        metrics = self._evaluator.evaluate(test_samples)

        eval_cfg = self.config.get("evaluation", {})
        report_path = eval_cfg.get(
            "report_path",
            str(Path(model_path) / "evaluation_report.json"),
        )
        self._evaluator.save_report(metrics, report_path)
        return metrics

    def export(self, metrics: Dict[str, Any]) -> None:
        """Export to MLflow and publish model-ready event (graceful on failure)."""
        export_cfg = self.config.get("export", {})
        mlflow_cfg = export_cfg.get("mlflow", {})
        model_path = self._trainer.get_model_path() if self._trainer else ""
        pipeline_name = self.config.get("pipeline_name", "dynamic_pipeline")

        if mlflow_cfg.get("enabled", False):
            try:
                from mlops.infrastructure.experiment_tracker import ExperimentTracker
                tracker = ExperimentTracker(
                    experiment_name=mlflow_cfg.get("experiment_name", pipeline_name),
                    tracking_uri=mlflow_cfg.get("tracking_uri", "http://localhost:5000"),
                )
                tracker.start_run(run_name=pipeline_name)
                # Log training params (filter non-serialisable values)
                safe_params = {
                    k: str(v) for k, v in self.config.get("training", {}).items()
                    if k != "trainer_type"
                }
                tracker.log_params(safe_params)
                # Log overall metrics
                overall = metrics.get("overall", {})
                if overall:
                    tracker.log_metrics({k: v for k, v in overall.items() if isinstance(v, (int, float))})
                if model_path:
                    tracker.log_model(model_path)
                if mlflow_cfg.get("register_model", False) and tracker.current_run:
                    tracker.register_model(
                        tracker.current_run.info.run_id,
                        mlflow_cfg.get("model_name", "intent-classifier"),
                    )
                tracker.end_run()
                print("  MLflow export complete")
            except Exception as exc:
                print(f"  Warning: MLflow export failed ({exc}) — model still saved locally")

        if export_cfg.get("publish_model_ready", False):
            try:
                import asyncio
                from mlops.model_registry.model_registrar import ModelRegistrar
                registrar = ModelRegistrar()
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
                asyncio.run(registrar.register_and_publish(
                    model_name=mlflow_cfg.get("model_name", "intent-classifier"),
                    version=version,
                    model_path=model_path,
                    metadata={
                        "accuracy": metrics.get("overall", {}).get("accuracy"),
                        "f1": metrics.get("overall", {}).get("f1"),
                        "pipeline_name": pipeline_name,
                    },
                ))
                print("  Model-ready event published")
            except Exception as exc:
                print(f"  Warning: model-ready event failed ({exc}) — continuing")

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline end-to-end."""
        pipeline_name = self.config.get("pipeline_name", "dynamic_pipeline")
        print("\n" + "=" * 60)
        print(f"DynamicTrainingPipeline: {pipeline_name}")
        print("=" * 60)

        print("\n[1/5] Setting up data sources...")
        self.setup_data_sources()

        print("\n[2/5] Loading data...")
        samples = self.load_data()
        if not samples:
            raise RuntimeError("No samples loaded — check your data source config")

        print("\n[3/5] Preprocessing...")
        samples = self.preprocess(samples)

        print("\n[4/5] Splitting data...")
        train, val, test = self.split_data(samples)

        print("\n[5a] Training...")
        train_metrics = self.train(train, val)

        eval_metrics: Dict[str, Any] = {}
        if self.config.get("evaluation", {}).get("enabled", True) and test:
            print("\n[5b] Evaluating...")
            eval_metrics = self.evaluate(test)
            overall = eval_metrics.get("overall", {})
            print(f"  Accuracy : {overall.get('accuracy', 0):.4f}")
            print(f"  F1       : {overall.get('f1', 0):.4f}")

        print("\n[5c] Exporting...")
        self.export(eval_metrics or train_metrics)

        result = eval_metrics or train_metrics
        print("\n" + "=" * 60)
        print(f"Pipeline '{pipeline_name}' complete.")
        print("=" * 60 + "\n")
        return result
