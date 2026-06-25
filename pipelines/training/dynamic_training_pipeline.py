"""
DynamicTrainingPipeline: config-driven training orchestrator that can be
pointed at any data source — static JSONL, Redis streams, synthetic generation,
self-feedback, or a composite of any of the above.

Pipeline flow:
  1. Load data from all configured sources
  2. Preprocess (TextCleaner + ExactDeduplicator via deepiri-dataset-processor)
  3. Split into train / val / test
  4. Train (IntentClassifierTrainer wrapped in TrainingOrchestrator callbacks)
  5. Evaluate (ModelEvaluator)
  6. Export to MLflow + publish model-ready event (graceful if unavailable)

Shared infrastructure:
  - deepiri-training-orchestrator: ExperimentTracker, ReproducibilityController,
    CheckpointCallback, EarlyStoppingCallback (consistent with all Deepiri services)
  - deepiri-dataset-processor: TextCleaner, ExactDeduplicator
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from deepiri_training_orchestrator import (
    CheckpointCallback,
    EarlyStoppingCallback,
    ExperimentTracker,
    LoggingCallback,
    initialize_deterministic_training,
)

from data_sources.base import DataSample
from data_sources.factory import create_data_sources_from_config


class DynamicTrainingPipeline:
    """
    Config-driven end-to-end training pipeline.

    Uses deepiri-training-orchestrator for reproducibility seeding, experiment
    tracking (MLflow), and training callbacks — keeping behaviour consistent with
    other Deepiri services that run training loops.

    Args:
        config (dict): pipeline configuration (load via from_file())

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
        self._sources: List[Any] = []
        self._trainer: Any = None
        self._evaluator: Any = None
        self._ingestion_logger: Any = None

    @classmethod
    def from_file(cls, path: str) -> "DynamicTrainingPipeline":
        with open(path) as f:
            config = json.load(f)
        config = cls._expand_env_vars(config)
        return cls(config)

    @staticmethod
    def _expand_env_vars(obj: Any) -> Any:
        """Recursively expand ${VAR} environment variables in config string values."""
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        if isinstance(obj, dict):
            return {k: DynamicTrainingPipeline._expand_env_vars(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [DynamicTrainingPipeline._expand_env_vars(v) for v in obj]
        return obj

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup_data_sources(self) -> None:
        """
        Instantiate all configured data sources.

        If multiple top-level sources are configured with non-default weights,
        automatically wrap them in a CompositeDataSource so weights are applied.
        """
        self._sources = create_data_sources_from_config(self.config.get("data_sources", []))
        if len(self._sources) <= 1:
            return

        has_weighted_sources = any(abs(float(s.config.weight) - 1.0) > 1e-9 for s in self._sources)
        if not has_weighted_sources:
            return

        from data_sources.base import DataSourceConfig
        from data_sources.composite_source import CompositeDataSource

        seed = self.config.get("split", {}).get("seed", 42)
        composite_cfg = DataSourceConfig(
            source_type="composite",
            name="auto_weighted_sources",
            params={"seed": seed},
        )
        self._sources = [CompositeDataSource(composite_cfg, self._sources)]

    def load_data(self) -> List[DataSample]:
        """Collect samples from all sources."""
        pipeline_name = self.config.get("pipeline_name", "pipeline")
        log_cfg = self.config.get("observability", {})
        log_path = log_cfg.get("ingestion_log_path")
        try:
            from observability.data_ingestion_logger import DataIngestionLogger

            self._ingestion_logger = DataIngestionLogger(
                log_path=log_path, pipeline_name=pipeline_name
            )
        except ImportError:
            self._ingestion_logger = None

        all_samples: List[DataSample] = []
        for source in self._sources:
            info = source.get_info()
            mode = info.get("mode", "")
            print(
                f"  Loading source '{source.name}'"
                f" ({source.source_type}{', ' + mode if mode else ''})..."
            )
            samples = source.load()
            print(f"    -> {len(samples)} samples")
            all_samples.extend(samples)
        print(f"  Total loaded: {len(all_samples)} samples")

        if self._ingestion_logger:
            self._ingestion_logger.record("ingest", all_samples)

        return all_samples

    def preprocess(self, samples: List[DataSample]) -> List[DataSample]:
        """
        Apply text cleaning and deduplication via deepiri-dataset-processor.
        Falls back gracefully if the package is unavailable.

        Optional (off by default): quality scoring via deepiri-dataset-processor
        QualityChecker. Enable with:
          preprocessing.use_quality_checker = true
        """
        cfg = self.config.get("preprocessing", {})
        min_len = cfg.get("min_text_length", 5)

        samples = [s for s in samples if s.text and len(s.text.strip()) >= min_len]

        if cfg.get("use_text_cleaner", True):
            try:
                from deepiri_dataset_processor.cleaning.text_cleaner import TextCleaner

                cleaner = TextCleaner(min_length=min_len, remove_boilerplate=False)
                cleaned = []
                for s in samples:
                    text = cleaner.clean(s.text) or s.text.strip()
                    cleaned.append(
                        DataSample(
                            text=text,
                            label=s.label,
                            label_name=s.label_name,
                            metadata=s.metadata,
                            source=s.source,
                        )
                    )
                samples = cleaned
                print(f"  Text cleaning applied: {len(samples)} samples remain")
            except ImportError:
                print("  Warning: deepiri-dataset-processor not available, skipping text cleaning")

        if cfg.get("use_deduplication", True):
            try:
                from collections import Counter

                from deepiri_dataset_processor.deduplication.exact_dedup import ExactDeduplicator

                dedup = ExactDeduplicator()
                texts = [s.text for s in samples]
                deduped_texts = dedup.filter_duplicates(texts)
                keep_budget = Counter(deduped_texts)
                before = len(samples)
                filtered_samples: List[DataSample] = []
                for sample in samples:
                    if keep_budget[sample.text] <= 0:
                        continue
                    filtered_samples.append(sample)
                    keep_budget[sample.text] -= 1
                samples = filtered_samples
                print(f"  Deduplication: {before} -> {len(samples)} samples")
            except ImportError:
                print("  Warning: deepiri-dataset-processor not available, skipping deduplication")

        # Optional quality scoring stage (non-blocking unless explicitly enforced).
        if cfg.get("use_quality_checker", False):
            try:
                from deepiri_dataset_processor.quality.checker import QualityChecker, QualityConfig

                qc_cfg_raw = cfg.get("quality_checker", {})
                qc_cfg = QualityConfig(
                    **{
                        k: v
                        for k, v in qc_cfg_raw.items()
                        if k in QualityConfig.__dataclass_fields__
                    }
                )
                checker = QualityChecker(config=qc_cfg)

                rows = [
                    {
                        "text": s.text,
                        "label": s.label_name if s.label_name is not None else s.label,
                        "source": s.source,
                    }
                    for s in samples
                ]
                report = checker.check_quality(
                    rows,
                    dataset_id=self.config.get("pipeline_name", "dynamic_training_pipeline"),
                )

                overall = float(report.overall_score)
                min_quality = float(cfg.get("min_quality_score", 0.0))
                print(
                    "  QualityChecker: "
                    f"overall={overall:.3f}, failed_metrics={report.summary.get('failed_metrics', 0)}"
                )
                if min_quality > 0 and overall < min_quality:
                    msg = f"Quality score {overall:.3f} below configured min_quality_score={min_quality:.3f}"
                    if cfg.get("enforce_quality_threshold", False):
                        raise ValueError(msg)
                    print(f"  Warning: {msg}")
            except ImportError:
                print("  Warning: quality checker unavailable, skipping quality scoring")

        if self._ingestion_logger:
            self._ingestion_logger.record("preprocess", samples)

        return samples

    def split_data(
        self, samples: List[DataSample]
    ) -> Tuple[List[DataSample], List[DataSample], List[DataSample]]:
        """
        Split into train / val / test.

        Uses ReproducibilityController (deepiri-training-orchestrator) to seed
        the shuffle, so the split is deterministic and fingerprinted consistently
        with the rest of the training run.
        """
        split_cfg = self.config.get("split", {})
        train_ratio = split_cfg.get("train_ratio", 0.70)
        val_ratio = split_cfg.get("val_ratio", 0.15)
        seed = split_cfg.get("seed", 42)

        initialize_deterministic_training(seed=seed)

        import random

        rng = random.Random(seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]

        print(f"  Split: {len(train)} train / {len(val)} val / {len(test)} test")
        if self._ingestion_logger:
            self._ingestion_logger.record("split_train", train)
            self._ingestion_logger.record("split_val", val)
            self._ingestion_logger.record("split_test", test)
            self._ingestion_logger.print_summary()
        return train, val, test

    def train(
        self, train_samples: List[DataSample], val_samples: List[DataSample]
    ) -> Dict[str, Any]:
        """
        Train using the configured trainer type.

        Supported trainer_type values:
          - "intent_classifier"       — BERT/DeBERTa fine-tuning for 31-category intent
                                        classification (default). Used by Cyrex for task routing.
          - "instruction_finetuning"  — Causal LM fine-tuning with response-only loss masking.
                                        Used for Persola personality fine-tuning: samples must
                                        have instruction/response format in metadata.
          - "bandit"                  — Contextual multi-armed bandit (Thompson sampling) for
                                        challenge selection. Samples must carry challenge_type
                                        and reward in metadata. Used by Cyrex engagement service.

        All trainer_types (except "bandit") are wrapped with deepiri-training-orchestrator
        callbacks for structured logging, checkpointing, and early stopping.
        """
        training_cfg = dict(self.config.get("training", {}))
        trainer_type = training_cfg.pop("trainer_type", "intent_classifier")

        model_output_dir = Path(training_cfg.get("output_dir", "models/intent_classifier"))

        if trainer_type == "intent_classifier":
            from training.intent_classifier_trainer import IntentClassifierTrainer

            # HF evaluate metrics keep the "eval_" prefix in the callback adapter.
            _callbacks = [
                LoggingCallback(every=10),
                CheckpointCallback(
                    directory=model_output_dir / "orchestrator_checkpoints",
                    every=50,
                ),
                EarlyStoppingCallback(monitor="eval_f1", patience=3, mode="max"),
            ]
            self._trainer = IntentClassifierTrainer(
                orchestrator_callbacks=_callbacks, **training_cfg
            )
            metrics: Dict[str, Any] = self._trainer.train(train_samples, val_samples)
            self._trainer.save()
            return metrics

        if trainer_type == "instruction_finetuning":
            # Causal LM fine-tuning with instruction masking — for Persola personality models.
            # Samples are expected to have 'instruction' and 'response' in metadata; falls back
            # to using text as the full sequence if not present.
            from training.hf_instruction_finetuning_trainer import HFInstructionFinetuningTrainer

            _callbacks = [
                LoggingCallback(every=10),
                CheckpointCallback(
                    directory=model_output_dir / "orchestrator_checkpoints",
                    every=50,
                ),
                EarlyStoppingCallback(monitor="eval_loss", patience=3, mode="min"),
            ]
            self._trainer = HFInstructionFinetuningTrainer(
                orchestrator_callbacks=_callbacks, **training_cfg
            )
            metrics = self._trainer.train(train_samples, val_samples)
            self._trainer.save()
            return metrics

        if trainer_type == "bandit":
            # Contextual multi-armed bandit for challenge selection (Thompson sampling).
            # Samples must have 'challenge_type' and 'reward' in metadata.
            # Trained bandit is saved as a pickle alongside the model output dir.
            return self._train_bandit(train_samples, training_cfg)

        raise ValueError(
            f"Unknown trainer_type: '{trainer_type}'. "
            "Valid options: 'intent_classifier', 'instruction_finetuning', 'bandit'."
        )

    def _train_bandit(
        self, samples: List[DataSample], training_cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train the contextual multi-armed bandit on challenge-engagement feedback samples.

        Each DataSample must carry in metadata:
          - challenge_type (str): the challenge type arm that was selected
          - reward         (float): engagement reward signal (0.0–1.0)
          - context        (list[float], optional): feature vector for the user/session context

        The trained bandit is saved to <output_dir>/bandit.pkl and loaded by Cyrex's
        engagement service at inference time to pick personalized challenges.
        """
        from pipelines.training.bandit_training import ContextualBandit

        challenge_types = training_cfg.get(
            "challenge_types",
            ["quiz", "code", "creative", "analysis", "debugging", "design"],
        )
        context_dim = training_cfg.get("context_dim", 10)
        output_dir = Path(training_cfg.get("output_dir", "models/bandit"))

        bandit = ContextualBandit(challenge_types=challenge_types, context_dim=context_dim)

        training_data = []
        skipped = 0
        for s in samples:
            challenge_type = s.metadata.get("challenge_type") or s.label_name
            if not challenge_type:
                skipped += 1
                continue
            training_data.append(
                {
                    "challenge_type": challenge_type,
                    "reward": float(s.metadata.get("reward", 1.0)),
                    "context": s.metadata.get("context", []),
                }
            )

        if skipped:
            print(f"  Warning: {skipped} samples skipped (missing challenge_type/label_name)")

        bandit.train(training_data)
        output_dir.mkdir(parents=True, exist_ok=True)
        bandit_path = str(output_dir / "bandit.pkl")
        bandit.save(bandit_path)

        counts = bandit.counts
        metrics = {
            "trainer_type": "bandit",
            "samples_trained": len(training_data),
            "challenge_counts": counts,
            "bandit_path": bandit_path,
        }
        print(f"  Bandit trained on {len(training_data)} samples → {bandit_path}")
        return metrics

    def evaluate(self, test_samples: List[DataSample]) -> Dict[str, Any]:
        """
        Evaluate the trained model on test samples.

        Bandit trainer_type uses its own metrics from training and does not run
        ModelEvaluator (there is no classification model to evaluate).
        """
        trainer_type = self.config.get("training", {}).get("trainer_type", "intent_classifier")
        if trainer_type == "bandit":
            # Bandit evaluation is reward-based, not classification-based.
            # Return empty so the pipeline continues without crashing.
            return {}

        from evaluation.model_evaluator import ModelEvaluator

        model_path = (
            self._trainer.get_model_path()
            if self._trainer
            else self.config.get("training", {}).get("output_dir", "models/intent_classifier")
        )

        self._evaluator = ModelEvaluator(model_path=model_path)
        metrics: Dict[str, Any] = self._evaluator.evaluate(test_samples)

        eval_cfg = self.config.get("evaluation", {})
        report_path = eval_cfg.get(
            "report_path",
            str(Path(model_path) / "evaluation_report.json"),
        )
        self._evaluator.save_report(metrics, report_path)
        return metrics

    def export(self, metrics: Dict[str, Any]) -> None:
        """
        Export to MLflow and publish model-ready event.

        Uses ExperimentTracker from deepiri-training-orchestrator — the same
        tracker used across Cyrex and other Deepiri services — for consistent
        run naming, param logging, and model registration.
        """
        export_cfg = self.config.get("export", {})
        mlflow_cfg = export_cfg.get("mlflow", {})
        model_path = self._trainer.get_model_path() if self._trainer else ""
        pipeline_name = self.config.get("pipeline_name", "dynamic_pipeline")

        if mlflow_cfg.get("enabled", False):
            try:
                tracker = ExperimentTracker(
                    experiment_name=mlflow_cfg.get("experiment_name", pipeline_name),
                    tracking_uri=mlflow_cfg.get("tracking_uri", "http://localhost:5000"),
                    use_wandb=mlflow_cfg.get("use_wandb", False),
                    wandb_project=mlflow_cfg.get("wandb_project"),
                )
                tracker.start_run(run_name=pipeline_name)
                tracker.log_git_info()

                training_params = {
                    k: v for k, v in self.config.get("training", {}).items() if k != "trainer_type"
                }
                safe_params = {k: str(v) for k, v in training_params.items()}

                # Generate a config fingerprint for reproducibility tracing
                from deepiri_training_orchestrator import ReproducibilityController

                repro = ReproducibilityController(seed=self.config.get("split", {}).get("seed", 42))
                fingerprint = repro.generate_training_fingerprint(training_params)
                safe_params["training_fingerprint"] = fingerprint

                tracker.log_params(safe_params)

                overall = metrics.get("overall", {})
                if overall:
                    tracker.log_metrics(
                        {k: v for k, v in overall.items() if isinstance(v, (int, float))}
                    )

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
                asyncio.run(
                    registrar.register_and_publish(
                        model_name=mlflow_cfg.get("model_name", "intent-classifier"),
                        version=version,
                        model_path=model_path,
                        metadata={
                            "accuracy": metrics.get("overall", {}).get("accuracy"),
                            "f1": metrics.get("overall", {}).get("f1"),
                            "pipeline_name": pipeline_name,
                        },
                    )
                )
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
