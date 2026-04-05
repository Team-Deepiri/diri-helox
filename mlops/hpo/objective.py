"""
Shared objective for HPO: run one training trial and return the metric to optimize.

VersionedTrainingPipeline is imported lazily so `from mlops.hpo import ...` works
without peft/torch until you actually run a trial.
"""

from typing import Dict, Any

# Default metric key to minimize (from Trainer.evaluate())
DEFAULT_METRIC = "eval_loss"


def run_one_trial(
    config: Dict[str, Any],
    metric_key: str = DEFAULT_METRIC,
    disable_tracking: bool = True,
) -> float:
    """
    Run one training trial with the given config and return the objective metric.

    Args:
        config: Full pipeline config (dataset_spec, base_model, learning_rate, etc.).
                Tunable keys are merged by the Optuna/Hyperopt wrappers.
        metric_key: Key from eval_results to optimize (e.g. "eval_loss", "eval_perplexity").
        disable_tracking: If True, disable MLflow/W&B for this trial to avoid
                          polluting the main experiment (HPO runs many trials).

    Returns:
        The value to minimize (e.g. eval_loss). Raises if the metric is missing.
    """
    from pipelines.training.versioned_training_pipeline import VersionedTrainingPipeline

    config = dict(config)
    if disable_tracking:
        config["mlflow_uri"] = ""
        config["use_wandb"] = False

    pipeline = VersionedTrainingPipeline(config)
    if not disable_tracking:
        pipeline.setup_experiment_tracking()
    pipeline.setup_model()
    train_dataset, val_dataset = pipeline.load_and_prepare_data()
    output_dir, metadata = pipeline.train(train_dataset, val_dataset)

    if not disable_tracking and pipeline.tracker:
        pipeline.tracker.end_run()

    eval_results = metadata.get("eval_results") or {}
    if metric_key not in eval_results:
        raise KeyError(
            f"Metric '{metric_key}' not in eval_results. Available: {list(eval_results.keys())}"
        )
    return float(eval_results[metric_key])
