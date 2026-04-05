"""
Optuna-based hyperparameter sweeps for Helox.

Runs trials in parallel via n_jobs. Minimizes eval_loss by default.
"""

from typing import Dict, Any, Optional
import optuna
from optuna.samplers import TPESampler

from .objective import run_one_trial, DEFAULT_METRIC


# Default search space (ranges) for the versioned training pipeline
DEFAULT_OPTUNA_SPACE = {
    "learning_rate": (1e-5, 5e-4),      # log scale, QLoRA sweet spot
    "batch_size": (2, 8),                # bounded by GPU memory
    "weight_decay": (0.01, 0.1),         # AdamW standard range
    "warmup_steps": (50, 200),           # ~5-10% of typical total steps
    "num_epochs": (1, 3),                # QLoRA overfits fast on small data
    "lora_rank": [8, 16, 32],            # categorical instead of range
    "lora_alpha": [16, 32, 64],          # categorical, usually 2x rank
    "lora_dropout": (0.03, 0.1),         # tight range around common values
}


def _suggest_value(trial: optuna.Trial, key: str, space: tuple, log_scale: bool = False) -> Any:
    """Suggest one value from the trial. space = (low, high) for float/int."""
    low, high = space[0], space[1]
    if isinstance(low, int) and isinstance(high, int):
        return trial.suggest_int(key, low, high)
    if log_scale:
        return trial.suggest_float(key, low, high, log=True)
    return trial.suggest_float(key, low, high)


def run_optuna_sweep(
    base_config: Dict[str, Any],
    n_trials: int = 20,
    n_jobs: int = 1,
    study_name: Optional[str] = None,
    metric_key: str = DEFAULT_METRIC,
    space: Optional[Dict[str, tuple]] = None,
    log_scale_keys: Optional[set] = None,
    seed: Optional[int] = None,
    disable_tracking: bool = True,
) -> optuna.Study:
    """
    Run an Optuna study to minimize the given metric (e.g. eval_loss).

    Args:
        base_config: Base pipeline config (dataset_spec, base_model, etc.).
                     Tunable keys will be overridden by suggested values.
        n_trials: Number of trials to run.
        n_jobs: Number of parallel trials (e.g. 4 for 4 runs at once).
        study_name: Optuna study name (for storage/display).
        metric_key: Key from eval_results to minimize.
        space: Search space: dict of param_name -> (low, high). Defaults to DEFAULT_OPTUNA_SPACE.
        log_scale_keys: Param names that use log scale (e.g. {"learning_rate"}).
        seed: Random seed for reproducibility.
        disable_tracking: If True, do not log each trial to MLflow/W&B.

    Returns:
        The completed Optuna Study (study.best_params, study.best_value).
    """
    space = space or DEFAULT_OPTUNA_SPACE
    log_scale_keys = log_scale_keys or {"learning_rate"}

    def objective(trial: optuna.Trial) -> float:
        overrides = {}
        for key, (low, high) in space.items():
            overrides[key] = _suggest_value(
                trial, key, (low, high), log_scale=(key in log_scale_keys)
            )
        config = {**base_config, **overrides}
        return run_one_trial(
            config,
            metric_key=metric_key,
            disable_tracking=disable_tracking,
        )

    sampler = TPESampler(seed=seed, n_startup_trials=min(5, n_trials))
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name or "helox_hpo",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study
