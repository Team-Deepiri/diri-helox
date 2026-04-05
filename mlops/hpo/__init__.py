"""
Hyperparameter optimization (HPO) for Helox via Optuna.

Parallel trials supported via n_jobs. Use run_one_trial with a merged config,
or run_optuna_sweep for a full study.
"""

from .objective import run_one_trial
from .optuna_sweep import run_optuna_sweep

__all__ = ["run_one_trial", "run_optuna_sweep"]
