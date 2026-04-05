#!/usr/bin/env python3
"""
Run hyperparameter optimization with Optuna for the versioned training pipeline.

Usage (from diri-helox):
  python scripts/run_hpo.py --config configs/hpo_base.json --n-trials 8 --n-jobs 2

Requires a base config JSON with dataset_spec, base_model, and other fixed settings.
Tunable parameters (learning_rate, batch_size, etc.) are overridden by Optuna.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root so pipelines and mlops are importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna HPO for versioned training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to base config JSON (dataset_spec, base_model, version_db_url, etc.)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (trials run in parallel)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="eval_loss",
        help="Metric to minimize (e.g. eval_loss, eval_perplexity)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--log-trials",
        action="store_true",
        help="Log each trial to MLflow/W&B (default: disabled for HPO)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_config = json.load(f)

    from mlops.hpo.optuna_sweep import run_optuna_sweep

    study = run_optuna_sweep(
        base_config,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        metric_key=args.metric,
        seed=args.seed,
        disable_tracking=not args.log_trials,
    )
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
