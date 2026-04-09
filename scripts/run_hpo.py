#!/usr/bin/env python3
"""
Run hyperparameter optimization with Optuna for the versioned training pipeline.

Two modes:
  CLI:         python scripts/run_hpo.py --config <file> --n-trials 8 --n-jobs 2
  Interactive: python scripts/run_hpo.py --interactive
"""

import argparse
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Interactive TUI helpers (Rich)
# ---------------------------------------------------------------------------

def _pick_config() -> Path:
    """Prompt user to pick a training config JSON from configs/.

    Only shows configs that contain 'dataset_spec' or 'base_model'
    (i.e. training configs, not data/model architecture configs).
    """
    from rich.prompt import Prompt

    configs_dir = _project_root / "configs"
    all_jsons = sorted(configs_dir.glob("*.json"))

    candidates = []
    for p in all_jsons:
        try:
            with open(p, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and ("dataset_spec" in data or "base_model" in data):
                candidates.append(p)
        except (json.JSONDecodeError, OSError):
            continue

    if not candidates:
        print(f"No training configs found in {configs_dir}")
        print("A training config must contain 'dataset_spec' or 'base_model'.")
        sys.exit(1)

    print()
    for i, p in enumerate(candidates, 1):
        print(f"  [{i}] {p.name}")
    choice = Prompt.ask(
        "\nSelect config file",
        choices=[str(i) for i in range(1, len(candidates) + 1)],
        default="1",
    )
    return candidates[int(choice) - 1]


def _interactive_setup() -> dict:
    """Collect sweep settings interactively via Rich prompts."""
    from rich.prompt import IntPrompt, Prompt, Confirm

    config_path = _pick_config()
    with open(config_path, "r") as f:
        base_config = json.load(f)
    print(f"\n  Config: {config_path.name}")

    n_trials = IntPrompt.ask("Number of trials", default=20)
    n_jobs = IntPrompt.ask("Parallel jobs", default=1)
    metric = Prompt.ask(
        "Metric to minimize",
        choices=["eval_loss", "eval_perplexity"],
        default="eval_loss",
    )
    study_name = Prompt.ask("Study name", default="helox_hpo")
    log_trials = Confirm.ask("Log each trial to MLflow/W&B?", default=False)

    return {
        "base_config": base_config,
        "n_trials": n_trials,
        "n_jobs": n_jobs,
        "metric_key": metric,
        "study_name": study_name,
        "disable_tracking": not log_trials,
    }


def _build_trial_callback():
    """Return an Optuna callback that updates a Rich Live table after each trial."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    def callback(study, trial):
        table = Table(
            title=f"Optuna Study: {study.study_name}",
            show_lines=True,
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Value", style="cyan", width=12)
        table.add_column("Params", style="white")
        table.add_column("", width=6)

        for t in study.trials:
            is_best = t.number == study.best_trial.number
            params_str = ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in t.params.items()
            )
            marker = "[bold green]best[/]" if is_best else ""
            value_str = f"{t.value:.6f}" if t.value is not None else "failed"
            table.add_row(str(t.number), value_str, params_str, marker)

        console.clear()
        console.print(table)

    return callback


def _print_results(study):
    """Print final results as a Rich table."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    best = study.best_params
    params_text = "\n".join(
        f"  {k} = {v:.6g}" if isinstance(v, float) else f"  {k} = {v}"
        for k, v in best.items()
    )
    console.print()
    console.print(Panel(
        f"[bold green]Best value:[/] {study.best_value:.6f}\n\n{params_text}",
        title="HPO Results",
        expand=False,
    ))
    console.print()

    table = Table(title="All Trials", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Value", style="cyan", width=12)
    table.add_column("Params")

    for t in sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf")):
        params_str = ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in t.params.items()
        )
        value_str = f"{t.value:.6f}" if t.value is not None else "failed"
        table.add_row(str(t.number), value_str, params_str)

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Optuna HPO for versioned training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interactive", action="store_true", help="Launch interactive TUI mode")
    parser.add_argument("--config", type=Path, default=None, help="Base config JSON")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs")
    parser.add_argument("--study-name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--metric", type=str, default="eval_loss", help="Metric to minimize")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--log-trials", action="store_true", help="Log each trial to MLflow/W&B")
    args = parser.parse_args()

    from mlops.hpo.optuna_sweep import run_optuna_sweep

    if args.interactive:
        sweep_args = _interactive_setup()
        callback = _build_trial_callback()
        study = run_optuna_sweep(
            **sweep_args,
            callbacks=[callback],
        )
        _print_results(study)
        return 0

    # CLI mode
    if not args.config:
        parser.error("--config is required in CLI mode (or use --interactive)")
    with open(args.config, "r") as f:
        base_config = json.load(f)

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
