# Hyperparameter Optimization with Optuna — Concepts and Helox Integration

This guide explains (1) what you need to know about hyperparameter optimization and Optuna, and (2) how the Helox integration works.

---

## Part 1: What You Need to Know

### 1.1 What is hyperparameter optimization (HPO)?

- **Hyperparameters** are settings you choose before training (e.g. learning rate, batch size, number of layers). They are **not** learned from data (unlike model weights).
- **Hyperparameter optimization** is the process of searching over many choices of hyperparameters to find a combination that gives better validation/test performance (e.g. lower loss, higher accuracy).

**Why it matters:** Picking hyperparameters by hand is slow and often suboptimal. HPO automates the search and can find better configs in fewer “manual” tries.

---

### 1.2 Key concepts

| Term | Meaning |
|------|--------|
| **Search space** | The set of hyperparameters to tune and their possible values (e.g. learning_rate in [1e-5, 1e-3]). |
| **Trial** | One complete training run with one specific set of hyperparameters. |
| **Objective** | The metric to optimize (e.g. minimize validation loss or maximize accuracy). Optuna runs trials and uses this metric to decide what to try next. |
| **Sampler** | The strategy for choosing the next hyperparameters (random, grid, or “smart”/Bayesian). |
| **Pruning** | Stopping a trial early if it looks bad (saving time). |
| **Parallel trials** | Running multiple trials at the same time (e.g. `n_jobs=4`). |

---

### 1.3 Optuna — what it is and how it works

**What Optuna is:** A Python library for hyperparameter optimization. You define a search space and an objective function; Optuna runs many trials and uses a **sampler** to pick the next hyperparameters.

**Core ideas:**

1. **Study** = one optimization run (e.g. “tune learning_rate and batch_size for this model”).
2. **Trial** = one evaluation of the objective (one training run with one config).
3. **Suggest API** = you ask Optuna for the next value of each hyperparameter; it returns a value from the search space according to the sampler:
   - `trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)`
   - `trial.suggest_int("batch_size", 4, 64)`
   - `trial.suggest_categorical("optimizer", ["adamw", "adam"])`
4. **Objective function** = a function that takes a `trial`, gets suggested values, runs training with those values, and returns the metric to optimize (e.g. validation loss).
5. **Samplers:**
   - **TPE (Tree-structured Parzen Estimator)** — default; uses past trials to propose promising configs (good when trials are expensive).
   - **Random** — random search.
   - **Grid** — full grid (usually only for very small search spaces).
   - **CmaEs** — evolution strategy; good for continuous spaces.
6. **Pruning** — report intermediate values (e.g. validation loss at step 1000); Optuna can stop bad trials early via a pruner (e.g. `MedianPruner`).
7. **Parallelism** — `study.optimize(objective, n_trials=20, n_jobs=4)` runs up to 4 trials in parallel on one machine. For multiple machines, use a shared database (RDB storage) and multiple workers.

---

### 1.4 Running sweeps in parallel

- **Single machine:** `study.optimize(objective, n_trials=20, n_jobs=4)` runs up to 4 trials in parallel (multiprocessing). Each trial = one full training run.
- **Multiple machines:** Use a shared database (e.g. MySQL/PostgreSQL) as Optuna study storage; start workers that run `study.optimize` with the same storage.
- **Important:** Each trial must be **independent** (no shared in-process state). In Helox, each trial = one training run with one config.

---

## Part 2: How Optuna is Integrated in Helox

### Components

- **`mlops/hpo/objective.py`** — `run_one_trial(config)` runs the versioned training pipeline once and returns the eval metric (e.g. `eval_loss`).
- **`mlops/hpo/optuna_sweep.py`** — `run_optuna_sweep(base_config, n_trials, n_jobs, ...)` creates an Optuna study with TPE sampler and (optionally) parallel trials.
- **`scripts/run_hpo.py`** — CLI to run a sweep: loads base config, runs Optuna, prints best params and best value.

### What gets tuned (default space)

- `learning_rate`, `batch_size`, `weight_decay`, `warmup_steps`, `num_epochs`
- LoRA: `lora_rank`, `lora_alpha`, `lora_dropout`

You can override the space by passing a custom `space` dict to `run_optuna_sweep`.

### Dependencies

- **Optuna** is in `requirements.txt` (`optuna>=3.4.0`). Install with: `pip install -r requirements.txt`.

---

## Quick start

1. **Install:** `pip install -r requirements.txt` (includes optuna).

2. **Base config:** Use a JSON that has everything the versioned pipeline needs; tunable keys will be overridden by Optuna. Example (save as `configs/hpo_base.json`):
   ```json
   {
     "experiment_name": "hpo_sweep",
     "base_model": "mistralai/Mistral-7B-v0.1",
     "use_qlora": true,
     "dataset_spec": "lease_abstraction_training@latest",
     "dataset_type": "lease_abstraction",
     "version_db_url": "sqlite:///dataset_versions.db",
     "storage_backend": "local",
     "output_dir": "./models/hpo_trials",
     "max_length": 512,
     "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
     "logging_steps": 10,
     "save_steps": 500,
     "eval_steps": 500,
     "use_wandb": false,
     "mlflow_uri": "http://localhost:5000"
   }
   ```
   Omit or leave placeholders for: `num_epochs`, `batch_size`, `learning_rate`, `weight_decay`, `warmup_steps`, `lora_rank`, `lora_alpha`, `lora_dropout` — they are tuned by default.

3. **Run from project root (diri-helox):**
   ```bash
   cd diri-helox
   # 8 trials, 2 in parallel
   python scripts/run_hpo.py --config configs/hpo_base.json --n-trials 8 --n-jobs 2
   ```

4. **Output:** Best value and best params are printed; use the best params in your final training config.
