# deepiri-helox-sdk

Installable package shipped from **diri-helox** so other services (Mudspeed, CI jobs,
downstream training pipelines) can depend on Helox-aligned helpers **without** putting
the full monorepo on `PYTHONPATH`.

## Packages

| Module | Purpose |
|--------|---------|
| `deepiri_helox_sdk.device` | Torch device resolution (CUDA → MPS → CPU via `deepiri-gpu-utils`) |
| `deepiri_helox_sdk.evaluation` | **Post-training evaluation harness** — classifier metrics, generation suites, parity, latency benchmarks, regression tracking, model comparison |

## Post-training evaluation harness

Run after training completes to gate promotion, compare checkpoints, and detect regressions.

### Install from git

Pin a revision for reproducible builds:

```bash
poetry add "deepiri-helox-sdk @ git+https://github.com/Team-Deepiri/diri-helox.git#subdirectory=helox_sdk&revision=<GIT_SHA>"
```

Or in `pyproject.toml`:

```toml
deepiri-helox-sdk = { git = "https://github.com/Team-Deepiri/diri-helox.git", rev = "<GIT_SHA>", subdirectory = "helox_sdk" }
```

### Python API

```python
from pathlib import Path
from deepiri_helox_sdk.evaluation import (
    PostTrainingEvalHarness,
    EvalRunConfig,
    EvalThresholds,
)

config = EvalRunConfig(
    model_path=Path("models/intent_classifier"),
    output_dir=Path("evaluation_runs"),
    suite_name="intent_holdout",
    thresholds=EvalThresholds(min_f1=0.75, min_accuracy=0.80),
    run_benchmark=True,
)
harness = PostTrainingEvalHarness(config)
harness.load_suite("intent_holdout", Path("data/eval/intent_holdout.jsonl"))
result = harness.run(mode="classifier")
assert result.passed, result.failures
```

### Generation suites

JSONL rows support `prompt`, `expected`, and `type` (`exact_match`, `contains`, `similarity`, `rouge_l`):

```python
from deepiri_helox_sdk.evaluation import GenerationEvaluator, EvaluationSample

evaluator = GenerationEvaluator(max_new_tokens=64)
samples = [
    EvaluationSample(prompt="Summarize:", expected="short summary", test_type="contains"),
]
report = evaluator.evaluate_callable(lambda p, n: "short summary here", samples)
```

### CLI

```bash
poetry run helox-eval run \
  --model-path models/intent_classifier \
  --suite-path data/eval/intent_holdout.jsonl \
  --min-f1 0.75 \
  --benchmark \
  --fail-on-threshold

poetry run helox-eval compare models/run_a models/run_b
```

### Components

- **ClassifierEvaluator** — accuracy, weighted F1, per-class metrics, confusion matrix
- **GenerationEvaluator** — fixed prompt suites with pluggable backends
- **InferenceParityTester** — train/eval mode, quantization, batch-size parity
- **InferenceBenchmark** — latency (avg, p95) and throughput
- **RegressionTracker** — JSONL history + regression detection vs prior best
- **ModelComparisonReport** — rank checkpoints by quality × throughput

## Monorepo / in-repo usage

If you already run inside diri-helox with `core` importable, prefer:

```python
from core.mudspeed_gpu import resolve_mudspeed_torch_device
```

For evaluation, the monorepo `evaluation/` package mirrors this SDK surface; new work
should import from `deepiri_helox_sdk.evaluation` for portable installs.

## Local development

```bash
cd helox_sdk
poetry install
poetry run pytest
poetry run python -c "from deepiri_helox_sdk import resolve_mudspeed_torch_device; print(resolve_mudspeed_torch_device())"
```

## Dependencies (included)

All libraries required for post-training evaluation are declared in this package:

- `torch`, `transformers`, `numpy`, `scikit-learn`
- `evaluate`, `rouge-score`
- `deepiri-gpu-utils` (device policy)

No separate eval extras — install this package and you get the full harness.
