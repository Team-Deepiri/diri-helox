# deepiri-helox-trace

**Reproducible** trace ML pipeline: ingest PyTorch profiler JSON → fixed-length features →
train/val/test **CSV** splits → optional **sklearn** runtime adapter (`joblib`).

Lives in **diri-helox** so Mudspeed and other services **depend on this package** instead of
duplicating pipeline code.

## Install

From git (pin `revision` for reproducible builds):

```bash
poetry add "deepiri-helox-trace @ git+https://github.com/Team-Deepiri/diri-helox.git#subdirectory=helox_trace_ml&revision=<SHA>"
```

Or in `pyproject.toml`:

```toml
deepiri-helox-trace = { git = "https://github.com/Team-Deepiri/diri-helox.git", rev = "<SHA>", subdirectory = "helox_trace_ml" }
```

Local dev:

```bash
cd helox_trace_ml && poetry install
```

## API (high level)

```python
from pathlib import Path
from deepiri_helox_trace import (
    TraceDatasetPipeline,
    default_data_roots,
    fit_trace_runtime_adapter,
)

roots = default_data_roots(Path("/path/to/your/repo"))
pipe = TraceDatasetPipeline(roots)
pipe.process_raw_json_dir(raw_dir=roots["raw_traces"], out_prefix="trace_tensors")

fit_trace_runtime_adapter(
    roots["processed"] / "trace_tensors_train.csv",
    roots["processed"] / "trace_tensors_val.csv",
    roots["artifacts"] / "trace_runtime_adapter.joblib",
)
```

## Layout (convention)

| Path | Role |
|------|------|
| `data/raw/traces/` | `pytorch_traces_*.json` from your collector |
| `data/processed/trace_ml/` | `*_{train,val,test}.csv` + `*_meta.json` |
| `data/artifacts/` | `joblib` bundles |

## Research notes (Mudspeed / GPU traces)

See the **Mudspeed** repo `docs/TRACE_DATA_PIPELINE.md` for Nsight, ROCm, and design context.
The **implementation** of ingest/featurize/split/train in this package is the shared source of truth.

## Dependencies

- `numpy`, `scikit-learn`, `joblib` (Python **3.10+**)

Install scikit-learn with `pip install scikit-learn`, not the deprecated PyPI name `sklearn`.
