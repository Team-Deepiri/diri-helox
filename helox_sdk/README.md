# deepiri-helox-sdk

Small **installable** package shipped from the **diri-helox** repository so other
services (for example **Mudspeed**) can depend on Helox-aligned helpers via Poetry
or pip **without** adding the full training repo to `PYTHONPATH`.

## Contents

- `resolve_mudspeed_torch_device` — same device policy as `core.gpu_utils` /
  `deepiri-gpu-utils` (CUDA → MPS → CPU, etc.).

## Monorepo / in-repo usage

If you already run inside diri-helox with `core` importable, prefer:

```python
from core.mudspeed_gpu import resolve_mudspeed_torch_device
```

## Install from git (Mudspeed and others)

After this path exists on your chosen branch, pin a revision for reproducible builds:

```bash
poetry add "deepiri-helox-sdk @ git+https://github.com/Team-Deepiri/diri-helox.git#subdirectory=helox_sdk&revision=<GIT_SHA>"
```

Or add to `pyproject.toml`:

```toml
deepiri-helox-sdk = { git = "https://github.com/Team-Deepiri/diri-helox.git", rev = "<GIT_SHA>", subdirectory = "helox_sdk" }
```

Then run `poetry lock`.

## Local development

From this directory:

```bash
poetry install
poetry run python -c "from deepiri_helox_sdk import resolve_mudspeed_torch_device; print(resolve_mudspeed_torch_device())"
```
