"""Load split CSVs and fit a small sklearn runtime adapter (reproducible recipe)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def load_trace_split_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load train/val CSV written by :meth:`TraceDatasetPipeline.process_raw_json_dir`."""
    n_feat = len(FEATURE_COLUMNS)
    path = Path(path)
    if not path.exists():
        return np.zeros((0, n_feat), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= 1:
        return np.zeros((0, n_feat), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64)
    if raw.size == 0:
        return np.zeros((0, n_feat), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] != n_feat + 1:
        raise ValueError(f"Expected {n_feat + 1} columns in {path}, got {raw.shape[1]}")
    return raw[:, :-1], raw[:, -1].ravel()


def fit_trace_runtime_adapter(
    train_csv: Path,
    val_csv: Path,
    artifact_path: Path,
    *,
    hidden_layer_sizes: Sequence[int] = (64, 32),
    max_iter: int = 500,
    learning_rate_init: float = 1e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fit StandardScaler + MLPRegressor and save joblib bundle. Returns metrics dict."""
    train_csv = Path(train_csv)
    val_csv = Path(val_csv)
    artifact_path = Path(artifact_path)

    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    X_train, y_train = load_trace_split_csv(train_csv)
    X_val, y_val = load_trace_split_csv(val_csv)

    use_early_stopping = X_train.shape[0] >= 32
    pipe: Pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=tuple(hidden_layer_sizes),
                    max_iter=max_iter,
                    random_state=seed,
                    early_stopping=use_early_stopping,
                    validation_fraction=0.15 if use_early_stopping else 0.0,
                    n_iter_no_change=20,
                    learning_rate_init=learning_rate_init,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    train_mae = float(np.mean(np.abs(pipe.predict(X_train) - y_train)))
    val_mae = float(np.mean(np.abs(pipe.predict(X_val) - y_val)))

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "sklearn_pipeline": pipe,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "target": "log1p(cuda_time_per_call_us)",
    }
    joblib.dump(bundle, artifact_path)
    logger.info("Saved adapter to %s (train_mae=%.4f val_mae=%.4f)", artifact_path, train_mae, val_mae)
    return bundle
