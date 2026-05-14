"""JSON → processed feature CSVs (+ meta)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .features import FEATURE_COLUMNS, operator_rows_to_arrays
from .ingest import list_trace_json_paths, merge_operator_stats

logger = logging.getLogger(__name__)


def _paths_for_meta(paths: List[Path], anchor: Optional[Path]) -> List[str]:
    """Stable, shareable paths: relative to ``anchor`` if set, else basenames only."""
    if anchor is None:
        return [p.name for p in paths]
    anchor_r = anchor.resolve()
    out: List[str] = []
    for p in paths:
        try:
            out.append(str(p.resolve().relative_to(anchor_r)))
        except ValueError:
            out.append(p.name)
    return out


def _output_path_for_meta(path: Path, anchor: Optional[Path]) -> str:
    if anchor is None:
        return path.name
    try:
        return str(path.resolve().relative_to(anchor.resolve()))
    except ValueError:
        return path.name


def _write_xy_csv(path: Path, X_part: np.ndarray, y_part: np.ndarray, header: str) -> None:
    """Write one split: feature columns plus ``target`` column."""
    if X_part.shape[0] == 0:
        table = np.zeros((0, len(FEATURE_COLUMNS) + 1), dtype=np.float64)
    else:
        table = np.hstack([X_part, y_part.reshape(-1, 1)])
    np.savetxt(path, table, delimiter=",", header=header, comments="")


@dataclass
class TraceDatasetPipeline:
    roots: Dict[str, Path]

    def ensure_dirs(self) -> None:
        for p in self.roots.values():
            p.mkdir(parents=True, exist_ok=True)

    def process_raw_json_dir(
        self,
        raw_dir: Optional[Path] = None,
        out_prefix: str = "trace_tensors",
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42,
        meta_path_anchor: Optional[Path] = None,
    ) -> Path:
        """Read all ``pytorch_traces_*.json`` under ``raw_dir``, write split CSVs + ``meta.json``.

        ``meta_path_anchor``: repository (or project) root — source/output paths in
        ``*_meta.json`` are stored **relative** to this anchor so metadata is portable
        across machines (no absolute ``/Users/...`` paths).
        """
        raw_dir = Path(raw_dir or self.roots["raw_traces"])
        paths = list_trace_json_paths(raw_dir)
        rows = merge_operator_stats(paths)
        X, y = operator_rows_to_arrays(rows)

        self.ensure_dirs()
        out_dir = self.roots["processed"]
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_header = ",".join(FEATURE_COLUMNS + ["target"])
        train_path = out_dir / f"{out_prefix}_train.csv"
        val_path = out_dir / f"{out_prefix}_val.csv"
        test_path = out_dir / f"{out_prefix}_test.csv"

        n = X.shape[0]
        if n == 0:
            empty_X = np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float64)
            empty_y = np.zeros((0,), dtype=np.float64)
            _write_xy_csv(train_path, empty_X, empty_y, csv_header)
            _write_xy_csv(val_path, empty_X, empty_y, csv_header)
            _write_xy_csv(test_path, empty_X, empty_y, csv_header)
            meta = {
                "num_rows": 0,
                "source_files": _paths_for_meta(paths, meta_path_anchor),
                "feature_columns": FEATURE_COLUMNS,
                "target": "log1p(cuda_time_per_call_us)",
                "format": "csv",
                "output": _output_path_for_meta(train_path, meta_path_anchor),
                "outputs": {
                    "train": _output_path_for_meta(train_path, meta_path_anchor),
                    "val": _output_path_for_meta(val_path, meta_path_anchor),
                    "test": _output_path_for_meta(test_path, meta_path_anchor),
                },
            }
            (out_dir / f"{out_prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.warning("No operator rows found; wrote empty CSV splits under %s", out_dir)
            return train_path

        if n == 1:
            X_train, y_train = X, y
            X_val, y_val = X, y
            X_test, y_test = X, y
        else:
            try:
                from sklearn.model_selection import train_test_split
            except ImportError as e:
                raise ImportError(
                    "The process step requires scikit-learn. Install with: pip install scikit-learn"
                ) from e
            X_tv, X_test, y_tv, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )
            if X_tv.shape[0] <= 1:
                X_train, y_train = X_tv, y_tv
                X_val, y_val = X_tv, y_tv
            else:
                val_rel = val_size / max(1e-9, (1.0 - test_size))
                val_rel = min(max(val_rel, 0.1), 0.9)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_tv, y_tv, test_size=val_rel, random_state=seed
                )

        _write_xy_csv(train_path, X_train, y_train, csv_header)
        _write_xy_csv(val_path, X_val, y_val, csv_header)
        _write_xy_csv(test_path, X_test, y_test, csv_header)

        meta: Dict[str, Any] = {
            "num_rows": int(n),
            "train": int(X_train.shape[0]),
            "val": int(X_val.shape[0]),
            "test": int(X_test.shape[0]),
            "source_files": _paths_for_meta(paths, meta_path_anchor),
            "feature_columns": FEATURE_COLUMNS,
            "target": "log1p(cuda_time_per_call_us)",
            "format": "csv",
            "output": _output_path_for_meta(train_path, meta_path_anchor),
            "outputs": {
                "train": _output_path_for_meta(train_path, meta_path_anchor),
                "val": _output_path_for_meta(val_path, meta_path_anchor),
                "test": _output_path_for_meta(test_path, meta_path_anchor),
            },
            "seed": seed,
        }
        (out_dir / f"{out_prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Wrote dataset CSVs under %s (%d rows)", out_dir, n)
        return train_path
