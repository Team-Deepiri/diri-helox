"""Turn operator-level profiler rows into fixed-length vectors for small models.

**Provenance (what each column means)** — all computed in this module from PyTorch
Kineto-style ``operator_stats`` rows (see ``row_to_features`` / ``FEATURE_COLUMNS``):

- ``name_hash``: SHA-256 of the operator name, reduced to a stable float in ``[0, 1)`` (not semantic, but repeatable).
- ``log1p_*``: log1p of counts and CPU/CUDA time totals and per-call CUDA time (targets often use per-call).
- ``shape_rank_*``, ``shape_prod_*``, ``shape_maxdim_*``: three input-shape slots; rank, log1p(product dims), log1p(max dim) or zeros if missing.
- ``log1p_shape_slots``: how many non-empty shape slots were present.
- ``zero_pad``: structural constant (reserved for future flags).

For kernel *clustering* (see Mudspeed ``kernel_cluster``), tune ``num_clusters`` / algorithm against
trace diversity; featurization here is regression-oriented, not identical to cluster features.

**Scalability:** each operator row maps to a **fixed-length** vector (constant width); cost is
**O(number of input shape slots)** (three slots) per row, not O(sequence length), so batch
featurization scales linearly with row count for typical profiler exports.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Tuple

import numpy as np

FEATURE_COLUMNS = [
    "name_hash",
    "log1p_count",
    "log1p_cuda_total_us",
    "log1p_cpu_total_us",
    "log1p_cuda_per_call_us",
    "shape_rank_0",
    "shape_prod_0",
    "shape_maxdim_0",
    "shape_rank_1",
    "shape_prod_1",
    "shape_maxdim_1",
    "shape_rank_2",
    "shape_prod_2",
    "shape_maxdim_2",
    "log1p_shape_slots",
    "zero_pad",
]


def _stable_name_hash(name: str) -> float:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    h = int(digest, 16)
    return (h % 1000003) / 1000003.0


def _shape_feats(input_shapes: Any, slot: int) -> Tuple[float, float, float]:
    if not input_shapes or slot >= len(input_shapes):
        return 0.0, 0.0, 0.0
    sh = input_shapes[slot]
    if not sh:
        return 0.0, 0.0, 0.0
    rank = float(len(sh))
    vals = [float(x) for x in sh if x is not None and x >= 0]
    if not vals:
        return rank, 0.0, 0.0
    prod = float(math.prod(vals)) if vals else 0.0
    mx = float(max(vals))
    return rank, math.log1p(prod), math.log1p(mx)


def row_to_features(row: Dict[str, Any]) -> np.ndarray:
    name = str(row.get("name", ""))
    count = float(row.get("count") or 0.0)
    cuda_total = float(row.get("cuda_time_total_us") or 0.0)
    cpu_total = float(row.get("cpu_time_total_us") or 0.0)
    cuda_per = float(row.get("cuda_time_per_call_us") or 0.0)
    shapes = row.get("input_shapes") or []

    f0 = _shape_feats(shapes, 0)
    f1 = _shape_feats(shapes, 1)
    f2 = _shape_feats(shapes, 2)
    n_slots = float(min(len(shapes), 3))

    return np.array(
        [
            _stable_name_hash(name),
            math.log1p(max(count, 0.0)),
            math.log1p(max(cuda_total, 0.0)),
            math.log1p(max(cpu_total, 0.0)),
            math.log1p(max(cuda_per, 0.0)),
            f0[0],
            f0[1],
            f0[2],
            f1[0],
            f1[1],
            f1[2],
            f2[0],
            f2[1],
            f2[2],
            math.log1p(n_slots),
            0.0,
        ],
        dtype=np.float64,
    )


def operator_rows_to_arrays(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``X`` (features) and ``y`` (log1p CUDA µs per call) for regression."""
    if not rows:
        return np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    X = np.stack([row_to_features(r) for r in rows], axis=0)
    y = np.array(
        [math.log1p(float(r.get("cuda_time_per_call_us") or 0.0)) for r in rows],
        dtype=np.float64,
    )
    return X, y
