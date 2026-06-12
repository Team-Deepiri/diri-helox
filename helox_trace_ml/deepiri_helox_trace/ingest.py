"""Load structured trace JSON produced by PyTorch profiler collectors."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def default_data_roots(project_root: Optional[Path] = None) -> Dict[str, Path]:
    """Standard ``data/`` layout under the given project root.

    If ``project_root`` is omitted, uses :func:`pathlib.Path.cwd()` so callers
    (e.g. Mudspeed) should pass their repo root for stable paths in scripts.
    """
    base = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    return {
        "raw_traces": base / "data" / "raw" / "traces",
        "processed": base / "data" / "processed" / "trace_ml",
        "artifacts": base / "data" / "artifacts",
    }


def load_pytorch_trace_json(path: Path) -> Dict[str, Any]:
    """Load a single ``save_traces`` JSON file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_trace_json_paths(root: Path, glob: str = "**/pytorch_traces_*.json") -> List[Path]:
    """Discover trace JSON files under ``root``."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.glob(glob))


def merge_operator_stats(paths: List[Path], source_tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """Concatenate ``operator_stats`` rows from many JSON files."""
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            doc = load_pytorch_trace_json(p)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Skip unreadable trace file %s: %s", p, e)
            continue
        stats = doc.get("operator_stats") or []
        for row in stats:
            out = dict(row)
            out["_source_file"] = str(p)
            if source_tag is not None:
                out["_source_tag"] = source_tag
            rows.append(out)
    return rows
