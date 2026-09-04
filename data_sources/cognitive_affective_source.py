"""
CognitiveAffectiveDataSource: consumes structured Cognitive-Affective
Resolution Trace (CART) records — see
docs/COGNITIVE_AFFECTIVE_RESOLUTION_TRACES.md for the schema and the capture
pipeline this depends on.

This source only reads already-structured JSONL records. Capture tooling
(bot triggers, debrief prompts, consent flow) lives in deepiri-control-plane,
not here — this adapter is intentionally a thin, dependency-free reader so it
can plug into `composite_source.py` alongside `self_feedback_source.py` and
other existing sources once real trace volume exists.

Scaffold only: field validation and anonymization enforcement are open
design questions (see doc, section 5) and are not implemented yet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .base import DataSample, DataSource, DataSourceConfig

logger = logging.getLogger(__name__)


class CognitiveAffectiveDataSource(DataSource):
    """
    Loads Cognitive-Affective Resolution Trace (CART) records as training
    samples.

    Config params:
        trace_log_path (str): path to JSONL file of CART records
                               (see docs/COGNITIVE_AFFECTIVE_RESOLUTION_TRACES.md)
        domain_filter  (list[str] | None): restrict to specific domains
        max_samples    (int | None): cap on samples loaded
    """

    def __init__(self, config: DataSourceConfig) -> None:
        super().__init__(config)
        self._path = Path(config.params["trace_log_path"])
        self._domain_filter: Optional[List[str]] = config.params.get("domain_filter")
        self._max_samples: Optional[int] = config.params.get("max_samples")

    def stream(self) -> Iterator[DataSample]:
        if not self._path.exists():
            logger.warning("CART trace log not found: %s", self._path)
            return

        count = 0
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record: Dict[str, Any] = json.loads(line)

                if self._domain_filter and record.get("domain") not in self._domain_filter:
                    continue

                yield DataSample(
                    text=json.dumps(record, ensure_ascii=False),
                    label=None,
                    label_name=record.get("domain"),
                    metadata={
                        "trace_id": record.get("trace_id"),
                        "domain": record.get("domain"),
                        "source_ref": record.get("source_ref"),
                    },
                    source="cognitive_affective",
                )

                count += 1
                if self._max_samples is not None and count >= self._max_samples:
                    return

    def get_info(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "name": self.name,
            "trace_log_path": str(self._path),
            "domain_filter": self._domain_filter,
            "status": "scaffold — schema/capture pipeline pending design review",
        }
