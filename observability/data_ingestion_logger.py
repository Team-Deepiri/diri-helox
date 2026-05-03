"""
DataIngestionLogger: records what happens to data at each pipeline stage.

Logs sample counts, per-source breakdown, quality distribution, and label
distribution as data flows through ingest -> preprocess -> split.

Can write a JSONL log file for later analysis.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_sources.base import DataSample


class DataIngestionLogger:
    """
    Tracks data through pipeline stages and prints a summary table.

    Usage:
        logger = DataIngestionLogger(log_path="logs/ingestion.jsonl")
        logger.record("ingest", samples)
        logger.record("preprocess", samples)
        logger.record("split_train", train_samples)
        logger.print_summary()
    """

    def __init__(self, log_path: Optional[str] = None, pipeline_name: str = "pipeline") -> None:
        self._pipeline_name = pipeline_name
        self._log_path = Path(log_path) if log_path else None
        self._stages: List[Dict[str, Any]] = []
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def record(self, stage: str, samples: List[DataSample]) -> None:
        """Record samples at a pipeline stage."""
        entry = self._build_entry(stage, samples)
        self._stages.append(entry)
        self._print_stage(entry)
        if self._log_path:
            self._append_jsonl(entry)

    def _build_entry(self, stage: str, samples: List[DataSample]) -> Dict[str, Any]:
        # Per-source breakdown
        source_counts: Dict[str, int] = {}
        stream_counts: Dict[str, int] = {}
        quality_scores: List[float] = []
        label_counts: Dict[str, int] = {}

        for s in samples:
            source_counts[s.source] = source_counts.get(s.source, 0) + 1

            stream = s.metadata.get("source_stream")
            if stream:
                stream_counts[stream] = stream_counts.get(stream, 0) + 1

            quality = s.metadata.get("quality_score")
            if quality is not None:
                quality_scores.append(float(quality))

            label = s.label_name or (str(s.label) if s.label is not None else None)
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1

        return {
            "run_id": self._run_id,
            "pipeline": self._pipeline_name,
            "stage": stage,
            "timestamp": time.time(),
            "count": len(samples),
            "sources": source_counts,
            "stream_types": stream_counts,
            "quality": {
                "min": round(min(quality_scores), 3) if quality_scores else None,
                "max": round(max(quality_scores), 3) if quality_scores else None,
                "mean": (
                    round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else None
                ),
            },
            "unique_labels": len(label_counts),
            "label_counts": dict(sorted(label_counts.items(), key=lambda x: -x[1])[:10]),
        }

    def _print_stage(self, entry: Dict[str, Any]) -> None:
        stage = entry["stage"]
        count = entry["count"]
        stream_info = ""
        if entry["stream_types"]:
            parts = [f"{k}: {v}" for k, v in entry["stream_types"].items()]
            stream_info = f"  [{', '.join(parts)}]"
        quality = entry["quality"]
        quality_info = ""
        if quality["mean"] is not None:
            quality_info = f"  quality avg={quality['mean']:.2f}"

        print(f"  [{stage}] {count} samples{stream_info}{quality_info}")

    def print_summary(self) -> None:
        if not self._stages:
            return
        print("\n  --- Ingestion Summary ---")
        for entry in self._stages:
            count = entry["count"]
            stage = entry["stage"].ljust(16)
            labels = entry["unique_labels"]
            print(f"  {stage}: {count:>5} samples  ({labels} unique labels)")
        if self._log_path:
            print(f"  Log written to: {self._log_path}")

    def _append_jsonl(self, entry: Dict[str, Any]) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
